//! Export what the rest of the engine reads: an evaluation set, and the text to train on.
//!
//! The two exports are one module because they are two halves of one split.
//! Every sentence drawn into the evaluation set must be absent from the training
//! text, or the baseline is scored on sentences it memorised, so the exclusion
//! list and the evaluation draw have to agree on what a sentence *is* -- the exact
//! target string, after the corpus normaliser has already run.
//!
//! Three constraints shape what is eligible. The line's `pinyin` field is what a
//! user *types*, so it is the toneless syllables run together with no separator --
//! the harness re-segments it, which is half of what is being evaluated. Only
//! sentences both annotators agreed on can appear, because a disputed reading
//! would score a correct conversion as wrong. And the target must be entirely
//! Han: a comma or a digit inside it has no keystrokes behind it, so it would
//! make the syllable count disagree with the character count.
//!
//! An exclusion that matches nothing is an error rather than a warning: the
//! held-out sentences came out of these very shards, so a miss means the wrong
//! directory or a normaliser that has moved under the file, and both of those
//! silently put the evaluation sentences back into training.

use crate::annotate::{AnnotatedRow, Sample};
use crate::error::{Error, Result};
use crate::shards::{read_frame, shard_paths};
use crate::text::{han_characters, toneless};
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::io::{BufWriter, Write as _};
use std::path::Path;
use tracing::info;

/// One line of the evaluation file: what was typed, what it should become.
///
/// The field order is the order `ime-eval` reads them in, and `context` is
/// written even when it is null so that a record without context is visibly a
/// record without context rather than a record from an older format.
#[derive(Clone, PartialEq, Eq, Debug, Serialize)]
pub struct EvalItem {
    /// The keystrokes: toneless syllables run together, with no separator.
    pub pinyin: String,
    /// What the engine should produce from them.
    pub text: String,
    /// What was already on screen, if anything.
    pub context: Option<String>,
}

/// The one field an exclusion file has to carry, whatever else it holds.
///
/// Both the evaluation set and the pool it was drawn from are JSON Lines with a
/// `text` field, so either can be handed to `--exclude` unchanged.
#[derive(Debug, Deserialize)]
struct HeldOutLine {
    text: String,
}

/// Rows both annotators agreed on whose target is Han characters only.
#[must_use]
pub fn eligible(annotated: &[AnnotatedRow]) -> Vec<&AnnotatedRow> {
    annotated
        .iter()
        .filter(|row| row.agree_all && row.text.chars().count() == row.characters.len())
        .collect()
}

/// Build one evaluation line, checking the syllables really do cover the target.
///
/// # Errors
///
/// If the row's syllables, its Han characters and its target do not all have the
/// same length, which means the row should never have been eligible.
pub fn to_item(row: &AnnotatedRow) -> Result<EvalItem> {
    let characters = han_characters(&row.text);
    let length = row.text.chars().count();
    if row.g2pw.len() != characters.len() || characters.len() != length {
        return Err(Error::Invariant(format!(
            "{:?} has {length} characters, {} of them Han, against {} syllables",
            row.text,
            characters.len(),
            row.g2pw.len()
        )));
    }
    Ok(EvalItem {
        pinyin: row.g2pw.iter().map(|syllable| toneless(syllable)).collect(),
        text: row.text.clone(),
        context: row.context.clone(),
    })
}

/// Split `size` as evenly as possible across sources, spilling any shortfall.
///
/// A source that cannot fill its share does not shrink the export; its unused
/// quota is offered to the sources that can, so a small dialogue slice never
/// silently caps the whole evaluation set.
///
/// # Errors
///
/// If the sources together cannot supply `size` rows.
pub fn allocate(
    available: &BTreeMap<String, usize>,
    size: usize,
) -> Result<BTreeMap<String, usize>> {
    let total: usize = available.values().sum();
    if total < size {
        return Err(Error::NotEnough {
            available: total,
            sources: available.len(),
            wanted: size,
        });
    }
    let mut quotas: BTreeMap<String, usize> =
        available.keys().map(|name| (name.clone(), 0)).collect();
    let mut remaining = size;
    let mut open: HashSet<&String> = available
        .iter()
        .filter(|(_, count)| **count > 0)
        .map(|(name, _)| name)
        .collect();
    while remaining > 0 && !open.is_empty() {
        let share = (remaining / open.len()).max(1);
        let mut exhausted = Vec::new();
        for name in available.keys() {
            if !open.contains(name) {
                continue;
            }
            if remaining == 0 {
                break;
            }
            let quota = quotas.entry(name.clone()).or_insert(0);
            let take = share.min(available[name] - *quota).min(remaining);
            *quota += take;
            remaining -= take;
            if *quota == available[name] {
                exhausted.push(name);
            }
        }
        for name in exhausted {
            open.remove(name);
        }
    }
    Ok(quotas)
}

/// A row the stratified draw can see: it has an identity and a source.
///
/// Both the prepared samples and the annotated rows are drawn the same way, and
/// the two draws have to stay identical -- the evaluation set is a subset of the
/// pool, so a pool drawn by one rule and an evaluation set drawn by another would
/// silently stop being nested.
pub trait Stratified {
    /// The row's stable identifier, which the draw sorts on before it samples.
    fn id(&self) -> &str;
    /// Which corpus the row came from, which the draw stratifies across.
    fn source(&self) -> &str;
}

impl Stratified for Sample {
    fn id(&self) -> &str {
        &self.id
    }

    fn source(&self) -> &str {
        &self.source
    }
}

impl Stratified for AnnotatedRow {
    fn id(&self) -> &str {
        &self.id
    }

    fn source(&self) -> &str {
        &self.source
    }
}

/// Deterministically draw `size` rows, stratified across the `source` column.
///
/// Rows are sorted by id before the draw so that the shard order they arrived in
/// -- which is whatever the filesystem listed -- cannot change what a seed selects.
///
/// # Errors
///
/// If the eligible rows cannot supply `size`.
pub fn sample_rows<'a, R: Stratified>(
    rows: &[&'a R],
    size: usize,
    seed: u64,
) -> Result<Vec<&'a R>> {
    let mut sorted: Vec<&R> = rows.to_vec();
    sorted.sort_by(|left, right| left.id().cmp(right.id()));

    let mut by_source: BTreeMap<String, Vec<&R>> = BTreeMap::new();
    for row in sorted {
        by_source
            .entry(row.source().to_owned())
            .or_default()
            .push(row);
    }
    let available: BTreeMap<String, usize> = by_source
        .iter()
        .map(|(name, group)| (name.clone(), group.len()))
        .collect();
    let quotas = allocate(&available, size)?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut drawn = Vec::with_capacity(size);
    for (name, group) in &by_source {
        let quota = quotas.get(name).copied().unwrap_or(0);
        let mut indices: Vec<usize> =
            rand::seq::index::sample(&mut rng, group.len(), quota).into_vec();
        indices.sort_unstable();
        drawn.extend(indices.into_iter().map(|index| group[index]));
    }
    Ok(drawn)
}

/// Write the sampled evaluation set to `out_path` as JSON Lines.
///
/// # Errors
///
/// If too few rows are eligible, or the file cannot be written.
pub fn export_eval_set(
    annotated: &[AnnotatedRow],
    out_path: &Path,
    size: usize,
    seed: u64,
) -> Result<usize> {
    let rows = eligible(annotated);
    info!(
        annotated = annotated.len(),
        eligible = rows.len(),
        "eligible for evaluation"
    );
    let selected = sample_rows(&rows, size, seed)?;
    let mut handle = create(out_path)?;
    for row in &selected {
        let item = to_item(row)?;
        let line = serde_json::to_string(&item).map_err(|error| {
            Error::Invariant(format!("an evaluation item would not serialise: {error}"))
        })?;
        writeln!(handle, "{line}").map_err(|source| Error::Read {
            path: out_path.to_owned(),
            source,
        })?;
    }
    handle.flush().map_err(|source| Error::Read {
        path: out_path.to_owned(),
        source,
    })?;
    let mut by_source: BTreeMap<&str, usize> = BTreeMap::new();
    for row in &selected {
        *by_source.entry(row.source.as_str()).or_insert(0) += 1;
    }
    info!(
        path = %out_path.display(),
        items = selected.len(),
        by_source = ?by_source,
        "evaluation set written"
    );
    Ok(selected.len())
}

/// One line of a pool file: the sentence, and nothing that would date it.
///
/// The pool is what the annotation run reads and what `--exclude` is later handed,
/// and both of those only ever look at `text`, so that is all it carries.
#[derive(Clone, PartialEq, Eq, Debug, Serialize)]
pub struct PoolLine {
    /// The target sentence.
    pub text: String,
}

/// How a pool draw came out, per source.
pub type PoolCounts = BTreeMap<String, usize>;

/// Draw a seeded, source-stratified pool out of the prepared samples.
///
/// A pool is a run's whole working set: the samples are written back out as
/// shards under `out_dir/samples`, so every later stage can be pointed at
/// `out_dir` and see a self-contained data root, and `out_dir/pool.jsonl` lists
/// the same sentences in the shape `--exclude` reads. Annotating a run means
/// annotating a pool, and this is what makes a run reproducible from a seed
/// rather than from a scratch script.
///
/// # Errors
///
/// If the samples directory holds no shards, if fewer than `size` samples exist,
/// or if the pool cannot be written.
pub fn export_pool(
    samples_dir: &Path,
    out_dir: &Path,
    size: usize,
    seed: u64,
) -> Result<PoolCounts> {
    let samples = crate::annotate::read_samples(samples_dir)?;
    let borrowed: Vec<&Sample> = samples.iter().collect();
    let drawn = sample_rows(&borrowed, size, seed)?;

    let mut by_source: BTreeMap<String, Vec<Sample>> = BTreeMap::new();
    for sample in &drawn {
        by_source
            .entry(sample.source.clone())
            .or_default()
            .push((*sample).clone());
    }

    let shard_dir = out_dir.join("samples");
    for (source, rows) in &by_source {
        let mut writer = crate::shards::ShardWriter::new(&shard_dir, source, usize::MAX)?;
        for row in rows {
            writer.write(row.clone())?;
        }
        writer.finish()?;
    }

    let pool_path = out_dir.join("pool.jsonl");
    let mut handle = create(&pool_path)?;
    for sample in &drawn {
        let line = serde_json::to_string(&PoolLine {
            text: sample.text.clone(),
        })
        .map_err(|error| Error::Invariant(format!("a pool line would not serialise: {error}")))?;
        writeln!(handle, "{line}").map_err(|source| Error::Read {
            path: pool_path.clone(),
            source,
        })?;
    }
    handle.flush().map_err(|source| Error::Read {
        path: pool_path.clone(),
        source,
    })?;

    let counts: PoolCounts = by_source
        .iter()
        .map(|(source, rows)| (source.clone(), rows.len()))
        .collect();
    info!(
        path = %pool_path.display(),
        shards = %shard_dir.display(),
        drawn = drawn.len(),
        by_source = ?counts,
        "pool drawn"
    );
    Ok(counts)
}

/// The target sentences held out of training, from the `text` field of each line.
///
/// # Errors
///
/// If a file is empty -- drop the flag rather than passing it -- or if a line has
/// no usable `text` field.
pub fn read_exclusions(paths: &[std::path::PathBuf]) -> Result<HashSet<String>> {
    let mut held_out = HashSet::new();
    for path in paths {
        let raw = std::fs::read_to_string(path).map_err(|source| Error::Read {
            path: path.clone(),
            source,
        })?;
        let lines: Vec<&str> = raw.lines().collect();
        if lines.is_empty() {
            return Err(Error::Invariant(format!(
                "{} holds no exclusions; drop the flag rather than passing it",
                path.display()
            )));
        }
        for (index, line) in lines.iter().enumerate() {
            let parsed: HeldOutLine =
                serde_json::from_str(line).map_err(|source| Error::JsonLine {
                    path: path.clone(),
                    line: index + 1,
                    source,
                })?;
            held_out.insert(parsed.text);
        }
    }
    info!(
        files = paths.len(),
        sentences = held_out.len(),
        "exclusions loaded"
    );
    Ok(held_out)
}

/// Write every prepared target except `held_out` to `out_path`, one per line.
///
/// Shards are read one at a time: the corpus this feeds is millions of lines and
/// the whole point of the parquet shards is that no stage has to hold them all.
///
/// # Errors
///
/// If the samples directory holds no shards, if one cannot be read, or if any
/// held-out sentence is not among them.
pub fn export_ngram_corpus<H: std::hash::BuildHasher>(
    samples_dir: &Path,
    out_path: &Path,
    held_out: &HashSet<String, H>,
) -> Result<usize> {
    let paths = shard_paths(samples_dir, "*")?;
    if paths.is_empty() {
        return Err(Error::Missing {
            what: "sample shards",
            path: samples_dir.to_owned(),
            hint: "run the corpus preparation stage first",
        });
    }
    let mut unmatched: HashSet<&str> = held_out.iter().map(String::as_str).collect();
    let mut written = 0;
    let mut excluded = 0;
    let mut handle = create(out_path)?;
    for path in &paths {
        let frame = read_frame(path)?;
        let column = frame.column("text")?;
        for value in column.str()?.iter() {
            let text = value.ok_or_else(|| {
                Error::Invariant(format!("{} holds a null target", path.display()))
            })?;
            if held_out.contains(text) {
                unmatched.remove(text);
                excluded += 1;
                continue;
            }
            writeln!(handle, "{text}").map_err(|source| Error::Read {
                path: out_path.to_owned(),
                source,
            })?;
            written += 1;
        }
    }
    handle.flush().map_err(|source| Error::Read {
        path: out_path.to_owned(),
        source,
    })?;
    if !unmatched.is_empty() {
        let mut examples: Vec<String> = unmatched.iter().map(|text| (*text).to_owned()).collect();
        examples.sort();
        examples.truncate(3);
        return Err(Error::HeldOutMissing {
            missing: unmatched.len(),
            held_out: held_out.len(),
            examples,
        });
    }
    info!(
        path = %out_path.display(),
        lines = written,
        excluded,
        shards = paths.len(),
        "n-gram corpus written"
    );
    Ok(written)
}

fn create(path: &Path) -> Result<BufWriter<std::fs::File>> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|source| Error::Read {
            path: parent.to_owned(),
            source,
        })?;
    }
    let file = std::fs::File::create(path).map_err(|source| Error::Read {
        path: path.to_owned(),
        source,
    })?;
    Ok(BufWriter::new(file))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        id: &str,
        source: &str,
        text: &str,
        syllables: &[&str],
        agree_all: bool,
    ) -> AnnotatedRow {
        AnnotatedRow {
            id: id.to_owned(),
            source: source.to_owned(),
            text: text.to_owned(),
            context: None,
            characters: han_characters(text)
                .iter()
                .map(ToString::to_string)
                .collect(),
            g2pw: syllables.iter().map(|s| (*s).to_owned()).collect(),
            llm: syllables.iter().map(|s| (*s).to_owned()).collect(),
            agree: syllables.iter().map(|_| agree_all).collect(),
            agree_all,
        }
    }

    #[test]
    fn only_agreed_all_han_rows_are_eligible() {
        let rows = vec![
            row("a", "wiki", "中国", &["zhong1", "guo2"], true),
            row("b", "wiki", "中，国", &["zhong1", "guo2"], true),
            row("c", "wiki", "重要", &["chong2", "yao4"], false),
        ];
        let kept = eligible(&rows);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].id, "a");
    }

    #[test]
    fn the_typed_pinyin_is_the_toneless_syllables_run_together() {
        let item = to_item(&row("a", "wiki", "绿色", &["lü4", "se4"], true)).expect("aligned");
        assert_eq!(item.pinyin, "lvse");
        assert_eq!(item.text, "绿色");
        assert_eq!(item.context, None);
        assert_eq!(
            serde_json::to_string(&item).expect("serialises"),
            r#"{"pinyin":"lvse","text":"绿色","context":null}"#
        );
    }

    #[test]
    fn a_shortfall_in_one_source_spills_onto_the_others() {
        let available = BTreeMap::from([
            ("dialogue".to_owned(), 2),
            ("news".to_owned(), 100),
            ("wiki".to_owned(), 100),
        ]);
        let quotas = allocate(&available, 60).expect("60 fit");
        assert_eq!(quotas.values().sum::<usize>(), 60);
        assert_eq!(quotas["dialogue"], 2);
        assert!(quotas["news"] > 20 && quotas["wiki"] > 20);
    }

    #[test]
    fn asking_for_more_than_exists_is_an_error_not_a_short_file() {
        let available = BTreeMap::from([("wiki".to_owned(), 3)]);
        let error = allocate(&available, 4).expect_err("only three exist");
        assert!(matches!(
            error,
            Error::NotEnough {
                available: 3,
                wanted: 4,
                ..
            }
        ));
    }

    #[test]
    fn a_pool_draw_writes_shards_and_a_jsonl_of_the_same_sentences() {
        let root = std::env::temp_dir().join(format!("ime-g2p-pool-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let samples_dir = root.join("samples");
        let mut writer = crate::shards::ShardWriter::new(&samples_dir, "wiki", 100)
            .expect("the fixture directory is writable");
        for index in 0..20 {
            writer
                .write(Sample {
                    id: format!("w{index:03}"),
                    source: "wiki".to_owned(),
                    text: format!("中国{index}"),
                    context: None,
                })
                .expect("the row is buffered");
        }
        writer.finish().expect("the shard is written");
        let mut writer = crate::shards::ShardWriter::new(&samples_dir, "news", 100)
            .expect("the fixture directory is writable");
        for index in 0..20 {
            writer
                .write(Sample {
                    id: format!("n{index:03}"),
                    source: "news".to_owned(),
                    text: format!("重要{index}"),
                    context: Some("前文".to_owned()),
                })
                .expect("the row is buffered");
        }
        writer.finish().expect("the shard is written");

        let out = root.join("run");
        let counts = export_pool(&samples_dir, &out, 10, 3).expect("ten fit");
        assert_eq!(counts["wiki"], 5);
        assert_eq!(counts["news"], 5);

        let pool = std::fs::read_to_string(out.join("pool.jsonl")).expect("the pool is written");
        assert_eq!(pool.lines().count(), 10);
        let drawn: Vec<String> = pool
            .lines()
            .map(|line| {
                serde_json::from_str::<HeldOutLine>(line)
                    .expect("every line carries a text field")
                    .text
            })
            .collect();

        let written: Vec<Sample> =
            crate::annotate::read_samples(&out.join("samples")).expect("the shards read back");
        assert_eq!(written.len(), 10);
        let mut shard_texts: Vec<String> = written.iter().map(|row| row.text.clone()).collect();
        let mut pool_texts = drawn.clone();
        shard_texts.sort();
        pool_texts.sort();
        assert_eq!(shard_texts, pool_texts);
        assert!(written.iter().any(|row| row.context.is_some()));

        let repeat = export_pool(&samples_dir, &root.join("again"), 10, 3).expect("ten fit");
        let again = std::fs::read_to_string(root.join("again").join("pool.jsonl"))
            .expect("the pool is written");
        assert_eq!(repeat, counts);
        assert_eq!(again, pool);

        std::fs::remove_dir_all(&root).expect("the fixture directory is removed");
    }

    #[test]
    fn the_same_seed_draws_the_same_rows() {
        let rows: Vec<AnnotatedRow> = (0..50)
            .map(|index| {
                row(
                    &format!("{index:03}"),
                    if index % 2 == 0 { "wiki" } else { "news" },
                    "中国",
                    &["zhong1", "guo2"],
                    true,
                )
            })
            .collect();
        let borrowed = eligible(&rows);
        let first = sample_rows(&borrowed, 10, 7).expect("ten fit");
        let second = sample_rows(&borrowed, 10, 7).expect("ten fit");
        let third = sample_rows(&borrowed, 10, 8).expect("ten fit");
        let ids = |drawn: &[&AnnotatedRow]| -> Vec<String> {
            drawn.iter().map(|row| row.id.clone()).collect()
        };
        assert_eq!(ids(&first), ids(&second));
        assert_ne!(ids(&first), ids(&third));
        assert_eq!(first.len(), 10);
        assert_eq!(first.iter().filter(|row| row.source == "wiki").count(), 5);
    }
}
