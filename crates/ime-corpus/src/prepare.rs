//! The offline half: turn raw documents into filtered, deduplicated samples.
//!
//! This is where every rule that might change lives -- cleaning, normalisation,
//! splitting a document into units, cutting those units into the runs a person
//! types in one go, the length filter, the duplicate check -- so that tightening
//! any of them costs a minute of local work rather than another pass over the
//! network.
//!
//! A document becomes samples in two cuts. The first is the *unit*: a sentence
//! for prose, a whole turn for dialogue, which is the span a writer composed
//! before moving on. The second is the *typing segment*: inside a unit, every
//! maximal run of Han characters, because a run is what one press of the
//! conversion key produces and the punctuation between two runs is typed with a
//! key of its own. Everything before a run inside its unit -- earlier runs,
//! punctuation, a stretch of Latin -- lands in that sample's context instead,
//! which is what the writer's screen actually held when they started typing it.
//!
//! The work splits cleanly in two, and the split is what makes it fast. Cleaning
//! and normalising a document is pure and expensive (the traditional-to-simplified
//! conversion dominates), so it runs across every core with rayon. Filtering is
//! cheap but *stateful*: the duplicate check is a set that every candidate in the
//! source has to be checked against in a fixed order, or two runs of the same
//! input would keep different segments. So one shard's documents are split in
//! parallel and then filtered, identified and written in the order they were read.

use crate::clean::{CleaningCounts, clean};
use crate::error::Result;
use crate::filter::{FilterCounts, SampleFilter};
use crate::segment::typing_segments;
use crate::source::{
    MAX_CONTEXT_CHARACTERS, RawDocument, SAMPLES_PER_SHARD, SegmentUnit, SourceSpec,
};
use crate::text::{Normalizer, content_id, split_sentences};
use ime_g2p::DataLayout;
use ime_g2p::annotate::Sample;
use ime_g2p::shards::{ShardWriter, Shardable as _, read_frame, shard_paths};
use ime_pinyin::Lexicon;
use rayon::prelude::{IntoParallelRefIterator as _, ParallelIterator as _};
use std::time::Instant;
use tracing::info;

/// How one source's preparation went, in the numbers worth putting in a table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PrepareReport {
    /// Which source was prepared.
    pub source: &'static str,
    /// How many raw documents were read.
    pub documents: usize,
    /// What the line-level cleaning removed.
    pub cleaning: CleaningCounts,
    /// Why candidate targets were kept or dropped.
    pub counts: FilterCounts,
    /// How many samples reached the shards.
    pub written: usize,
    /// How long the whole pass took, in milliseconds.
    pub milliseconds: u128,
}

impl PrepareReport {
    /// Samples written per second, which is what a full run is planned against.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "a throughput figure is reported to one decimal place"
    )]
    pub fn samples_per_second(&self) -> f64 {
        if self.milliseconds == 0 {
            return 0.0;
        }
        self.written as f64 * 1000.0 / self.milliseconds as f64
    }
}

/// One document reduced to the ordered units its samples are cut out of.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Segmented {
    units: Vec<String>,
    cleaning: CleaningCounts,
}

/// Clean, normalise and split one document into its units.
fn segment(document: &RawDocument, spec: SourceSpec, normalizer: &Normalizer) -> Result<Segmented> {
    let mut units = Vec::new();
    let mut counts = CleaningCounts::default();
    for part in &document.parts {
        let (cleaned, removed) = clean(part, spec.cleaning);
        counts.merge(removed);
        let body = normalizer.normalize(&cleaned)?;
        match spec.unit {
            SegmentUnit::Sentence => units.extend(split_sentences(&body)),
            SegmentUnit::Turn => {
                if !body.is_empty() {
                    units.push(body);
                }
            }
        }
    }
    Ok(Segmented {
        units,
        cleaning: counts,
    })
}

/// What was on the writer's screen when they started typing one segment.
///
/// Two things, in the order they were typed: the run of units before the segment's
/// own, and the part of that unit the segment follows. Only the last
/// [`MAX_CONTEXT_CHARACTERS`] survive, because that is as much as the model reads.
fn context_of(units: &[String], index: usize, prefix: &str, spec: SourceSpec) -> Option<String> {
    let start = index.saturating_sub(spec.context_units);
    let mut pieces: Vec<&str> = units[start..index].iter().map(String::as_str).collect();
    if !prefix.is_empty() {
        pieces.push(prefix);
    }
    let joined = pieces.join(spec.unit.joiner());
    let characters = joined.chars().count();
    let tail: String = joined
        .chars()
        .skip(characters.saturating_sub(MAX_CONTEXT_CHARACTERS))
        .collect();
    (!tail.trim().is_empty()).then_some(tail)
}

/// Build filtered samples for one source out of its raw document shards.
///
/// # Errors
///
/// If the source has no raw shards, if one of them cannot be read, if the
/// normaliser's length invariant is violated, or if a sample shard cannot be
/// written.
pub fn prepare(
    spec: SourceSpec,
    layout: &DataLayout,
    lexicon: &Lexicon,
    limit: Option<usize>,
) -> Result<PrepareReport> {
    let started = Instant::now();
    let paths = shard_paths(&layout.documents(), spec.name)?;
    if paths.is_empty() {
        return Err(ime_g2p::Error::Missing {
            what: "raw document shards",
            path: layout.documents().join(format!("{}-*.parquet", spec.name)),
            hint: "run `ime-cli corpus fetch` for this source first",
        }
        .into());
    }
    let normalizer = Normalizer::new()?;
    let mut filter = SampleFilter::new(lexicon);
    let mut writer = ShardWriter::new(&layout.samples(), spec.name, SAMPLES_PER_SHARD)?;
    let mut cleaning = CleaningCounts::default();
    let mut documents = 0_usize;
    let mut written = 0_usize;

    'shards: for path in &paths {
        let frame = read_frame(path)?;
        let batch = RawDocument::from_frame(&frame)?;
        documents += batch.len();
        let segmented: Vec<Segmented> = batch
            .par_iter()
            .map(|document| segment(document, spec, &normalizer))
            .collect::<Result<Vec<_>>>()?;
        for document in &segmented {
            cleaning.merge(document.cleaning);
            for (index, unit) in document.units.iter().enumerate() {
                for candidate in typing_segments(unit) {
                    let text = candidate.text();
                    if !filter.accepts(text) {
                        continue;
                    }
                    let context = context_of(&document.units, index, candidate.prefix(), spec);
                    writer.write(Sample {
                        id: content_id(spec.name, text, context.as_deref()),
                        source: spec.name.to_owned(),
                        text: text.to_owned(),
                        context,
                    })?;
                    written += 1;
                    if limit.is_some_and(|limit| written >= limit) {
                        break 'shards;
                    }
                }
            }
        }
        info!(
            source = spec.name,
            shard = %path.display(),
            documents,
            written,
            "shard prepared"
        );
    }
    writer.finish()?;

    let report = PrepareReport {
        source: spec.name,
        documents,
        cleaning,
        counts: filter.counts(),
        written,
        milliseconds: started.elapsed().as_millis(),
    };
    info!(
        source = report.source,
        documents = report.documents,
        lines = report.cleaning.lines,
        infobox_lines = report.cleaning.infobox_lines,
        kept = report.counts.kept,
        too_short_run = report.counts.too_short_run,
        too_long_run = report.counts.too_long_run,
        unknown_character = report.counts.unknown_character,
        duplicate = report.counts.duplicate,
        written = report.written,
        per_second = report.samples_per_second(),
        "samples prepared"
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{BILIBILI, DIALOGUE, DOUYIN, MOEGIRL};
    use ime_g2p::shards::read_shards;
    use ime_pinyin::SyllableTable;

    fn lexicon() -> Lexicon {
        Lexicon::load(&SyllableTable::load()).expect("the generated pinyin tables agree")
    }

    fn normalizer() -> Normalizer {
        Normalizer::new().expect("the bundled t2s configuration loads")
    }

    fn document(source: &SourceSpec, text: &str) -> RawDocument {
        RawDocument {
            document_id: "fixture".to_owned(),
            source: source.name.to_owned(),
            parts: vec![text.to_owned()],
        }
    }

    fn units(source: &str) -> Vec<String> {
        source.split('|').map(ToOwned::to_owned).collect()
    }

    #[test]
    fn an_infobox_block_is_gone_before_the_prose_is_split() {
        let raw = "初音未来\n本名=初音ミク\n发色=青\n萌点=双马尾、吐槽\n\
                   她是世界上最有名的虚拟歌手。很多人都喜欢她的歌曲。";
        let segmented = segment(&document(&MOEGIRL, raw), MOEGIRL, &normalizer())
            .expect("the fixture normalises");
        assert_eq!(segmented.cleaning.infobox_lines, 3);
        assert!(segmented.units.iter().all(|line| !line.contains('=')));
        assert_eq!(
            segmented.units.last().map(String::as_str),
            Some("很多人都喜欢她的歌曲。")
        );
    }

    #[test]
    fn a_douyin_caption_loses_its_hashtags_and_keeps_its_sentence() {
        let segmented = segment(
            &document(
                &DOUYIN,
                "这个女人可不好惹！  #GQ说电影  #智取威虎山电影解说 ",
            ),
            DOUYIN,
            &normalizer(),
        )
        .expect("the fixture normalises");
        assert_eq!(segmented.units, vec!["这个女人可不好惹！".to_owned()]);
    }

    #[test]
    fn a_dialogue_record_keeps_one_unit_per_turn_rather_than_per_sentence() {
        let raw = RawDocument {
            document_id: "fixture".to_owned(),
            source: DIALOGUE.name.to_owned(),
            parts: vec![
                "火锅 我 在 重庆 吃 了 七八 顿。真 的".to_owned(),
                "哈哈哈哈 ！ 那 我 的 嘴巴 要 烂掉".to_owned(),
            ],
        };
        let segmented = segment(&raw, DIALOGUE, &normalizer()).expect("the fixture normalises");
        assert_eq!(
            segmented.units,
            vec![
                "火锅我在重庆吃了七八顿。真的".to_owned(),
                "哈哈哈哈！那我的嘴巴要烂掉".to_owned(),
            ]
        );
    }

    #[test]
    fn a_sample_carries_the_units_before_it_and_what_it_follows_inside_its_own() {
        let units = units("第一句。|第二句，第三段落。|第四句。");
        assert_eq!(
            context_of(&units, 1, "第二句，", MOEGIRL),
            Some("第一句。第二句，".to_owned())
        );
        assert_eq!(
            context_of(&units, 1, "", MOEGIRL),
            Some("第一句。".to_owned())
        );
        assert_eq!(context_of(&units, 0, "", MOEGIRL), None);
        // A social source carries no preceding unit, but still what it follows.
        assert_eq!(context_of(&units, 1, "", BILIBILI), None);
        assert_eq!(
            context_of(&units, 1, "第二句，", BILIBILI),
            Some("第二句，".to_owned())
        );
    }

    #[test]
    fn dialogue_context_puts_a_line_break_between_turns_and_before_the_prefix() {
        let units = units("你好|今天天气不错，真的");
        assert_eq!(
            context_of(&units, 1, "今天天气不错，", DIALOGUE),
            Some("你好\n今天天气不错，".to_owned())
        );
        assert_eq!(context_of(&units, 1, "", DIALOGUE), Some("你好".to_owned()));
    }

    #[test]
    fn a_context_longer_than_the_cap_keeps_its_last_characters() {
        let units = vec!["前".repeat(400), "目标".to_owned()];
        let context = context_of(&units, 1, "", MOEGIRL).expect("there is a preceding unit");
        assert_eq!(context.chars().count(), MAX_CONTEXT_CHARACTERS);
    }

    #[test]
    fn a_run_too_long_to_type_is_counted_and_dropped_rather_than_cut() {
        let lexicon = lexicon();
        let mut filter = SampleFilter::new(&lexicon);
        let unit = "中".repeat(100);
        for candidate in typing_segments(&unit) {
            assert!(!filter.accepts(candidate.text()));
        }
        assert_eq!(filter.counts().too_long_run, 1);
        assert_eq!(filter.counts().kept, 0);
    }

    #[test]
    fn preparing_a_fixture_corpus_writes_the_sample_schema_the_pipeline_reads() {
        let root = std::env::temp_dir().join(format!("ime-corpus-prepare-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let layout = DataLayout::new(&root);
        let mut writer = ShardWriter::new(&layout.documents(), MOEGIRL.name, 100)
            .expect("the fixture directory is writable");
        writer
            .write(document(
                &MOEGIRL,
                "初音未来\n发色=青\n她是虛擬歌手，唱過很多歌。她是虛擬歌手。",
            ))
            .expect("the row is buffered");
        writer.finish().expect("the shard is written");

        let lexicon = lexicon();
        let report = prepare(MOEGIRL, &layout, &lexicon, None).expect("the fixture prepares");
        assert_eq!(report.documents, 1);
        assert_eq!(report.cleaning.infobox_lines, 1);
        assert_eq!(report.counts.duplicate, 1);
        assert_eq!(report.written, report.counts.kept);

        let samples: Vec<Sample> =
            read_shards(&layout.samples(), MOEGIRL.name).expect("the samples read back");
        assert_eq!(samples.len(), report.written);
        assert!(samples.iter().all(|sample| sample.source == MOEGIRL.name));
        // Every target is a run of Han characters and nothing else.
        assert!(
            samples
                .iter()
                .all(|sample| crate::segment::is_typable_target(&sample.text))
        );
        // The traditional fixture comes out simplified, as the wiki register demands.
        let comma = samples
            .iter()
            .find(|sample| sample.text == "唱过很多歌")
            .expect("the run after the comma is its own target");
        assert_eq!(comma.context.as_deref(), Some("初音未来她是虚拟歌手，"));
        assert!(samples.iter().any(|sample| sample.text == "她是虚拟歌手"));
        assert!(samples.iter().all(|sample| sample.id
            == content_id(&sample.source, &sample.text, sample.context.as_deref())));
        std::fs::remove_dir_all(&root).expect("the fixture directory is removed");
    }
}
