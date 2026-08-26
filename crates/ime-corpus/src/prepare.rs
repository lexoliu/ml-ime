//! The offline half: turn raw documents into filtered, deduplicated samples.
//!
//! This is where every rule that might change lives -- cleaning, normalisation,
//! sentence splitting, the length and script filters, the duplicate check -- so
//! that tightening any of them costs a minute of local work rather than another
//! pass over the network.
//!
//! The work splits cleanly in two, and the split is what makes it fast. Cleaning
//! and normalising a document is pure and expensive (the traditional-to-simplified
//! conversion dominates), so it runs across every core with rayon. Filtering is
//! cheap but *stateful*: the duplicate check is a set that every candidate in the
//! source has to be checked against in a fixed order, or two runs of the same
//! input would keep different sentences. So one shard's documents are segmented in
//! parallel and then filtered, identified and written in the order they were read.

use crate::clean::{Cleaning, CleaningCounts, clean};
use crate::error::Result;
use crate::filter::{FilterCounts, SampleFilter};
use crate::source::{MAX_CONTEXT_CHARACTERS, RawDocument, SAMPLES_PER_SHARD, SourceSpec};
use crate::text::{Normalizer, content_id, split_sentences, strip_terminal_delimiter};
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

/// One document reduced to the segments a sample can be built from.
///
/// Prose segments are sentences and carry their own terminal punctuation, which
/// is why a run of them reassembles into readable context with no joiner.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Segmented {
    segments: Vec<String>,
    cleaning: CleaningCounts,
}

/// Clean, normalise and split one document into its segments.
fn segment(
    document: &RawDocument,
    cleaning: Cleaning,
    normalizer: &Normalizer,
) -> Result<Segmented> {
    let mut segments = Vec::new();
    let mut counts = CleaningCounts::default();
    for part in &document.parts {
        let (cleaned, removed) = clean(part, cleaning);
        counts.merge(removed);
        segments.extend(split_sentences(&normalizer.normalize(&cleaned)?));
    }
    Ok(Segmented {
        segments,
        cleaning: counts,
    })
}

/// The context a sample at `index` carries: the run of segments before it.
///
/// Empty when the source carries no context at all, which is what turns the
/// column null for a post or a comment without a branch anywhere else.
fn context_of(segments: &[String], index: usize, span: usize) -> Option<String> {
    let start = index.saturating_sub(span);
    let joined: String = segments[start..index].concat();
    let characters = joined.chars().count();
    let trimmed: String = joined
        .chars()
        .skip(characters.saturating_sub(MAX_CONTEXT_CHARACTERS))
        .collect();
    (!trimmed.is_empty()).then_some(trimmed)
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
            .map(|document| segment(document, spec.cleaning, &normalizer))
            .collect::<Result<Vec<_>>>()?;
        for document in &segmented {
            cleaning.merge(document.cleaning);
            for (index, raw) in document.segments.iter().enumerate() {
                let text = strip_terminal_delimiter(raw).trim();
                if !filter.accepts(text) {
                    continue;
                }
                let context = context_of(&document.segments, index, spec.context_segments);
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
        too_short = report.counts.too_short,
        too_long = report.counts.too_long,
        not_chinese_enough = report.counts.not_chinese_enough,
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
    use crate::source::{BILIBILI, DOUYIN, MOEGIRL};
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

    #[test]
    fn an_infobox_block_is_gone_before_the_prose_is_split() {
        let raw = "初音未来\n本名=初音ミク\n发色=青\n萌点=双马尾、吐槽\n\
                   她是世界上最有名的虚拟歌手。很多人都喜欢她的歌曲。";
        let segmented = segment(&document(&MOEGIRL, raw), MOEGIRL.cleaning, &normalizer())
            .expect("the fixture normalises");
        assert_eq!(segmented.cleaning.infobox_lines, 3);
        assert!(segmented.segments.iter().all(|line| !line.contains('=')));
        assert_eq!(
            segmented.segments.last().map(String::as_str),
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
            DOUYIN.cleaning,
            &normalizer(),
        )
        .expect("the fixture normalises");
        assert_eq!(segmented.segments, vec!["这个女人可不好惹！".to_owned()]);
    }

    #[test]
    fn a_prose_sample_carries_its_preceding_sentences_and_a_comment_carries_none() {
        let segments: Vec<String> = ["第一句。", "第二句。", "第三句。", "第四句。"]
            .iter()
            .map(|line| (*line).to_owned())
            .collect();
        assert_eq!(
            context_of(&segments, 3, MOEGIRL.context_segments),
            Some("第一句。第二句。第三句。".to_owned())
        );
        assert_eq!(
            context_of(&segments, 1, MOEGIRL.context_segments),
            Some("第一句。".to_owned())
        );
        assert_eq!(context_of(&segments, 0, MOEGIRL.context_segments), None);
        assert_eq!(context_of(&segments, 3, BILIBILI.context_segments), None);
    }

    #[test]
    fn a_context_longer_than_the_cap_keeps_its_last_characters() {
        let segments = vec!["前".repeat(400), "目标".to_owned()];
        let context = context_of(&segments, 1, 3).expect("there is a preceding segment");
        assert_eq!(context.chars().count(), MAX_CONTEXT_CHARACTERS);
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
                "初音未来\n发色=青\n她是虛擬歌手。很多人都喜歡她的歌曲。她是虛擬歌手。",
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
        assert!(samples.iter().all(|sample| !sample.text.contains('=')));
        // The traditional fixture comes out simplified, as the wiki register demands.
        assert!(samples.iter().any(|sample| sample.text == "她是虚拟歌手"));
        assert!(samples.iter().any(|sample| sample.context.is_some()));
        assert!(samples.iter().all(|sample| sample.id
            == content_id(&sample.source, &sample.text, sample.context.as_deref())));
        std::fs::remove_dir_all(&root).expect("the fixture directory is removed");
    }
}
