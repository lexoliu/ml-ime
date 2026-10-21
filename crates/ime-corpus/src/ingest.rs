//! Streaming ingestion: an upstream too large to keep, one file at a time.
//!
//! [`crate::prepare`] reads a source whose raw documents are all on disk. The
//! web-scale sources are not: FineWeb-2's Mandarin is 1.6 TB, and only a slice
//! of it is wanted. So each upstream file is downloaded, run through the same
//! rules -- cleaning, normalisation, the unit split, the personal-data drop, the
//! typing segments and [`SampleFilter`] -- and deleted before the next one is
//! needed. Two more gates run here that `prepare` does not need, because a web
//! crawl overlaps everything else this project holds: the [`EvalGuard`] keeps
//! every evaluation record out, and run3's own targets are not taken twice.
//!
//! A run is resumable per file. A file's outputs are written aside and moved
//! into place, and only then is its record written; on start, the records name
//! the finished files, their accepted digests restore the duplicate check, and
//! anything an unfinished file left behind is removed. A run takes a bounded
//! number of files, and stops cleanly before a download would take the disk
//! below its floor. The next file downloads while the current one is processed.
//!
//! Two passes share all of this: [`GeneralPass`] keeps every target that
//! passes, and [`MinePass`] keeps only sentences that use an entry run3 never
//! shows, through the gates in [`crate::mine`].

use crate::download::{Hub, RepoFile};
use crate::error::{Error, Result};
use crate::filter::{FilterCounts, SampleFilter};
use crate::leak::EvalGuard;
use crate::mine::{
    CharacterPairs, DocumentProfile, EntryHit, EntryMatcher, PER_ENTRY_CAP, SpunReason,
};
use crate::prepare::{context_of, drop_personal, split_units};
use crate::privacy::{PersonalCounts, PersonalData};
use crate::segment::typing_segments;
use crate::source::{RawDocument, SAMPLES_PER_SHARD, SourceSpec};
use crate::text::{DEDUPE_DIGEST_BYTES, Normalizer, content_id, dedupe_digest};
use crate::upstream::{Flow, Upstream};
use foldhash::fast::RandomState;
use ime_g2p::annotate::Sample;
use ime_g2p::shards::{ShardWriter, column_of_strings, read_frame, shard_paths};
use ime_pinyin::Lexicon;
use rayon::prelude::{IntoParallelRefIterator as _, ParallelIterator as _};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::info;

/// A content digest of one target, as the duplicate check keys it.
type Digest = [u8; DEDUPE_DIGEST_BYTES];

/// The most upstream files a source may have, so an output name stays four digits.
const MAX_FILES: usize = 10_000;

/// The most sample shards one upstream file may produce, so a name stays three digits.
const MAX_PARTS: usize = 1_000;

/// run3's targets: what a new source must not take twice, and the pairs it writes.
pub struct Run3 {
    digests: HashSet<Digest, RandomState>,
    pairs: CharacterPairs,
}

impl std::fmt::Debug for Run3 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Run3")
            .field("digests", &self.digests.len())
            .field("pairs", &self.pairs.len())
            .finish()
    }
}

impl Run3 {
    /// Read every sample shard under `samples`, with its character pairs when asked.
    ///
    /// # Errors
    ///
    /// If the directory holds no shard, or one cannot be read.
    pub fn load(samples: &Path, with_pairs: bool) -> Result<Self> {
        let paths = shard_paths(samples, "*")?;
        if paths.is_empty() {
            return Err(ime_g2p::Error::Missing {
                what: "run3 sample shards",
                path: samples.join("*.parquet"),
                hint: "pass the samples directory of the run3 data root",
            }
            .into());
        }
        let shards: Vec<(Vec<Digest>, CharacterPairs)> = paths
            .par_iter()
            .map(|path| {
                let texts = column_of_strings(&read_frame(path)?, "text")?;
                let digests = texts.iter().map(|text| dedupe_digest(text)).collect();
                let pairs = if with_pairs {
                    CharacterPairs::new(texts.iter().map(String::as_str))
                } else {
                    CharacterPairs::new([])
                };
                Ok((digests, pairs))
            })
            .collect::<Result<_>>()?;
        let mut digests = HashSet::with_hasher(RandomState::default());
        let mut pairs = CharacterPairs::new([]);
        for (shard_digests, shard_pairs) in shards {
            digests.extend(shard_digests);
            pairs.merge(shard_pairs);
        }
        info!(
            shards = paths.len(),
            targets = digests.len(),
            pairs = pairs.len(),
            "run3 loaded"
        );
        Ok(Self { digests, pairs })
    }

    /// A run3 of exactly these targets, for tests.
    #[must_use]
    pub fn of<'a>(texts: impl IntoIterator<Item = &'a str> + Clone) -> Self {
        Self {
            digests: texts.clone().into_iter().map(dedupe_digest).collect(),
            pairs: CharacterPairs::new(texts),
        }
    }

    fn holds(&self, digest: &Digest) -> bool {
        self.digests.contains(digest)
    }
}

/// What every pass reads besides the upstream itself.
pub struct Shared<'a> {
    /// The pinyin lexicon the length and coverage gate reads.
    pub lexicon: &'a Lexicon,
    /// The evaluation and held-out records nothing may leak into.
    pub guard: &'a EvalGuard,
    /// run3's targets.
    pub run3: &'a Run3,
    normalizer: Normalizer,
    personal: PersonalData,
}

impl std::fmt::Debug for Shared<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Shared")
            .field("guard", self.guard)
            .field("run3", self.run3)
            .finish_non_exhaustive()
    }
}

impl<'a> Shared<'a> {
    /// Gather what a pass reads, building the normaliser and the personal-data rules.
    ///
    /// # Errors
    ///
    /// If the normaliser cannot be built.
    pub fn new(lexicon: &'a Lexicon, guard: &'a EvalGuard, run3: &'a Run3) -> Result<Self> {
        Ok(Self {
            lexicon,
            guard,
            run3,
            normalizer: Normalizer::new()?,
            personal: PersonalData::new(),
        })
    }

    /// Normalise and split one document, refuse it whole if it quotes a record,
    /// and drop its sentences that hold personal data.
    fn prepare(&self, document: &RawDocument, spec: SourceSpec) -> Result<Prepared> {
        let segmented = split_units(document, spec, &self.normalizer)?;
        if self
            .guard
            .shares_window(segmented.units.iter().map(String::as_str))
        {
            return Ok(Prepared::Leaked);
        }
        let mut units = segmented.units;
        let personal = drop_personal(&mut units, &self.personal);
        Ok(Prepared::Kept { units, personal })
    }
}

/// One document after the parallel stage.
enum Prepared {
    /// It shares a window with an evaluation record and is gone whole.
    Leaked,
    /// Its units, personal data already dropped.
    Kept {
        units: Vec<String>,
        personal: PersonalCounts,
    },
}

/// Everything counted over one upstream file, or summed over many.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Census {
    /// Upstream files processed.
    pub files: usize,
    /// Bytes downloaded.
    pub downloaded_bytes: u64,
    /// Bytes of document text read out of them.
    pub text_bytes: u64,
    /// Documents read.
    pub documents: usize,
    /// Documents normalised; for mining, those the raw prefilter let through.
    pub normalised_documents: usize,
    /// Documents dropped whole for sharing a window with an evaluation record.
    pub leaked_documents: usize,
    /// Sentences (or turns) left once personal data is dropped.
    pub units: usize,
    /// Sentences dropped for personal data, by kind.
    pub personal: PersonalCounts,
    /// Candidate targets the length, lexicon and duplicate gate saw.
    pub considered: usize,
    /// Of those, runs shorter than four characters.
    pub too_short_run: usize,
    /// Runs longer than sixty-four characters.
    pub too_long_run: usize,
    /// Targets with a character the lexicon cannot read.
    pub unknown_character: usize,
    /// Targets this source already accepted.
    pub duplicate: usize,
    /// Targets equal to an evaluation or held-out text.
    pub eval_exact: usize,
    /// Targets run3 already holds.
    pub in_run3: usize,
    /// Targets written.
    pub written: usize,
    /// Mining: target sentences holding an absent entry.
    pub hit_sentences: usize,
    /// Mining: hits the sentence gate refused.
    pub unusable_hits: usize,
    /// Mining: hits in documents refused for idiom density.
    pub spun_idiom_density: usize,
    /// Mining: hits in documents holding too many absent entries.
    pub spun_absent_entries: usize,
    /// Mining: hits in documents of pairs run3 never writes.
    pub spun_novel_pairs: usize,
    /// Mining: hits whose every entry already had its cap.
    pub capped: usize,
    /// Milliseconds spent downloading.
    pub download_ms: u64,
    /// Milliseconds spent processing.
    pub process_ms: u64,
}

impl Census {
    /// Add another census into this one.
    pub fn add(&mut self, other: &Self) {
        self.files += other.files;
        self.downloaded_bytes += other.downloaded_bytes;
        self.text_bytes += other.text_bytes;
        self.documents += other.documents;
        self.normalised_documents += other.normalised_documents;
        self.leaked_documents += other.leaked_documents;
        self.units += other.units;
        self.personal.merge(other.personal);
        self.considered += other.considered;
        self.too_short_run += other.too_short_run;
        self.too_long_run += other.too_long_run;
        self.unknown_character += other.unknown_character;
        self.duplicate += other.duplicate;
        self.eval_exact += other.eval_exact;
        self.in_run3 += other.in_run3;
        self.written += other.written;
        self.hit_sentences += other.hit_sentences;
        self.unusable_hits += other.unusable_hits;
        self.spun_idiom_density += other.spun_idiom_density;
        self.spun_absent_entries += other.spun_absent_entries;
        self.spun_novel_pairs += other.spun_novel_pairs;
        self.capped += other.capped;
        self.download_ms += other.download_ms;
        self.process_ms += other.process_ms;
    }

    /// Targets written per candidate the filter considered.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "a share printed to two decimals"
    )]
    pub fn filter_yield(&self) -> f64 {
        if self.considered == 0 {
            return 0.0;
        }
        self.written as f64 / self.considered as f64
    }

    fn set_filter_counts(&mut self, counts: FilterCounts) {
        self.considered = counts.considered();
        self.too_short_run = counts.too_short_run;
        self.too_long_run = counts.too_long_run;
        self.unknown_character = counts.unknown_character;
        self.duplicate = counts.duplicate;
    }
}

fn counts_since(now: FilterCounts, before: FilterCounts) -> FilterCounts {
    FilterCounts {
        kept: now.kept - before.kept,
        too_short_run: now.too_short_run - before.too_short_run,
        too_long_run: now.too_long_run - before.too_long_run,
        unknown_character: now.unknown_character - before.unknown_character,
        duplicate: now.duplicate - before.duplicate,
    }
}

/// What one finished upstream file left, written last so it marks the file done.
#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub struct FileRecord {
    /// The file's place in the source's processing order.
    pub ordinal: usize,
    /// Its path in the dataset.
    pub path: String,
    /// Whether the target budget was reached part way through it.
    pub truncated: bool,
    /// What was counted over it.
    pub census: Census,
    /// Mining: how many sentences each entry kept from it.
    pub entries: BTreeMap<String, usize>,
}

/// What became of one mining hit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    /// The sentence gate refused every entry in it.
    Unusable,
    /// The document gate refused the page it is on.
    Spun(SpunReason),
    /// The length, lexicon or duplicate gate refused it.
    Filtered,
    /// It is an evaluation or held-out text.
    Eval,
    /// run3 already holds it.
    Run3,
    /// Every entry it uses already has its cap.
    Capped,
    /// It was written.
    Kept,
}

impl Verdict {
    fn label(self) -> &'static str {
        match self {
            Self::Unusable => "unusable",
            Self::Spun(SpunReason::IdiomDensity) => "idiom-density",
            Self::Spun(SpunReason::AbsentEntries) => "absent-entries",
            Self::Spun(SpunReason::NovelPairs) => "novel-pairs",
            Self::Filtered => "filtered",
            Self::Eval => "eval",
            Self::Run3 => "run3",
            Self::Capped => "capped",
            Self::Kept => "kept",
        }
    }

    fn count(self, census: &mut Census) {
        match self {
            Self::Unusable => census.unusable_hits += 1,
            Self::Spun(SpunReason::IdiomDensity) => census.spun_idiom_density += 1,
            Self::Spun(SpunReason::AbsentEntries) => census.spun_absent_entries += 1,
            Self::Spun(SpunReason::NovelPairs) => census.spun_novel_pairs += 1,
            Self::Eval => census.eval_exact += 1,
            Self::Run3 => census.in_run3 += 1,
            Self::Capped => census.capped += 1,
            Self::Filtered | Self::Kept => {}
        }
    }
}

/// One mining hit as the audit log records it.
#[derive(serde::Serialize)]
struct HitRow<'a> {
    entries: String,
    usable: bool,
    verdict: &'static str,
    han: usize,
    idioms_per_hundred: String,
    absent: usize,
    novel_pair_share: String,
    rare3_pair_share: String,
    rare30_pair_share: String,
    text: &'a str,
    context: &'a str,
    document: &'a str,
}

/// Where one file's outputs gather before they are moved into place.
pub struct FileOutput {
    directory: PathBuf,
    source: &'static str,
    writer: ShardWriter<Sample>,
    census: Census,
    digests: Vec<Digest>,
    entries: BTreeMap<String, usize>,
    entry_rows: Vec<(String, String)>,
    hits: Option<csv::Writer<std::fs::File>>,
}

impl std::fmt::Debug for FileOutput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FileOutput")
            .field("directory", &self.directory)
            .field("census", &self.census)
            .finish_non_exhaustive()
    }
}

impl FileOutput {
    fn new(directory: PathBuf, source: &'static str) -> Result<Self> {
        if directory.exists() {
            remove_dir(&directory)?;
        }
        let writer = ShardWriter::new(&directory, source, SAMPLES_PER_SHARD)?;
        Ok(Self {
            directory,
            source,
            writer,
            census: Census::default(),
            digests: Vec::new(),
            entries: BTreeMap::new(),
            entry_rows: Vec::new(),
            hits: None,
        })
    }

    /// Append one row to the hit log, opening it on the first.
    fn log_hit(&mut self, row: &HitRow<'_>) -> Result<()> {
        let path = self.directory.join("hits.tsv");
        let hits = match &mut self.hits {
            Some(hits) => hits,
            None => self.hits.insert(
                csv::WriterBuilder::new()
                    .delimiter(b'\t')
                    .from_path(&path)
                    .map_err(|source| Error::Csv {
                        path: path.clone(),
                        source,
                    })?,
            ),
        };
        hits.serialize(row)
            .map_err(|source| Error::Csv { path, source })
    }

    /// Count a prepared document in, handing back its units when it survived.
    fn admit(&mut self, prepared: Prepared) -> Option<Vec<String>> {
        match prepared {
            Prepared::Leaked => {
                self.census.leaked_documents += 1;
                None
            }
            Prepared::Kept { units, personal } => {
                self.census.personal.merge(personal);
                self.census.units += units.len();
                Some(units)
            }
        }
    }

    fn write(&mut self, spec: SourceSpec, text: &str, context: Option<String>) -> Result<String> {
        let id = content_id(spec.name, text, context.as_deref());
        self.writer.write(Sample {
            id: id.clone(),
            source: spec.name.to_owned(),
            text: text.to_owned(),
            context,
        })?;
        self.census.written += 1;
        Ok(id)
    }
}

/// One way of turning a source's documents into samples.
pub trait Pass {
    /// Restore what a finished file left: its accepted digests and kept entries.
    ///
    /// # Errors
    ///
    /// If the record names an entry this pass does not mine for.
    fn restore(&mut self, record: &FileRecord, digests: Vec<Digest>) -> Result<()>;

    /// The filter's tally so far, so a file's share can be told apart.
    fn filter_counts(&self) -> FilterCounts;

    /// Process one batch of documents, in file order, into `out`.
    ///
    /// # Errors
    ///
    /// If a document breaks the normaliser's invariant or a sample cannot be written.
    fn batch(
        &mut self,
        spec: SourceSpec,
        documents: &[RawDocument],
        out: &mut FileOutput,
    ) -> Result<Flow>;
}

/// Keep every target that passes, up to an optional budget over the whole source.
pub struct GeneralPass<'a> {
    shared: &'a Shared<'a>,
    filter: SampleFilter<'a>,
    budget: Option<usize>,
    written: usize,
}

impl std::fmt::Debug for GeneralPass<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GeneralPass")
            .field("budget", &self.budget)
            .field("written", &self.written)
            .finish_non_exhaustive()
    }
}

impl<'a> GeneralPass<'a> {
    /// A pass over `shared`, stopping the source once `budget` targets are written.
    #[must_use]
    pub fn new(shared: &'a Shared<'a>, budget: Option<usize>) -> Self {
        Self {
            shared,
            filter: SampleFilter::new(shared.lexicon),
            budget,
            written: 0,
        }
    }

    fn full(&self, out: &FileOutput) -> bool {
        self.budget
            .is_some_and(|budget| self.written + out.census.written >= budget)
    }
}

impl Pass for GeneralPass<'_> {
    fn restore(&mut self, record: &FileRecord, digests: Vec<Digest>) -> Result<()> {
        self.filter.preload(digests);
        self.written += record.census.written;
        Ok(())
    }

    fn filter_counts(&self) -> FilterCounts {
        self.filter.counts()
    }

    fn batch(
        &mut self,
        spec: SourceSpec,
        documents: &[RawDocument],
        out: &mut FileOutput,
    ) -> Result<Flow> {
        let shared = self.shared;
        let prepared: Vec<Prepared> = documents
            .par_iter()
            .map(|document| shared.prepare(document, spec))
            .collect::<Result<_>>()?;
        out.census.normalised_documents += prepared.len();
        for document in prepared {
            let Some(units) = out.admit(document) else {
                continue;
            };
            for (index, unit) in units.iter().enumerate() {
                for candidate in typing_segments(unit) {
                    if self.full(out) {
                        return Ok(Flow::Stop);
                    }
                    let text = candidate.text();
                    if !self.filter.accepts(text) {
                        continue;
                    }
                    let digest = dedupe_digest(text);
                    out.digests.push(digest);
                    if shared.guard.holds_text(text) {
                        out.census.eval_exact += 1;
                        continue;
                    }
                    if shared.run3.holds(&digest) {
                        out.census.in_run3 += 1;
                        continue;
                    }
                    out.write(
                        spec,
                        text,
                        context_of(&units, index, candidate.prefix(), spec),
                    )?;
                }
            }
        }
        Ok(if self.full(out) {
            Flow::Stop
        } else {
            Flow::More
        })
    }
}

/// Keep only sentences that use an absent entry, through the sentence and
/// document gates, at most [`PER_ENTRY_CAP`] per entry.
pub struct MinePass<'a> {
    shared: &'a Shared<'a>,
    matcher: &'a EntryMatcher,
    filter: SampleFilter<'a>,
    index: HashMap<&'a str, usize>,
    kept: Vec<usize>,
}

impl std::fmt::Debug for MinePass<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MinePass")
            .field("matcher", &self.matcher)
            .field("kept", &self.kept.iter().sum::<usize>())
            .finish_non_exhaustive()
    }
}

impl<'a> MinePass<'a> {
    /// A mining pass over `shared` for `matcher`'s targets.
    ///
    /// # Errors
    ///
    /// If run3 was loaded without its character pairs, which the document gate reads.
    pub fn new(shared: &'a Shared<'a>, matcher: &'a EntryMatcher) -> Result<Self> {
        if shared.run3.pairs.is_empty() {
            return Err(Error::Invariant(
                "mining reads run3's character pairs, and none were loaded".to_owned(),
            ));
        }
        Ok(Self {
            shared,
            matcher,
            filter: SampleFilter::new(shared.lexicon),
            index: matcher
                .targets()
                .iter()
                .enumerate()
                .map(|(index, target)| (target.entry.as_str(), index))
                .collect(),
            kept: vec![0; matcher.targets().len()],
        })
    }

    /// How many sentences each target entry has kept, in target order.
    #[must_use]
    pub fn kept(&self) -> &[usize] {
        &self.kept
    }
}

/// One target sentence holding an absent entry, found in the parallel stage.
struct Candidate {
    unit: usize,
    prefix_bytes: usize,
    text: String,
    hits: Vec<EntryHit>,
}

/// A document the mining prefilter let through, after the parallel stage.
struct Mined {
    id: String,
    prepared: Prepared,
    candidates: Vec<Candidate>,
    profile: DocumentProfile,
}

impl Pass for MinePass<'_> {
    fn restore(&mut self, record: &FileRecord, digests: Vec<Digest>) -> Result<()> {
        self.filter.preload(digests);
        for (entry, count) in &record.entries {
            let index = self.index.get(entry.as_str()).ok_or_else(|| {
                Error::Invariant(format!(
                    "file {} kept {entry:?}, which is not among the entries mined for",
                    record.ordinal
                ))
            })?;
            self.kept[*index] += count;
        }
        Ok(())
    }

    fn filter_counts(&self) -> FilterCounts {
        self.filter.counts()
    }

    fn batch(
        &mut self,
        spec: SourceSpec,
        documents: &[RawDocument],
        out: &mut FileOutput,
    ) -> Result<Flow> {
        let shared = self.shared;
        let matcher = self.matcher;
        let mined: Vec<Option<Mined>> = documents
            .par_iter()
            .map(|document| {
                if !document
                    .parts
                    .iter()
                    .any(|part| matcher.may_hold_target(part))
                {
                    return Ok(None);
                }
                let prepared = shared.prepare(document, spec)?;
                let Prepared::Kept { units, .. } = &prepared else {
                    return Ok(Some(Mined {
                        id: document.document_id.clone(),
                        prepared,
                        candidates: Vec::new(),
                        profile: DocumentProfile::default(),
                    }));
                };
                let candidates: Vec<Candidate> = units
                    .iter()
                    .enumerate()
                    .flat_map(|(unit, text)| {
                        typing_segments(text).filter_map(move |candidate| {
                            let hits = matcher.hits(candidate.text());
                            (!hits.is_empty()).then(|| Candidate {
                                unit,
                                prefix_bytes: candidate.prefix().len(),
                                text: candidate.text().to_owned(),
                                hits,
                            })
                        })
                    })
                    .collect();
                let profile = if candidates.is_empty() {
                    DocumentProfile::default()
                } else {
                    matcher.profile(units.iter().map(String::as_str), &shared.run3.pairs)
                };
                Ok(Some(Mined {
                    id: document.document_id.clone(),
                    prepared,
                    candidates,
                    profile,
                }))
            })
            .collect::<Result<_>>()?;
        for document in mined.into_iter().flatten() {
            out.census.normalised_documents += 1;
            let profile = document.profile;
            let Some(units) = out.admit(document.prepared) else {
                continue;
            };
            for candidate in document.candidates {
                let context = context_of(
                    &units,
                    candidate.unit,
                    &units[candidate.unit][..candidate.prefix_bytes],
                    spec,
                );
                let verdict = self.judge(spec, &candidate, &profile, context.clone(), out)?;
                out.census.hit_sentences += 1;
                verdict.count(&mut out.census);
                let entries: Vec<&str> = candidate
                    .hits
                    .iter()
                    .map(|hit| self.matcher.targets()[hit.target as usize].entry.as_str())
                    .collect();
                out.log_hit(&HitRow {
                    entries: entries.join(","),
                    usable: candidate.hits.iter().any(|hit| hit.usable),
                    verdict: verdict.label(),
                    han: profile.han,
                    idioms_per_hundred: format!("{:.2}", profile.idioms_per_hundred()),
                    absent: profile.absent,
                    novel_pair_share: format!("{:.4}", profile.novel_pair_share()),
                    rare3_pair_share: format!("{:.4}", profile.rare_pair_share(1)),
                    rare30_pair_share: format!("{:.4}", profile.rare_pair_share(2)),
                    text: &candidate.text,
                    context: context.as_deref().unwrap_or_default(),
                    document: &document.id,
                })?;
            }
        }
        Ok(Flow::More)
    }
}

impl MinePass<'_> {
    /// Run one hit through every gate, writing it when it passes; returns the verdict.
    fn judge(
        &mut self,
        spec: SourceSpec,
        candidate: &Candidate,
        profile: &DocumentProfile,
        context: Option<String>,
        out: &mut FileOutput,
    ) -> Result<Verdict> {
        let usable: Vec<usize> = candidate
            .hits
            .iter()
            .filter(|hit| hit.usable)
            .map(|hit| hit.target as usize)
            .collect();
        if usable.is_empty() {
            return Ok(Verdict::Unusable);
        }
        if let Some(reason) = profile.spun() {
            return Ok(Verdict::Spun(reason));
        }
        if !self.filter.accepts(&candidate.text) {
            return Ok(Verdict::Filtered);
        }
        let digest = dedupe_digest(&candidate.text);
        out.digests.push(digest);
        if self.shared.guard.holds_text(&candidate.text) {
            return Ok(Verdict::Eval);
        }
        if self.shared.run3.holds(&digest) {
            return Ok(Verdict::Run3);
        }
        let open: Vec<usize> = usable
            .into_iter()
            .filter(|target| self.kept[*target] < PER_ENTRY_CAP)
            .collect();
        if open.is_empty() {
            return Ok(Verdict::Capped);
        }
        let id = out.write(spec, &candidate.text, context)?;
        for target in open {
            self.kept[target] += 1;
            let entry = &self.matcher.targets()[target].entry;
            *out.entries.entry(entry.clone()).or_default() += 1;
            out.entry_rows.push((id.clone(), entry.clone()));
        }
        Ok(Verdict::Kept)
    }
}

/// How far a run goes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Plan {
    /// The data root the samples and the run state go under.
    pub root: PathBuf,
    /// The most upstream files this run processes.
    pub files: usize,
    /// When set, the source is only the first this many files in processing order.
    pub upstream_files: Option<usize>,
    /// The disk space no download may take the root's volume below, in bytes.
    pub floor_bytes: u64,
}

/// Why a run ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ending {
    /// Every file of the source is finished.
    Exhausted,
    /// The run processed as many files as it was allowed.
    Chunk,
    /// The source wrote its whole target budget.
    Budget,
    /// The next download would have taken the disk below the floor.
    DiskFloor {
        /// Bytes free when the run stopped.
        available: u64,
        /// Bytes the next file needs.
        needed: u64,
    },
}

/// What one run did.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RunReport {
    /// Why it stopped.
    pub ending: Ending,
    /// What it counted over the files it processed itself.
    pub run: Census,
    /// What the source has counted over every finished file.
    pub total: Census,
    /// Finished files out of the files the source has.
    pub finished: usize,
    /// Files the source has.
    pub available: usize,
    /// Wall-clock milliseconds the run took, listing to last commit.
    pub wall_ms: u64,
}

/// The run state's directories for one source.
struct StateDirs {
    files: PathBuf,
    seen: PathBuf,
    entries: PathBuf,
    hits: PathBuf,
    scratch: PathBuf,
    samples: PathBuf,
    downloads: PathBuf,
    root: PathBuf,
}

impl StateDirs {
    fn new(root: &Path, source: &str) -> Result<Self> {
        let state = root.join("ingest").join(source);
        let dirs = Self {
            files: state.join("files"),
            seen: state.join("seen"),
            entries: state.join("entries"),
            hits: state.join("hits"),
            scratch: state.join("scratch"),
            samples: root.join("samples"),
            downloads: root.join("downloads").join(source),
            root: state,
        };
        for directory in [
            &dirs.files,
            &dirs.seen,
            &dirs.entries,
            &dirs.hits,
            &dirs.samples,
            &dirs.downloads,
        ] {
            std::fs::create_dir_all(directory).map_err(|source| Error::Read {
                path: directory.clone(),
                source,
            })?;
        }
        Ok(dirs)
    }

    fn record(&self, ordinal: usize) -> PathBuf {
        self.files.join(format!("{ordinal:04}.json"))
    }
}

/// Bytes free to an unprivileged writer on the volume holding `path`.
///
/// # Errors
///
/// If the volume cannot be queried.
pub fn available_bytes(path: &Path) -> Result<u64> {
    let stats = rustix::fs::statvfs(path).map_err(|error| Error::Read {
        path: path.to_owned(),
        source: error.into(),
    })?;
    Ok(stats.f_bavail * stats.f_frsize)
}

/// Run one bounded chunk of a source, resuming from its finished files.
///
/// # Errors
///
/// If the listing, a download, a file's processing or the run state fails, or
/// if the finished files' records disagree with the listing.
pub async fn run<U: Upstream, P: Pass>(hub: &Hub, plan: &Plan, pass: &mut P) -> Result<RunReport> {
    let started = Instant::now();
    let spec = U::SPEC;
    let dirs = StateDirs::new(&plan.root, spec.name)?;
    let files = listing::<U>(hub, plan).await?;

    let records = restore(&dirs, &files, spec, pass)?;
    let mut total = Census::default();
    for record in records.values() {
        total.add(&record.census);
    }
    let budget_spent = records.values().any(|record| record.truncated);
    let pending: Vec<(usize, RepoFile)> = files
        .iter()
        .cloned()
        .enumerate()
        .filter(|(ordinal, _)| !records.contains_key(ordinal))
        .take(plan.files)
        .collect();
    let mut report = RunReport {
        ending: Ending::Exhausted,
        run: Census::default(),
        total,
        finished: records.len(),
        available: files.len(),
        wall_ms: 0,
    };
    if budget_spent {
        report.ending = Ending::Budget;
        return Ok(report);
    }

    let mut next: Option<Download> = None;
    for (position, (ordinal, file)) in pending.iter().enumerate() {
        let download = match next.take() {
            Some(handle) => handle,
            None => match start_download(hub, spec, *ordinal, file, &dirs, plan)? {
                Ok(handle) => handle,
                Err(ending) => {
                    report.ending = ending;
                    break;
                }
            },
        };
        let (path, download_ms) = download
            .await
            .map_err(|error| Error::Invariant(format!("a download task panicked: {error}")))??;
        if let Some((following_ordinal, following)) = pending.get(position + 1) {
            // A refusal here is not the end yet: the current file still runs,
            // and the floor is checked again when the following file is due.
            next = start_download(hub, spec, *following_ordinal, following, &dirs, plan)?.ok();
        }
        let record = tokio::task::block_in_place(|| {
            process_file::<U, P>(&dirs, spec, *ordinal, file, &path, download_ms, pass)
        })?;
        info!(
            source = spec.name,
            ordinal,
            path = file.path,
            documents = record.census.documents,
            written = record.census.written,
            process_ms = record.census.process_ms,
            "file finished"
        );
        report.run.add(&record.census);
        report.total.add(&record.census);
        report.finished += 1;
        write_summary(&dirs, &report.total)?;
        if record.truncated {
            report.ending = Ending::Budget;
            break;
        }
        report.ending = if report.finished == files.len() {
            Ending::Exhausted
        } else {
            Ending::Chunk
        };
    }
    if let Some(handle) = next {
        // The chunk ended with the next file still downloading: let it finish so
        // the next run finds it complete rather than resuming a partial file.
        handle
            .await
            .map_err(|error| Error::Invariant(format!("a download task panicked: {error}")))??;
    }
    report.wall_ms = millis(started);
    Ok(report)
}

/// The source's files in processing order, cut to the plan's share.
async fn listing<U: Upstream>(hub: &Hub, plan: &Plan) -> Result<Vec<RepoFile>> {
    let spec = U::SPEC;
    let mut files: Vec<RepoFile> = U::order(
        hub.files(spec.dataset, U::PREFIX)
            .await?
            .into_iter()
            .filter(|file| U::selects(&file.path))
            .collect(),
    );
    if files.is_empty() {
        return Err(Error::Layout {
            dataset: spec.dataset,
            detail: format!(
                "nothing under {} is a file of the {} source",
                U::PREFIX,
                spec.name
            ),
        });
    }
    if let Some(limit) = plan.upstream_files {
        files.truncate(limit);
    }
    if files.len() > MAX_FILES {
        return Err(Error::Invariant(format!(
            "{} has {} files, more than the {MAX_FILES} an output name can number",
            spec.name,
            files.len()
        )));
    }
    Ok(files)
}

/// A download in flight: the local path once it lands, and how long it took.
type Download = tokio::task::JoinHandle<Result<(PathBuf, u64)>>;

/// Start downloading `file`, or say why the disk floor forbids it.
fn start_download(
    hub: &Hub,
    spec: SourceSpec,
    ordinal: usize,
    file: &RepoFile,
    dirs: &StateDirs,
    plan: &Plan,
) -> Result<std::result::Result<Download, Ending>> {
    let available = available_bytes(&dirs.downloads)?;
    // One directory per file, because two boards can name their files alike
    // and the next file downloads while the current one is still being read.
    let directory = dirs.downloads.join(format!("{ordinal:04}"));
    let local = directory.join(file.path.rsplit('/').next().unwrap_or(&file.path));
    let have = std::fs::metadata(&local).map_or(0, |meta| meta.len());
    let needed = file.bytes().saturating_sub(have);
    if available < plan.floor_bytes + needed {
        return Ok(Err(Ending::DiskFloor { available, needed }));
    }
    let hub = hub.clone();
    let file = file.clone();
    let dataset = spec.dataset;
    Ok(Ok(tokio::spawn(async move {
        let started = Instant::now();
        let path = hub.download(dataset, &file, &directory).await?;
        Ok((path, millis(started)))
    })))
}

/// Read the finished files' records, restore the pass, and clear what unfinished ones left.
fn restore<P: Pass>(
    dirs: &StateDirs,
    files: &[RepoFile],
    spec: SourceSpec,
    pass: &mut P,
) -> Result<BTreeMap<usize, FileRecord>> {
    let mut records = BTreeMap::new();
    for entry in read_dir(&dirs.files)? {
        if entry
            .extension()
            .is_none_or(|extension| extension != "json")
        {
            continue;
        }
        let text = std::fs::read_to_string(&entry).map_err(|source| Error::Read {
            path: entry.clone(),
            source,
        })?;
        let record: FileRecord = serde_json::from_str(&text).map_err(|source| Error::JsonLine {
            path: entry.clone(),
            line: 1,
            source,
        })?;
        match files.get(record.ordinal) {
            Some(file) if file.path == record.path => {}
            other => {
                return Err(Error::Invariant(format!(
                    "{} records file {} as {}, but the listing has {:?} there; the upstream changed",
                    entry.display(),
                    record.ordinal,
                    record.path,
                    other.map(|file| &file.path)
                )));
            }
        }
        let digests = read_digests(&dirs.seen.join(format!("{:04}.bin", record.ordinal)))?;
        pass.restore(&record, digests)?;
        records.insert(record.ordinal, record);
    }
    if dirs.scratch.exists() {
        remove_dir(&dirs.scratch)?;
    }
    // Outputs of a file whose record was never written are from a run that died.
    let prefix = format!("{}-", spec.name);
    for path in read_dir(&dirs.samples)? {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(number) = name
            .strip_prefix(&prefix)
            .and_then(|rest| rest.strip_suffix(".parquet"))
        else {
            continue;
        };
        let ordinal: Option<usize> = (number.len() == 7)
            .then(|| number[..4].parse().ok())
            .flatten();
        if ordinal.is_some_and(|ordinal| !records.contains_key(&ordinal)) {
            remove_file(&path)?;
        }
    }
    for directory in [&dirs.seen, &dirs.entries, &dirs.hits] {
        for path in read_dir(directory)? {
            let ordinal: Option<usize> = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .and_then(|stem| stem.parse().ok());
            if ordinal.is_none_or(|ordinal| !records.contains_key(&ordinal)) {
                remove_file(&path)?;
            }
        }
    }
    Ok(records)
}

/// Process one downloaded file, move its outputs into place, and record it.
fn process_file<U: Upstream, P: Pass>(
    dirs: &StateDirs,
    spec: SourceSpec,
    ordinal: usize,
    file: &RepoFile,
    path: &Path,
    download_ms: u64,
    pass: &mut P,
) -> Result<FileRecord> {
    let started = Instant::now();
    let before = pass.filter_counts();
    let mut out = FileOutput::new(dirs.scratch.join(format!("{ordinal:04}")), spec.name)?;
    let mut truncated = false;
    let text_bytes = U::read(path, &mut |batch| {
        out.census.documents += batch.len();
        let flow = pass.batch(spec, &batch, &mut out)?;
        truncated |= flow == Flow::Stop;
        Ok(flow)
    })?;
    let FileOutput {
        directory,
        source,
        writer,
        mut census,
        digests,
        entries,
        entry_rows,
        hits,
    } = out;
    census.files = 1;
    census.text_bytes = text_bytes;
    census.downloaded_bytes = file.bytes();
    census.download_ms = download_ms;
    census.set_filter_counts(counts_since(pass.filter_counts(), before));
    let parts = writer.finish()?.div_ceil(SAMPLES_PER_SHARD);
    if parts > MAX_PARTS {
        return Err(Error::Invariant(format!(
            "file {ordinal} produced {parts} sample shards, more than an output name can number"
        )));
    }
    if let Some(mut hits) = hits {
        hits.flush().map_err(|source| Error::Read {
            path: directory.join("hits.tsv"),
            source,
        })?;
    }

    // Everything is written aside; move it into place, the record last.
    for part in 0..parts {
        rename(
            &directory.join(format!("{source}-{part:05}.parquet")),
            &dirs
                .samples
                .join(format!("{source}-{ordinal:04}{part:03}.parquet")),
        )?;
    }
    let seen = directory.join("seen.bin");
    write_file(&seen, digests.concat().as_slice())?;
    rename(&seen, &dirs.seen.join(format!("{ordinal:04}.bin")))?;
    if !entry_rows.is_empty() {
        let rows = directory.join("entries.tsv");
        let mut writer = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .has_headers(false)
            .from_path(&rows)
            .map_err(|source| Error::Csv {
                path: rows.clone(),
                source,
            })?;
        for row in &entry_rows {
            writer.serialize(row).map_err(|source| Error::Csv {
                path: rows.clone(),
                source,
            })?;
        }
        writer.flush().map_err(|source| Error::Read {
            path: rows.clone(),
            source,
        })?;
        rename(&rows, &dirs.entries.join(format!("{ordinal:04}.tsv")))?;
    }
    if directory.join("hits.tsv").exists() {
        rename(
            &directory.join("hits.tsv"),
            &dirs.hits.join(format!("{ordinal:04}.tsv")),
        )?;
    }
    census.process_ms = millis(started);
    let record = FileRecord {
        ordinal,
        path: file.path.clone(),
        truncated,
        census,
        entries,
    };
    let staged = directory.join("record.json");
    write_file(&staged, &serialize(&record)?)?;
    rename(&staged, &dirs.record(ordinal))?;
    remove_dir(&directory)?;
    match path.parent() {
        Some(download) => remove_dir(download)?,
        None => remove_file(path)?,
    }
    Ok(record)
}

/// The source's running totals, rewritten after every file.
#[derive(serde::Serialize)]
struct Summary<'a> {
    census: &'a Census,
    filter_yield: f64,
}

fn write_summary(dirs: &StateDirs, total: &Census) -> Result<()> {
    let path = dirs.root.join("census.json");
    let staged = dirs.root.join("census.json.part");
    write_file(
        &staged,
        &serialize(&Summary {
            census: total,
            filter_yield: total.filter_yield(),
        })?,
    )?;
    rename(&staged, &path)
}

/// Every target's kept count, summed over the finished files' records.
///
/// # Errors
///
/// If a record cannot be read.
pub fn entry_counts(root: &Path, source: &str) -> Result<BTreeMap<String, usize>> {
    let directory = root.join("ingest").join(source).join("files");
    let mut counts = BTreeMap::new();
    for path in read_dir(&directory)? {
        let text = std::fs::read_to_string(&path).map_err(|source| Error::Read {
            path: path.clone(),
            source,
        })?;
        let record: FileRecord = serde_json::from_str(&text).map_err(|source| Error::JsonLine {
            path: path.clone(),
            line: 1,
            source,
        })?;
        for (entry, count) in record.entries {
            *counts.entry(entry).or_default() += count;
        }
    }
    Ok(counts)
}

fn serialize<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    serde_json::to_vec_pretty(value)
        .map_err(|error| Error::Invariant(format!("a run record does not serialise: {error}")))
}

fn read_digests(path: &Path) -> Result<Vec<Digest>> {
    let bytes = std::fs::read(path).map_err(|source| Error::Read {
        path: path.to_owned(),
        source,
    })?;
    if bytes.len() % DEDUPE_DIGEST_BYTES != 0 {
        return Err(Error::Invariant(format!(
            "{} is {} bytes, not a whole number of digests",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes.as_chunks::<DEDUPE_DIGEST_BYTES>().0.to_vec())
}

fn read_dir(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(directory)
        .map_err(|source| Error::Read {
            path: directory.to_owned(),
            source,
        })?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<_>>()
        .map_err(|source| Error::Read {
            path: directory.to_owned(),
            source,
        })?;
    paths.sort();
    Ok(paths)
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes).map_err(|source| Error::Read {
        path: path.to_owned(),
        source,
    })
}

fn rename(from: &Path, to: &Path) -> Result<()> {
    std::fs::rename(from, to).map_err(|source| Error::Read {
        path: from.to_owned(),
        source,
    })
}

fn remove_file(path: &Path) -> Result<()> {
    std::fs::remove_file(path).map_err(|source| Error::Read {
        path: path.to_owned(),
        source,
    })
}

fn remove_dir(path: &Path) -> Result<()> {
    std::fs::remove_dir_all(path).map_err(|source| Error::Read {
        path: path.to_owned(),
        source,
    })
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "a run lasts far less than 2^64 milliseconds"
)]
fn millis(started: Instant) -> u64 {
    started.elapsed().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mine::{Category, Target, Word};
    use crate::source::FINEWEB2_HQ;
    use ime_g2p::shards::read_shards;
    use ime_pinyin::SyllableTable;

    /// A local upstream: one document per line of a plain text file.
    struct Lines;

    impl Upstream for Lines {
        const SPEC: SourceSpec = FINEWEB2_HQ;
        const PREFIX: &'static str = "";

        fn selects(_: &str) -> bool {
            true
        }

        fn order<T>(files: Vec<T>) -> Vec<T> {
            files
        }

        fn read(
            path: &Path,
            sink: &mut dyn FnMut(Vec<RawDocument>) -> Result<Flow>,
        ) -> Result<u64> {
            let text = std::fs::read_to_string(path).expect("the fixture reads");
            let batch: Vec<RawDocument> = text
                .lines()
                .map(|line| RawDocument {
                    document_id: line.to_owned(),
                    source: Self::SPEC.name.to_owned(),
                    parts: vec![line.to_owned()],
                })
                .collect();
            sink(batch)?;
            Ok(text.len() as u64)
        }
    }

    const CONTEXT: &str = "今天下午的雨下得特别大啊，这下我们只好躲在书店里等雨停了，";

    struct Fixture {
        root: PathBuf,
        dirs: StateDirs,
        lexicon: Lexicon,
        guard: EvalGuard,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let root = std::env::temp_dir()
                .join(format!("ime-corpus-ingest-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            let dirs =
                StateDirs::new(&root, Lines::SPEC.name).expect("the fixture root is writable");
            Self {
                root,
                dirs,
                lexicon: Lexicon::load(&SyllableTable::load()).expect("the pinyin tables agree"),
                guard: EvalGuard::new([("窗外桂花真香", Some(CONTEXT)), ("一张旧车票", None)]),
            }
        }

        /// Write `documents` as upstream file `ordinal` and describe it as listed.
        fn upstream(&self, ordinal: usize, documents: &[&str]) -> (RepoFile, PathBuf) {
            let directory = self.dirs.downloads.join(format!("{ordinal:04}"));
            std::fs::create_dir_all(&directory).expect("writable");
            let path = directory.join(format!("{ordinal}.txt"));
            let body = documents.join("\n");
            std::fs::write(&path, &body).expect("writable");
            let file = RepoFile {
                kind: "file".to_owned(),
                path: format!("fixture/{ordinal}.txt"),
                size: body.len() as u64,
                lfs: None,
            };
            (file, path)
        }

        fn process<P: Pass>(&self, ordinal: usize, documents: &[&str], pass: &mut P) -> FileRecord {
            let (file, path) = self.upstream(ordinal, documents);
            process_file::<Lines, P>(&self.dirs, Lines::SPEC, ordinal, &file, &path, 0, pass)
                .expect("the fixture file processes")
        }

        fn samples(&self) -> Vec<Sample> {
            read_shards(&self.dirs.samples, Lines::SPEC.name).expect("the samples read back")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn planted_evaluation_collisions_personal_data_and_run3_targets_never_reach_a_sample() {
        let fixture = Fixture::new("planted");
        let run3 = Run3::of(["这句话很正常哦"]);
        let shared = Shared::new(&fixture.lexicon, &fixture.guard, &run3).expect("builds");
        let mut pass = GeneralPass::new(&shared, None);
        let record = fixture.process(
            0,
            &[
                // Quotes the record's context and text, re-punctuated: dropped whole,
                // the unrelated sentence after it included.
                "他说只好躲在书店里等雨停了！窗外桂花真香。另外今天去公园散步了。",
                // A target equal to a held-out text.
                "一张旧车票。我们明天再见吧。",
                // A phone number takes its sentence, and run3's target is not taken twice.
                "我的电话是13812345678，有事找我。这句话很正常哦。",
            ],
            &mut pass,
        );
        let census = record.census;
        assert_eq!(census.documents, 3);
        assert_eq!(census.leaked_documents, 1);
        assert_eq!(census.eval_exact, 1);
        assert_eq!(census.personal.phone, 1);
        assert_eq!(census.in_run3, 1);
        assert_eq!(census.written, 1);

        let samples = fixture.samples();
        let texts: Vec<&str> = samples.iter().map(|sample| sample.text.as_str()).collect();
        assert_eq!(texts, vec!["我们明天再见吧"]);
        for sample in &samples {
            let context = sample.context.as_deref().unwrap_or_default();
            // The quoting document and the personal sentence are gone as targets
            // and as context alike.
            for planted in [
                "窗外桂花真香",
                "称得上还原",
                "公园散步",
                "13812345678",
                "有事找我",
            ] {
                assert!(
                    !sample.text.contains(planted) && !context.contains(planted),
                    "{planted} leaked"
                );
            }
            // A record too short for a window is guarded as a target, which is
            // what the exact check covers; its sentence may still be context.
            assert_ne!(sample.text, "一张旧车票");
            assert_eq!(sample.source, "fineweb2-hq");
        }
    }

    #[test]
    fn a_resumed_run_deduplicates_against_the_files_already_finished() {
        let fixture = Fixture::new("resume");
        let run3 = Run3::of(["无关的句子在这里"]);
        let shared = Shared::new(&fixture.lexicon, &fixture.guard, &run3).expect("builds");
        let mut first = GeneralPass::new(&shared, None);
        fixture.process(0, &["今天天气真不错啊。"], &mut first);

        // A new process: everything it knows comes back from the run state.
        let mut second = GeneralPass::new(&shared, None);
        let (listed, _) = fixture.upstream(0, &["今天天气真不错啊。"]);
        let (listed_next, _) = fixture.upstream(1, &[]);
        let records = restore(
            &fixture.dirs,
            &[listed, listed_next],
            Lines::SPEC,
            &mut second,
        )
        .expect("the state restores");
        assert_eq!(records.len(), 1);
        let record = fixture.process(1, &["今天天气真不错啊。明天也会是好天气。"], &mut second);
        assert_eq!(record.census.duplicate, 1);
        assert_eq!(record.census.written, 1);
        let mut texts: Vec<String> = fixture
            .samples()
            .into_iter()
            .map(|sample| sample.text)
            .collect();
        texts.sort();
        assert_eq!(texts, vec!["今天天气真不错啊", "明天也会是好天气"]);
        // Each file's samples carry its ordinal, so one file's outputs are findable.
        assert!(
            fixture
                .dirs
                .samples
                .join("fineweb2-hq-0000000.parquet")
                .exists()
        );
        assert!(
            fixture
                .dirs
                .samples
                .join("fineweb2-hq-0001000.parquet")
                .exists()
        );
    }

    #[test]
    fn what_an_unfinished_file_left_is_removed_on_restore() {
        let fixture = Fixture::new("leftovers");
        let run3 = Run3::of(["无关的句子在这里"]);
        let shared = Shared::new(&fixture.lexicon, &fixture.guard, &run3).expect("builds");
        let mut pass = GeneralPass::new(&shared, None);
        fixture.process(0, &["今天天气真不错啊。"], &mut pass);
        let stale_sample = fixture.dirs.samples.join("fineweb2-hq-0001000.parquet");
        let stale_seen = fixture.dirs.seen.join("0001.bin");
        std::fs::copy(
            fixture.dirs.samples.join("fineweb2-hq-0000000.parquet"),
            &stale_sample,
        )
        .expect("copied");
        std::fs::write(&stale_seen, [0_u8; DEDUPE_DIGEST_BYTES]).expect("written");
        let (zero, _) = fixture.upstream(0, &[]);
        let (one, _) = fixture.upstream(1, &[]);
        let mut fresh = GeneralPass::new(&shared, None);
        restore(&fixture.dirs, &[zero, one], Lines::SPEC, &mut fresh).expect("the state restores");
        assert!(!stale_sample.exists());
        assert!(!stale_seen.exists());
        assert!(
            fixture
                .dirs
                .samples
                .join("fineweb2-hq-0000000.parquet")
                .exists()
        );
    }

    #[test]
    fn a_listing_that_moved_under_a_finished_file_is_refused() {
        let fixture = Fixture::new("moved");
        let run3 = Run3::of(["无关的句子在这里"]);
        let shared = Shared::new(&fixture.lexicon, &fixture.guard, &run3).expect("builds");
        let mut pass = GeneralPass::new(&shared, None);
        fixture.process(0, &["今天天气真不错啊。"], &mut pass);
        let (mut other, _) = fixture.upstream(0, &[]);
        other.path = "fixture/elsewhere.txt".to_owned();
        let mut fresh = GeneralPass::new(&shared, None);
        assert!(restore(&fixture.dirs, &[other], Lines::SPEC, &mut fresh).is_err());
    }

    #[test]
    fn the_target_budget_truncates_the_file_it_is_reached_in() {
        let fixture = Fixture::new("budget");
        let run3 = Run3::of(["无关的句子在这里"]);
        let shared = Shared::new(&fixture.lexicon, &fixture.guard, &run3).expect("builds");
        let mut pass = GeneralPass::new(&shared, Some(2));
        let record = fixture.process(
            0,
            &["第一句话很短。第二句话也很短。第三句话还是很短。"],
            &mut pass,
        );
        assert!(record.truncated);
        assert_eq!(record.census.written, 2);
    }

    fn matcher() -> EntryMatcher {
        EntryMatcher::new(
            vec![
                Word {
                    text: "一丁不识".to_owned(),
                    category: Category::Idiom,
                },
                Word {
                    text: "画蛇添足".to_owned(),
                    category: Category::Idiom,
                },
            ],
            vec![Target {
                entry: "一丁不识".to_owned(),
                category: Category::Idiom,
            }],
        )
        .expect("the fixture lexicon builds")
    }

    #[test]
    fn mining_keeps_real_usage_once_per_entry_up_to_the_cap() {
        let fixture = Fixture::new("mine");
        let matcher = matcher();
        let tail = "后来他慢慢学会了读书写字也成了一个很有学问的人。";
        let names: Vec<char> = "一二三四五六七八九十百千万天地人日月山水火木金土上下左右前后东西南北春夏秋冬风雨雷电花草树叶红黄蓝白黑大小多少高低长短"
            .chars()
            .collect();
        let documents: Vec<String> = names
            .iter()
            .take(PER_ENTRY_CAP + 1)
            .map(|name| format!("{name}{name}小时候家里穷得一丁不识。{tail}"))
            .chain([
                "他家里一丁不识。".to_owned(),
                "这是一句没有目标的话。".to_owned(),
            ])
            .collect();
        let run3 = Run3::of(documents.iter().map(String::as_str));
        let shared = Shared::new(&fixture.lexicon, &fixture.guard, &run3).expect("builds");
        let mut pass = MinePass::new(&shared, &matcher).expect("run3 has pairs");
        let borrowed: Vec<&str> = documents.iter().map(String::as_str).collect();
        let record = fixture.process(0, &borrowed, &mut pass);
        let census = record.census;
        // The sentence without a target is never normalised.
        assert_eq!(census.documents, PER_ENTRY_CAP + 3);
        assert_eq!(census.normalised_documents, PER_ENTRY_CAP + 2);
        assert_eq!(census.hit_sentences, PER_ENTRY_CAP + 2);
        assert_eq!(census.unusable_hits, 1);
        assert_eq!(census.capped, 1);
        assert_eq!(census.written, PER_ENTRY_CAP);
        assert_eq!(record.entries.get("一丁不识"), Some(&PER_ENTRY_CAP));
        assert_eq!(pass.kept(), &[PER_ENTRY_CAP]);
        assert!(fixture.dirs.hits.join("0000.tsv").exists());
        assert!(fixture.dirs.entries.join("0000.tsv").exists());
        let samples = fixture.samples();
        assert!(
            samples
                .iter()
                .all(|sample| sample.text.contains("一丁不识"))
        );

        // A resumed pass starts from the counts the finished file left.
        let mut resumed = MinePass::new(&shared, &matcher).expect("run3 has pairs");
        resumed
            .restore(&record, Vec::new())
            .expect("the entry is mined for");
        assert_eq!(resumed.kept(), &[PER_ENTRY_CAP]);
    }

    #[test]
    fn mining_drops_a_hit_in_a_document_that_quotes_an_evaluation_record() {
        let fixture = Fixture::new("mine-leak");
        let matcher = matcher();
        let document =
            "只好躲在书店里等雨停了，窗外桂花真香。他小时候家里穷得一丁不识，后来学会了读书。";
        let run3 = Run3::of([document]);
        let shared = Shared::new(&fixture.lexicon, &fixture.guard, &run3).expect("builds");
        let mut pass = MinePass::new(&shared, &matcher).expect("run3 has pairs");
        let record = fixture.process(0, &[document], &mut pass);
        assert_eq!(record.census.leaked_documents, 1);
        assert_eq!(record.census.written, 0);
        assert_eq!(
            shard_paths(&fixture.dirs.samples, Lines::SPEC.name).expect("the directory lists"),
            Vec::<PathBuf>::new()
        );
    }
}
