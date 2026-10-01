//! `ime-cli replay`: an evaluation slice through the session API, one
//! keystroke at a time, measuring what each keystroke costs.
//!
//! A keystroke's cost is the product's: the input method calls
//! `Session::key` and reads `Session::candidates`, and this harness times
//! that per key pressed, on a fresh session per record, over the slice the
//! same hash `fused-eval` cuts by. The gate is structural but opt-in:
//! `--reference-first` first decodes every record through a baseline
//! session with every reuse lever off, then runs the measured pass alone
//! and compares tops after the baseline session has dropped — so peak RSS
//! and pool occupancy stay the product's. Without it the measured session
//! is the pool's only client, as it is in the product, and `mismatches`
//! in the report is `null`.

use crate::neural::{BackendArg, SliceArgs, load_emittable, load_set};
use crate::{load_ngram, tables};
use anyhow::{Context, Result};
use clap::Args;
use ime_decode::{Asked, Both, NoTransition, Transition};
use ime_lm::{CharLm, LmState};
use ime_neural::RouteA;
use ime_ngram::NgramModel;
use ime_pinyin::CharId;
use ime_session::{
    CacheReport, Cached, EngineStats, Fused, FusedConfig, Session, SessionOptions, Stages, Timed,
    TransitionStats,
};
use serde::Serialize;
use std::cell::RefCell;
#[cfg(target_os = "macos")]
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::info;

/// The `replay` command's arguments.
#[expect(
    clippy::struct_excessive_bools,
    reason = "the harness's levers are independent flags, as fused-eval's are"
)]
#[derive(Debug, Clone, Args)]
pub(crate) struct ReplayArgs {
    /// A model written by `train-ngram`. Required unless the run drops the
    /// transition or decodes with the character model alone.
    #[arg(
        long,
        required_unless_present_any = ["no_transition", "lm"],
        conflicts_with = "no_transition"
    )]
    model: Option<PathBuf>,
    /// A directory holding `charlm.onnx` and `charlm.json` from `mlime export
    /// char-lm`: the character model as the transition, alone or fused with
    /// the trigram when `--model` is given too.
    #[arg(long, conflicts_with = "no_transition")]
    lm: Option<PathBuf>,
    /// Weight on the character model's scores when it is fused with the
    /// trigram; alone, it is scored at one.
    #[arg(long, default_value = "1.0", requires = "lm")]
    lm_weight: f32,
    /// The JSON Lines evaluation set to replay.
    #[arg(long)]
    eval_set: PathBuf,
    /// A directory written by `mlime export route-a`: the sessions' emissions,
    /// computed live by the towers.
    #[arg(long)]
    route_a: PathBuf,
    /// Score the records without their contexts: every record's gate is
    /// zeroed, the `context off` twin.
    #[arg(long)]
    no_context: bool,
    /// Intra-op threads for the towers' sessions.
    #[arg(long, default_value = "1")]
    route_a_threads: NonZeroUsize,
    /// Intra-op threads for the character model's sessions.
    #[arg(long, default_value = "1", requires = "lm")]
    lm_threads: NonZeroUsize,
    /// The precision the Metal step packs its matmul weights in — the
    /// export's own dtype by default, fp16 to measure the half-precision
    /// pack. Ignored on every other backend.
    #[arg(long, value_enum, default_value_t, requires = "lm")]
    metal_weights: crate::neural::MetalWeightsArg,
    /// The character-prefix state cache's bound. For a model whose state
    /// row carries a sequence axis — the transformer — the bound counts
    /// prefix-trie nodes, one position's K and V per node across every
    /// layer, so the memory it stands for is the bound times the node's
    /// bytes (`cache_node_bytes` in the report); a state without a
    /// sequence axis — the LSTM — counts resident rows instead, one row
    /// per cached state. The replay's baseline session runs it at zero —
    /// every reuse lever off — and `--incremental` turns it on for the
    /// measured one.
    #[arg(long, default_value = "4096")]
    lm_cache: usize,
    /// Drop the n-gram and decode on the emissions alone.
    #[arg(long)]
    no_transition: bool,
    /// The same emittable set the lattice was written with.
    #[arg(long)]
    emittable: PathBuf,
    /// What a candidate the model has no output row for scores. Must match
    /// the floor the score file was written under.
    #[arg(long, default_value = "-30.0")]
    unscored: f32,
    /// Which backend the ONNX Runtime sessions run on -- the character
    /// model's and the towers' alike. Asking for a provider the binary was
    /// not compiled with is an error, never a silent CPU session.
    #[arg(long, value_enum)]
    backend: Option<BackendArg>,
    /// ONNX Runtime's verbose session logging: the provider each graph node
    /// lands on, the evidence for whether a backend runs the step.
    #[arg(long)]
    pub(crate) ort_verbose: bool,
    /// Reuse what a keystroke did not change -- the context tower's output
    /// while the context stands. Off is the baseline: every keystroke
    /// re-runs the whole pipeline.
    #[arg(long)]
    incremental: bool,
    /// Run the all-off reference pass to completion before the measured
    /// session starts, then compare the measured tops against it. The
    /// default has no gate at all: the measured session is the LM pool's
    /// only client — the product picture — and the report's `mismatches`
    /// is `null`.
    #[arg(long)]
    reference_first: bool,
    /// Pause the measured pass at this many pending keystrokes for the
    /// given seconds, so `vmmap --summary <pid>` samples the resident set
    /// at steady state. The keystroke continues when the sleep returns.
    #[arg(long, value_name = "TYPED")]
    pause_at: Option<usize>,
    /// Seconds `--pause-at` sleeps; the default gives time to attach
    /// vmmap and let it finish.
    #[arg(long, default_value_t = 90, requires = "pause_at")]
    pause_secs: u64,
    /// Cap the records replayed.
    #[arg(long)]
    records: Option<usize>,
    /// Where the JSON report is written.
    #[arg(long)]
    out: PathBuf,
    /// Replay only the record at this eval-set index with the LM's real
    /// `advance` calls recorded: at the first keystroke the incremental
    /// and baseline sessions' candidates differ, both top lists print,
    /// and every recorded call is re-scored on a fresh `CharLm` — each
    /// row alone and in the batch the call ran — so a score that moved
    /// with its batchmates shows directly as a bit-diff. Requires `--lm`.
    #[arg(long, value_name = "INDEX")]
    probe: Option<usize>,
    #[command(flatten)]
    slice: SliceArgs,
    #[command(flatten)]
    search: crate::SearchArgs,
}

/// A latency summary in milliseconds.
#[derive(Serialize)]
struct Summary {
    count: usize,
    median_ms: f64,
    p95_ms: f64,
    max_ms: f64,
}

/// One typed-length band's latency summary.
#[derive(Serialize)]
struct BandReport {
    /// The band's label, `"9-16"`-shaped or `"65+"`.
    band: &'static str,
    #[serde(flatten)]
    summary: Summary,
}

/// One keystroke's cost and where it went, in the JSON report.
#[derive(Serialize)]
struct Keystroke {
    /// The record's index in the eval set.
    record: usize,
    /// How many keystrokes are pending after this one.
    typed: usize,
    /// The whole `key` call.
    key_ns: u64,
    /// Of it: segmentation plus the candidate masks.
    segment_ns: u64,
    /// The engine's emission — the towers' forwards.
    emission_ns: u64,
    /// The beam decode.
    decode_ns: u64,
    /// Of it: candidate expansion — the score calls and the merge index.
    beam_expand_ns: u64,
    /// Survivor selection — guards, partition, head sort.
    beam_select_ns: u64,
    /// The finish pass — `Transition::finish`, reconstruct, `keep_diverse`.
    beam_finish_ns: u64,
    /// (beam, character) pairs the expansion scored.
    beam_expanded: u64,
}

/// A record whose session answer disagreed with the baseline's.
#[derive(Serialize)]
struct Mismatch {
    /// The record's index in the eval set.
    record: usize,
    /// Its keystrokes.
    pinyin: String,
    /// The baseline session's top candidate.
    expected: Option<String>,
    /// The measured session's.
    got: Option<String>,
    /// The first keystroke whose top candidate already disagreed — the
    /// record replayed in lockstep to localise the divergence.
    first_divergence: Option<usize>,
}

/// The JSON report `replay` writes.
#[derive(Serialize)]
struct Report {
    /// What the run was given, so the numbers carry their shape.
    config: ConfigEcho,
    /// Records replayed.
    records: usize,
    /// Records whose top candidate disagreed with the baseline's — `None`
    /// when no gate ran (the measured session was the pool's only client).
    mismatches: Option<Vec<Mismatch>>,
    /// Latency summaries: overall, then per typed-length band.
    latency: LatencyReport,
    /// Where the time went, cumulative.
    stages: StageReport,
    /// How much of the lattice each keystroke changed.
    invalidation: InvalidationReport,
    /// The process's peak resident set, bytes.
    peak_rss_bytes: i64,
    /// Its physical footprint: resident plus compressed, bytes.
    phys_footprint_bytes: i64,
    /// Every keystroke's timings, in order.
    samples: Vec<Keystroke>,
}

/// The run's shape, echoed into the report.
#[derive(Serialize)]
struct ConfigEcho {
    eval_set: String,
    slice: String,
    dev_share: f64,
    records_cap: Option<usize>,
    incremental: bool,
    backend: String,
    route_a_threads: usize,
    lm_threads: usize,
    lm_cache: usize,
    with_context: bool,
    transition: String,
    lm_weight: f32,
    unscored: f32,
    beam_width: usize,
    top_k: usize,
    max_paths: usize,
    incomplete_tail: bool,
}

/// Latency at whole-run and per-band grain.
#[derive(Serialize)]
struct LatencyReport {
    all: Summary,
    by_typed: Vec<BandReport>,
}

/// Cumulative stage totals, in nanos where they are timings.
#[derive(Serialize)]
struct StageReport {
    refreshes: u64,
    undecodable: u64,
    segment_ns: u64,
    emission_ns: u64,
    decode_ns: u64,
    candidates_ns: u64,
    context_calls: u64,
    context_cached: u64,
    context_ns: u64,
    fill_ns: u64,
    /// Fill forwards run — one per keystroke's refresh.
    fill_calls: u64,
    /// Tokens the fill graph encoded, `rows * width` summed over calls.
    fill_tokens: u64,
    transition_start_calls: u64,
    transition_start_ns: u64,
    transition_advance_calls: u64,
    transition_advance_ns: u64,
    transition_advanced_states: u64,
    /// Inside `beam_expand_ns`: the transition's own `score` calls and
    /// their nanos — the trigram/LM term of every scored pair.
    transition_score_calls: u64,
    /// Their nanos.
    transition_score_ns: u64,
    /// The state cache's `start` hits and misses.
    cache_start_hits: u64,
    /// `start` misses.
    cache_start_misses: u64,
    /// The state cache's `advance` row hits and misses.
    cache_step_hits: u64,
    /// `advance` row misses.
    cache_step_misses: u64,
    /// Entries evicted under the bound, and held at the end.
    cache_evictions: u64,
    /// Entries held.
    cache_entries: u64,
    /// The character model's bytes per trie node — `0` when the run had no
    /// page cache to size.
    cache_node_bytes: usize,
    /// Its live trie nodes at the end.
    cache_live_nodes: usize,
    /// Host-side nanos the LM spent on its page bookkeeping — the `page_row`
    /// builds, scratch materialisation, claims and page writes — inside
    /// `transition_advance_ns`. The misses' own compute is the difference.
    lm_bookkeeping_ns: u64,
    /// Inside `decode_ns` + `candidates_ns`: candidate expansion (the
    /// score calls and the merge index).
    beam_expand_ns: u64,
    /// Survivor selection (guards, partition, head sort).
    beam_select_ns: u64,
    /// The finish pass (`Transition::finish`, reconstruct, `keep_diverse`).
    beam_finish_ns: u64,
    /// (beam, character) pairs the expansion scored, cumulative.
    beam_expanded: u64,
}

/// The lattice-invalidation counters, echoed.
#[derive(Serialize)]
struct InvalidationReport {
    compared: u64,
    stable: u64,
    first_divergence: std::collections::BTreeMap<usize, u64>,
}

/// The typed-length bands the table reports at.
const BANDS: &[(&str, usize, usize)] = &[
    ("1-4", 1, 4),
    ("5-8", 5, 8),
    ("9-16", 9, 16),
    ("17-24", 17, 24),
    ("25-32", 25, 32),
    ("33-48", 33, 48),
    ("49-64", 49, 64),
    ("65+", 65, usize::MAX),
];

/// Which band a typed length lands in.
fn band_index(typed: usize) -> usize {
    BANDS
        .iter()
        .position(|(_, lo, hi)| *lo <= typed && typed <= *hi)
        .expect("the bands cover every length")
}

/// The `num`/`den` quantile's nearest rank in a sorted slice.
fn rank(len: usize, num: usize, den: usize) -> usize {
    (len.saturating_mul(num).saturating_add(den - 1) / den).clamp(1, len.max(1))
}

/// Median, p95 (nearest-rank) and max of *samples*, nanos in, millis out.
#[expect(
    clippy::cast_precision_loss,
    reason = "keystroke nanos never approach 2^53; the f64 mantissa holds them"
)]
fn summarize(samples: &[u64]) -> Summary {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let millis = |ns: u64| ns as f64 / 1e6;
    if sorted.is_empty() {
        return Summary {
            count: 0,
            median_ms: 0.0,
            p95_ms: 0.0,
            max_ms: 0.0,
        };
    }
    Summary {
        count: sorted.len(),
        median_ms: millis(sorted[rank(sorted.len(), 1, 2) - 1]),
        p95_ms: millis(sorted[rank(sorted.len(), 95, 100) - 1]),
        max_ms: millis(sorted[sorted.len() - 1]),
    }
}

/// The process's peak resident set: `ru_maxrss`, bytes on the platforms the
/// product runs on.
fn peak_rss() -> i64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `usage` points at valid, owned memory and the call writes it.
    let ok = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if ok == 0 {
        // SAFETY: getrusage succeeded, so the struct is initialised.
        unsafe { usage.assume_init() }.ru_maxrss
    } else {
        0
    }
}

/// The physical footprint `vmmap` and `/usr/bin/time` report: resident
/// plus compressed pages, unlike `ru_maxrss`'s resident-only set. Zero
/// where `proc_pid_rusage` does not exist.
fn phys_footprint() -> i64 {
    #[cfg(target_os = "macos")]
    {
        let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::uninit();
        // SAFETY: `info` points at valid, owned memory the call writes.
        let ok = unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                info.as_mut_ptr().cast::<libc::rusage_info_t>(),
            )
        };
        if ok == 0 {
            // SAFETY: the call succeeded, so the struct is initialised.
            return i64::try_from(unsafe { info.assume_init() }.ri_phys_footprint)
                .unwrap_or(i64::MAX);
        }
    }
    0
}

/// `vmmap --summary`'s data read straight out of the kernel: `mach_vm_region`
/// on `mach_task_self` needs no `task_for_pid` permission, so the breakdown
/// works where external inspection does not. A file-backed region's
/// `external_pager` is set — mapped file pages are clean and cheap to drop,
/// so they inflate `ru_maxrss` without costing real memory pressure.
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Default)]
struct VmRegionExtendedInfo {
    protection: i32,
    user_tag: u32,
    pages_resident: u32,
    pages_shared_now_private: u32,
    pages_swapped_out: u32,
    pages_dirtied: u32,
    ref_count: u32,
    shadow_depth: u16,
    external_pager: u8,
    share_mode: u8,
    pages_reusable: u32,
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    /// Walk one region of the task's address map; `info` holds
    /// `infoCnt` elements on the way in.
    fn mach_vm_region(
        task: u32,
        address: *mut u64,
        size: *mut u64,
        flavor: i32,
        info: *mut VmRegionExtendedInfo,
        info_cnt: *mut u32,
        object_name: *mut u32,
    ) -> i32;
    /// Release the send right `mach_vm_region` reports for a mapped file.
    fn mach_port_deallocate(task: u32, name: u32) -> i32;
    /// This task's kernel handle — declared here because `libc`'s copy is
    /// deprecated in favour of a crate we do not depend on.
    fn mach_task_self() -> u32;
    /// The path a file-backed region maps — `libproc`, `PATH_MAX` buffer.
    fn proc_regionfilename(pid: i32, address: u64, buffer: *mut u8, buffer_size: u32) -> i32;
    /// Our own process id — `libc`'s `getpid`.
    fn getpid() -> i32;
    /// `libmalloc`'s purge of freed-but-retained pages across all zones —
    /// the answer to whether a resident heap is live data or allocator
    /// retention: live bytes stay, retained bytes go back to the kernel.
    fn malloc_zone_pressure_relief(zone: *mut u8, goal: usize) -> usize;
}

/// A `user_tag`'s display name — the subset `vmmap` prints that this
/// process's regions can carry.
#[cfg(target_os = "macos")]
fn vm_tag_name(tag: u32) -> String {
    let name = match tag {
        0 => "unclassified",
        1 => "MALLOC",
        2 => "MALLOC small",
        3 => "MALLOC large",
        4 => "MALLOC huge",
        5 => "SBRK",
        7 => "MALLOC tiny",
        10 => "analysis tool",
        11 => "MALLOC nano",
        30 => "STACK",
        42 => "shared pmap",
        43 => "shared region",
        61 => "dylib",
        74 | 75 => "dyld",
        76 => "shared cache",
        97 => "page table",
        99 => "untagged jit",
        _ => return format!("tag {tag}"),
    };
    name.to_owned()
}

/// The basename of the file a mapped region points at — `None` for
/// `proc_regionfilename` failures (anonymous regions and the dyld shared
/// cache report no path).
#[cfg(target_os = "macos")]
fn file_name(address: u64) -> Option<String> {
    let mut buffer = [0u8; 1024];
    // SAFETY: `buffer` is ours and `buffer_size` is its real length.
    let len = unsafe {
        proc_regionfilename(
            getpid(),
            address,
            buffer.as_mut_ptr(),
            u32::try_from(buffer.len()).expect("a path buffer fits u32"),
        )
    };
    if len <= 0 {
        return None;
    }
    let path = String::from_utf8_lossy(
        &buffer[..usize::try_from(len).expect("a positive result length fits usize")],
    );
    Some(
        path.rsplit('/')
            .next()
            .filter(|name| !name.is_empty())
            .unwrap_or(path.as_ref())
            .to_owned(),
    )
}

/// The address space's resident set grouped by `(user tag, file-backed)` —
/// the steady-state breakdown the memory note quotes; it prints where the
/// RSS and footprint actually sit rather than inferring them from totals.
#[cfg(target_os = "macos")]
fn memory_breakdown(label: &str) {
    const EXTENDED: i32 = 13;
    let page_size = u64::try_from(
        usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap_or(16_384),
    )
    .unwrap_or(16_384);
    // SAFETY: reads our own task port; no lifetimes to violate.
    let task = unsafe { mach_task_self() };
    let mut rows: HashMap<(String, bool), [u64; 4]> = HashMap::new();
    let mut address = 0u64;
    loop {
        let mut size = 0u64;
        let mut info = VmRegionExtendedInfo::default();
        let mut count =
            u32::try_from(std::mem::size_of::<VmRegionExtendedInfo>() / 4).unwrap_or(20);
        let mut object = 0u32;
        // SAFETY: every out pointer is an owned local; `task` is our own
        // kernel handle; `info` has room for `count` ints.
        let rc = unsafe {
            mach_vm_region(
                task,
                &raw mut address,
                &raw mut size,
                EXTENDED,
                &raw mut info,
                &raw mut count,
                &raw mut object,
            )
        };
        if rc != 0 || size == 0 {
            break;
        }
        if object != 0 {
            // SAFETY: a live send right `mach_vm_region` just returned.
            unsafe { mach_port_deallocate(task, object) };
        }
        let name = if info.external_pager == 0 {
            vm_tag_name(info.user_tag)
        } else {
            file_name(address).unwrap_or_else(|| vm_tag_name(info.user_tag))
        };
        let entry = rows.entry((name, info.external_pager != 0)).or_default();
        entry[0] += u64::from(info.pages_resident) * page_size;
        entry[1] += u64::from(info.pages_swapped_out) * page_size;
        entry[2] += u64::from(info.pages_dirtied) * page_size;
        entry[3] += 1;
        address += size;
    }
    let mut sorted: Vec<((String, bool), [u64; 4])> = rows.into_iter().collect();
    sorted.sort_by_key(|(_, v)| std::cmp::Reverse(v[0]));
    for ((tag, mapped), [resident, swapped, dirtied, regions]) in sorted.iter().take(20) {
        info!(
            at = label,
            tag = tag.as_str(),
            mapped = mapped,
            regions = regions,
            resident_mb = resident / 1_000_000,
            swapped_mb = swapped / 1_000_000,
            dirtied_mb = dirtied / 1_000_000,
            "memory region"
        );
    }
    let resident: u64 = sorted.iter().map(|(_, v)| v[0]).sum();
    let swapped: u64 = sorted.iter().map(|(_, v)| v[1]).sum();
    info!(
        at = label,
        resident_mb = resident / 1_000_000,
        swapped_mb = swapped / 1_000_000,
        footprint_mb = phys_footprint() / 1_000_000,
        peak_rss_mb = peak_rss() / 1_000_000,
        "memory breakdown"
    );
}

#[cfg(not(target_os = "macos"))]
fn memory_breakdown(_: &str) {}

/// One typing session's state shared by every transition kind.
struct Harness<'a> {
    towers: &'a RouteA,
    emittable: &'a ime_decode::Emittable,
    lexicon: &'a ime_pinyin::Lexicon,
    table: &'a ime_pinyin::SyllableTable,
    args: &'a ReplayArgs,
}

/// The session type one transition kind makes: the transition timed for
/// the stage split, wrapped in the character-prefix state cache.
type FusedSession<'a, T> = Session<'a, Fused<'a, Cached<Timed<T>>>>;

impl<'a> Harness<'a> {
    /// A session over *transition*, decoding under the run's options;
    /// *incremental* selects the engine's reuse — the context tower's and
    /// the state cache's alike, the cache's rows bound dropping to zero
    /// when it is off.
    fn session<T: Transition>(&self, transition: T, incremental: bool) -> FusedSession<'a, T> {
        self.session_cached(transition, incremental, self.args.lm_cache)
    }

    /// The baseline session — the same towers, every reuse lever off.
    fn baseline_session<T: Transition>(&self, transition: T) -> FusedSession<'a, T> {
        let engine = Fused::new(
            self.towers,
            Cached::new(Timed::new(transition), 0),
            FusedConfig {
                emittable: self.emittable,
                lexicon: self.lexicon,
                weight: 1.0,
                floor: self.args.unscored,
                with_context: !self.args.no_context,
                incremental: false,
            },
        );
        Session::new(
            engine,
            SessionOptions {
                lexicon: self.lexicon,
                table: self.table,
                segment: self.args.search.segment(),
                beam: self.args.search.beam(),
                sentences: 3,
            },
        )
    }

    /// The same session with the state cache's capacity given directly —
    /// the probe's discriminating runs sit between the flags' two corners.
    fn session_cached<T: Transition>(
        &self,
        transition: T,
        incremental: bool,
        lm_cache: usize,
    ) -> FusedSession<'a, T> {
        let cache_rows = if incremental { lm_cache } else { 0 };
        let engine = Fused::new(
            self.towers,
            Cached::new(Timed::new(transition), cache_rows),
            FusedConfig {
                emittable: self.emittable,
                lexicon: self.lexicon,
                weight: 1.0,
                floor: self.args.unscored,
                with_context: !self.args.no_context,
                incremental,
            },
        );
        Session::new(
            engine,
            SessionOptions {
                lexicon: self.lexicon,
                table: self.table,
                segment: self.args.search.segment(),
                beam: self.args.search.beam(),
                sentences: 3,
            },
        )
    }
}

/// Copy the engine's `Cell` stats into an owned struct.
fn copy_engine(stats: &EngineStats) -> EngineStats {
    EngineStats {
        context_calls: std::cell::Cell::new(stats.context_calls.get()),
        context_cached: std::cell::Cell::new(stats.context_cached.get()),
        context_ns: std::cell::Cell::new(stats.context_ns.get()),
        fill_ns: std::cell::Cell::new(stats.fill_ns.get()),
        fill_calls: std::cell::Cell::new(stats.fill_calls.get()),
        fill_tokens: std::cell::Cell::new(stats.fill_tokens.get()),
    }
}

/// Copy the transition's `Cell` stats into an owned struct.
fn copy_transition(stats: &TransitionStats) -> TransitionStats {
    TransitionStats {
        start_calls: std::cell::Cell::new(stats.start_calls.get()),
        start_ns: std::cell::Cell::new(stats.start_ns.get()),
        advance_calls: std::cell::Cell::new(stats.advance_calls.get()),
        advance_ns: std::cell::Cell::new(stats.advance_ns.get()),
        advanced_states: std::cell::Cell::new(stats.advanced_states.get()),
        score_calls: std::cell::Cell::new(stats.score_calls.get()),
        score_ns: std::cell::Cell::new(stats.score_ns.get()),
    }
}

/// The collected timings one replay produces.
struct Replays {
    /// Every keystroke, in order.
    samples: Vec<Keystroke>,
    /// Records whose measured top 1 differed from the baseline's — `None`
    /// when the reference pass did not run and there was no gate.
    mismatches: Option<Vec<Mismatch>>,
    /// The measured session's cumulative stages.
    stages: Stages,
    /// Its invalidation counts.
    invalidation: ime_session::Invalidation,
    /// Its engine's cumulative stats.
    engine: EngineStats,
    /// Its transition's cumulative stats.
    transition: TransitionStats,
    /// Its state cache's cumulative counters.
    cache: CacheReport,
}

/// Replay the records under transition kind *kind* — the generic holds the
/// decode's exact `HISTORY` and `State` for each of the four configurations.
#[expect(
    clippy::too_many_lines,
    reason = "the pass reads top-down: sessions, keystrokes, tallies"
)]
fn replay_kind<T: Transition>(
    harness: &Harness<'_>,
    transition: impl Fn() -> T,
    records: &[(usize, &ime_eval::EvalRecord)],
) -> Result<Replays> {
    // The reference pass: the all-off session decodes each record once to
    // its final top 1, then drops before the measured session opens — the
    // only gate there is. Without `--reference-first` the measured session
    // is the pool's only client and `mismatches` reports `null`.
    let reference_tops = harness
        .args
        .reference_first
        .then(|| -> Result<Vec<Option<String>>> {
            let mut baseline = harness.baseline_session(transition());
            let mut tops = Vec::with_capacity(records.len());
            for &(index, record) in records {
                baseline.reset();
                if let Some(context) = record.context.as_deref() {
                    baseline
                        .set_context(context)
                        .map_err(anyhow::Error::from)
                        .with_context(|| format!("could not set record {index}'s context"))?;
                }
                for ch in record.pinyin.chars() {
                    baseline
                        .key(ch)
                        .map_err(anyhow::Error::from)
                        .with_context(|| format!("could not press {ch:?} on record {index}"))?;
                }
                tops.push(baseline.candidates().first().map(|c| c.text.clone()));
            }
            Ok(tops)
        })
        .transpose()?;

    let mut measured = harness.session(transition(), harness.args.incremental);
    let mut samples = Vec::new();
    let mut mismatches = reference_tops.as_ref().map(|_| Vec::new());
    let mut paused = false;
    for (position, &(index, record)) in records.iter().enumerate() {
        measured.reset();
        if let Some(context) = record.context.as_deref() {
            measured
                .set_context(context)
                .map_err(anyhow::Error::from)
                .with_context(|| format!("could not set record {index}'s context"))?;
        }
        let mut before = measured.stages().clone();
        for ch in record.pinyin.chars() {
            let tick = Instant::now();
            measured
                .key(ch)
                .map_err(anyhow::Error::from)
                .with_context(|| format!("could not press {ch:?} on record {index}"))?;
            let key_ns = nanos(tick);
            let now = measured.stages();
            let typed = measured.pending();
            samples.push(Keystroke {
                record: index,
                typed,
                key_ns,
                segment_ns: now.segment_ns - before.segment_ns,
                emission_ns: now.emission_ns - before.emission_ns,
                decode_ns: now.decode_ns - before.decode_ns
                    + (now.candidates_ns - before.candidates_ns),
                beam_expand_ns: now.breakdown.expand_ns - before.breakdown.expand_ns,
                beam_select_ns: now.breakdown.select_ns - before.breakdown.select_ns,
                beam_finish_ns: now.breakdown.finish_ns - before.breakdown.finish_ns,
                beam_expanded: now.breakdown.expanded - before.breakdown.expanded,
            });
            if !paused && harness.args.pause_at == Some(typed) {
                paused = true;
                info!(
                    pid = std::process::id(),
                    secs = harness.args.pause_secs,
                    footprint_mb = phys_footprint() / 1_000_000,
                    rss_mb = peak_rss() / 1_000_000,
                    "paused for memory sampling — `vmmap --summary <pid>` now"
                );
                memory_breakdown("pause");
                std::thread::sleep(std::time::Duration::from_secs(harness.args.pause_secs));
            }
            before = now.clone();
        }
        // The gate: the session's last top candidate is the full decode's,
        // which the finished reference pass computed with every lever off.
        if let Some(tops) = &reference_tops {
            let expected = tops[position].clone();
            let got = measured.candidates().first().map(|c| c.text.clone());
            if expected != got {
                // The reference pass dropped before the measured one; a
                // one-off baseline session replays this record in lockstep.
                let mut once = harness.baseline_session(transition());
                let first_divergence = first_divergence(record, &mut measured, &mut once)?;
                mismatches
                    .as_mut()
                    .expect("the gate vec lives when the reference pass ran")
                    .push(Mismatch {
                        record: index,
                        pinyin: record.pinyin.clone(),
                        expected,
                        got,
                        first_divergence,
                    });
            }
        }
    }
    Ok(Replays {
        samples,
        mismatches,
        stages: measured.stages().clone(),
        invalidation: measured.invalidation().clone(),
        engine: copy_engine(measured.engine().stats()),
        transition: copy_transition(measured.engine().transition().inner().stats()),
        cache: measured.engine().transition().stats(),
    })
}

/// The index of the first keystroke where the sessions' top candidates
/// disagree — the record replayed with the sessions stepping in lockstep,
/// so a mismatched record names where its states first parted rather than
/// just that they did.
fn first_divergence<T: Transition>(
    record: &ime_eval::EvalRecord,
    measured: &mut FusedSession<'_, T>,
    baseline: &mut FusedSession<'_, T>,
) -> Result<Option<usize>> {
    measured.reset();
    baseline.reset();
    if let Some(context) = record.context.as_deref() {
        measured
            .set_context(context)
            .map_err(anyhow::Error::from)
            .with_context(|| "could not set the divergence replay's context")?;
        baseline
            .set_context(context)
            .map_err(anyhow::Error::from)
            .with_context(|| "could not set the divergence replay's context")?;
    }
    for (keystroke, ch) in record.pinyin.chars().enumerate() {
        measured
            .key(ch)
            .map_err(anyhow::Error::from)
            .with_context(|| "could not press a key in the divergence replay")?;
        baseline
            .key(ch)
            .map_err(anyhow::Error::from)
            .with_context(|| "could not press a key in the divergence replay")?;
        let expected = baseline.candidates().first().map(|c| c.text.as_str());
        let got = measured.candidates().first().map(|c| c.text.as_str());
        if expected != got {
            return Ok(Some(keystroke));
        }
    }
    Ok(None)
}

/// An [`Instant::elapsed`] in nanos, saturating at a span no keystroke
/// reaches.
fn nanos(tick: Instant) -> u64 {
    u64::try_from(tick.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Which transition the flags named, for the report's echo.
fn transition_label(args: &ReplayArgs) -> &'static str {
    match (args.model.is_some(), args.lm.is_some()) {
        (true, true) => "ngram+lm",
        (true, false) => "ngram",
        (false, true) => "lm",
        (false, false) => "none",
    }
}

/// Build the report, log the table, and write the JSON. *cache* is the
/// character model's pool accounting — node bytes and live nodes — where
/// the run carried one.
fn report(
    args: &ReplayArgs,
    count: usize,
    replays: Replays,
    cache: Option<ime_lm::CacheStats>,
    bookkeeping_ns: u64,
    bk_debug: (u64, u64, u64, u64),
) -> Result<()> {
    let mut bands: Vec<Vec<u64>> = (0..BANDS.len()).map(|_| Vec::new()).collect();
    for sample in &replays.samples {
        bands[band_index(sample.typed)].push(sample.key_ns);
    }
    let all: Vec<u64> = replays.samples.iter().map(|s| s.key_ns).collect();
    let report = Report {
        config: ConfigEcho {
            eval_set: args.eval_set.display().to_string(),
            slice: args.slice.slice.label().to_owned(),
            dev_share: args.slice.dev_share,
            records_cap: args.records,
            incremental: args.incremental,
            backend: format!("{:?}", args.backend.unwrap_or_default()),
            route_a_threads: args.route_a_threads.get(),
            lm_threads: args.lm_threads.get(),
            lm_cache: if args.incremental { args.lm_cache } else { 0 },
            with_context: !args.no_context,
            transition: transition_label(args).to_owned(),
            lm_weight: args.lm_weight,
            unscored: args.unscored,
            beam_width: args.search.beam_width.get(),
            top_k: args.search.top_k.get(),
            max_paths: args.search.max_paths.get(),
            incomplete_tail: args.search.incomplete_tail,
        },
        records: count,
        mismatches: replays.mismatches,
        latency: LatencyReport {
            all: summarize(&all),
            by_typed: BANDS
                .iter()
                .zip(&bands)
                .filter(|(_, samples)| !samples.is_empty())
                .map(|((label, _, _), samples)| BandReport {
                    band: label,
                    summary: summarize(samples),
                })
                .collect(),
        },
        stages: StageReport {
            refreshes: replays.stages.refreshes,
            undecodable: replays.stages.undecodable,
            segment_ns: replays.stages.segment_ns,
            emission_ns: replays.stages.emission_ns,
            decode_ns: replays.stages.decode_ns,
            candidates_ns: replays.stages.candidates_ns,
            context_calls: replays.engine.context_calls.get(),
            context_cached: replays.engine.context_cached.get(),
            context_ns: replays.engine.context_ns.get(),
            fill_ns: replays.engine.fill_ns.get(),
            fill_calls: replays.engine.fill_calls.get(),
            fill_tokens: replays.engine.fill_tokens.get(),
            transition_start_calls: replays.transition.start_calls.get(),
            transition_start_ns: replays.transition.start_ns.get(),
            transition_advance_calls: replays.transition.advance_calls.get(),
            transition_advance_ns: replays.transition.advance_ns.get(),
            transition_advanced_states: replays.transition.advanced_states.get(),
            transition_score_calls: replays.transition.score_calls.get(),
            transition_score_ns: replays.transition.score_ns.get(),
            cache_start_hits: replays.cache.start_hits,
            cache_start_misses: replays.cache.start_misses,
            cache_step_hits: replays.cache.step_hits,
            cache_step_misses: replays.cache.step_misses,
            cache_evictions: replays.cache.evictions,
            cache_entries: replays.cache.entries,
            cache_node_bytes: cache.map_or(0, |stats| stats.node_bytes),
            cache_live_nodes: cache.map_or(0, |stats| stats.live),
            lm_bookkeeping_ns: bookkeeping_ns,
            beam_expand_ns: replays.stages.breakdown.expand_ns,
            beam_select_ns: replays.stages.breakdown.select_ns,
            beam_finish_ns: replays.stages.breakdown.finish_ns,
            beam_expanded: replays.stages.breakdown.expanded,
        },
        invalidation: InvalidationReport {
            compared: replays.invalidation.compared,
            stable: replays.invalidation.stable,
            first_divergence: replays.invalidation.first_divergence.clone(),
        },
        peak_rss_bytes: peak_rss(),
        phys_footprint_bytes: phys_footprint(),
        samples: replays.samples,
    };
    log_table(&report, bk_debug);
    let file = std::fs::File::create(&args.out)
        .with_context(|| format!("could not create {}", args.out.display()))?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &report)
        .context("could not write the replay report")
}

/// The short table to stderr: the summary, the bands, the stage split.
#[expect(
    clippy::cast_precision_loss,
    reason = "stage nanos never approach 2^53; the f64 mantissa holds them"
)]
fn log_table(report: &Report, bk_debug: (u64, u64, u64, u64)) {
    info!(
        count = report.latency.all.count,
        median_ms = report.latency.all.median_ms,
        p95_ms = report.latency.all.p95_ms,
        max_ms = report.latency.all.max_ms,
        peak_rss_mb = report.peak_rss_bytes / 1_000_000,
        phys_footprint_mb = report.phys_footprint_bytes / 1_000_000,
        mismatches = tracing::field::display(
            report
                .mismatches
                .as_ref()
                .map_or_else(|| "null".to_owned(), |m| m.len().to_string(),)
        ),
        "replay finished"
    );
    info!(
        backend = report.config.backend.as_str(),
        route_a_threads = report.config.route_a_threads,
        lm_threads = report.config.lm_threads,
        "thread counts"
    );
    memory_breakdown("finished");
    #[cfg(target_os = "macos")]
    {
        // SAFETY: `NULL` walks every zone; the return is bytes purged.
        let purged = unsafe { malloc_zone_pressure_relief(std::ptr::null_mut(), 0) };
        info!(purged_mb = purged / 1_048_576, "malloc pressure relief");
    }
    memory_breakdown("relieved");
    for band in &report.latency.by_typed {
        info!(
            band = band.band,
            count = band.summary.count,
            median_ms = band.summary.median_ms,
            p95_ms = band.summary.p95_ms,
            max_ms = band.summary.max_ms,
            "keystroke latency"
        );
    }
    info!(
        segment_ms = report.stages.segment_ns as f64 / 1e6,
        emission_ms = report.stages.emission_ns as f64 / 1e6,
        fill_ms = report.stages.fill_ns as f64 / 1e6,
        fill_calls = report.stages.fill_calls,
        fill_ms_per_call =
            report.stages.fill_ns as f64 / 1e6 / report.stages.fill_calls.max(1) as f64,
        fill_avg_tokens = report.stages.fill_tokens as f64 / report.stages.fill_calls.max(1) as f64,
        context_ms = report.stages.context_ns as f64 / 1e6,
        context_cached = report.stages.context_cached,
        decode_ms = report.stages.decode_ns as f64 / 1e6,
        candidates_ms = report.stages.candidates_ns as f64 / 1e6,
        transition_advance_ms = report.stages.transition_advance_ns as f64 / 1e6,
        lm_bookkeeping_ms = report.stages.lm_bookkeeping_ns as f64 / 1e6,
        bk_marshal_ms = bk_debug.0 as f64 / 1e6,
        bk_resolve_ms = bk_debug.1 as f64 / 1e6,
        bk_mat_ms = bk_debug.2 as f64 / 1e6,
        bk_mat_n = bk_debug.3,
        "stage split"
    );
    info!(
        expand_ms = report.stages.beam_expand_ns as f64 / 1e6,
        of_which_score_ms = report.stages.transition_score_ns as f64 / 1e6,
        select_ms = report.stages.beam_select_ns as f64 / 1e6,
        finish_ms = report.stages.beam_finish_ns as f64 / 1e6,
        expanded = report.stages.beam_expanded,
        score_calls = report.stages.transition_score_calls,
        "beam split"
    );
    let cache_rows = report.stages.cache_start_hits
        + report.stages.cache_start_misses
        + report.stages.cache_step_hits
        + report.stages.cache_step_misses;
    info!(
        cache_hit_rate = if cache_rows == 0 {
            0.0
        } else {
            (report.stages.cache_start_hits + report.stages.cache_step_hits) as f64
                / cache_rows as f64
        },
        cache_entries = report.stages.cache_entries,
        cache_evictions = report.stages.cache_evictions,
        cache_node_bytes = report.stages.cache_node_bytes,
        cache_live_nodes = report.stages.cache_live_nodes,
        "state cache"
    );
}

/// The `replay` command.
///
/// # Errors
///
/// If the tables, models, or eval set do not load, a session's decode fails,
/// or the report cannot be written.
#[expect(
    clippy::too_many_lines,
    reason = "arg validation, asset loading, and dispatch read top-down"
)]
pub fn run(args: &ReplayArgs) -> Result<()> {
    let (table, lexicon) = tables()?;
    memory_breakdown("tables");
    let backend = args.backend.unwrap_or_default();
    let towers = RouteA::open(
        &args.route_a,
        ime_neural::SessionShape {
            backend: backend.neural_backend(),
            intra_threads: args.route_a_threads,
            verbose_logging: args.ort_verbose,
        },
    )
    .with_context(|| format!("could not open the towers at {}", args.route_a.display()))?;
    let ngram = args
        .model
        .as_deref()
        .map(|path| load_ngram(path, &lexicon))
        .transpose()?;
    let lm = args
        .lm
        .as_deref()
        .map(|dir| {
            CharLm::open(
                dir,
                &lexicon,
                ime_lm::SessionShape {
                    backend: backend.backend(),
                    intra_threads: args.lm_threads,
                    width: args.search.beam_width,
                    verbose_logging: args.ort_verbose,
                    cache_rows: args.lm_cache,
                    metal_weights: args.metal_weights.weights(),
                },
            )
        })
        .transpose()?;
    let emittable = load_emittable(&args.emittable, &lexicon)?;
    let set = load_set(&args.eval_set)?;
    memory_breakdown("opened");
    if let Some(index) = args.probe {
        let record = set
            .records()
            .get(index)
            .with_context(|| format!("the eval set has no record {index}"))?;
        let lm = lm
            .as_ref()
            .context("--probe requires --lm: the calls it records are the model's")?;
        return probe(
            &Harness {
                towers: &towers,
                emittable: &emittable,
                lexicon: &lexicon,
                table: &table,
                args,
            },
            index,
            record,
            lm,
            ngram.as_ref(),
            args.lm.as_deref().expect("the model dir is checked"),
        );
    }
    let records: Vec<(usize, &ime_eval::EvalRecord)> = set
        .records()
        .iter()
        .enumerate()
        .filter(|(_, record)| args.slice.slice.slice().holds(record, args.slice.dev_share))
        .take(args.records.unwrap_or(usize::MAX))
        .collect();
    info!(
        records = records.len(),
        slice = args.slice.slice.label(),
        incremental = args.incremental,
        transition = transition_label(args),
        "replaying"
    );
    let harness = Harness {
        towers: &towers,
        emittable: &emittable,
        lexicon: &lexicon,
        table: &table,
        args,
    };
    let replays = match (ngram.as_ref(), lm.as_ref()) {
        (Some(model), Some(model_lm)) => replay_kind(
            &harness,
            || Both {
                first: model,
                first_weight: 1.0,
                second: model_lm,
                second_weight: args.lm_weight,
            },
            &records,
        )?,
        (Some(model), None) => replay_kind(&harness, || model, &records)?,
        (None, Some(model_lm)) => replay_kind(&harness, || model_lm, &records)?,
        (None, None) => replay_kind(&harness, || NoTransition, &records)?,
    };
    report(
        args,
        records.len(),
        replays,
        lm.as_ref().and_then(CharLm::cache_stats),
        lm.as_ref().map_or(0, CharLm::bookkeeping_ns),
        {
            let (mat_ms, mat_n) = lm.as_ref().map_or((0, 0), CharLm::bk_mat);
            (
                lm.as_ref().map_or(0, CharLm::bk_marshal_ns),
                lm.as_ref().map_or(0, CharLm::bk_resolve_ns),
                mat_ms,
                mat_n,
            )
        },
    )
}

/// One input row of a recorded `CharLm::advance` call: the produced
/// state's history, the character asked of it, and its request — enough
/// to rebuild the same step on a fresh model, alone or in the batch the
/// session ran it in.
#[derive(Debug)]
struct ProbeStep {
    /// The characters the input state covers, as [`LmState::history`]
    /// reports them.
    history: Vec<CharId>,
    /// The character this row extends the state by.
    ch: CharId,
    /// The request the produced state is scored under.
    candidates: Vec<CharId>,
    /// Whether `<eos>` rides the request.
    eos: bool,
}

/// A `CharLm` front recording every `advance` call's input rows. Sitting
/// inside the session's `Cached` it sees only the misses the model
/// really ran — the real batches a score's bits may depend on. The
/// session borrows it by reference, so the tape stays readable outside.
struct Probe<'a> {
    /// The model the calls forward to.
    inner: &'a CharLm,
    /// One entry per `advance` call, each a row list in call order.
    calls: RefCell<Vec<Vec<ProbeStep>>>,
    /// Every `score` the model emitted: the scored state's history, the
    /// candidate and the result's bits — the tape two sessions can diff
    /// to name a score that moved.
    scores: RefCell<Vec<(Vec<CharId>, CharId, u32)>>,
    /// Every `finish` the model emitted, same idea.
    finishes: RefCell<Vec<(Vec<CharId>, u32)>>,
    /// The scores attributed to the call that emitted them:
    /// `(call_index, row, candidate, bits)` — the `scores` tape is keyed
    /// by when the beam asked, which lets a corrupted state's moved
    /// scores hide in a later call's emission; this tape names the
    /// producing call.
    emitted: RefCell<Vec<(u32, u32, CharId, u32)>>,
}

impl Transition for Probe<'_> {
    const HISTORY: usize = CharLm::HISTORY;
    type State = LmState;

    fn start(&self, context: Option<&str>, asked: &Asked<'_>) -> Self::State {
        self.inner.start(context, asked)
    }

    fn score(&self, state: &Self::State, candidate: CharId) -> f32 {
        let score = self.inner.score(state, candidate);
        self.scores
            .borrow_mut()
            .push((state.history(), candidate, score.to_bits()));
        score
    }

    fn finish(&self, state: &Self::State) -> f32 {
        let score = self.inner.finish(state);
        self.finishes
            .borrow_mut()
            .push((state.history(), score.to_bits()));
        score
    }

    fn advance(&self, steps: &[(&Self::State, CharId, Asked<'_>)]) -> Vec<Self::State> {
        self.calls.borrow_mut().push(
            steps
                .iter()
                .map(|(state, ch, asked)| ProbeStep {
                    history: state.history(),
                    ch: *ch,
                    candidates: asked.candidates.to_vec(),
                    eos: asked.eos,
                })
                .collect(),
        );
        let produced = self.inner.advance(steps);
        let call = u32::try_from(self.calls.borrow().len() - 1).unwrap_or(u32::MAX);
        for (row, ((_, _, asked), state)) in steps.iter().zip(&produced).enumerate() {
            for &candidate in asked.candidates {
                self.emitted.borrow_mut().push((
                    call,
                    u32::try_from(row).unwrap_or(u32::MAX),
                    candidate,
                    self.inner.score(state, candidate).to_bits(),
                ));
            }
        }
        produced
    }

    fn compact(&self, state: &Self::State) -> Self::State {
        self.inner.compact(state)
    }
}

/// A text a probe prints: the character chain as the lexicon reads it.
fn probe_text(lexicon: &ime_pinyin::Lexicon, ids: &[CharId]) -> String {
    ids.iter()
        .map(|&id| lexicon.character(id))
        .collect::<String>()
}

/// Rebuild one recorded call's inputs on *model* — `start` plus one
/// `advance` per history character — then score the produced states
/// twice: once batched exactly as the recorded call ran them, once solo.
/// Bit-diffs between the two name a kernel whose result depends on its
/// batchmates.
fn probe_call(
    model: &CharLm,
    lexicon: &ime_pinyin::Lexicon,
    context: Option<&str>,
    call: &[ProbeStep],
) -> (usize, Vec<String>) {
    let inputs: Vec<LmState> = call
        .iter()
        .map(|row| {
            let asked = probe_asked(row);
            let mut state = model.start(context, &asked);
            for &ch in &row.history {
                state = model
                    .advance(&[(&state, ch, asked)])
                    .into_iter()
                    .next()
                    .expect("advance returns one state per step");
            }
            state
        })
        .collect();
    let steps: Vec<(&LmState, CharId, Asked<'_>)> = inputs
        .iter()
        .zip(call)
        .map(|(state, row)| (state, row.ch, probe_asked(row)))
        .collect();
    let batched = model.advance(&steps);
    let mut checked = 0usize;
    let mut diffs = Vec::new();
    for (row, (input, (produced, step))) in inputs.iter().zip(batched.iter().zip(call)).enumerate()
    {
        let asked = probe_asked(step);
        let solo = model
            .advance(&[(input, step.ch, asked)])
            .into_iter()
            .next()
            .expect("advance returns one state per step");
        for &candidate in &step.candidates {
            checked += 1;
            let batched_score = model.score(produced, candidate);
            let solo_score = model.score(&solo, candidate);
            if batched_score.to_bits() != solo_score.to_bits() {
                diffs.push(format!(
                    "row {row} (history {:?}+{}) candidate {}: batched {batched_score} 0x{:08x} solo {solo_score} 0x{:08x}",
                    probe_text(lexicon, &step.history),
                    lexicon.character(step.ch),
                    lexicon.character(candidate),
                    batched_score.to_bits(),
                    solo_score.to_bits(),
                ));
            }
        }
    }
    (checked, diffs)
}

/// The keystroke range `[start, end)` of a session's recorded calls at
/// each keystroke, in keystroke order.
type CallSpans = Vec<(usize, usize)>;

/// The `Asked` a recorded row asked for — its candidate list plus the
/// eos flag, borrowed so rebuilds reuse the step's own vectors.
fn probe_asked(row: &ProbeStep) -> Asked<'_> {
    Asked {
        candidates: &row.candidates,
        eos: row.eos,
    }
}

/// The produced candidate bits for a recorded call on *model* — the same
/// rebuild [`probe_call`] does, kept raw so the live pool's answers can be
/// put next to a fresh model's: a live-vs-fresh diff on identical logical
/// inputs says the pool's bytes for that chain are corrupted.
fn probe_call_bits(
    model: &CharLm,
    context: Option<&str>,
    call: &[ProbeStep],
) -> Vec<(usize, CharId, CharId, u32)> {
    let inputs: Vec<LmState> = call
        .iter()
        .map(|row| {
            let asked = probe_asked(row);
            let mut state = model.start(context, &asked);
            for &ch in &row.history {
                state = model
                    .advance(&[(&state, ch, asked)])
                    .into_iter()
                    .next()
                    .expect("advance returns one state per step");
            }
            state
        })
        .collect();
    let steps: Vec<(&LmState, CharId, Asked<'_>)> = inputs
        .iter()
        .zip(call)
        .map(|(state, row)| (state, row.ch, probe_asked(row)))
        .collect();
    let produced = model.advance(&steps);
    let mut bits = Vec::new();
    for (row, (state, step)) in produced.iter().zip(call).enumerate() {
        for &candidate in &step.candidates {
            bits.push((
                row,
                step.ch,
                candidate,
                model.score(state, candidate).to_bits(),
            ));
        }
    }
    bits
}

/// Every intermediate state's score for *asked*'s first candidate along
/// *history* — the prefix positions where two models first disagree on the
/// same logical input is where the corrupt byte lives.
fn probe_path_bits(
    model: &CharLm,
    context: Option<&str>,
    history: &[CharId],
    asked: Asked<'_>,
) -> Vec<u32> {
    let sentinel = asked
        .candidates
        .first()
        .copied()
        .expect("a recorded step asks for a candidate");
    let mut state = model.start(context, &asked);
    let mut bits = Vec::with_capacity(history.len());
    for &ch in history {
        state = model
            .advance(&[(&state, ch, asked)])
            .into_iter()
            .next()
            .expect("advance returns one state per step");
        bits.push(model.score(&state, sentinel).to_bits());
    }
    bits
}

/// What [`probe_lockstep`] returns: the first divergent keystroke, both
/// sessions' per-keystroke call spans, and the two candidate lists at the
/// divergence.
type Lockstep = (
    Option<usize>,
    CallSpans,
    CallSpans,
    Vec<String>,
    Vec<String>,
);

/// Step the measured and baseline sessions through *record* in lockstep,
/// returning the first keystroke their candidate texts differ at plus
/// the call spans each tape wrote per keystroke. `None` means identical.
fn probe_lockstep<M: Transition, B: Transition>(
    measured: &mut FusedSession<'_, M>,
    baseline: &mut FusedSession<'_, B>,
    record: &ime_eval::EvalRecord,
    tape: &RefCell<Vec<Vec<ProbeStep>>>,
    base_tape: &RefCell<Vec<Vec<ProbeStep>>>,
) -> Result<Lockstep> {
    let mut m_spans: CallSpans = Vec::new();
    let mut b_spans: CallSpans = Vec::new();
    let mut m_list = Vec::new();
    let mut b_list = Vec::new();
    let mut divergence = None;
    for (keystroke, ch) in record.pinyin.chars().enumerate() {
        let m_start = tape.borrow().len();
        let b_start = base_tape.borrow().len();
        measured
            .key(ch)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("could not press {ch:?} on the measured session"))?;
        baseline
            .key(ch)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("could not press {ch:?} on the baseline session"))?;
        m_spans.push((m_start, tape.borrow().len()));
        b_spans.push((b_start, base_tape.borrow().len()));
        let identical = measured
            .candidates()
            .iter()
            .map(|c| c.text.as_str())
            .eq(baseline.candidates().iter().map(|c| c.text.as_str()));
        if !identical {
            divergence = Some(keystroke);
            m_list = measured
                .candidates()
                .iter()
                .map(|c| c.text.clone())
                .collect();
            b_list = baseline
                .candidates()
                .iter()
                .map(|c| c.text.clone())
                .collect();
            break;
        }
    }
    Ok((divergence, m_spans, b_spans, m_list, b_list))
}

/// What one lockstep pair leaves behind: the keystroke the candidate
/// lists diverged at (`None` = identical), the call spans and candidate
/// lists of that point, and both sessions' full tapes — the LM calls
/// each ran and every score and finish each emitted.
struct ProbePair {
    /// First keystroke the lists differed at; `None` when identical.
    divergence: Option<usize>,
    /// The measured session's call span per keystroke.
    measured_spans: CallSpans,
    /// The baseline session's call span per keystroke.
    baseline_spans: CallSpans,
    /// Candidate texts at the divergence, measured then baseline.
    measured_list: Vec<String>,
    /// The baseline's candidate texts at the divergence.
    baseline_list: Vec<String>,
    /// The measured session's recorded `advance` calls.
    calls: Vec<Vec<ProbeStep>>,
    /// The baseline session's recorded `advance` calls.
    base_calls: Vec<Vec<ProbeStep>>,
    /// Every `(history, candidate, score bits)` the measured LM emitted.
    scores: Vec<(Vec<CharId>, CharId, u32)>,
    /// The baseline's emitted `(history, candidate, score bits)`.
    base_scores: Vec<(Vec<CharId>, CharId, u32)>,
    /// Measured `(call, row, candidate, bits)` — production-attributed.
    emitted: Vec<(u32, u32, CharId, u32)>,
    /// The baseline's production-attributed scores.
    base_emitted: Vec<(u32, u32, CharId, u32)>,
}

/// `--probe`: replay *record* against itself. The incremental session
/// and the all-off baseline step in lockstep with every `CharLm::advance`
/// call recorded; at the first keystroke their candidate lists differ,
/// both lists print, and each recorded call is re-run on a fresh
/// `CharLm` — batched as the session ran it and solo — so a score that
/// moved with its batchmates shows as a bit-diff. A second pair repeats
/// the lockstep with the session cache at zero: whether it still
/// diverges splits cache-restore causes from kernel or tower ones.
#[expect(
    clippy::too_many_lines,
    reason = "the report reads top-down: lockstep, divergence, tapes"
)]
fn probe(
    harness: &Harness<'_>,
    index: usize,
    record: &ime_eval::EvalRecord,
    lm: &CharLm,
    ngram: Option<&NgramModel>,
    lm_dir: &Path,
) -> Result<()> {
    info!(
        "probe record {index}: pinyin {:?} context {:?}",
        record.pinyin, record.context
    );
    let context = record.context.as_deref();
    // A lockstep pair on one probe pair; the session cache bound picks
    // which corner the measured side sits on.
    let lockstep = |lm_cache: usize| -> Result<ProbePair> {
        let tape = Probe {
            inner: lm,
            calls: RefCell::new(Vec::new()),
            scores: RefCell::new(Vec::new()),
            finishes: RefCell::new(Vec::new()),
            emitted: RefCell::new(Vec::new()),
        };
        let base_tape = Probe {
            inner: lm,
            calls: RefCell::new(Vec::new()),
            scores: RefCell::new(Vec::new()),
            finishes: RefCell::new(Vec::new()),
            emitted: RefCell::new(Vec::new()),
        };
        let (divergence, m_spans, b_spans, m_list, b_list) = if let Some(model) = ngram {
            let inc = Both {
                first: model,
                first_weight: 1.0,
                second: &tape,
                second_weight: harness.args.lm_weight,
            };
            let base = Both {
                first: model,
                first_weight: 1.0,
                second: &base_tape,
                second_weight: harness.args.lm_weight,
            };
            let mut measured = harness.session_cached(&inc, true, lm_cache);
            let mut baseline = harness.session(&base, false);
            if let Some(context) = context {
                measured
                    .set_context(context)
                    .map_err(anyhow::Error::from)
                    .context("could not set the measured session's context")?;
                baseline
                    .set_context(context)
                    .map_err(anyhow::Error::from)
                    .context("could not set the baseline session's context")?;
            }
            probe_lockstep(
                &mut measured,
                &mut baseline,
                record,
                &tape.calls,
                &base_tape.calls,
            )?
        } else {
            let mut measured = harness.session_cached(&tape, true, lm_cache);
            let mut baseline = harness.session(&base_tape, false);
            if let Some(context) = context {
                measured
                    .set_context(context)
                    .map_err(anyhow::Error::from)
                    .context("could not set the measured session's context")?;
                baseline
                    .set_context(context)
                    .map_err(anyhow::Error::from)
                    .context("could not set the baseline session's context")?;
            }
            probe_lockstep(
                &mut measured,
                &mut baseline,
                record,
                &tape.calls,
                &base_tape.calls,
            )?
        };
        Ok(ProbePair {
            divergence,
            measured_spans: m_spans,
            baseline_spans: b_spans,
            measured_list: m_list,
            baseline_list: b_list,
            calls: tape.calls.into_inner(),
            base_calls: base_tape.calls.into_inner(),
            scores: tape.scores.into_inner(),
            base_scores: base_tape.scores.into_inner(),
            emitted: tape.emitted.into_inner(),
            base_emitted: base_tape.emitted.into_inner(),
        })
    };
    let pair = lockstep(harness.args.lm_cache)?;
    let (divergence, m_spans, b_spans, calls, base_calls) = (
        pair.divergence,
        &pair.measured_spans,
        &pair.baseline_spans,
        &pair.calls,
        &pair.base_calls,
    );
    let keystrokes = record.pinyin.chars().count();
    if let Some(keystroke) = divergence {
        info!(
            "probe: candidates diverge at keystroke {} of {keystrokes}",
            keystroke + 1
        );
        info!("probe: measured  top-8 {:?}", pair.measured_list);
        info!("probe: baseline  top-8 {:?}", pair.baseline_list);
    } else {
        info!("probe: no divergence over {keystrokes} keystrokes");
    }
    // The discriminating run: the same pair with the session cache at
    // zero leaves the towers' incremental reuse standing alone. A
    // divergence here cannot come from the LM's cache.
    let zero = lockstep(0)?;
    if let Some(keystroke) = zero.divergence {
        info!(
            "probe: cache-off lockstep diverges at keystroke {} — the state cache is not required for the flip",
            keystroke + 1
        );
    } else {
        info!("probe: cache-off lockstep is identical");
    }
    // The tape diff: every score the two sessions emitted for the same
    // (history, candidate) must agree bit-for-bit, whatever batch each
    // came from. A moved score names the flipped value directly, and the
    // earliest one in the baseline's emission order is where corruption
    // first surfaces.
    let measured_scores: std::collections::HashMap<(Vec<CharId>, CharId), u32> = pair
        .scores
        .iter()
        .cloned()
        .map(|(h, c, b)| ((h, c), b))
        .collect();
    let baseline_scores: std::collections::HashMap<(Vec<CharId>, CharId), u32> = pair
        .base_scores
        .iter()
        .cloned()
        .map(|(h, c, b)| ((h, c), b))
        .collect();
    let mut moved = 0usize;
    let mut shared = 0usize;
    let mut first_moved: Option<(Vec<CharId>, CharId, u32, u32)> = None;
    for (history, candidate, bits) in &pair.base_scores {
        if let Some(&first) = measured_scores.get(&(history.clone(), *candidate)) {
            shared += 1;
            if first != *bits {
                moved += 1;
                if first_moved.is_none() {
                    first_moved = Some((history.clone(), *candidate, first, *bits));
                }
                if moved <= 8 {
                    info!(
                        "probe: score moved — history {:?} candidate {}: measured 0x{first:08x} baseline 0x{bits:08x}",
                        probe_text(harness.lexicon, history),
                        harness.lexicon.character(*candidate),
                    );
                }
            }
        }
    }
    info!(
        "probe: score tape — {shared} shared (history, candidate) scores, {moved} with differing bits (measured {} entries, baseline {})",
        pair.scores.len(),
        pair.base_scores.len()
    );
    // The first moved score's state came out of some recorded call:
    // which session ran it, and which call index produced a state whose
    // history ends at this one.
    let mut producer: Option<(usize, usize)> = None;
    if let Some((history, candidate, _, _)) = &first_moved {
        'outer: for (session_index, (session_name, session_calls)) in
            [("measured", calls), ("baseline", base_calls)]
                .iter()
                .enumerate()
        {
            for (call_index, call) in session_calls.iter().enumerate() {
                for (row, step) in call.iter().enumerate() {
                    let mut produced_history = step.history.clone();
                    produced_history.push(step.ch);
                    if &produced_history == history {
                        info!(
                            "probe: first moved score's state ({}…+{}) came from {session_name} call {call_index} row {row} of {}",
                            probe_text(
                                harness.lexicon,
                                &history[..history.len().saturating_sub(1)]
                            ),
                            harness.lexicon.character(*candidate),
                            call.len()
                        );
                        producer = Some((session_index, call_index));
                        break 'outer;
                    }
                }
            }
        }
    }
    // Which call emitted the moved scores: `emitted` attributes every
    // score to the `advance` call that produced it, so grouping the
    // moved entries by call names the call that first wrote wrong bits.
    let base_emitted: std::collections::HashMap<(Vec<CharId>, CharId), u32> = pair
        .base_emitted
        .iter()
        .map(|&(call, row, candidate, bits)| {
            let step = &pair.base_calls[call as usize][row as usize];
            let mut history = step.history.clone();
            history.push(step.ch);
            ((history, candidate), bits)
        })
        .collect();
    let mut moved_by_call: std::collections::BTreeMap<u32, usize> =
        std::collections::BTreeMap::new();
    for &(call, row, candidate, bits) in &pair.emitted {
        let step = &pair.calls[call as usize][row as usize];
        let mut history = step.history.clone();
        history.push(step.ch);
        if let Some(&base) = base_emitted.get(&(history, candidate))
            && base != bits
        {
            *moved_by_call.entry(call).or_default() += 1;
        }
    }
    if let Some((&call, &count)) = moved_by_call.iter().next() {
        info!(
            "probe: first corrupt-emitting call {call} — {count} moved scores; moved-by-call {:?}",
            moved_by_call.iter().take(12).collect::<Vec<_>>()
        );
    }
    // The rebuild: each recorded call re-scored on a fresh CharLm,
    // batched as recorded and solo. Bit-diffs name a batch-dependent
    // kernel; none narrows the cause to cache-restore or the n-gram.
    // Rebuilding a row replays its whole history, so the analysis covers
    // the divergent keystroke's calls (the batches that matter) plus a
    // bounded prefix for an aggregate signal.
    let fresh = CharLm::open(
        lm_dir,
        harness.lexicon,
        ime_lm::SessionShape {
            backend: harness.args.backend.unwrap_or_default().backend(),
            intra_threads: harness.args.lm_threads,
            width: harness.args.search.beam_width,
            verbose_logging: false,
            cache_rows: harness.args.lm_cache,
            metal_weights: harness.args.metal_weights.weights(),
        },
    )
    .context("could not open the probe's rebuild model")?;
    let mut total_checked = 0usize;
    let mut total_diffs = 0usize;
    let mut diff_calls = 0usize;
    let keystroke_of = |call_index: usize, spans: &CallSpans| -> usize {
        spans
            .iter()
            .position(|&(start, end)| call_index >= start && call_index < end)
            .map_or(0, |keystroke| keystroke + 1)
    };
    let mut analysed = 0usize;
    for (session_name, session_calls, spans) in [
        ("measured", calls, m_spans),
        ("baseline", base_calls, b_spans),
    ] {
        // The bound: every call of the divergent keystroke (or, clean
        // run, the last one) plus the first calls — the rest repeat the
        // same experiment at real cost.
        let interesting = |call_index: usize| -> bool {
            let focus = divergence.unwrap_or(keystrokes - 1);
            keystroke_of(call_index, spans) == focus + 1 || call_index < 16
        };
        for (call_index, call) in session_calls.iter().enumerate() {
            if !interesting(call_index) {
                continue;
            }
            analysed += 1;
            let (checked, diffs) = probe_call(&fresh, harness.lexicon, context, call);
            total_checked += checked;
            total_diffs += diffs.len();
            if !diffs.is_empty() {
                diff_calls += 1;
                info!(
                    "probe: {session_name} call {call_index} (keystroke {}, {} rows): {} of {checked} candidate scores bit-differ",
                    keystroke_of(call_index, spans),
                    call.len(),
                    diffs.len()
                );
                for line in diffs.iter().take(4) {
                    info!("probe:   {line}");
                }
            }
        }
    }
    info!(
        "probe: rebuild of {analysed} calls (of {} recorded), {total_checked} scores: {total_diffs} bit-diffs across {diff_calls} calls",
        calls.len() + base_calls.len()
    );
    // The three-way replay: the recorded call's inputs rebuilt on the
    // session's own pool, on a clean model, and the bits the tape kept.
    // recorded == live != fresh says the pool still holds the bad bytes;
    // recorded != live == fresh says the corruption rode the input state
    // and the pool has since healed; three distinct values say the bytes
    // changed again. The calls covered are the divergent keystroke's and
    // the producing call's neighbourhood, so call 202 itself is scored.
    if first_moved.is_some() {
        let focus = divergence.unwrap_or(keystrokes - 1);
        for (session_index, (session_name, session_calls, spans)) in [
            ("measured", calls, m_spans),
            ("baseline", base_calls, b_spans),
        ]
        .iter()
        .enumerate()
        {
            let mut compared = 0usize;
            for (call_index, call) in session_calls.iter().enumerate() {
                let interesting_call = keystroke_of(call_index, spans) == focus + 1
                    || producer
                        .is_some_and(|(s, c)| s == session_index && call_index.abs_diff(c) <= 2);
                if !interesting_call {
                    continue;
                }
                compared += 1;
                let live = probe_call_bits(lm, context, call);
                let clean = probe_call_bits(&fresh, context, call);
                let tape = if session_index == 0 {
                    &measured_scores
                } else {
                    &baseline_scores
                };
                let mut moved = 0usize;
                let mut recorded_differs_live = 0usize;
                let mut first_row: Option<(usize, CharId, CharId, u32, u32, u32)> = None;
                for (&(row, ch, candidate, lb), &(_, _, _, cb)) in live.iter().zip(&clean) {
                    let mut produced_history = call[row].history.clone();
                    produced_history.push(ch);
                    let recorded = tape.get(&(produced_history, candidate)).copied();
                    let rb = recorded.unwrap_or(lb);
                    if recorded.is_some_and(|rb| rb != lb) {
                        recorded_differs_live += 1;
                    }
                    if lb != cb {
                        moved += 1;
                        if first_row.is_none() {
                            first_row = Some((row, ch, candidate, lb, cb, rb));
                        }
                    }
                }
                info!(
                    "probe: {session_name} call {call_index} three-way — {moved} of {} live-vs-fresh, {recorded_differs_live} recorded-vs-live",
                    clean.len()
                );
                if let Some((row, ch, candidate, lb, cb, rb)) = first_row {
                    let step = &call[row];
                    info!(
                        "probe:   row {row} history {:?}+{} candidate {}: recorded 0x{rb:08x} live 0x{lb:08x} fresh 0x{cb:08x}",
                        probe_text(harness.lexicon, &step.history),
                        harness.lexicon.character(ch),
                        harness.lexicon.character(candidate),
                    );
                    let asked = probe_asked(step);
                    let live_path = probe_path_bits(lm, context, &step.history, asked);
                    let clean_path = probe_path_bits(&fresh, context, &step.history, asked);
                    for (position, (lb, cb)) in live_path.iter().zip(&clean_path).enumerate() {
                        if lb != cb {
                            info!(
                                "probe:   first content diff at history position {} of {}: live 0x{lb:08x} fresh 0x{cb:08x}",
                                position + 1,
                                live_path.len()
                            );
                            break;
                        }
                    }
                }
                if compared >= 6 {
                    break;
                }
            }
        }
    }
    if divergence.is_none() && total_diffs == 0 && zero.divergence.is_none() {
        info!("probe: clean — no divergence and every score bit-stable under batching");
    }
    Ok(())
}
