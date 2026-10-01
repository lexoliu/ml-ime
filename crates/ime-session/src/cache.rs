//! The state cache a session wraps its transition in.
//!
//! A keystroke's decode asks the transition for a state per `(prefix,
//! request)` — the character prefix a beam stands for plus the candidates
//! the next position admits — and almost all of them recur keystroke after
//! keystroke, because the text so far is mostly the same. [`Cached`] keeps a
//! map from that pair to the state the model produced for it, so the beam's
//! `advance` computes only the states the map misses. An entry is stored as
//! it was produced and moved into its [`Transition::compact`] form on its
//! first hit — only a state that provably recurs pays the row copy — under
//! an LRU bound, and the whole map is dropped on a context change, since a
//! state means nothing under a different context.
//!
//! The map is `RefCell`-local: a session is single-threaded by contract, so
//! the wrapper never synchronises.

use hashbrown::HashMap;
use ime_decode::{Asked, Transition};
use ime_pinyin::CharId;
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;

/// A beam's state plus the character prefix it stands for.
///
/// The prefix is what keys the cache: two beams whose histories merged are
/// still distinguished here, because a long-memory model's state depends on
/// the whole prefix, not the part the search keeps apart.
#[derive(Clone)]
pub struct CachedState<S> {
    /// The model's own state.
    inner: S,
    /// The characters emitted so far, oldest first; empty for a `start`
    /// state. Shared with the map key that produced it.
    prefix: Arc<[CharId]>,
}

/// What a stored state answers: the sorted candidate ids its scores were
/// gathered under, and whether `<eos>` was among them.
#[derive(Clone, PartialEq, Eq, Hash)]
struct AskedSig {
    /// The request's candidate ids, sorted — the order nothing depends on.
    candidates: Box<[CharId]>,
    /// Whether the request could end there.
    eos: bool,
}

impl AskedSig {
    /// The signature of *asked*.
    fn of(asked: &Asked<'_>) -> Self {
        let mut candidates = asked.candidates.to_vec().into_boxed_slice();
        candidates.sort_unstable();
        Self {
            candidates,
            eos: asked.eos,
        }
    }

    /// Whether a state stored under `self` answers *asked*: every id it
    /// names was gathered, and `<eos>` too when the path can end there.
    fn covers(&self, asked: &Asked<'_>) -> bool {
        (!asked.eos || self.eos)
            && asked
                .candidates
                .iter()
                .all(|id| self.candidates.binary_search(id).is_ok())
    }

    /// The sig-to-sig form of [`AskedSig::covers`]: a miss queued under
    /// `self` resolves a duplicate requesting *other*.
    fn covers_sig(&self, other: &Self) -> bool {
        (!other.eos || self.eos)
            && other
                .candidates
                .iter()
                .all(|id| self.candidates.binary_search(id).is_ok())
    }
}

/// One cached state: the request its scores cover, its prefix, the state,
/// and the tick the LRU ordering compares against.
struct Entry<S> {
    /// What the state's scores answer — shared with the entry's queue
    /// place, since a stored sig is read, never rewritten.
    sig: Rc<AskedSig>,
    /// The prefix the entry's map key holds; kept here so a hit need not
    /// reach back through the key.
    prefix: Arc<[CharId]>,
    /// The produced state until its first hit, then the compacted row —
    /// see `packed`.
    state: S,
    /// Whether `state` is the compacted form yet. A miss stores the state
    /// as produced — sharing its step's buffers costs nothing until the
    /// entry proves it recurs, which is what the copy buys.
    packed: bool,
    /// Last touch, in the cache's own tick count.
    tick: u64,
}

/// One entry's place in the LRU queue.
struct Order {
    /// The entry's prefix.
    prefix: Arc<[CharId]>,
    /// The request it covers — the entry's own `Rc`.
    sig: Rc<AskedSig>,
    /// The tick it was queued at — stale once the entry moved on.
    tick: u64,
}

/// A snapshot of the cache's counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheReport {
    /// `start` calls the cache answered.
    pub start_hits: u64,
    /// `start` calls that ran the model.
    pub start_misses: u64,
    /// `advance` rows the cache answered.
    pub step_hits: u64,
    /// `advance` rows the model computed.
    pub step_misses: u64,
    /// Entries evicted under the bound.
    pub evictions: u64,
    /// Entries held now.
    pub entries: u64,
}

/// A transition wrapped in the prefix-keyed state cache.
///
/// `capacity` bounds the entries: for the LM a compacted entry is one
/// resident row, so the cap is in the resident layout's natural unit and
/// the memory it stands for is `capacity` times a row — entries not yet
/// hit share their step's buffers, freed as the LRU evicts the cohort.
/// Zero means no caching — the wrapper asks
/// the model for everything, the baseline the replay's equality gate checks
/// against.
pub struct Cached<T: Transition> {
    /// The model doing the work on a miss.
    inner: T,
    /// The context every entry was produced under — compared at `start`,
    /// which a refresh always reaches before `advance` does. `None` also
    /// means "never set": the entries are empty then, so the skipped clear
    /// is a no-op.
    context: RefCell<Option<String>>,
    /// prefix -> the states stored for it, one per distinct request.
    entries: RefCell<Entries<T::State>>,
    /// Oldest-first entry locations; stale places skip themselves.
    order: RefCell<VecDeque<Order>>,
    /// Entries held now.
    count: Cell<usize>,
    /// The entry bound.
    capacity: usize,
    /// Monotonic touch counter.
    tick: Cell<u64>,
    /// The counters the report reads: `start` hits and misses, `advance`
    /// rows hit and missed, and entries evicted.
    stats: CacheCells,
}

/// The entry map's own type: one bucket of stored states per prefix, one
/// [`Entry`] per request shape asked under it.
type Entries<S> = HashMap<Arc<[CharId]>, Vec<Entry<S>>>;

/// The cache's counters as interior cells, so `Transition`'s `&self` methods
/// can count.
#[derive(Default)]
struct CacheCells {
    /// `start` calls the cache answered.
    start_hits: Cell<u64>,
    /// `start` calls that ran the model.
    start_misses: Cell<u64>,
    /// `advance` rows the cache answered.
    step_hits: Cell<u64>,
    /// `advance` rows the model computed.
    step_misses: Cell<u64>,
    /// Entries evicted under the bound.
    evictions: Cell<u64>,
}

impl<T: Transition> Cached<T> {
    /// Wrap *transition* keeping up to *capacity* produced states.
    #[must_use]
    pub fn new(transition: T, capacity: usize) -> Self {
        Self {
            inner: transition,
            context: RefCell::new(None),
            entries: RefCell::new(HashMap::new()),
            order: RefCell::new(VecDeque::new()),
            count: Cell::new(0),
            capacity,
            tick: Cell::new(0),
            stats: CacheCells::default(),
        }
    }

    /// The wrapped model — the way to its own stats.
    #[must_use]
    pub const fn inner(&self) -> &T {
        &self.inner
    }

    /// The cache's counters, cumulative.
    #[must_use]
    pub fn stats(&self) -> CacheReport {
        CacheReport {
            start_hits: self.stats.start_hits.get(),
            start_misses: self.stats.start_misses.get(),
            step_hits: self.stats.step_hits.get(),
            step_misses: self.stats.step_misses.get(),
            evictions: self.stats.evictions.get(),
            entries: self.count.get() as u64,
        }
    }

    /// Clear the entries when *context* moved on; called from `start`, the
    /// first call a refresh always makes.
    fn refresh_context(&self, context: Option<&str>) {
        let context = context.map(str::to_owned);
        let mut current = self.context.borrow_mut();
        if *current == context {
            return;
        }
        self.entries.borrow_mut().clear();
        self.order.borrow_mut().clear();
        self.count.set(0);
        *current = context;
    }

    /// The stored state covering (*prefix*, *asked*), when one is held.
    /// The hit is the entry's proof of recurrence: its state is compacted
    /// into a resident row before it is served, then marked most-recent.
    fn get(&self, prefix: &[CharId], asked: &Asked<'_>) -> Option<CachedState<T::State>> {
        let tick = self.tick.get() + 1;
        self.tick.set(tick);
        let (state, key, sig) = {
            let mut entries = self.entries.borrow_mut();
            let node = entries.get_mut(prefix)?;
            let entry = node.iter_mut().find(|entry| entry.sig.covers(asked))?;
            if !entry.packed {
                entry.state = self.inner.compact(&entry.state);
                entry.packed = true;
            }
            entry.tick = tick;
            (
                entry.state.clone(),
                Arc::clone(&entry.prefix),
                entry.sig.clone(),
            )
        };
        self.order.borrow_mut().push_back(Order {
            prefix: Arc::clone(&key),
            sig,
            tick,
        });
        self.evict();
        Some(CachedState {
            inner: state,
            prefix: key,
        })
    }

    /// Store *state* under (*prefix*, *sig*) as produced — the row copy is
    /// a hit's job; a zero capacity keeps nothing.
    fn store(&self, prefix: Arc<[CharId]>, sig: Rc<AskedSig>, state: &T::State) {
        if self.capacity == 0 {
            return;
        }
        let tick = self.tick.get() + 1;
        self.tick.set(tick);
        self.entries
            .borrow_mut()
            .entry(Arc::clone(&prefix))
            .or_default()
            .push(Entry {
                sig: Rc::clone(&sig),
                prefix: Arc::clone(&prefix),
                state: state.clone(),
                packed: false,
                tick,
            });
        self.count.set(self.count.get() + 1);
        self.order
            .borrow_mut()
            .push_back(Order { prefix, sig, tick });
        self.evict();
    }

    /// Evict the oldest live entries while over the bound; stale queue
    /// places — an entry touched or removed since it was queued — skip.
    /// The queue is compacted once its length doubles the bound, since a
    /// hit queues a place too and stale ones would otherwise grow with the
    /// hit count, not the entry count.
    fn evict(&self) {
        if self.order.borrow().len() > self.capacity.saturating_mul(2) {
            let mut order = self.order.borrow_mut();
            let entries = self.entries.borrow();
            order.retain(|oldest| {
                entries
                    .get(&oldest.prefix)
                    .and_then(|node| node.iter().find(|entry| entry.sig == oldest.sig))
                    .is_some_and(|entry| entry.tick == oldest.tick)
            });
        }
        while self.count.get() > self.capacity {
            let Some(oldest) = self.order.borrow_mut().pop_front() else {
                break;
            };
            let stale = self
                .entries
                .borrow()
                .get(&oldest.prefix)
                .and_then(|node| node.iter().find(|entry| entry.sig == oldest.sig))
                .is_none_or(|entry| entry.tick != oldest.tick);
            if stale {
                continue;
            }
            let mut entries = self.entries.borrow_mut();
            let Some(node) = entries.get_mut(&oldest.prefix) else {
                continue;
            };
            node.retain(|entry| entry.sig != oldest.sig);
            if node.is_empty() {
                entries.remove(&oldest.prefix);
            }
            self.count.set(self.count.get() - 1);
            self.stats.evictions.set(self.stats.evictions.get() + 1);
        }
    }
}

impl<T: Transition> Transition for Cached<T> {
    const HISTORY: usize = T::HISTORY;

    type State = CachedState<T::State>;

    /// # Panics
    ///
    /// If the model fails its start, as `T::start` does.
    fn start(&self, context: Option<&str>, asked: &Asked<'_>) -> Self::State {
        self.refresh_context(context);
        if let Some(state) = self.get(&[], asked) {
            self.stats.start_hits.set(self.stats.start_hits.get() + 1);
            return state;
        }
        self.stats
            .start_misses
            .set(self.stats.start_misses.get() + 1);
        let produced = self.inner.start(context, asked);
        self.store(Arc::from(&[][..]), Rc::new(AskedSig::of(asked)), &produced);
        CachedState {
            inner: produced,
            prefix: Arc::from(&[][..]),
        }
    }

    fn score(&self, state: &Self::State, candidate: CharId) -> f32 {
        self.inner.score(&state.inner, candidate)
    }

    fn finish(&self, state: &Self::State) -> f32 {
        self.inner.finish(&state.inner)
    }

    /// The model runs only the steps the cache misses; a hit's stored state
    /// comes back shared with the map's key, and a step an earlier miss of
    /// this same call covers resolves against what the miss stored.
    ///
    /// # Panics
    ///
    /// If the model fails its run or returns a wrong row count, as
    /// `T::advance` does.
    fn advance(&self, steps: &[(&Self::State, CharId, Asked<'_>)]) -> Vec<Self::State> {
        let mut out: Vec<Option<Self::State>> = (0..steps.len()).map(|_| None).collect();
        let mut misses: Vec<(&T::State, CharId, Asked<'_>)> = Vec::new();
        // `steps`'s index, produced prefix and request for each miss, in
        // call order.
        let mut origins: Vec<(usize, Arc<[CharId]>, Rc<AskedSig>)> = Vec::new();
        for (index, (state, ch, asked)) in steps.iter().enumerate() {
            let mut prefix = state.prefix.to_vec();
            prefix.push(*ch);
            if let Some(hit) = self.get(&prefix, asked) {
                self.stats.step_hits.set(self.stats.step_hits.get() + 1);
                out[index] = Some(hit);
                continue;
            }
            let sig = Rc::new(AskedSig::of(asked));
            let prefix: Arc<[CharId]> = Arc::from(prefix);
            if self.capacity > 0
                && origins
                    .iter()
                    .any(|(_, prior, prior_sig)| *prior == prefix && prior_sig.covers_sig(&sig))
            {
                continue;
            }
            origins.push((index, prefix, sig));
            misses.push((&state.inner, *ch, *asked));
        }
        let produced = self.inner.advance(&misses);
        assert_eq!(
            produced.len(),
            misses.len(),
            "a transition model returns one state per step"
        );
        for ((index, prefix, sig), state) in origins.into_iter().zip(produced) {
            self.store(Arc::clone(&prefix), sig, &state);
            out[index] = Some(CachedState {
                inner: state,
                prefix,
            });
            self.stats.step_misses.set(self.stats.step_misses.get() + 1);
        }
        // The duplicates left open take what this call's misses stored —
        // or, when an eviction raced them, their own step's run.
        for (index, (state, ch, asked)) in steps.iter().enumerate() {
            if out[index].is_some() {
                continue;
            }
            let mut prefix = state.prefix.to_vec();
            prefix.push(*ch);
            let resolved = self.get(&prefix, asked).unwrap_or_else(|| {
                let mut produced = self.inner.advance(&[(&state.inner, *ch, *asked)]);
                CachedState {
                    inner: produced.pop().expect("one state per step"),
                    prefix: Arc::from(prefix),
                }
            });
            out[index] = Some(resolved);
            self.stats.step_hits.set(self.stats.step_hits.get() + 1);
        }
        out.into_iter()
            .map(|state| state.expect("every step resolves"))
            .collect()
    }

    fn compact(&self, state: &Self::State) -> Self::State {
        CachedState {
            inner: self.inner.compact(&state.inner),
            prefix: Arc::clone(&state.prefix),
        }
    }
}
