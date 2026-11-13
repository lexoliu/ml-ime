//! The key engine over a model-free synthetic model.
//!
//! [`Synthetic`] answers the keys function and the tree calls with
//! `SplitMix64` draws, as the key search's fixture does: a key's CTC row is a
//! function of the whole key string and the key's position (the keys tower
//! is bidirectional, so every row moves with a keystroke), and a node's
//! decoder row a function of the keystroke's encoding and the node's prefix —
//! plus, when `vary` is on, of the call that computed it, the way an fp16 row
//! may depend on its bucket and neighbours. It logs every call it serves, so
//! the tests can hold the engine to the speculation and caching contract.

use std::cell::RefCell;
use std::collections::HashSet;
use std::convert::Infallible;
use std::num::NonZeroUsize;

use ime_decode::{
    Combine, DEFAULT_ARC_WIDTH, KeyHypothesis, KeyLogProbs, KeySearchOptions, KeySearchStats,
    KeyTransition, PrefixId, PrefixTrie, Weight, key_search,
};
use ime_neural::keys::{KeyModel, TreeCall, plan};
use ime_pinyin::{EmitId, Keys, Legality, LegalityOptions};
use ime_session::{
    Engine, KeyEngine, KeyEngineOptions, Pending, Placed, Sentence, Session, SessionOptions,
};

/// The characters the tests emit, in emission order.
const ALPHABET: &str = "我爱北京被经窝哎西安先现线县显鲜";

/// `SplitMix64`'s finaliser.
const fn splitmix(value: u64) -> u64 {
    let mut value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// A multiple of 2^-16 in (-16, 0], exact in f32.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a 20-bit multiple of 2^-16 is exact in f32"
)]
fn draw(seed: u64, value: u64) -> f32 {
    -(f64::from(u32::try_from(splitmix(seed ^ value) >> 44).expect("20 bits")) / 65536.0) as f32
}

fn wide(value: usize) -> u64 {
    u64::try_from(value).expect("a usize fits u64")
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(0x5EED, |hash, &byte| splitmix(hash ^ u64::from(byte)))
}

fn hash_ids(ids: &[EmitId]) -> u64 {
    ids.iter()
        .fold(0, |hash, id| splitmix(hash ^ wide(id.index())))
}

/// The row-wise log-softmax of *row* into *out*, in f32.
fn log_softmax(row: &[f32], out: &mut [f32]) {
    let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f32 = row.iter().map(|value| (value - top).exp()).sum();
    let norm = top + total.ln();
    for (slot, value) in out.iter_mut().zip(row) {
        *slot = value - norm;
    }
}

/// A node's logits: the keystroke's encoding and the prefix, and the call
/// when the rows vary by call.
#[expect(
    clippy::cast_precision_loss,
    reason = "a test's call counter is far below 2^24"
)]
fn node_row(emittable: usize, encoding: u64, prefix: &[EmitId], call: Option<usize>) -> Vec<f32> {
    let hash = hash_ids(prefix);
    (0..emittable)
        .map(|column| {
            let shift = match call {
                Some(call) if column == 0 => call as f32 / 16.0,
                _ => 0.0,
            };
            draw(encoding, hash ^ wide(column)) + shift
        })
        .collect()
}

/// What the synthetic model served, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    /// A keys call over these letters.
    Encode(String),
    /// A tree call: its global number, bucket and nodes.
    Tree {
        call: usize,
        bucket: usize,
        nodes: Vec<Vec<EmitId>>,
    },
    /// A row the engine read: the call and the node.
    Read { call: usize, node: usize },
}

/// The model-free key model.
struct Synthetic {
    emittable: usize,
    vary: bool,
    calls: usize,
    events: RefCell<Vec<Event>>,
}

impl Synthetic {
    fn new(emittable: usize, vary: bool) -> Self {
        Self {
            emittable,
            vary,
            calls: 0,
            events: RefCell::new(Vec::new()),
        }
    }

    fn events(&self) -> Vec<Event> {
        self.events.borrow().clone()
    }
}

/// One tree call's output.
struct Logits {
    call: usize,
    rows: Vec<f32>,
}

impl KeyModel for Synthetic {
    type Error = Infallible;
    type Encoding = u64;
    type Logits = Logits;

    fn encode(
        &mut self,
        keys: &Keys,
        context: Option<&str>,
        log_probs: &mut Vec<f32>,
    ) -> Result<u64, Infallible> {
        let seed = hash_bytes(keys.as_str().as_bytes())
            ^ context.map_or(0, |text| hash_bytes(text.as_bytes()).rotate_left(1));
        log_probs.clear();
        let mut row = vec![0.0; self.emittable + 1];
        for key in 0..keys.len() {
            let logits: Vec<f32> = (0..=self.emittable)
                .map(|column| draw(seed, (wide(key) << 32) | wide(column)))
                .collect();
            log_softmax(&logits, &mut row);
            log_probs.extend_from_slice(&row);
        }
        self.events
            .borrow_mut()
            .push(Event::Encode(keys.as_str().to_owned()));
        Ok(seed)
    }

    fn tree(&mut self, encoding: &u64, call: &TreeCall<'_, EmitId>) -> Result<Logits, Infallible> {
        let number = self.calls;
        self.calls += 1;
        let rows = call
            .nodes()
            .iter()
            .flat_map(|node| node_row(self.emittable, *encoding, node, self.vary.then_some(number)))
            .collect();
        self.events.borrow_mut().push(Event::Tree {
            call: number,
            bucket: call.bucket(),
            nodes: call.nodes().iter().map(|node| node.to_vec()).collect(),
        });
        Ok(Logits { call: number, rows })
    }

    fn log_softmax(&self, logits: &Logits, node: usize, out: &mut [f32]) {
        self.events.borrow_mut().push(Event::Read {
            call: logits.call,
            node,
        });
        let width = self.emittable;
        log_softmax(&logits.rows[node * width..(node + 1) * width], out);
    }
}

/// The direct search's transition: every row from the prefix alone.
struct Reference {
    emittable: usize,
    encoding: u64,
}

impl KeyTransition for Reference {
    type Error = Infallible;

    fn rows(
        &mut self,
        trie: &PrefixTrie,
        queries: &[PrefixId],
        out: &mut [f32],
    ) -> Result<(), Infallible> {
        let mut ids = Vec::new();
        for (&query, row) in queries.iter().zip(out.chunks_exact_mut(self.emittable)) {
            trie.ids_into(query, &mut ids);
            log_softmax(&node_row(self.emittable, self.encoding, &ids, None), row);
        }
        Ok(())
    }
}

fn legality() -> Legality {
    Legality::over(&ALPHABET.chars().collect::<Vec<_>>()).expect("the alphabet is emittable")
}

fn options(legality: LegalityOptions) -> KeyEngineOptions {
    KeyEngineOptions {
        legality,
        search: KeySearchOptions {
            beta: Weight::try_from(1.0).expect("a weight"),
            gamma: Weight::try_from(1.0).expect("a weight"),
            beam: NonZeroUsize::new(8),
            top_k: NonZeroUsize::new(8),
            combine: Combine::Sum,
            prior: Weight::ZERO,
            arc_width: DEFAULT_ARC_WIDTH,
        },
    }
}

fn engine(legality: &Legality, vary: bool) -> KeyEngine<'_, Synthetic> {
    KeyEngine::new(
        Synthetic::new(legality.emittable().len(), vary),
        legality,
        options(LegalityOptions::LIVE),
    )
}

fn pending(typed: &str) -> Pending {
    Pending::parse(typed)
        .expect("the input is letters and apostrophes")
        .expect("the input has letters")
}

/// *engine*'s sentences for *typed*.
fn sentences(engine: &mut KeyEngine<'_, Synthetic>, typed: &str) -> Vec<Sentence> {
    engine
        .sentences(&pending(typed), None)
        .expect("the synthetic model never fails")
}

/// A hypothesis as the engine reports it.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the session's scores are f32"
)]
fn as_sentence(legality: &Legality, hypothesis: &KeyHypothesis) -> Sentence {
    Sentence {
        chars: hypothesis
            .ids
            .iter()
            .zip(&hypothesis.cuts)
            .zip(&hypothesis.steps)
            .map(|((id, &end), &step)| Placed {
                ch: legality.emittable()[id.index()],
                end,
                score: step as f32,
            })
            .collect(),
        score: hypothesis.score as f32,
    }
}

/// The key search run directly over *typed* with the same lattice, log-probs
/// and rows the engine's model gives.
fn direct(legality: &Legality, typed: &str) -> Vec<Sentence> {
    let pending = pending(typed);
    let mut lattice = legality.lattice(
        pending.letters(),
        LegalityOptions {
            unfinished_tail: !pending.closed(),
            ..LegalityOptions::LIVE
        },
    );
    if !pending.cuts().is_empty() {
        lattice = lattice.with_forced_cuts(pending.cuts());
    }
    let emittable = legality.emittable().len();
    let mut model = Synthetic::new(emittable, false);
    let mut data = Vec::new();
    let Ok(encoding) = model.encode(pending.letters(), None, &mut data);
    let log_probs =
        KeyLogProbs::new(pending.letters().len(), emittable + 1, &data).expect("a table");
    let mut reference = Reference {
        emittable,
        encoding,
    };
    let Ok(found) = key_search(
        &lattice,
        &log_probs,
        &[],
        &mut reference,
        &options(LegalityOptions::LIVE).search,
        &mut KeySearchStats::default(),
    );
    found
        .iter()
        .map(|hypothesis| as_sentence(legality, hypothesis))
        .collect()
}

/// The tree calls among *events*, as `(call, nodes)`.
fn trees(events: &[Event]) -> Vec<(usize, Vec<Vec<EmitId>>)> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Tree { call, nodes, .. } => Some((*call, nodes.clone())),
            _ => None,
        })
        .collect()
}

fn closure(prefixes: &[Box<[EmitId]>]) -> HashSet<Vec<EmitId>> {
    prefixes
        .iter()
        .flat_map(|prefix| (0..=prefix.len()).map(move |length| prefix[..length].to_vec()))
        .collect()
}

#[test]
fn the_last_keystroke_equals_a_direct_search() {
    let legality = legality();
    for typed in ["woaibeijing", "woaibj", "xian", "xi'an", "xia'"] {
        let mut engine = engine(&legality, false);
        let mut last = Vec::new();
        for end in 1..=typed.len() {
            if Pending::parse(&typed[..end]).expect("valid").is_some() {
                last = sentences(&mut engine, &typed[..end]);
            }
        }
        assert!(!last.is_empty(), "{typed} reads");
        assert_eq!(last, direct(&legality, typed), "{typed}");
    }
}

#[test]
fn typing_key_by_key_equals_typing_at_once() {
    let legality = legality();
    let mut session = Session::new(engine(&legality, false), SessionOptions { sentences: 3 });
    for ch in "woaibeijing".chars() {
        session.key(ch).expect("a keystroke decodes");
    }
    let mut once = engine(&legality, false);
    assert_eq!(session.sentences(), sentences(&mut once, "woaibeijing"));
}

#[test]
fn backspace_then_retype_speculates_and_answers_as_typed() {
    let legality = legality();
    let mut typing = engine(&legality, false);
    for end in 1..=5 {
        sentences(&mut typing, &"woaib"[..end]);
    }
    let at_woaib = typing.queried().to_vec();
    assert_ne!(at_woaib, [] as [std::boxed::Box<[ime_pinyin::EmitId]>; 0]);
    sentences(&mut typing, "woaibk");
    // Back to five letters: the speculation is what "woaib" asked before,
    // not what "woaibk" just asked.
    let mark = typing.model().events().len();
    let back = sentences(&mut typing, "woaib");
    let made = trees(&typing.model().events()[mark..]);
    let planned = plan(&at_woaib).expect("the queries fit");
    assert!(made.len() >= planned.len());
    for (call, (_, nodes)) in planned.iter().zip(&made) {
        let want: Vec<Vec<EmitId>> = call.nodes().iter().map(|node| node.to_vec()).collect();
        assert_eq!(
            *nodes, want,
            "the backspace speculates with the shorter input's queries"
        );
    }
    assert_eq!(back, sentences(&mut engine(&legality, false), "woaib"));
    let retyped = sentences(&mut typing, "woaibj");
    assert_eq!(retyped, direct(&legality, "woaibj"));
}

#[test]
fn an_apostrophe_forces_a_cut() {
    let legality = legality();
    let cut = sentences(&mut engine(&legality, false), "xi'an");
    assert_ne!(cut, [] as [ime_session::Sentence; 0]);
    for sentence in &cut {
        assert!(
            sentence.chars.iter().any(|placed| placed.end == 2),
            "{sentence:?} places a boundary at the apostrophe"
        );
    }
    let free = sentences(&mut engine(&legality, false), "xian");
    assert!(
        free.iter()
            .any(|sentence| sentence.chars.iter().all(|placed| placed.end != 2)),
        "without the apostrophe some reading spans it"
    );
    // A closing apostrophe ends the last syllable: the live input reads as
    // complete input does.
    let closed = sentences(&mut engine(&legality, false), "xia'");
    let mut offline = KeyEngine::new(
        Synthetic::new(legality.emittable().len(), false),
        &legality,
        options(LegalityOptions::OFFLINE),
    );
    assert_eq!(closed, sentences(&mut offline, "xia"));
}

#[test]
fn every_row_read_is_the_first_value_its_keystroke_computed() {
    let legality = legality();
    let mut engine = engine(&legality, true);
    let typed = "woaibeijing";
    for end in 1..=typed.len() {
        sentences(&mut engine, &typed[..end]);
    }
    let events = engine.model().events();
    let (mut reads, mut recomputed) = (0, 0);
    for keystroke in events.split(|event| matches!(event, Event::Encode(_))) {
        let made = trees(keystroke);
        let nodes_of = |call: usize| -> &Vec<Vec<EmitId>> {
            &made
                .iter()
                .find(|(number, _)| *number == call)
                .expect("a read names a call of its keystroke")
                .1
        };
        for event in keystroke {
            let Event::Read { call, node } = event else {
                continue;
            };
            let prefix = &nodes_of(*call)[*node];
            let first = made
                .iter()
                .find(|(_, nodes)| nodes.contains(prefix))
                .expect("the read prefix was computed")
                .0;
            assert_eq!(*call, first, "{prefix:?} is read from its first call");
            reads += 1;
            if made
                .iter()
                .filter(|(_, nodes)| nodes.contains(prefix))
                .count()
                > 1
            {
                recomputed += 1;
            }
        }
    }
    assert!(reads > 0);
    assert!(
        recomputed > 0,
        "some read prefix was computed by more than one call, so a later value existed"
    );
}

#[test]
fn the_speculative_pass_replays_the_previous_queries() {
    let legality = legality();
    let mut engine = engine(&legality, false);
    let typed = "woaibeijing";
    sentences(&mut engine, &typed[..1]);
    let (mut speculated, mut missed) = (0, 0);
    for end in 2..=typed.len() {
        let previous = engine.queried().to_vec();
        let mark = engine.model().events().len();
        sentences(&mut engine, &typed[..end]);
        let made = trees(&engine.model().events()[mark..]);
        let planned = plan(&previous).expect("the queries fit");
        assert!(made.len() >= planned.len(), "{end}: the pass runs first");
        for (call, (_, nodes)) in planned.iter().zip(&made) {
            let want: Vec<Vec<EmitId>> = call.nodes().iter().map(|node| node.to_vec()).collect();
            assert_eq!(*nodes, want, "{end}: the speculative calls are the plan");
        }
        let covered = closure(&previous);
        let stats = engine.last_keystroke();
        assert_eq!(stats.speculative_rows, covered.len(), "{end}");
        assert_eq!(stats.miss_calls, made.len() - planned.len(), "{end}");
        assert_eq!(
            usize::try_from(stats.calls.iter().sum::<u64>()).expect("small"),
            made.len()
        );
        for (_, nodes) in &made[planned.len()..] {
            assert!(
                nodes.iter().any(|node| !covered.contains(node)),
                "{end}: a miss call asks for something the pass did not compute"
            );
        }
        speculated += planned.len();
        missed += stats.miss_calls;
    }
    assert!(speculated > 0 && missed > 0);
}

/// How many keystrokes of *typed* hold *letters* letters, a closing `'`
/// included.
fn keystrokes(typed: &str, letters: usize) -> usize {
    let mut seen = 0;
    let mut end = typed
        .bytes()
        .position(|byte| {
            if byte != b'\'' {
                seen += 1;
            }
            seen == letters
        })
        .expect("the letters are typed")
        + 1;
    while typed.as_bytes().get(end) == Some(&b'\'') {
        end += 1;
    }
    end
}

#[test]
fn the_session_lists_prefixes_over_the_key_engine() {
    let legality = legality();
    for typed in ["woaibj", "wo'aibj"] {
        let mut session = Session::new(engine(&legality, false), SessionOptions { sentences: 3 });
        for ch in typed.chars() {
            session.key(ch).expect("a keystroke decodes");
        }
        let top = session.sentences()[0].clone();
        let leading = session.sentences().len().min(3);
        let candidates = session.candidates().to_vec();
        assert!(
            candidates.len() > leading,
            "{typed}: prefix candidates exist"
        );
        for candidate in &candidates[..leading] {
            assert_eq!(candidate.consumes, typed.len());
        }
        for candidate in &candidates[leading..] {
            let k = candidate.text.chars().count();
            assert!(k < top.chars.len());
            assert_eq!(
                candidate.consumes,
                keystrokes(typed, top.chars[k - 1].end),
                "{typed}: {candidate:?} ends where the top sentence's character {k} ends"
            );
            assert!(
                session.sentences().iter().any(|sentence| {
                    sentence.chars.len() >= k
                        && sentence.chars[..k]
                            .iter()
                            .zip(&top.chars[..k])
                            .all(|(a, b)| a.end == b.end)
                        && sentence.chars[..k]
                            .iter()
                            .map(|placed| placed.ch)
                            .collect::<String>()
                            == candidate.text
                }),
                "{typed}: {candidate:?} is a sentence's prefix"
            );
        }
        let index = leading;
        let prefix = candidates[index].clone();
        let commit = session.select(index).expect("selecting a prefix commits");
        assert_eq!(commit.text, prefix.text);
        assert_eq!(session.pending(), typed.len() - prefix.consumes);
        if session.pending() > 0 {
            let top = session.candidates().first().expect("a new top candidate");
            assert_eq!(top.consumes, session.pending());
        }
    }
}
