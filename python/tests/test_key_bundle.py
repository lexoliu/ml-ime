"""The key-level model's tree recompute, manifest and bundle (#266).

A tiny randomly initialised joint-stage model stands for the real one. The tree
recompute (:meth:`~mlime.train.key_model.KeyE2EModel.tree_log_probs`) must give
the rows :meth:`~mlime.train.key_model.KeyE2EModel.transition_log_probs` gives
for the same prefixes, at float32 and to a tight tolerance, however the
prefixes are packed and however much padding a bucket adds. The bundle carries
the weights the three function families read, a manifest that round trips and
refuses what it should, and a reference the conversion verifies against.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open
from test_keys import (
    KeyTokenizer,
    key_collator,
    legality_fixture,  # noqa: F401 -- the fixture, by name
    reader_vocab_fixture,  # noqa: F401
    tiny_joint_model,
    tiny_key_model,
    vocabularies_fixture,  # noqa: F401
)
from typer.testing import CliRunner

from mlime.__main__ import app
from mlime.legality import Legality
from mlime.train.charlm_vocab import BOS, CharVocab
from mlime.train.key_bundle import (
    CLOSE,
    REFERENCE_PROBES,
    bundle_weights,
    read_reference,
    reference_requests,
    write_key_bundle,
)
from mlime.train.key_eval import DecodeTimes, ModelTransition
from mlime.train.key_manifest import (
    KEYS_WIDTHS,
    LAYOUT,
    LEGALITY_PROBES,
    MANIFEST,
    N_MAX,
    NEG,
    KeyManifest,
    families,
)
from mlime.train.key_model import KeyE2EModel, KeyStates
from mlime.train.key_tree import POLICY_VERSION, TREE_BUCKETS, Prefix, plan
from mlime.train.lexicon import Lexicon
from mlime.train.spans import SpanVocab

#: Equal to the transition at float32: the two compute the same sums in another order.
TIGHT = 1e-5


@pytest.fixture(name="model")
def model_fixture(lexicon: Lexicon, reader_vocab: CharVocab) -> KeyE2EModel:
    return tiny_joint_model(lexicon, reader_vocab).eval()


def encode(
    model: KeyE2EModel, spans: SpanVocab, lexicon: Lexicon, keys: str, context: str | None
) -> KeyStates:
    with torch.no_grad():
        return model.key_states(key_collator(spans, lexicon).inputs([keys], [context]))


def reference_rows(model: KeyE2EModel, states: KeyStates, prefixes: list[Prefix]) -> torch.Tensor:
    """The search's own transition over *prefixes*, ``[N, E]``."""
    with torch.no_grad():
        return ModelTransition(model, states, DecodeTimes())([0] * len(prefixes), prefixes)


def tree_rows(model: KeyE2EModel, states: KeyStates, requests: list[Prefix]) -> list[torch.Tensor]:
    """Each request's row through the packed tree calls, in request order."""
    cross = model.cross_memory(states)
    rows: list[torch.Tensor | None] = [None] * len(requests)
    for call in plan(requests):
        feeds = model.tree_feeds(call)
        with torch.no_grad():
            got = model.tree_log_probs(feeds.reader_ids, feeds.positions, feeds.tree_bias, cross)
        for request, row in call.answers:
            rows[request] = got[0, row]
    assert all(row is not None for row in rows)
    return [row for row in rows if row is not None]


def random_requests(rng: random.Random, emitted: int, count: int, depth: int) -> list[Prefix]:
    return [
        tuple(rng.randrange(emitted) for _ in range(rng.randint(0, depth))) for _ in range(count)
    ]


# --- the tree recompute -----------------------------------------------------------


@pytest.mark.parametrize(("keys", "context"), [("wmzyq", "我们吃水饺"), ("zhonghe", None)])
def test_the_tree_rows_are_the_transitions_rows(
    model: KeyE2EModel, spans: SpanVocab, lexicon: Lexicon, keys: str, context: str | None
) -> None:
    states = encode(model, spans, lexicon, keys, context)
    rng = random.Random(7)
    for _ in range(6):
        requests = random_requests(rng, lexicon.size, rng.randint(1, 30), 9)
        expected = reference_rows(model, states, requests)
        for got, want in zip(tree_rows(model, states, requests), expected, strict=True):
            torch.testing.assert_close(got, want, atol=TIGHT, rtol=0.0)
            assert abs(float(got.exp().sum()) - 1.0) < 1e-5


def test_padded_nodes_change_no_real_row_at_either_edge_of_a_bucket(
    model: KeyE2EModel, spans: SpanVocab, lexicon: Lexicon
) -> None:
    states = encode(model, spans, lexicon, "woaibj", "北京")
    cross = model.cross_memory(states)
    rng = random.Random(3)
    chain = tuple(rng.randrange(lexicon.size) for _ in range(40))
    sets = {
        # One node alone: 31 padding rows. 32 nodes: none. 33: a bucket of 64 with 31.
        "alone": [()],
        "full-32": [chain[:31]],
        "past-32": [chain[:32]],
        "full-64": [chain[:63]],
        "past-64": [chain[:40], *random_requests(rng, lexicon.size, 25, 3)],
    }
    for name, requests in sets.items():
        reference = reference_rows(model, states, requests)
        results = []
        for buckets in ((32, 64, 128, 256), (64, 128, 256), (128, 256), (256,)):
            [*calls] = plan(requests, buckets=buckets)
            assert len(calls) == 1, name
            [call] = calls
            feeds = model.tree_feeds(call)
            with torch.no_grad():
                got = model.tree_log_probs(
                    feeds.reader_ids, feeds.positions, feeds.tree_bias, cross
                )[0]
            real = got[: len(call.nodes)]
            results.append((call, real))
            for request, row in call.answers:
                torch.testing.assert_close(got[row], reference[request], atol=TIGHT, rtol=0.0)
            assert torch.isfinite(got).all(), (name, buckets)
        # The same node set in every bucket: padding added rows, changed none.
        first = results[0][1]
        for call, real in results[1:]:
            assert call.nodes == results[0][0].nodes
            torch.testing.assert_close(real, first, atol=TIGHT, rtol=0.0)


def test_a_padded_row_sees_only_itself_and_the_bias_is_the_trees(
    model: KeyE2EModel,
) -> None:
    [call] = plan([(1, 2), (1, 3), (0,)])
    feeds = model.tree_feeds(call)
    bias = feeds.tree_bias[0, 0]
    assert bias.shape == (32, 32) and feeds.reader_ids.shape == feeds.positions.shape == (1, 32)
    visible = bias == 0.0
    assert bias[~visible].eq(NEG).all()
    for row in range(32):
        assert visible[row].nonzero().flatten().tolist() == list(call.visible(row))
    # (), (0,), (1,), (1, 2), (1, 3): the root, then depth one in id order.
    assert call.nodes == ((), (0,), (1,), (1, 2), (1, 3))
    assert feeds.positions[0, :5].tolist() == [0, 1, 1, 2, 2]
    assert feeds.positions[0, 5:].eq(0).all()
    assert feeds.reader_ids[0, 0] == feeds.reader_ids[0, 5]  # the root and a pad are <bos>
    emit = model.decoder.emit_ids
    assert feeds.reader_ids[0, 3] == emit[2] and feeds.reader_ids[0, 4] == emit[3]


def test_the_cross_memory_pads_the_key_states_and_masks_the_padding(
    model: KeyE2EModel, spans: SpanVocab, lexicon: Lexicon
) -> None:
    states = encode(model, spans, lexicon, "wmzyq", "我们")
    cross = model.cross_memory(states)
    config = model.decoder.reader.config
    layers, heads = config.layers, config.heads
    assert cross.keys.shape == cross.values.shape == (layers, heads, N_MAX, config.hidden // heads)
    assert cross.bias.shape == (1, 1, 1, N_MAX)
    assert cross.bias[..., :5].eq(0).all() and cross.bias[..., 5:].eq(NEG).all()
    assert cross.keys[:, :, 5:].eq(0).all() and cross.values[:, :, 5:].eq(0).all()
    for layer, gate in enumerate(model.decoder.gated_layers()):
        keys, values = gate.kv(states.states)
        torch.testing.assert_close(cross.keys[layer, :, :5], keys[0])
        torch.testing.assert_close(cross.values[layer, :, :5], values[0])
    short = model.cross_memory(states, length=8)
    assert short.keys.shape[2] == 8
    with pytest.raises(ValueError, match="do not fit"):
        model.cross_memory(states, length=4)
    gap = KeyStates(states=states.states, mask=torch.tensor([[True, False, True, True, True]]))
    with pytest.raises(ValueError, match="non-empty prefix"):
        model.cross_memory(gap)


def test_the_ctc_log_probs_are_the_logits_log_softmaxed(
    model: KeyE2EModel, spans: SpanVocab, lexicon: Lexicon
) -> None:
    states = encode(model, spans, lexicon, "zhonghe", None)
    with torch.no_grad():
        logits = model.ctc_logits(states.states[0])
        assert logits.shape == (7, lexicon.size + 1)
        torch.testing.assert_close(
            model.ctc_log_probs(states.states[0]), logits.float().log_softmax(-1)
        )


# --- the manifest -----------------------------------------------------------------


@pytest.fixture(name="bundle")
def bundle_fixture(
    model: KeyE2EModel, lexicon: Lexicon, legality: Legality, tmp_path: Path
) -> Path:
    directory = tmp_path / "bundle"
    write_key_bundle(model, 11, lexicon, legality, KeyTokenizer(), directory)
    return directory


def test_the_manifest_describes_the_bundle(
    bundle: Path, model: KeyE2EModel, lexicon: Lexicon, legality: Legality
) -> None:
    manifest = KeyManifest.read(bundle)
    config = model.decoder.reader.config
    assert manifest.layout == LAYOUT == "key-bundle"
    assert manifest.step == 11 and manifest.base_model == model.config.base_model
    assert manifest.policy_version == POLICY_VERSION and manifest.n_max == N_MAX
    assert manifest.neg == NEG and manifest.context_tokens == 64
    assert manifest.specials.pad == 0 and manifest.specials.mask == 3
    assert (manifest.specials.cls, manifest.specials.sep) == (1, 2)
    assert manifest.letter_pad == 26
    assert manifest.characters == tuple(lexicon.characters)
    assert manifest.reader_ids == tuple(model.decoder.emit_ids.tolist())
    assert manifest.reader_bos == BOS
    assert manifest.dims.emitted == lexicon.size
    assert manifest.dims.decoder.layers == config.layers
    assert manifest.dims.decoder.head_dim * manifest.dims.decoder.heads == config.hidden
    assert manifest.dims.fill.hidden == model.fill.config.hidden_size
    assert manifest.dims.gated == len(model.gated_layers())
    assert manifest.dims.key_front.layers == len(model.key_front.blocks)
    assert manifest.dims.fill.layer_norm_eps == model.fill.config.layer_norm_eps
    assert manifest.correct_syllables == model.keys_config.correct_syllables
    assert [(p.keys, p.unfinished_tail) for p in manifest.probes] == list(LEGALITY_PROBES)
    assert any(p.arcs > 0 for p in manifest.probes)
    assert manifest.families == families()
    by_name = {family.name: family for family in manifest.families}
    assert [f.name for f in by_name["context"].functions] == ["ctx_64"]
    assert [f.name for f in by_name["keys"].functions] == [f"keys_{w}" for w in KEYS_WIDTHS]
    assert [f.capacity for f in by_name["keys"].functions] == [w - 2 for w in KEYS_WIDTHS]
    assert [f.name for f in by_name["tree"].functions] == [f"tree_{s}" for s in TREE_BUCKETS]
    assert [f.length for f in by_name["tree"].functions] == list(TREE_BUCKETS)
    assert [probe.name for probe in manifest.reference] == [p[0] for p in REFERENCE_PROBES]
    # What a loader checks against its own tables.
    manifest.check_lexicon(lexicon)
    manifest.check_legality(legality)


def test_the_manifest_is_json_the_dataclasses_serialise_to(bundle: Path) -> None:
    manifest = KeyManifest.read(bundle)
    record = json.loads((bundle / MANIFEST).read_text(encoding="utf-8"))
    assert record["characters"] == list(manifest.characters)
    assert record["families"][2]["functions"][0] == {
        "name": "tree_32",
        "length": 32,
        "capacity": 32,
    }
    written = manifest.write(bundle.parent)
    assert written.read_bytes() == (bundle / MANIFEST).read_bytes()
    assert KeyManifest.read(bundle.parent) == manifest


def test_a_moved_digest_another_order_or_another_policy_is_refused(
    bundle: Path, lexicon: Lexicon, legality: Legality, tmp_path: Path
) -> None:
    manifest = KeyManifest.read(bundle)
    index = next(index for index, probe in enumerate(manifest.probes) if probe.arcs)
    probe = manifest.probes[index]
    changed = replace(probe, digest=f"{int(probe.digest, 16) ^ 1:016x}")
    moved = replace(
        manifest, probes=(*manifest.probes[:index], changed, *manifest.probes[index + 1 :])
    )
    with pytest.raises(ValueError, match=rf"{probe.keys!r}.*pins"):
        moved.check_legality(legality)
    reversed_order = replace(manifest, characters=tuple(reversed(manifest.characters)))
    with pytest.raises(ValueError, match=r"another row"):
        reversed_order.check_lexicon(lexicon)
    record = json.loads((bundle / MANIFEST).read_text(encoding="utf-8"))
    for name, value, message in (
        ("layout", "keys-pages", "layout"),
        ("policy_version", POLICY_VERSION + 1, "packing policy"),
        ("n_max", "many", "n_max"),
        ("surprise", 1, "surprise"),
    ):
        wrong = tmp_path / f"wrong-{name}"
        wrong.mkdir()
        (wrong / MANIFEST).write_text(json.dumps({**record, name: value}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            KeyManifest.read(wrong)
    other = tmp_path / "families"
    other.mkdir()
    shrunk = {**record, "families": record["families"][:2]}
    (other / MANIFEST).write_text(json.dumps(shrunk), encoding="utf-8")
    with pytest.raises(ValueError, match="families"):
        KeyManifest.read(other)
    short = tmp_path / "short"
    short.mkdir()
    (short / MANIFEST).write_text(
        json.dumps({**record, "reader_ids": record["reader_ids"][:-1]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="reader ids"):
        KeyManifest.read(short)


# --- the bundle -------------------------------------------------------------------


def test_the_weights_are_what_the_families_read(bundle: Path, model: KeyE2EModel) -> None:
    manifest = KeyManifest.read(bundle)
    state = model.state_dict()
    with safe_open(bundle / "weights.safetensors", framework="pt") as weights:
        assert sorted(weights.keys()) == [t.name for t in manifest.tensors]
        for entry in manifest.tensors:
            tensor = weights.get_tensor(entry.name)
            assert tuple(tensor.shape) == entry.shape
            assert entry.dtype == ("int64" if entry.name == "decoder.emit_ids" else "float32")
            if entry.name != "key_front.letter_table":
                assert torch.equal(tensor, state[entry.name]), entry.name
        table = weights.get_tensor("key_front.letter_table")
    front = model.key_front
    assert table.shape == (27, model.fill.config.hidden_size)
    torch.testing.assert_close(table, front.letters.weight + front.place(front.coordinates))
    assert table[26].eq(0).all()
    names = {entry.name: entry.families for entry in manifest.tensors}
    # The span model's parts are not the key model's.
    assert not any(name.startswith(("span_embeddings", "letter_encoder")) for name in names)
    assert "candidate_mask" not in names and "key_front.letters.weight" not in names
    assert names["context.embeddings.word_embeddings.weight"] == ("context",)
    assert names["fill.embeddings.word_embeddings.weight"] == ("keys",)
    assert names["blank.weight"] == ("keys",)
    assert names["cross_attention.0.gate"] == ("keys",)
    assert names["decoder.reader.embed.weight"] == ("tree",)
    assert names["decoder.emit_ids"] == ("tree",)
    assert names["decoder.cross_attention.0.key.weight"] == ("keys",)
    assert names["decoder.cross_attention.0.value.weight"] == ("keys",)
    assert names["decoder.cross_attention.0.query.weight"] == ("tree",)
    assert names["decoder.cross_attention.0.gate"] == ("tree",)
    assert (bundle / "tokenizer.json").is_file()


def test_every_parameter_is_in_the_bundle_or_left_out_by_name(model: KeyE2EModel) -> None:
    chosen = bundle_weights(model)
    left_out = {name for name in model.state_dict() if name not in chosen}
    assert left_out == {
        "candidate_mask",
        "emittable_token_ids",
        "span_embeddings.weight",
        "letter_encoder.letters.weight",
        "letter_encoder.positions.weight",
        "letter_encoder.project.weight",
        "key_front.letters.weight",
        "key_front.place.weight",
    }


def test_the_reference_holds_the_probes_functions_outputs(
    bundle: Path, model: KeyE2EModel, spans: SpanVocab, lexicon: Lexicon
) -> None:
    manifest = KeyManifest.read(bundle)
    reference = read_reference(bundle)
    hidden = model.fill.config.hidden_size
    emitted = lexicon.size
    for probe in manifest.reference:
        n = len(probe.keys)
        name = probe.name
        assert reference[f"{name}.context_ids"].shape == (64,)
        assert reference[f"{name}.context_ids"].dtype == np.int32
        mask = reference[f"{name}.context_mask"]
        assert mask.shape == (64,) and mask.dtype == np.float32
        assert reference[f"{name}.context"].shape == (64, hidden)
        if probe.context is None:
            assert not mask.any() and not reference[f"{name}.context"].any()
        else:
            assert mask.sum() > 2
        for array in ("input_ids", "key_letters", "attention_mask"):
            assert reference[f"{name}.{array}"].shape == (n + 2,)
        assert reference[f"{name}.ctc_logits"].shape == (n, emitted + 1)
        assert reference[f"{name}.cross_keys"].shape[2] == N_MAX
        assert reference[f"{name}.cross_bias"].shape == (1, 1, 1, N_MAX)
        # The keys' logits are what the model gives for the same record.
        states = encode(model, spans, lexicon, probe.keys, probe.context)
        np.testing.assert_allclose(
            reference[f"{name}.ctc_logits"],
            model.ctc_logits(states.states[0]).detach().numpy(),
            atol=TIGHT,
        )
        for number, requests in enumerate(probe.tree):
            assert requests == reference_requests(emitted, manifest.reference.index(probe))[number]
            for position, call in enumerate(plan(requests)):
                prefix = f"{name}.tree{number}.call{position}"
                assert reference[f"{prefix}.logits"].shape == (call.bucket, emitted)
                assert reference[f"{prefix}.tree_bias"].shape == (call.bucket, call.bucket)
                assert reference[f"{prefix}.reader_ids"].shape == (call.bucket,)
                answers = reference[f"{prefix}.answers"]
                assert [tuple(pair) for pair in answers.tolist()] == list(call.answers)
                # The tree's rows are the transition's, as recorded.
                got = torch.from_numpy(reference[f"{prefix}.logits"][answers[:, 1]])
                np.testing.assert_allclose(
                    got.log_softmax(-1).numpy(), reference[f"{prefix}.log_probs"], atol=CLOSE
                )
    arrays = [name for name in reference if ".tree1." in name]
    assert any(".call0.log_probs" in name for name in arrays)


def test_the_bundle_is_deterministic(
    bundle: Path, model: KeyE2EModel, lexicon: Lexicon, legality: Legality, tmp_path: Path
) -> None:
    again = tmp_path / "again"
    write_key_bundle(model, 11, lexicon, legality, KeyTokenizer(), again)
    for name in (MANIFEST, "weights.safetensors", "tokenizer.json"):
        assert (bundle / name).read_bytes() == (again / name).read_bytes(), name
    first, second = read_reference(bundle), read_reference(again)
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)


def test_a_qwen_context_a_warm_up_and_mismatched_tables_are_refused(
    lexicon: Lexicon, legality: Legality, reader_vocab: CharVocab, tmp_path: Path
) -> None:
    class QwenLike(torch.nn.Module):
        pass

    model = tiny_joint_model(lexicon, reader_vocab)
    model.context = QwenLike()  # type: ignore[assignment]
    with pytest.raises(ValueError, match=r"MacBERT's tower.*QwenLike"):
        write_key_bundle(model, 1, lexicon, legality, KeyTokenizer(), tmp_path / "qwen")
    assert not (tmp_path / "qwen" / "weights.safetensors").exists()

    warm_up = tiny_key_model(lexicon, reader_vocab)
    with pytest.raises(ValueError, match="warm-up"):
        write_key_bundle(warm_up, 1, lexicon, legality, KeyTokenizer(), tmp_path / "warm")

    joint = tiny_joint_model(lexicon, reader_vocab)
    other = SimpleNamespace(characters=tuple(reversed(lexicon.characters)))
    with pytest.raises(ValueError, match="another set than the lexicon"):
        write_key_bundle(joint, 1, lexicon, other, KeyTokenizer(), tmp_path / "set")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="context function holds"):
        write_key_bundle(joint, 1, lexicon, legality, KeyTokenizer(), tmp_path / "ctx", 32)


def test_writing_a_bundle_leaves_the_models_mode_alone(
    lexicon: Lexicon, legality: Legality, reader_vocab: CharVocab, tmp_path: Path
) -> None:
    model = tiny_joint_model(lexicon, reader_vocab)
    model.train()
    write_key_bundle(model, 1, lexicon, legality, KeyTokenizer(), tmp_path / "mode")
    assert model.training


# --- the fixture command ----------------------------------------------------------


def test_the_fixture_command_writes_the_committed_bytes(tmp_path: Path) -> None:
    out = tmp_path / "key-tree.json"
    result = CliRunner().invoke(app, ["gen-key-tree-fixture", "--out", str(out)])
    assert result.exit_code == 0, result.output
    committed = (
        Path(__file__).resolve().parents[2] / "crates/ime-neural/tests/fixtures/key-tree.json"
    )
    assert out.read_bytes() == committed.read_bytes()
