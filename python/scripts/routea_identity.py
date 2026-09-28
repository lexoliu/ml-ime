"""The route A identity check: the Rust towers against the Python score file.

Run from the repository root:

    uv run --project python python/scripts/routea_identity.py \
        --eval-set data/eval3.jsonl --lattice data/lattice.jsonl \
        --scores data/scores-lattice-context-on.jsonl.gz \
        --export data/route-a-export/fp32 --emittable data/emittable.txt \
        [--no-context] [--work-dir DIR] [--emit BIN] [--ime-cli BIN]

On the eval set's dev slice (`--dev-share`, the record hash `ime-eval`
cuts with) it:

1. emits a score file through `emit --raw` -- the towers' unrounded log
   probabilities -- and diffs every candidate slot against `mlime train
   emit`'s file. The file was emitted by PyTorch and the live path by
   ONNX Runtime, so a slot differs by the file's rounding (5e-5 + one
   f32 ulp) plus the engines' fp32 drift; the script reports how many
   slots exceed the rounding bound and fails only past ENGINE_BOUND.
2. decodes the slice twice under `fused-eval --no-transition --weight 1`,
   once reading the score file, once computing emissions live, and diffs
   the dumps record by record: how many top-1 hypotheses differ, and for
   each that does, the margin the file path held over the live winner and
   the margin the live run held over the file winner. Any flip fails.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from mlime.logging import log

#: Half the score file's decimal quantum: `emit --raw` answers the full
#: precision values the file rounds away, so a slot beyond this plus one
#: f32 ulp owes to torch-versus-ORT kernel drift, not to rounding. The
#: count is reported, not enforced.
ROUND_BOUND = 5e-5

#: The hard bound on |live - file|: cross-engine fp32 drift on this
#: checkpoint measures <= 4.3e-4 on aarch64 and x86_64, so 1e-3 trips only
#: on a real divergence, not on kernel-order noise.
ENGINE_BOUND = 1e-3


def dev_slice(eval_set: Path, dev_share: float) -> list[int]:
    """The record indices `fused-eval --slice dev` decodes, replicating
    `EvalRecord::digest` in `crates/ime-eval`: blake2b-8 over
    `pinyin \\0 text \\0 context`, big-endian, below the share."""
    indices = []
    with eval_set.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            record = json.loads(line)
            hasher = hashlib.blake2b(digest_size=8)
            hasher.update(record["pinyin"].encode())
            hasher.update(b"\0")
            hasher.update(record["text"].encode())
            hasher.update(b"\0")
            hasher.update((record.get("context") or "").encode())
            if int.from_bytes(hasher.digest(), "big") / 18446744073709551615.0 < dev_share:
                indices.append(index)
    return indices


def load_scores(path: Path) -> dict[int, list[list[list[float]]]]:
    """A score file's records indexed by record number."""
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as source:
        return {json.loads(line)["record"]: json.loads(line)["paths"] for line in source}


def load_dump(dir_: Path) -> dict[int, list[dict[str, object]]]:
    """The one dev-section dump in *dir_*, as record -> hypotheses."""
    files = list(dir_.glob("*-dev.jsonl"))
    assert len(files) == 1, f"{dir_} holds {len(files)} dev dumps, expected one"
    with files[0].open(encoding="utf-8") as source:
        return {json.loads(line)["record"]: json.loads(line)["hypotheses"] for line in source}


def diff_scores(
    live: dict[int, list[list[list[float]]]],
    filed: dict[int, list[list[list[float]]]],
    records: list[int],
) -> tuple[float, int, int]:
    """Every dev record's max |live - file| and how many slots exceed the
    5e-5 + one-ulp rounding bound (torch-versus-ORT drift, reported)."""
    max_delta = 0.0
    violations = 0
    slots = 0
    for record in records:
        got = live[record]
        want = filed[record]
        assert len(got) == len(want), f"record {record}: path count differs"
        for got_path, want_path in zip(got, want, strict=True):
            for got_row, want_row in zip(got_path, want_path, strict=True):
                for value, reference in zip(got_row, want_row, strict=True):
                    delta = abs(value - reference)
                    bound = ROUND_BOUND + float(np.spacing(np.float32(reference)))
                    slots += 1
                    max_delta = max(max_delta, delta)
                    if delta > bound:
                        violations += 1
    return max_delta, slots, violations


def diff_dumps(
    live: dict[int, list[dict[str, object]]],
    filed: dict[int, list[dict[str, object]]],
    records: list[int],
) -> int:
    """How many dev records' top-1 differs, with each flip's margins."""
    flips = 0
    for record in records:
        live_top = live[record][0]
        file_top = filed[record][0]
        if live_top["text"] == file_top["text"]:
            continue
        flips += 1
        file_scores = {h["text"]: float(h["score"]) for h in filed[record]}
        live_scores = {h["text"]: float(h["score"]) for h in live[record]}
        # How far the winner was ahead of the flip partner in its own run.
        file_margin = float(file_top["score"]) - file_scores.get(live_top["text"], float("nan"))
        live_margin = float(live_top["score"]) - live_scores.get(file_top["text"], float("nan"))
        log.info(
            "top-1 flip",
            record=record,
            file=f"{file_top['text']}@{float(file_top['score']):.4f}",
            live=f"{live_top['text']}@{float(live_top['score']):.4f}",
            file_margin=round(file_margin, 4),
            live_margin=round(live_margin, 4),
        )
    return flips


def main() -> None:
    """Emit, decode and diff; exit 1 when a slot or a verdict fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--lattice", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--emittable", type=Path, required=True)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--dev-share", type=float, default=0.0905)
    parser.add_argument("--emit", type=Path, default=Path("target/release/examples/emit"))
    parser.add_argument("--ime-cli", type=Path, default=Path("target/release/ime-cli"))
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    records = dev_slice(args.eval_set, args.dev_share)
    log.info("dev slice", records=len(records), dev_share=args.dev_share)

    work = args.work_dir or Path(tempfile.mkdtemp(prefix="routea-identity-"))
    work.mkdir(parents=True, exist_ok=True)
    index_file = work / "dev.idx"
    index_file.write_text("".join(f"{record}\n" for record in records), encoding="utf-8")
    live_scores_path = work / "live.jsonl"
    emit_cmd = [
        str(args.emit.resolve()),
        str(args.export),
        str(args.lattice),
        str(live_scores_path),
        "--raw",
        "--records",
        str(index_file),
    ] + (["--no-context"] if args.no_context else [])
    subprocess.run(emit_cmd, check=True)
    log.info("emitted live scores", path=str(live_scores_path))

    live = load_scores(live_scores_path)
    filed = load_scores(args.scores)
    max_delta, slots, violations = diff_scores(live, filed, records)
    log.info(
        "emission diff",
        slots=slots,
        max_delta=round(max_delta, 6),
        bound=ROUND_BOUND,
        violations=violations,
    )

    dump_file_dir = work / "dump-file"
    dump_live_dir = work / "dump-live"
    fused = [
        str(args.ime_cli.resolve()),
        "fused-eval",
        "--eval-set",
        str(args.eval_set),
        "--emittable",
        str(args.emittable),
        "--no-transition",
        "--weight",
        "1",
        "--slice",
        "dev",
        "--dev-share",
        str(args.dev_share),
    ]
    subprocess.run([*fused, "--scores", str(args.scores), "--dump", str(dump_file_dir)], check=True)
    live_cmd = [*fused, "--route-a", str(args.export), "--dump", str(dump_live_dir)]
    if args.no_context:
        live_cmd.append("--no-context")
    subprocess.run(live_cmd, check=True)

    flips = diff_dumps(load_dump(dump_live_dir), load_dump(dump_file_dir), records)
    if max_delta > ENGINE_BOUND or flips:
        log.error(
            "identity check failed",
            violations=violations,
            flips=flips,
            records=len(records),
            max_delta=max_delta,
            engine_bound=ENGINE_BOUND,
        )
        sys.exit(1)
    log.info("identity check passed", records=len(records), slots=slots, max_delta=max_delta)


if __name__ == "__main__":
    main()
