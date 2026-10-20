"""The end-to-end evaluation hands its dump to ``ime-ambiguity`` rather than measuring the slice.

The slice's selection and metrics are Rust's (``ime-eval``), tested there and
against ``fused-eval``'s own beams. What this side owns is the hand-off: the
dump, the eval set and the slice reach the binary as it reads them, its report
comes back verbatim, and a refusal surfaces with the binary's reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mlime.train.e2e_eval import ambiguity_report


def stub(tmp_path: Path, body: str) -> Path:
    """An executable standing in for ``ime-ambiguity``."""
    binary = tmp_path / "ime-ambiguity"
    binary.write_text(f"#!{sys.executable}\nimport json, sys\n{body}\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


def test_the_dump_reaches_the_binary_and_its_report_comes_back(tmp_path: Path) -> None:
    binary = stub(tmp_path, "print(json.dumps(sys.argv[1:]))")
    report = ambiguity_report(
        binary, tmp_path / "eval3.jsonl", tmp_path / "dump.jsonl", "test", 0.0905
    )
    assert json.loads(report) == [
        "--eval-set",
        str(tmp_path / "eval3.jsonl"),
        "--dump",
        str(tmp_path / "dump.jsonl"),
        "--slice",
        "test",
        "--dev-share",
        "0.0905",
    ]


def test_a_refusal_carries_the_binary_s_reason(tmp_path: Path) -> None:
    binary = stub(
        tmp_path,
        "sys.stderr.write('ime-ambiguity: the dump holds no row for record 7\\n'); sys.exit(1)",
    )
    with pytest.raises(RuntimeError, match="no row for record 7"):
        ambiguity_report(binary, tmp_path / "eval3.jsonl", tmp_path / "dump.jsonl", "all")
