"""Packaging is deterministic, and each model directory's manifest says what a loader needs."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
import zstandard

from mlime_coreml.bundle import Bundle
from mlime_coreml.package import (
    DIRECTORY_MANIFEST,
    sha256_file,
    write_archive,
    write_directory_manifest,
)


def entries(tmp_path: Path) -> list[tuple[str, bytes | Path | None]]:
    loose = tmp_path / "loose.bin"
    loose.write_bytes(b"weights")
    return [("z.json", b"{}"), ("model/", None), ("model/w.bin", loose), ("a.txt", b"a")]


def test_two_archives_of_the_same_entries_are_the_same_bytes(tmp_path: Path) -> None:
    first, second = tmp_path / "one.tar.zst", tmp_path / "two.tar.zst"
    assert write_archive(first, entries(tmp_path)) == write_archive(second, entries(tmp_path))
    assert first.read_bytes() == second.read_bytes()
    assert (tmp_path / "one.tar.zst.sha256").read_text() == f"{sha256_file(first)}  one.tar.zst\n"


def test_an_archive_is_sorted_with_fixed_owner_time_and_mode(tmp_path: Path) -> None:
    out = tmp_path / "a.tar.zst"
    write_archive(out, entries(tmp_path))
    with (
        out.open("rb") as raw,
        zstandard.ZstdDecompressor().stream_reader(raw) as stream,
        tarfile.open(fileobj=stream, mode="r|") as tar,
    ):
        members = list(tar)
    assert [member.name for member in members] == ["a.txt", "model", "model/w.bin", "z.json"]
    assert {(m.mtime, m.uid, m.gid, m.uname) for m in members} == {(0, 0, 0, "")}
    assert [oct(m.mode) for m in members] == ["0o644", "0o755", "0o644", "0o644"]


@pytest.mark.parametrize(
    ("family", "default", "lengths"),
    [
        ("context", "ctx_64", {"ctx_64": 64}),
        ("keys", "keys_16", {"keys_16": 16, "keys_32": 32, "keys_64": 64, "keys_128": 128}),
        ("tree", "tree_32", {"tree_32": 32, "tree_64": 64, "tree_128": 128, "tree_256": 256}),
    ],
)
def test_a_directory_manifest_lists_the_functions_and_their_shapes(
    bundle: Bundle, tmp_path: Path, family: str, default: str, lengths: dict[str, int]
) -> None:
    path = write_directory_manifest(bundle.manifest, family, tmp_path)  # type: ignore[arg-type]
    assert path.name == DIRECTORY_MANIFEST
    document = json.loads(path.read_text())
    assert document["functions"] == lengths and document["default_function"] == default
    assert document["inputs"] and document["outputs"]
    for tensor in (*document["inputs"].values(), *document["outputs"].values()):
        assert tensor["dtype"] in ("int32", "float16")
        assert all(isinstance(dim, int) or dim == "length" for dim in tensor["shape"])
