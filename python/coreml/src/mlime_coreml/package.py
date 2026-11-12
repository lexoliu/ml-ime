"""Each model directory's ``manifest.json`` and the deterministic ``tar.zst`` of the three.

Entries are sorted by name and stamped with a fixed mtime, owner and mode, so two runs over the
same compiled models produce identical bytes (the compiled models themselves are deterministic
because the converted spec is re-serialised with ordered maps).
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import zstandard

from mlime_coreml import spec
from mlime_coreml.bundle import MANIFEST, TOKENIZER, Manifest
from mlime_coreml.signature import Tensor, functions, signature
from mlime_coreml.spec import Family

#: The report the archive carries beside the models.
VERIFICATION = "verification.json"

#: The manifest each model directory ships, which the loader reads (ondevice's format).
DIRECTORY_MANIFEST = "manifest.json"


def _tensor_spec(tensor: Tensor) -> dict[str, object]:
    return {"dtype": tensor.dtype, "shape": list(tensor.shape)}


def write_directory_manifest(manifest: Manifest, family: Family, directory: Path) -> Path:
    """Write *directory*'s ``manifest.json``: the functions, the shared inputs and the outputs.

    ``length`` in a shape is the function's own length. The fields beyond ``functions``,
    ``inputs`` and ``outputs`` are the model's own.
    """
    members = functions(manifest, family)
    sig = signature(manifest, family)
    document = {
        "model": "mlime-key-model",
        "family": family,
        "step": manifest.step,
        "policy_version": manifest.policy_version,
        "functions": {function.name: function.length for function in members},
        "default_function": members[0].name,
        "inputs": {tensor.name: _tensor_spec(tensor) for tensor in sig.inputs},
        "outputs": {tensor.name: _tensor_spec(tensor) for tensor in sig.outputs},
        "n_max": spec.N_MAX,
        "mask": spec.NEG,
    }
    path = directory / DIRECTORY_MANIFEST
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path


def sha256_file(path: Path) -> str:
    """The hex sha256 of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _info(name: str, size: int, directory: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    if directory:
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        info.size = 0
    else:
        info.mode = 0o644
        info.size = size
    return info


def _tree(root: Path, base: Path) -> list[tuple[str, Path | None]]:
    """``(archive name, path or None for a directory)`` for *root* and below, relative to *base*."""
    entries: list[tuple[str, Path | None]] = [(f"{root.relative_to(base)}/", None)]
    for path in sorted(root.rglob("*")):
        name = str(path.relative_to(base))
        if path.is_dir():
            entries.append((f"{name}/", None))
        elif path.is_file():
            entries.append((name, path))
        else:
            raise RuntimeError(f"not a file or a directory, so not archivable: {path}")
    return entries


def write_archive(out: Path, entries: list[tuple[str, bytes | Path | None]]) -> str:
    """Write *entries* (name, then ``None`` for a directory, bytes or a file path) as a tar.zst.

    Sorted by name, with a ``.sha256`` sidecar; returns the archive's sha256.
    """
    with (
        out.open("wb") as raw,
        zstandard.ZstdCompressor().stream_writer(raw) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as tar,
    ):
        for name, payload in sorted(entries, key=lambda entry: entry[0]):
            if payload is None:
                tar.addfile(_info(name, 0, directory=True))
            elif isinstance(payload, bytes):
                tar.addfile(_info(name, len(payload)), io.BytesIO(payload))
            else:
                with payload.open("rb") as handle:
                    tar.addfile(_info(name, payload.stat().st_size), handle)
    digest = sha256_file(out)
    out.with_suffix(out.suffix + ".sha256").write_text(f"{digest}  {out.name}\n")
    return digest


def build_archive(
    manifest: Manifest, bundle_dir: Path, models_dir: Path, verification: Path, out_dir: Path
) -> tuple[Path, str]:
    """The archive of the model directories, the bundle's manifest and tokenizer, and the report.

    Returns its path and sha256.
    """
    entries: list[tuple[str, bytes | Path | None]] = []
    for family in spec.FAMILIES:
        entries.extend(_tree(models_dir / spec.DIRECTORIES[family], models_dir))
    entries += [
        (MANIFEST, bundle_dir / MANIFEST),
        (TOKENIZER, bundle_dir / TOKENIZER),
        (VERIFICATION, verification),
    ]
    out = out_dir / f"mlime-keys-coreml-step{manifest.step}.tar.zst"
    digest = write_archive(out, entries)
    return out, digest
