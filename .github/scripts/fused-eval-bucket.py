"""Move one fused-eval section's files to and from the private resume bucket (#170).

    python fused-eval-bucket.py put <dir> <prefix>
    python fused-eval-bucket.py get <dir> <prefix>

`put` sends every file under *dir* to `lexoliu/mlime-resume` under *prefix*,
at its path relative to *dir*; `get` brings every file under *prefix* back
into *dir* the same way, and finds none on a section's first run. The
repository is public and the dumps hold eval3's text, so they never become
workflow artifacts.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

BUCKET = "lexoliu/mlime-resume"


def put(api: HfApi, source: Path, prefix: str) -> None:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise SystemExit(f"{source} holds nothing to publish")
    api.batch_bucket_files(
        BUCKET,
        add=[(path, f"{prefix}/{path.relative_to(source).as_posix()}") for path in files],
    )
    for path in files:
        print(f"{BUCKET}/{prefix}/{path.relative_to(source).as_posix()}")


def get(api: HfApi, target: Path, prefix: str) -> None:
    entries = api.list_bucket_tree(BUCKET, prefix=f"{prefix}/", recursive=True)
    paths = [entry.path for entry in entries if entry.type == "file"]
    if not paths:
        print(f"{BUCKET}/{prefix}: nothing kept yet")
        return
    pairs = [(path, target / path.removeprefix(f"{prefix}/")) for path in paths]
    for _, local in pairs:
        local.parent.mkdir(parents=True, exist_ok=True)
    api.download_bucket_files(BUCKET, pairs, raise_on_missing_files=True)
    for path in paths:
        print(f"{BUCKET}/{path}")


command, directory, prefix = sys.argv[1], Path(sys.argv[2]), sys.argv[3].strip("/")
{"put": put, "get": get}[command](HfApi(), directory, prefix)
