"""Put one fused-eval section's outputs in the private resume bucket (#170).

    python fused-eval-publish.py <dir> <prefix>

Every file under *dir* goes to `lexoliu/mlime-resume` under *prefix*, at its
path relative to *dir*. The repository is public and the dumps hold eval3's
text, so they never become workflow artifacts.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

BUCKET = "lexoliu/mlime-resume"

source, prefix = Path(sys.argv[1]), sys.argv[2].strip("/")
files = sorted(path for path in source.rglob("*") if path.is_file())
if not files:
    raise SystemExit(f"{source} holds nothing to publish")
HfApi().batch_bucket_files(
    BUCKET,
    add=[(path, f"{prefix}/{path.relative_to(source).as_posix()}") for path in files],
)
for path in files:
    print(f"{BUCKET}/{prefix}/{path.relative_to(source).as_posix()}")
