"""Stage the char-lm kernel's inputs on a Colab VM and start it detached.

Runs on the VM as its own process, which `colab/char-lm.sh start` launches
(logging to `/kaggle/working/stage.out`) after it has uploaded the Hugging
Face token and the kernel script. It recreates the mount layout the Kaggle
kernel expects (`/kaggle/input/<slug>` holding each dataset's files,
`/kaggle/working` for outputs) by downloading the same datasets from their
private Hugging Face repos, all at once, then launches the unchanged kernel as
a process that outlives this one.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

INPUTS = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
KERNEL = Path("/kaggle/kernel.py")
HF_TOKEN = Path("/root/.cache/huggingface/token")
#: Private Hub dataset repos, named like the Kaggle slugs the kernel mounts.
DATASETS = (
    "lexoliu/mlime-src",
    "lexoliu/mlime-route-a-assets",
    "lexoliu/mlime-run3-v1-samples",
    "lexoliu/mlime-run3-rest-samples",
)


def download(dataset):
    """Download the Hub dataset repo *dataset* into its mount directory."""
    from huggingface_hub import snapshot_download

    mount = INPUTS / dataset.split("/")[1]
    snapshot_download(dataset, repo_type="dataset", local_dir=mount, max_workers=8)
    # The Hub keeps its download metadata under the mount's `.cache`.
    files = sum(1 for f in mount.rglob("*") if f.is_file() and ".cache" not in f.parts)
    print(f"{dataset}: {files} files", flush=True)


def main():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub"], check=True
    )
    if not HF_TOKEN.is_file():
        raise FileNotFoundError(f"upload ~/.cache/huggingface/token to {HF_TOKEN} first")
    if not KERNEL.is_file():
        raise FileNotFoundError(f"upload kaggle/char-lm/kernel.py to {KERNEL} first")
    WORKING.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(DATASETS)) as pool:
        list(pool.map(download, DATASETS))
    log = open(WORKING / "kernel.out", "ab")  # noqa: SIM115 -- handed to the child
    child = subprocess.Popen(
        [sys.executable, str(KERNEL)],
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=str(WORKING),
        env=dict(os.environ),
        start_new_session=True,
    )
    (WORKING / "kernel.pid").write_text(f"{child.pid}\n")
    print(f"kernel started, pid {child.pid}; log {WORKING / 'kernel.out'}", flush=True)


main()
