"""Stage the char-lm kernel's inputs on a Colab VM and start it detached.

Runs inside the VM's kernel through `colab exec -f colab/stage.py` after
`colab/char-lm.sh` has uploaded the Kaggle credentials and the kernel script.
It recreates the mount layout the Kaggle kernel expects (`/kaggle/input/<slug>`
holding each dataset's files, `/kaggle/working` for outputs) by downloading
the same datasets with the Kaggle API, then launches the unchanged kernel as a
process that outlives this cell.
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

INPUTS = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
KERNEL = Path("/kaggle/kernel.py")
DATASETS = (
    "lexoliu/mlime-src",
    "lexoliu/mlime-route-a-assets",
    "lexoliu/mlime-run3-v1-samples",
    "lexoliu/mlime-run3-rest-samples",
)


def download(dataset):
    """Fetch *dataset* into its mount directory unless it is already there."""
    mount = INPUTS / dataset.split("/")[1]
    if mount.is_dir() and any(mount.iterdir()):
        print(f"{dataset}: already staged", flush=True)
        return
    mount.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(mount)],
        check=True,
    )
    for archive in mount.glob("*.zip"):
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(mount)
        archive.unlink()
    print(f"{dataset}: {sum(1 for _ in mount.rglob('*') if _.is_file())} files", flush=True)


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    if not Path("/root/.kaggle/credentials.json").is_file():
        raise FileNotFoundError("upload ~/.kaggle/credentials.json to /root/.kaggle first")
    if not KERNEL.is_file():
        raise FileNotFoundError(f"upload kaggle/char-lm/kernel.py to {KERNEL} first")
    WORKING.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        download(dataset)
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
