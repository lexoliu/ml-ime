"""Stage the char-lm kernel's inputs on a Colab VM and start it detached.

Runs on the VM as its own process, which `colab/char-lm.sh start` launches
(logging to `/kaggle/working/stage.out`) after it has uploaded the Hugging
Face token and the kernel script. It recreates the mount layout the Kaggle
kernel expects (`/kaggle/input/<slug>` holding each dataset's files,
`/kaggle/working` for outputs) by downloading the same datasets from their
private Hugging Face repos, all of them in one `aria2c` run (`hub.py`), then
launches the unchanged kernel as a process that outlives this one.
"""

import os
import subprocess
import sys
from pathlib import Path

INPUTS = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
KERNEL = Path("/kaggle/kernel.py")
HUB = Path("/kaggle/hub.py")
HF_TOKEN = Path("/root/.cache/huggingface/token")
#: Private Hub dataset repos, named like the Kaggle slugs the kernel mounts.
DATASETS = (
    "lexoliu/mlime-route-a-assets",
    "lexoliu/mlime-run3-v1-samples",
    "lexoliu/mlime-run3-rest-samples",
)


def main():
    installs = [
        subprocess.Popen([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub"]),
        subprocess.Popen(["apt-get", "install", "-y", "-q", "aria2"]),
    ]
    for install in installs:
        if install.wait() != 0:
            raise subprocess.CalledProcessError(install.returncode, install.args)
    if not HF_TOKEN.is_file():
        raise FileNotFoundError(f"upload ~/.cache/huggingface/token to {HF_TOKEN} first")
    for uploaded in (KERNEL, HUB, HUB.parent / "aria2.py"):
        if not uploaded.is_file():
            raise FileNotFoundError(f"upload {uploaded.name} to {uploaded} first")
    WORKING.mkdir(parents=True, exist_ok=True)
    # Imported only now: the Hub client is the one just installed, and
    # `hub.py` finds `aria2.py` beside it.
    sys.path.insert(0, str(HUB.parent))
    import hub

    hub.download(DATASETS, INPUTS)
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
