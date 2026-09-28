#!/bin/bash
# Run the e2e chain's Colab leg on a T4 through the Colab CLI.
#
#   colab/e2e.sh start [--gpu T4]   provision, stage, resume the newest trio, launch
#   colab/e2e.sh status             tail of the kernel's and publisher's logs
#   colab/e2e.sh fetch <out dir>    download the metrics, the summary, the logs
#   colab/e2e.sh publish            version lexoliu/mlime-e2e-resume once, in the VM
#   colab/e2e.sh stop               release the VM
#
# The session is named e2e; `colab sessions` lists it. The kernel is the same
# script Kaggle runs (kaggle/e2e/kernel.py): on one T4 it trains the run's
# world of two as one process of two virtual ranks, and it resumes the newest
# checkpoint-paused trio mounted -- the latest COMPLETE Kaggle segment's
# output or the lexoliu/mlime-e2e-resume dataset, whichever reached the higher
# step. A publisher loop beside the kernel pushes each interval checkpoint to
# that dataset, so a VM that dies without warning loses minutes, not the leg.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
session=e2e
command=${1:?start|status|fetch|publish|stop}; shift
case $command in
  start)
    gpu=T4
    while [ $# -gt 0 ]; do case $1 in --gpu) gpu=$2; shift 2;; *) echo "unknown argument $1" >&2; exit 2;; esac; done
    colab new -s $session --gpu "$gpu"
    # Colab mounts an empty, read-only /kaggle/input for its own Kaggle
    # integration; the kernel reads the datasets staged there, so it goes first.
    printf 'import os, subprocess\nif os.path.ismount("/kaggle/input"):\n    subprocess.run(["umount", "/kaggle/input"], check=True)\nfor d in ("/root/.kaggle", "/kaggle/input", "/kaggle/working"): os.makedirs(d, exist_ok=True)\n' | colab exec -s $session
    colab upload -s $session "$HOME/.kaggle/credentials.json" /root/.kaggle/credentials.json
    colab upload -s $session "$here/kaggle/e2e/kernel.py" /kaggle/kernel.py
    colab upload -s $session "$here/kaggle/e2e/kernel-metadata.json" /kaggle/kernel-metadata.json
    colab upload -s $session "$here/colab/e2e-publish.py" /kaggle/e2e-publish.py
    # Staging downloads gigabytes, far past what one `colab exec` waits for,
    # so it runs as its own process and `status` reads its log.
    colab upload -s $session "$here/colab/e2e-stage.py" /kaggle/stage.py
    printf 'import subprocess, sys\nlog = open("/kaggle/working/stage.out", "ab")\nchild = subprocess.Popen([sys.executable, "/kaggle/stage.py"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)\nprint(f"staging started, pid {child.pid}; log /kaggle/working/stage.out")\n' | colab exec -s $session
    ;;
  status)
    printf 'import json, time\nfrom pathlib import Path\np = Path("/kaggle/working")\nfor name in ("stage.out", "kernel.out", "train.log", "publish.out"):\n    f = p / name\n    if f.is_file():\n        print(f"--- {name} ---")\n        print(f.read_text()[-3000:])\nstate = p / "publish-state.json"\nif state.is_file():\n    s = json.loads(state.read_text())\n    if "last_published_at" in s:\n        print(f"last publish: step {s[\"last_published_step\"]}, {round(time.time() - s[\"last_published_at\"])}s ago")\n    if s.get("last_error"):\n        print(f"last publish error: {s[\"last_error\"]}")\nrun = p / "run"\nif run.is_dir():\n    print("intervals:", sorted(d.name for d in run.glob("interval-*")))\n' | colab exec -s $session
    ;;
  fetch)
    out=${1:?out dir}
    mkdir -p "$out"
    for f in run-summary.json run-config.json train.log publish.out publish-state.json run/metrics.jsonl; do
      colab download -s $session "/kaggle/working/$f" "$out/$(basename "$f")" || echo "not on the VM yet: $f"
    done
    ;;
  publish)
    printf 'import os\nos.environ["ONCE"] = "1"\nexec(open("/kaggle/e2e-publish.py").read())\n' | colab exec -s $session
    ;;
  stop)
    colab stop -s $session
    ;;
  *) echo "unknown command $command" >&2; exit 2;;
esac
