#!/bin/bash
# Run a Colab leg of an e2e kernel through the Colab CLI.
#
#   colab/e2e.sh <kernel dir> start [--gpu T4|--cpu] [--census FILE]
#                                                provision, stage, resume the newest trio, launch
#   colab/e2e.sh <kernel dir> status             tail of the kernel's and publisher's logs
#   colab/e2e.sh <kernel dir> fetch <out dir>    download the metrics, the summary, the logs
#   colab/e2e.sh <kernel dir> publish            publish the newest trio to a resume slot once
#   colab/e2e.sh <kernel dir> stop               release the VM
#
# <kernel dir> holds a stamped kernel.py and its kernel-metadata.json --
# kaggle/e2e for v1's chain, or a stage stamped from it. Its slug,
# <run>-s<n>, names the run: the session is the run's name (`colab sessions`
# lists it), the Kaggle segments are <run>-s<n>, and the publisher's resume
# slots are <run>-resume-{a,b}. The kernel is the same script Kaggle runs: on
# one GPU it trains the run's world of two as one process of two virtual
# ranks, and it resumes the newest checkpoint-paused trio mounted -- the
# latest COMPLETE Kaggle segment's output or a resume slot, whichever reached
# the higher step. A publisher loop beside the kernel pushes each interval
# checkpoint to a slot, so a VM that dies without warning loses minutes, not
# the leg.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
kernel_dir=$(cd "${1:?kernel dir}" && pwd); shift
slug=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["id"])' "$kernel_dir/kernel-metadata.json")
session=$(basename "${slug%-s[0-9]*}")
command=${1:?start|status|fetch|publish|stop}; shift
case $command in
  start)
    gpu=T4
    census=
    while [ $# -gt 0 ]; do case $1 in
      --gpu) gpu=$2; shift 2;;
      --cpu) gpu=; shift;;
      --census) census=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); shift 2;;
      *) echo "unknown argument $1" >&2; exit 2;;
    esac; done
    colab new -s $session ${gpu:+--gpu "$gpu"}
    # Colab mounts an empty, read-only /kaggle/input for its own Kaggle
    # integration; the kernel reads the datasets staged there, so it goes first.
    printf 'import os, subprocess\nif os.path.ismount("/kaggle/input"):\n    subprocess.run(["umount", "/kaggle/input"], check=True)\nfor d in ("/root/.kaggle", "/root/.cache/huggingface", "/kaggle/input", "/kaggle/working"): os.makedirs(d, exist_ok=True)\n' | colab exec -s $session
    colab upload -s $session "$HOME/.kaggle/credentials.json" /root/.kaggle/credentials.json
    # Only the access token: the OAuth refresh token stays on the Mac, so no
    # VM ever refreshes it and invalidates the Mac's login.
    colab upload -s $session "$HOME/.cache/huggingface/token" /root/.cache/huggingface/token
    colab upload -s $session "$kernel_dir/kernel.py" /kaggle/kernel.py
    colab upload -s $session "$kernel_dir/kernel-metadata.json" /kaggle/kernel-metadata.json
    colab upload -s $session "$here/colab/e2e-publish.py" /kaggle/e2e-publish.py
    colab upload -s $session "$here/kaggle/output.py" /kaggle/output.py
    # A census counted for the same corpus and batch shape -- by a sibling
    # run, or by a CPU rehearsal -- spares the GPU the count; the kernel
    # recounts when even one field differs.
    if [ -n "$census" ]; then
      printf 'import os\nos.makedirs("/kaggle/input/census", exist_ok=True)\n' | colab exec -s $session
      colab upload -s $session "$census" /kaggle/input/census/batch-counts.json
    fi
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
