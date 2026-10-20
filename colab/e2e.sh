#!/bin/bash
# Run a Colab leg of an e2e kernel through the Colab CLI.
#
#   colab/e2e.sh <kernel dir> start --session-hours H [--commit REV] [--gpu G|--cpu]
#                [--census FILE] [--resume-from RUN] [--no-publish] [--segment-steps N]
#                [--allow-commit-change]
#                                                provision, fetch the code, stage, launch
#   colab/e2e.sh <kernel dir> wait --deadline SECONDS [--interval SECONDS]
#                                                beat the watchdog until the leg ends
#   colab/e2e.sh <kernel dir> status             tail of the kernel's and publisher's logs
#   colab/e2e.sh <kernel dir> fetch <out dir>    download the census, metrics, samples, logs
#   colab/e2e.sh <kernel dir> publish            publish the newest trio to the bucket once
#   colab/e2e.sh <kernel dir> stop [--force]     release the VM once nothing would be lost
#
# <kernel dir> holds a stamped kernel.py and its kernel-metadata.json --
# kaggle/e2e for v1's chain, or a run stamped from it (`mlime e2e stamp`). Its
# slug, <run>-s<n>, names the run: the session is the run's name (`colab
# sessions` lists it), the Kaggle segments are <run>-s<n>, and the publisher's
# trios live under <run>/ in the `lexoliu/mlime-resume` bucket. The kernel is
# the same script Kaggle runs: on one GPU it trains the run's world of two as
# one process of two virtual ranks, and it resumes the newest trio mounted.
#
# The code is one commit of this public repository, fetched by the VM from
# GitHub: --commit (default HEAD) must be on a branch of origin, and this
# checkout must have no uncommitted changes. The stager, the publisher and
# the probe run from that checkout too; only the stamped kernel and its
# metadata are uploaded. A ranking stage's `ime-segment` is built here from
# the same commit and uploaded beside the inputs.
#
# --session-hours is how long the runtime is held: the kernel pauses itself
# inside it (`SESSION_SECONDS`), so the leg ends on a known budget. `wait`
# exits with the leg's end -- 10 paused (its trio published), 11 finished,
# 12 failed, 13 past --deadline, 14 runtime lost -- and `stop` refuses while
# the newest trio is unpublished or a finished run is only on the VM.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
source "$here/colab/provision.sh"
source "$here/colab/code.sh"
# Where the VM checks out the leg's commit (the kernel's CHECKOUT).
checkout=/tmp/ml-ime
kernel_dir=$(cd "${1:?kernel dir}" && pwd); shift
slug=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["id"])' "$kernel_dir/kernel-metadata.json")
session=$(basename "${slug%-s[0-9]*}")
command=${1:?start|wait|status|fetch|publish|stop}; shift
case $command in
  start)
    gpu=T4
    census=
    hours=
    rev=HEAD
    stage_options=()
    while [ $# -gt 0 ]; do case $1 in
      --gpu) gpu=$2; shift 2;;
      --cpu) gpu=; shift;;
      --census) census=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); shift 2;;
      --session-hours) hours=$2; shift 2;;
      --commit) rev=$2; shift 2;;
      # Race another run's newest trio under this run's name.
      --resume-from) stage_options+=(--resume-from "$2"); shift 2;;
      # A check leaves the run's lineage in the bucket untouched.
      --no-publish) stage_options+=(--no-publish); shift;;
      # A check pauses after a fixed count of steps, not on the clock.
      --segment-steps) stage_options+=(--segment-steps "$2"); shift 2;;
      --allow-commit-change) stage_options+=(--allow-commit-change); shift;;
      *) echo "unknown argument $1" >&2; exit 2;;
    esac; done
    # Everything that can refuse does so before a runtime bills.
    if [ -z "$hours" ]; then
      echo "--session-hours is required: the leg pauses inside the session it is given" >&2
      exit 2
    fi
    seconds=$(python3 -c 'import sys; h = float(sys.argv[1]); sys.exit("--session-hours must be positive") if h <= 0 else print(round(h * 3600))' "$hours")
    commit=$(leg_commit "$here" "$rev")
    segmenter=
    if kernel_ranks "$kernel_dir/kernel.py"; then
      segmenter=$(build_segmenter "$here" "$commit")
    fi
    echo "leg of $session: commit $commit, session ${seconds}s"
    provision $session ${gpu:+--gpu "$gpu"}
    # Colab mounts an empty, read-only /kaggle/input for its own Kaggle
    # integration; the kernel reads the datasets staged there, so it goes first.
    printf 'import os, subprocess\nif os.path.ismount("/kaggle/input"):\n    subprocess.run(["umount", "/kaggle/input"], check=True)\nfor d in ("/root/.kaggle", "/root/.cache/huggingface", "/kaggle/input", "/kaggle/working"): os.makedirs(d, exist_ok=True)\n' | colab exec -s $session
    colab upload -s $session "$HOME/.kaggle/credentials.json" /root/.kaggle/credentials.json
    # Only the access token: the OAuth refresh token stays on the Mac, so no
    # VM ever refreshes it and invalidates the Mac's login.
    colab upload -s $session "$HOME/.cache/huggingface/token" /root/.cache/huggingface/token
    colab upload -s $session "$kernel_dir/kernel.py" /kaggle/kernel.py
    colab upload -s $session "$kernel_dir/kernel-metadata.json" /kaggle/kernel-metadata.json
    # The kernel's own `checkout` fetches the commit, so the VM's code and the
    # kernel's check of it are one function.
    printf 'import runpy\nrunpy.run_path("/kaggle/kernel.py")["checkout"](%s)\n' "'$commit'" | colab exec -s $session
    # A census counted for the same corpus and batch shape -- by a sibling
    # run, or by an earlier leg (`fetch`) -- spares the GPU the count; the
    # kernel recounts when even one field differs.
    if [ -n "$census" ]; then
      printf 'import os\nos.makedirs("/kaggle/input/census", exist_ok=True)\n' | colab exec -s $session
      colab upload -s $session "$census" /kaggle/input/census/batch-counts.json
    fi
    if [ -n "$segmenter" ]; then
      printf 'import os\nos.makedirs("/kaggle/input/ime-segment", exist_ok=True)\n' | colab exec -s $session
      colab upload -s $session "$segmenter" /kaggle/input/ime-segment/ime-segment
    fi
    # Staging downloads gigabytes, far past what one `colab exec` waits for,
    # so it runs as its own process; `wait` and `status` read what it leaves.
    # A JSON list of strings is a Python literal: the stager's argv, quoted once.
    stage_argv=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' \
      --commit "$commit" --session-seconds "$seconds" ${stage_options[@]+"${stage_options[@]}"})
    printf 'import subprocess, sys\nlog = open("/kaggle/working/stage.out", "ab")\nchild = subprocess.Popen([sys.executable, "%s/colab/e2e-stage.py", *%s], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)\nopen("/kaggle/working/stage.pid", "w").write(f"{child.pid}\\n")\nprint(f"staging started, pid {child.pid}; log /kaggle/working/stage.out")\n' \
      "$checkout" "$stage_argv" | colab exec -s $session
    provisioned
    ;;
  wait)
    exec python3 "$here/colab/control.py" wait "$session" "$@"
    ;;
  status)
    heartbeat $session
    printf 'import json, time\nfrom pathlib import Path\np = Path("/kaggle/working")\nfor name in ("stage.out", "kernel.out", "train.log", "publish.out"):\n    f = p / name\n    if f.is_file():\n        print(f"--- {name} ---")\n        print(f.read_text()[-3000:])\nstate = p / "publish-state.json"\nif state.is_file():\n    s = json.loads(state.read_text())\n    if "last_published_at" in s:\n        print(f"last publish: step {s[\"last_published_step\"]}, {round(time.time() - s[\"last_published_at\"])}s ago")\n    if s.get("last_error"):\n        print(f"last publish error: {s[\"last_error\"]}")\nrun = p / "run"\nif run.is_dir():\n    print("intervals:", sorted(d.name for d in run.glob("interval-*")))\n' | colab exec -s $session
    ;;
  fetch)
    out=${1:?out dir}
    mkdir -p "$out"
    # batch-counts.json is the census the next run of the same shape is
    # started with (`start --census`); gpu.csv and host.jsonl are what
    # `mlime train measure` reads beside the metrics.
    for f in batch-counts.json run-summary.json run-config.json leg.json train.log count.log kernel.out stage.out publish.out publish-state.json gpu.csv host.jsonl run/metrics.jsonl; do
      colab download -s $session "/kaggle/working/$f" "$out/$(basename "$f")" || echo "not on the VM yet: $f"
    done
    ;;
  publish)
    printf 'import os, runpy\nos.environ["ONCE"] = "1"\nrunpy.run_path("%s/colab/e2e-publish.py", run_name="__main__")\n' "$checkout" | colab exec -s $session
    ;;
  stop)
    force=
    while [ $# -gt 0 ]; do case $1 in
      --force) force=1; shift;;
      *) echo "unknown argument $1" >&2; exit 2;;
    esac; done
    if [ -z "$force" ]; then
      python3 "$here/colab/control.py" releasable "$session"
    fi
    colab stop -s $session
    ;;
  *) echo "unknown command $command" >&2; exit 2;;
esac
