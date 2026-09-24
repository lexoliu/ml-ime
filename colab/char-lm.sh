#!/bin/bash
# Train the character language model on a Colab GPU through the Colab CLI.
#
#   colab/char-lm.sh start [--gpu A100]   provision, stage the datasets, launch
#   colab/char-lm.sh status               tail of the kernel's log
#   colab/char-lm.sh fetch <out dir>      download the export and the summary
#   colab/char-lm.sh stop                 release the VM
#
# The session is named char-lm; `colab sessions` lists it. The kernel is the
# same script Kaggle runs (kaggle/char-lm/kernel.py); it sizes itself to the
# GPUs it finds, so one A100 trains the same batches as two T4s.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
session=char-lm
command=${1:?start|status|fetch|stop}; shift
case $command in
  start)
    gpu=A100
    while [ $# -gt 0 ]; do case $1 in --gpu) gpu=$2; shift 2;; *) echo "unknown argument $1" >&2; exit 2;; esac; done
    colab new -s $session --gpu "$gpu"
    printf 'import os\nfor d in ("/root/.kaggle", "/kaggle"): os.makedirs(d, exist_ok=True)\n' | colab exec -s $session
    colab upload -s $session "$HOME/.kaggle/credentials.json" /root/.kaggle/credentials.json
    colab upload -s $session "$here/kaggle/char-lm/kernel.py" /kaggle/kernel.py
    colab exec -s $session -f "$here/colab/stage.py"
    ;;
  status)
    printf 'from pathlib import Path\np = Path("/kaggle/working")\nfor name in ("kernel.out", "train.log"):\n    f = p / name\n    if f.is_file():\n        print(f"--- {name} ---")\n        print(f.read_text()[-3000:])\nprint("summary:", (p / "run-summary.json").is_file())\n' | colab exec -s $session
    ;;
  fetch)
    out=${1:?out dir}
    mkdir -p "$out/char-lm" "$out/run"
    for f in char-lm/charlm.onnx char-lm/charlm.json run-summary.json train.log export.log run/metrics.jsonl run/charlm-final.pt; do
      colab download -s $session "/kaggle/working/$f" "$out/$f"
    done
    ;;
  stop)
    colab stop -s $session
    ;;
  *) echo "unknown command $command" >&2; exit 2;;
esac
