#!/bin/bash
# Keep a segmented Kaggle training chain moving without anyone watching it.
#
#   kaggle/chain.sh                                     # route A v2, as ever
#   kaggle/chain.sh --slug lexoliu/mlime-e2e-s --kernel-dir kaggle/e2e \
#                   --data-dir data/e2e --finish none   # the e2e chain
#
# Run hourly (launchd on the Mac mini; see kaggle/README.md). Each run looks at
# the highest segment that exists on Kaggle: if it is COMPLETE its output is
# downloaded to <data dir>/s<n> (a download that breaks off is redone next
# hour; only a complete one leaves the `harvested` stamp), the segment that
# finished the run gets its finish step, and otherwise the next segment is
# pushed. The finish step is a script run on the segment directory -- route A
# v2's is finish.sh, which fuses with the trigram; e2e's is none, because that
# kernel evaluates in-kernel, so the chain only gathers its reports into
# results.md.
#
# A push that Kaggle refuses (the weekly quota, an expired token) is simply
# retried an hour later, and a push whose previous segment turns out to have
# *finished* fails inside the kernel within minutes at no cost, so the script
# does not need to know.
#
# Every decision is logged, one line per run, to <data dir>/chain.log.
set -u
REPO=${MLIME_REPO:-/Users/lexoliu/Coding/ml-ime}
SLUG=lexoliu/mlime-route-a-v2-s
KERNEL_DIR=$REPO/kaggle/route-a-v2
DATA_DIR=$REPO/data/route-a-v2
FINISH=$REPO/kaggle/finish.sh
MAX_SEGMENT=12
export PATH=/Users/lexoliu/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin

usage() {
  echo "usage: $0 [--slug PREFIX] [--kernel-dir DIR] [--data-dir DIR] [--finish SCRIPT|none]" >&2
  exit 2
}
while [ $# -gt 0 ]; do
  case "$1" in
    --slug|--kernel-dir|--data-dir|--finish)
      [ $# -ge 2 ] || usage
      case "$1" in
        --slug) SLUG=$2 ;;
        --kernel-dir) KERNEL_DIR=$2 ;;
        --data-dir) DATA_DIR=$2 ;;
        --finish) FINISH=$2 ;;
      esac
      shift 2 ;;
    *) usage ;;
  esac
done
# Relative paths resolve against the repository, not the caller's directory.
for var in KERNEL_DIR DATA_DIR FINISH; do
  eval "val=\$$var"
  if [ "$val" != "none" ] && [ "${val#/}" = "$val" ]; then eval "$var=$REPO/$val"; fi
done
[ -f "$KERNEL_DIR/kernel.py" ] || { echo "no kernel.py under $KERNEL_DIR" >&2; exit 2; }
LOG=${MLIME_CHAIN_LOG:-$DATA_DIR/chain.log}

mkdir -p "$(dirname "$LOG")"
say() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"; }

status() {
  local out
  out=$(kaggle kernels status "$SLUG$1" 2>&1)
  local s
  s=$(printf '%s' "$out" | grep -o 'KernelWorkerStatus\.[A-Z]*')
  if [ -n "$s" ]; then printf '%s' "$s"
  elif printf '%s' "$out" | grep -q "Cannot access kernel\|404 Client Error"; then printf 'UNPUSHED'
  else printf 'API_ERROR'
  fi
}

# The highest pushed segment, walking up until one is unpushed.
last=-1
for n in $(seq 0 $MAX_SEGMENT); do
  s=$(status "$n")
  case "$s" in
    UNPUSHED) break ;;
    API_ERROR) say "api error reading segment $n; retry next hour"; exit 0 ;;
    *) last=$n; last_status=$s ;;
  esac
done
if [ "$last" -lt 0 ]; then say "no segment pushed yet; nothing to chain from"; exit 0; fi
# GPU hours left this week; a segment needs a whole session, so a push into
# less than that is a session Kaggle will kill before it pauses.
SESSION_HOURS=${MLIME_SESSION_HOURS:-12}
quota_left() { kaggle quota 2>&1 | awk '/^GPU/{print $3}' | tr -d h; }

case "$last_status" in
  *RUNNING*|*QUEUED*) say "segment $last is $last_status; waiting"; exit 0 ;;
  *ERROR*|*CANCEL*)
    # A segment that died leaves no checkpoint, segment 0 included; re-push it
    # when a whole session of quota is available again, at most three times,
    # else a person looks.
    # grep -c exits 1 on zero matches and prints 0 anyway, so the default
    # applies to the missing file, not to the count itself.
    attempts=$(grep -c "re-pushed segment $last" "$LOG" 2>/dev/null || true)
    attempts=${attempts:-0}
    if [ "$attempts" -ge 3 ]; then say "segment $last is $last_status after 3 re-pushes; a person has to look"; exit 0; fi
    left=$(quota_left)
    if [ -z "$left" ] || ! python3 -c "import sys; sys.exit(0 if float('$left') >= $SESSION_HOURS else 1)"; then say "segment $last is $last_status; ${left:-?}h of quota left, waiting for $SESSION_HOURS"; exit 0; fi
    last=$((last - 1)); repush=1 ;;
esac

finished() { python3 -c "import json,sys;sys.exit(0 if json.load(open('$1/run-summary.json')).get('finished') else 1)"; }

# Harvest: every COMPLETE segment's output lands in <data dir>/s<n> once, and
# the one that finished the run gets the finish step.
harvest() {
  local n=$1 dir=$DATA_DIR/s$1
  if [ -e "$dir/harvested" ]; then return 0; fi
  say "downloading segment $n output"
  mkdir -p "$dir"
  # The CLI keeps going after one file breaks off, so a small file such as
  # run-summary.json can land while a checkpoint did not: the stamp, not any
  # downloaded file, is what says the segment is here in full.
  if ! kaggle kernels output "$SLUG$n" -p "$dir" >> "$dir/download.log" 2>&1 || [ ! -e "$dir/run-summary.json" ]; then
    say "download of segment $n failed; retry next hour"; return 1
  fi
  touch "$dir/harvested"
  say "segment $n harvested: $(python3 -c "import json;d=json.load(open('$dir/run-summary.json'));print('steps',d.get('first_step'),'->',d.get('last_step'),'loss',round(d.get('last_loss') or 0,3),'finished',d.get('finished'))")"
  if finished "$dir"; then
    if [ "$FINISH" != "none" ]; then
      say "segment $n finished the run; evaluating (see $dir/finish.log)"
      if "$FINISH" "$dir" > "$dir/finish.log" 2>&1; then say "results in $dir/results.md"; else say "finish step failed; see $dir/finish.log"; fi
    else
      # The kernel evaluated in-line; gather its report files into results.md.
      say "segment $n finished the run; the kernel evaluated in-line, gathering reports"
      for report in "$dir"/e2e-*-report.txt; do
        [ -e "$report" ] || continue
        { printf '# %s\n\n' "$(basename "$report" .txt)"; cat "$report"; echo; }
      done > "$dir/results.md"
      say "results in $dir/results.md"
    fi
  fi
}
if [ "${repush:-0}" -eq 0 ]; then harvest "$last" || exit 0; fi
if [ "${repush:-0}" -eq 0 ] && [ -e "$DATA_DIR/s$last/harvested" ] && finished "$DATA_DIR/s$last"; then
  say "the run is finished at segment $last; nothing more to push"; exit 0
fi

next=$((last + 1))
if [ "$next" -gt "$MAX_SEGMENT" ]; then say "segment $last complete and no room for segment $next"; exit 0; fi
left=$(quota_left)
if [ -z "$left" ] || ! python3 -c "import sys; sys.exit(0 if float('$left') >= $SESSION_HOURS else 1)"; then
  say "segment $last complete; ${left:-?}h of quota left, waiting for $SESSION_HOURS before pushing segment $next"; exit 0
fi
dir=$(mktemp -d)
sed "s/^SEGMENT = 0$/SEGMENT = $next/" "$KERNEL_DIR/kernel.py" > "$dir/kernel.py"
python3 - "$next" "$dir/kernel-metadata.json" "$KERNEL_DIR/kernel-metadata.json" "$SLUG" <<'PY'
import json, sys
n, out, template, slug = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
m = json.load(open(template))
m["id"] = f"{slug}{n}"
m["title"] = m["id"].split("/")[1]
m["kernel_sources"] = [f"{slug}{n - 1}"] if n else []
json.dump(m, open(out, "w"), indent=2)
PY
out=$(kaggle kernels push -p "$dir" 2>&1)
if printf '%s' "$out" | grep -q "successfully pushed"; then
  if [ "${repush:-0}" -eq 1 ]; then say "re-pushed segment $next after it died"; else say "segment $last complete; pushed segment $next"; fi
else
  say "segment $last complete; push of segment $next refused: $(printf '%s' "$out" | tail -1)"
fi
