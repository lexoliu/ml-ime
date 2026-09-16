#!/bin/bash
# Keep the route A v2 training chain moving without anyone watching it.
#
# Run hourly (launchd on the Mac mini; see kaggle/README.md). Each run looks at
# the highest segment that exists on Kaggle: if it is COMPLETE its output is
# downloaded once to data/route-a-v2/s<n>, the segment that finished the run is
# evaluated by kaggle/finish.sh, and otherwise the next segment is pushed. The
# notes stay with a person. A push that Kaggle refuses (the
# weekly quota, an expired token) is simply retried an hour later, and a push
# whose previous segment turns out to have *finished* fails inside the kernel
# within minutes at no cost, so the script does not need to know.
#
# Every decision is logged, one line per run, to $LOG.
set -u
REPO=${MLIME_REPO:-/Users/lexoliu/Coding/ml-ime}
LOG=${MLIME_CHAIN_LOG:-$REPO/data/route-a-v2/chain.log}
SLUG=lexoliu/mlime-route-a-v2-s
MAX_SEGMENT=12
export PATH=/Users/lexoliu/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin

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
case "$last_status" in
  *RUNNING*|*QUEUED*) say "segment $last is $last_status; waiting"; exit 0 ;;
  *ERROR*|*CANCEL*) say "segment $last is $last_status; a person has to look"; exit 0 ;;
esac

# Harvest: every COMPLETE segment's output lands in data/route-a-v2/s<n> once,
# and the one that finished the run is evaluated by kaggle/finish.sh.
harvest() {
  local n=$1 dir=$REPO/data/route-a-v2/s$1
  if [ -e "$dir/run-summary.json" ]; then return 0; fi
  say "downloading segment $n output"
  mkdir -p "$dir"
  if ! kaggle kernels output "$SLUG$n" -p "$dir" >> "$dir/download.log" 2>&1 || [ ! -e "$dir/run-summary.json" ]; then
    say "download of segment $n failed; retry next hour"; return 1
  fi
  say "segment $n harvested: $(python3 -c "import json;d=json.load(open('$dir/run-summary.json'));print('steps',d['first_step'],'->',d['last_step'],'loss',round(d['last_loss'],3),'finished',d['finished'])")"
  if python3 -c "import json,sys;sys.exit(0 if json.load(open('$dir/run-summary.json'))['finished'] else 1)"; then
    say "segment $n finished the run; evaluating (see $dir/finish.log)"
    if "$REPO/kaggle/finish.sh" "$dir" > "$dir/finish.log" 2>&1; then say "results in $dir/results.md"; else say "finish.sh failed; see $dir/finish.log"; fi
  fi
}
harvest "$last" || exit 0
if python3 -c "import json,sys;sys.exit(0 if json.load(open('$REPO/data/route-a-v2/s$last/run-summary.json'))['finished'] else 1)"; then
  say "the run is finished at segment $last; nothing more to push"; exit 0
fi

next=$((last + 1))
if [ "$next" -gt "$MAX_SEGMENT" ]; then say "segment $last complete and no room for segment $next"; exit 0; fi
dir=$(mktemp -d)
sed "s/^SEGMENT = 0$/SEGMENT = $next/" "$REPO/kaggle/route-a-v2/kernel.py" > "$dir/kernel.py"
python3 - "$next" "$dir/kernel-metadata.json" "$REPO/kaggle/route-a-v2/kernel-metadata.json" <<'PY'
import json, sys
n, out, template = int(sys.argv[1]), sys.argv[2], sys.argv[3]
m = json.load(open(template))
m["id"] = f"lexoliu/mlime-route-a-v2-s{n}"
m["title"] = m["id"].split("/")[1]
m["kernel_sources"] = [f"lexoliu/mlime-route-a-v2-s{n - 1}"]
json.dump(m, open(out, "w"), indent=2)
PY
out=$(kaggle kernels push -p "$dir" 2>&1)
if printf '%s' "$out" | grep -q "successfully pushed"; then
  say "segment $last complete; pushed segment $next"
else
  say "segment $last complete; push of segment $next refused: $(printf '%s' "$out" | tail -1)"
fi
