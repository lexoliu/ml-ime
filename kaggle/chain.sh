#!/bin/bash
# Keep the route A v2 training chain moving without anyone watching it.
#
# Run hourly (launchd on the Mac mini; see kaggle/README.md). Each run looks at
# the highest segment that exists on Kaggle: if it is COMPLETE and the next one
# has not been pushed, the next one is pushed. Nothing else -- harvesting,
# evaluation and the notes stay with a person. A push that Kaggle refuses (the
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
