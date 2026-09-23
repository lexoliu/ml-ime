#!/usr/bin/env bash
# Resume the stalled ml-ime session inside tmux so it survives SSH disconnects.
# Idempotent: creates the tmux session if absent, then attaches to it.
set -uo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:$PATH"

SESSION=mlime
REPO="$HOME/Coding/ml-ime"
SESSION_ID=bd573d13-8f60-40b2-b8cd-d9bc45316219
BRIEF="$REPO/.resume-brief.md"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  [ -r "$BRIEF" ] || { echo "missing $BRIEF" >&2; exit 1; }
  # caffeinate keeps the mini awake for as long as Claude runs
  tmux new-session -d -s "$SESSION" -c "$REPO" \
    caffeinate -is claude \
      --resume "$SESSION_ID" \
      --model claude-fable-5-1 \
      --remote-control \
      "$(cat "$BRIEF")"
  echo "started tmux session '$SESSION'"
else
  echo "tmux session '$SESSION' already running; attaching"
fi

exec tmux attach -t "$SESSION"
