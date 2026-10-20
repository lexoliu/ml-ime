# Sourced by the Colab launchers (colab/e2e.sh, colab/char-lm.sh).
#
# A launcher's `start` is a dozen CLI calls against a runtime that bills from
# the moment it exists, and any of them can fail -- the CLI drops its
# connection now and then. `provision SESSION [colab new options]` refuses a
# name that is already running, so a failure can only ever stop the runtime
# this start created; it then arms an ERR trap that stops that runtime and
# exits with the failing command named, and creates the runtime.
# `provisioned` disarms the trap once the start has handed the runtime its
# work: what runs after that on the VM is the waiter's to watch.
#
# The waiter can end before the runtime does, and then nothing would stop it.
# So the first thing a new runtime runs is `watchdog.py`, which releases the
# runtime from inside once its heartbeat goes stale; whatever controls the
# runtime touches the heartbeat each time it looks at it (`heartbeat`).

watchdog=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/watchdog.py
colab_cli=$(type -P colab) || { echo "the Colab CLI (colab) is not on PATH" >&2; exit 2; }
deadline=$(type -P gtimeout || type -P timeout) || {
  echo "GNU timeout is not on PATH (gtimeout from coreutils on macOS)" >&2
  exit 2
}

# The Colab CLI, bounded. After a dropped connection the CLI can wait forever,
# and a call that never returns never reaches the ERR trap that releases the
# runtime -- a start once hung for hours on a billed L4 that way. Each call
# ends within its subcommand's bound, or COLAB_SECONDS for a call known to
# run longer, and a call cut off fails like any other.
colab() {
  local seconds
  case $1 in
    sessions | stop) seconds=120 ;;
    new | upload | download) seconds=900 ;;
    *) seconds=600 ;;
  esac
  "$deadline" --kill-after=15 "${COLAB_SECONDS:-$seconds}" "$colab_cli" "$@"
}

# Run the Python on stdin in SESSION's kernel and fail when it does not finish.
# `colab exec` hands the code to the runtime's IPython kernel and exits 0 even
# when the code raises, so its status says only that the code was delivered.
# Success is a token printed after the code's last statement, which a raise
# never reaches; the code's own output is passed through.
remote() {
  local session=$1 token output
  token="remote-ok-$RANDOM$RANDOM$RANDOM"
  output=$({ cat; printf '\nprint("%s")\n' "$token"; } | colab exec -s "$session") || return
  if ! printf '%s\n' "$output" | grep -qxF "$token"; then
    printf '%s\n' "$output" >&2
    echo "the code sent to $session did not run to its end" >&2
    return 1
  fi
  printf '%s\n' "$output" | grep -vxF "$token" || true
}

provision() {
  local session=$1
  shift
  if colab sessions | grep -qF "[$session] "; then
    echo "a session named $session is already running; it is left as it is" >&2
    exit 2
  fi
  trap "release_on_failure $session" ERR
  colab new -s "$session" "$@"
  colab upload -s "$session" "$watchdog" /root/watchdog.py
  # A watchdog that died on start would leave the runtime with nothing to
  # release it, so the start waits to see it still running.
  printf 'import subprocess, sys, time\nlog = open("/root/watchdog.out", "ab")\nchild = subprocess.Popen([sys.executable, "/root/watchdog.py"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)\ntime.sleep(3)\nif child.poll() is not None:\n    raise SystemExit(f"the watchdog exited {child.returncode} on start; see /root/watchdog.out")\nprint(f"watchdog running, pid {child.pid}")\n' | remote "$session"
}

# The controller's beat: run on every look at the runtime.
heartbeat() {
  printf 'from pathlib import Path\nPath("/root/heartbeat").touch()\n' | remote "$1" > /dev/null
}

release_on_failure() {
  local status=$? session=$1
  trap - ERR
  echo "start failed (exit $status) at: $BASH_COMMAND; stopping $session" >&2
  colab stop -s "$session" || echo "could not stop $session; it is still billing" >&2
  exit "$status"
}

provisioned() {
  trap - ERR
}
