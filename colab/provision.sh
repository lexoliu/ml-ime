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

provision() {
  local session=$1
  shift
  if colab sessions | grep -qF "[$session] "; then
    echo "a session named $session is already running; it is left as it is" >&2
    exit 2
  fi
  trap "release_on_failure $session" ERR
  colab new -s "$session" "$@"
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
