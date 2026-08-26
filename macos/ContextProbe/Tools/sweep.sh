#!/usr/bin/env bash
# Visit each application in turn and let the probe record what it reports.
#
# Deliberately keystroke-free: posting synthetic key events needs Accessibility
# permission, which cannot be granted programmatically, while activating an
# application needs only Automation permission. IMKInputController.activateServer
# fires on each visit, which is the measurement.
set -euo pipefail
cd "$(dirname "$0")"

PROBE_SOURCE="cool.lexo.inputmethod.ContextProbe.probe"
FIXTURE="$(pwd)/fixture.html"
SEED="The probe needs a focused text field holding enough text that a request for the sixty-four characters before the caret can be satisfied."

original=$(swift input-source.swift current)
echo "current input source: ${original}"

restore() {
    swift input-source.swift select "${original}" >/dev/null || true
    echo "restored ${original}"
}
trap restore EXIT

# Give each application a focused text field without typing into anything the
# user owns: a scratch document, a fixture page, a temporary file.
osascript -e 'tell application "TextEdit" to make new document with properties {text:"'"${SEED}"'"}' >/dev/null
scratch=$(mktemp -t mlime-probe).txt
printf '%s\n' "${SEED}" > "${scratch}"

visit() {
    local app="$1"
    shift
    if ! open -Ra "${app}" 2>/dev/null; then
        echo "skip ${app}: not installed"
        return
    fi
    "$@" >/dev/null 2>&1 || true
    osascript -e "tell application \"${app}\" to activate" >/dev/null 2>&1 || true
    sleep 1
    swift input-source.swift select "${PROBE_SOURCE}" >/dev/null
    sleep 1
    echo "visited ${app}"
}

visit "TextEdit" true
visit "Safari" open -a Safari "${FIXTURE}"
visit "Google Chrome" open -a "Google Chrome" "${FIXTURE}"
visit "Visual Studio Code" open -a "Visual Studio Code" "${scratch}"
visit "Terminal" open -a Terminal
visit "Slack" true
visit "WeChat" true
