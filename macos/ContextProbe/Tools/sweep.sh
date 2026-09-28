#!/usr/bin/env bash
# Visit each application in turn and let the probe record what it reports.
#
# What this needs on a fresh machine, granted once:
#   - The probe installed and registered: ../build.sh --install (the install
#     launches it once, which calls TISRegisterInputSource).
#   - The input method enabled. Registration alone leaves the mode unselectable:
#     TISSelectInputSource returns -50 until the *bundle* source is in
#     com.apple.inputsources AppleEnabledThirdPartyInputSources. The System
#     Settings add (Keyboard -> Input Sources -> Edit -> +) writes it; the
#     preflight below writes the same entries directly, with no UI and no
#     logout.
#   - Automation consent for whichever process runs this script, for each app
#     seeded through AppleScript (Notes, Mail, Terminal) and for System Events
#     if the opportunistic arrow key is wanted. Accessibility is needed only
#     for that arrow key (`System Events` -> `key code`), which takes a second
#     measurement of each focused client; without it the sweep still visits
#     every app. A missing grant surfaces as a consent dialog that blocks until
#     the AppleEvent timeout; every call here is wrapped in `with timeout` so a
#     missing grant skips the app instead of stalling the sweep.
#
# Sampling is a property of the probe process, not of this script: run the
# probe under MLIME_PROBE_SAMPLE=1 (killall ContextProbe, then relaunch its app
# with that variable set) before the sweep if the recorded contexts should
# include their first characters.
set -uo pipefail
cd "$(dirname "$0")"

PROBE_MODE="cool.lexo.inputmethod.ContextProbe.probe"
PROBE_BUNDLE="cool.lexo.inputmethod.ContextProbe"
LOG="${HOME}/Library/Logs/mlime-context-probe.jsonl"
SEED="The probe needs a focused text field holding enough text that a request for the sixty-four characters before the caret can be satisfied."
FIXTURE="$(pwd)/fixture.html"
WORK="$(mktemp -d -t mlime-probe)"
SCRATCH="${WORK}/scratch.txt"
printf '%s\n' "${SEED}" > "${SCRATCH}"

# Compile the input-source helper once; `swift file.swift` re-parses every call.
swiftc -O -o "${WORK}/input-source" input-source.swift || { echo "cannot compile input-source.swift"; exit 1; }
IS="${WORK}/input-source"

# --- preflight: is the probe selectable? -------------------------------------
current=$("${IS}" current 2>/dev/null || true)
echo "current input source: ${current:-unknown}"

enabled=$(defaults read com.apple.inputsources AppleEnabledThirdPartyInputSources 2>/dev/null || true)
if ! printf '%s' "${enabled}" | grep -q "${PROBE_BUNDLE}"; then
    echo "input method not yet enabled; writing AppleEnabledThirdPartyInputSources"
    defaults write com.apple.inputsources AppleEnabledThirdPartyInputSources -array-add \
        "{\"Bundle ID\"=\"${PROBE_BUNDLE}\"; \"Input Mode\"=\"${PROBE_MODE}\"; InputSourceKind=\"Input Mode\";}" \
        "{\"Bundle ID\"=\"${PROBE_BUNDLE}\"; InputSourceKind=\"Keyboard Input Method\";}"
fi

if ! "${IS}" select "${PROBE_MODE}" >/dev/null 2>&1; then
    cat >&2 <<EOF
probe is registered but still not selectable (TISSelectInputSource failed).
Fix it by either re-running this script (the preflight write above just ran)
or adding the input source once via
System Settings -> Keyboard -> Input Sources -> Edit -> +.
Neither TISEnableInputSource nor a logout substitutes.
EOF
    exit 1
fi

restore() {
    [ -n "${current}" ] && "${IS}" select "${current}" >/dev/null 2>&1 || true
    rm -rf "${WORK}"
    echo "restored ${current:-nothing}"
}
trap restore EXIT

lines_so_far() { [ -f "${LOG}" ] && wc -l < "${LOG}" | tr -d ' ' || echo 0; }

# Wait for the probe to log a record for the app we just activated.
wait_for_client() {  # wait_for_client <bundle-id-regex> <lines-before> [seconds]
    local want="$1" before="$2" deadline=$(( $(date +%s) + ${3:-10} ))
    while [ "$(date +%s)" -lt "${deadline}" ]; do
        if tail -n +"$((before + 1))" "${LOG}" 2>/dev/null | grep -qE "${want}"; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

# AppleScript under a bounded timeout; nonfatal.
osascript_t() {  # osascript_t <timeout-seconds> <script>
    local t="$1" script="$2"
    osascript - <<EOF
with timeout of ${t} seconds
${script}
end timeout
EOF
}

visit() {  # visit <app-name> <bundle-id-regex> [seed-command...]
    local app="$1" want="$2"; shift 2
    if ! osascript -e "get id of application \"${app}\"" >/dev/null 2>&1; then
        echo "skip ${app}: not installed"
        return
    fi
    local before; before=$(lines_so_far)
    if [ "$#" -gt 0 ]; then
        "$@" >/dev/null 2>&1 || echo "note: seeding ${app} failed (grant Automation?)"
    fi
    osascript_t 10 "tell application \"${app}\" to activate" >/dev/null 2>&1 \
        || echo "note: activating ${app} failed (grant Automation?)"
    sleep 1
    "${IS}" select "${PROBE_MODE}" >/dev/null 2>&1 || true
    # A second measurement of the same client, only if key events are allowed.
    osascript_t 5 'tell application "System Events" to key code 124' >/dev/null 2>&1 || true
    if wait_for_client "${want}" "${before}" 10; then
        echo "visited ${app}"
    else
        echo "visited ${app}: no record matching ${want} (field not focused?)"
    fi
}

open_in() { open -a "$1" "$2"; }  # seed a focused document without AppleScript

# --- the sweep --------------------------------------------------------------
visit "TextEdit" "com\.apple\.TextEdit" open_in "TextEdit" "${SCRATCH}"
visit "Notes" "com\.apple\.Notes" osascript_t 15 \
    "tell application \"Notes\" to make new note at folder \"Notes\" of default account with properties {body:\"${SEED}\"}"
visit "Mail" "com\.apple\.mail" osascript_t 15 \
    "tell application \"Mail\" to make new outgoing message with properties {subject:\"probe\", content:\"${SEED}\", visible:true}"
visit "Messages" "com\.apple\.(MobileSMS|AuthKitUI)"
visit "Safari" "com\.apple\.Safari" open_in "Safari" "${FIXTURE}"
visit "Google Chrome" "com\.google\.Chrome" open_in "Google Chrome" "${FIXTURE}"
visit "Terminal" "com\.apple\.Terminal" osascript_t 15 \
    "tell application \"Terminal\" to do script \"printf '%s' '${SEED}'\""
visit "Xcode" "com\.apple\.dt\.Xcode" open_in "Xcode" "${SCRATCH}"
visit "Visual Studio Code" "com\.microsoft\.VSCode" open_in "Visual Studio Code" "${SCRATCH}"
visit "Pages" "com\.apple\.iWork\.Pages" osascript_t 15 \
    "tell application \"Pages\" to make new document with properties {body:\"${SEED}\"}"
visit "Obsidian" "md\.obsidian"
visit "Discord" "com\.hnc\.Discord"
visit "WeChat" "com\.tencent\.xinWeChat"
visit "Slack" "com\.tinyspeck\.slackmacgap"

# Discard the Mail draft the seed created.
osascript_t 10 'tell application "Mail" to delete (every outgoing message)' >/dev/null 2>&1 || true

echo "done; records are in ${LOG}"
