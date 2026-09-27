#!/usr/bin/env bash
# Build ime-drive and assemble it as an .app bundle.
#
# The tool types into its own text view through CGEvent and reads the
# candidate window through the Accessibility API. Both need the process to
# be a TCC client with a stable identity, which on macOS means a bundle:
# running the binary inside build/ImeDrive.app registers it in System
# Settings → Privacy & Security → Accessibility as "ImeDrive". Ad-hoc
# signing is required on Apple Silicon or the bundle will not load.
set -euo pipefail

cd "$(dirname "$0")"

APP_NAME="ImeDrive"
BINARY="ime-drive"
BUNDLE="build/${APP_NAME}.app"

swift build -c release

rm -rf "${BUNDLE}"
mkdir -p "${BUNDLE}/Contents/MacOS" "${BUNDLE}/Contents/Resources"
cp ".build/release/${APP_NAME}" "${BUNDLE}/Contents/MacOS/${BINARY}"
cp "Resources/Info.plist" "${BUNDLE}/Contents/Info.plist"
codesign --force --sign - --timestamp=none "${BUNDLE}"

# Convenience copy for PATH-less invocation; the bundled copy is the one
# TCC knows about, so prefer it for real runs.
cp ".build/release/${APP_NAME}" "build/${BINARY}"

echo "built ${BUNDLE} — run ${BUNDLE}/Contents/MacOS/${BINARY} --help"
