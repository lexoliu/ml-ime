#!/usr/bin/env bash
# Build the context probe and install it as an input method.
#
# InputMethodKit loads input methods as .app bundles, which SwiftPM does not
# produce, so the bundle is assembled here. Ad-hoc signing is required on Apple
# Silicon or the bundle will not load.
set -euo pipefail

cd "$(dirname "$0")"

APP_NAME="ContextProbe"
BUNDLE="build/${APP_NAME}.app"
INSTALL_DIR="${HOME}/Library/Input Methods"

swift build -c release

rm -rf "${BUNDLE}"
mkdir -p "${BUNDLE}/Contents/MacOS" "${BUNDLE}/Contents/Resources"
cp ".build/release/${APP_NAME}" "${BUNDLE}/Contents/MacOS/${APP_NAME}"
cp "Resources/Info.plist" "${BUNDLE}/Contents/Info.plist"
cp "Resources/probe.pdf" "${BUNDLE}/Contents/Resources/probe.pdf"
codesign --force --sign - --timestamp=none "${BUNDLE}"

if [[ "${1-}" == "--install" ]]; then
    # Replacing a bundle while its process is alive leaves a stale server bound
    # to the connection name, so stop it first.
    killall "${APP_NAME}" 2>/dev/null || true
    mkdir -p "${INSTALL_DIR}"
    rm -rf "${INSTALL_DIR}/${APP_NAME}.app"
    cp -R "${BUNDLE}" "${INSTALL_DIR}/${APP_NAME}.app"
    # Launching once registers the input source with Text Input Sources.
    open "${INSTALL_DIR}/${APP_NAME}.app"
    echo "installed to ${INSTALL_DIR}/${APP_NAME}.app"
else
    echo "built ${BUNDLE} (pass --install to register it as an input method)"
fi
