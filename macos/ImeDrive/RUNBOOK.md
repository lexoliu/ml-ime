# ime-drive runbook

How a sibling session reproduces the commercial-IME runs on a fresh macOS VM.
Everything below was done on macOS 26.5.2 (arm64) in the session that wrote
this; it is the tested path, not a sketch.

## Eval sets

`data/run3_pool/{eval3,eval3-abbreviated,eval3-mixed}.jsonl` are NOT in git
(`data/` is gitignored). They are handed to the session as attachments —
download them to `data/run3_pool/` first. SHA256SUMS (eval3 729115e6…,
abbreviated 6c5f2413…, mixed aaa14f1e…) can be checked when present.

## Build

    macos/ImeDrive/build.sh
    macos/ImeDrive/build/ImeDrive.app/Contents/MacOS/ime-drive --list   # dump input-source ids

Accessibility: the harness checks `AXIsProcessTrustedWithOptions` at start
and prints `trusted=true|false`. In Devin sessions trust is inherited — no
manual TCC grant was needed. If a run prints `trusted=false`, grant
Accessibility to the harness/terminal in System Settings → Privacy & Security.

## Install the engine

Sogou Pinyin for Mac — <https://shurufa.sogou.com/mac>
(zip ~186 MB → `SogouInstaller.app`; verified signer "Beijing Sogou
Technology Development Co.,Ltd. (DFD88F82SU)", universal arm64, seen version
6.25.1). Run the installer, click through. A first-run helper dialog may
appear; pressing Return on the default (highlighted) button works when AX
clicks do not.

Baidu Pinyin for Mac — <https://srf.baidu.com/input/mac.html>
(dmg `baiduinput_mac.dmg` ~83 MB; verified signer "Baidu (China) Co., Ltd
(738UU3Y57V)", universal arm64, seen version 6.0.3.66). Mount the dmg, run
the installer. Enabling the input source pops a GUI dialog "Allow ImeDrive
to enable Baidu输入法?" — approve it (Return on the default button). Until
it is approved `TISEnableInputSource` is a silent no-op and
`TISSelectInputSource` on the `.pinyin` mode fails with -50; the harness
falls back to the parent source automatically once the parent is enabled.

Verify with `codesign -dvv <app>` before trusting a download; the signers
above are the only ones accepted.

## Run

One engine, the three twins, sequentially (a GUI session has one keyboard
focus — do not run two engines in parallel on one VM):

    for twin in eval3 eval3-abbreviated eval3-mixed; do
      macos/ImeDrive/build/ImeDrive.app/Contents/MacOS/ime-drive \
        --engine <ENGINE> --eval-set data/run3_pool/$twin.jsonl \
        --out data/gui/<ENGINE>/$twin.jsonl
    done

The `--out` file is the journal: kill and restart at will — records already
in the file are skipped, the rest append. A record over the 20 s budget is
a failure line (`"failed": true`), never retried. Roughly 1–2.3 s per
record depending on engine → ~1.5–3 h per twin; the full sweep is many
hours and is expected to run unattended.

## Score

    uv run --project python mlime eval gui --engine <ENGINE> \
      --results data/gui/<ENGINE>/$twin.jsonl \
      --eval-set data/run3_pool/$twin.jsonl \
      --out data/gui/reports/<ENGINE>-$twin.json

## Per-engine notes

- apple — `com.apple.inputmethod.SCIM.ITABC`, built in, no install. The
  candidate window is a TextInputUI panel owned by the client process; its
  AX tree exposes every candidate (all pages). Works everywhere.
- sogou — `com.sogou.inputmethod.sogou.pinyin`. Engine-process candidate
  window with a clean 5-candidate first page in AX.
- baidu — `com.baidu.inputmethod.BaiduIM.pinyin`. Candidate panel exposes
  NO AX text (custom-drawn): `first_page` is always `[]` and
  `candidate_window` is detected by window-server geometry instead. This is
  recorded honestly, not skipped.
