# ime-drive

Types an eval set's keystrokes into a text view this process owns, reads the
selected input method's candidate window through the accessibility tree, and
accepts the default candidate until the composition is consumed — what a user
gets by pressing space. One JSON line per record; the output file is the
journal, so a run is resumable by repeating the command.

    ime-drive --engine <apple|sogou|baidu> --eval-set <jsonl> --out <jsonl> [--slice dev|test|all]

## Eval sets

`data/run3_pool/{eval3,eval3-abbreviated,eval3-mixed}.jsonl` are NOT in git
(`data/` is gitignored). They arrive as session attachments; place them at
`data/run3_pool/` and check SHA256SUMS when present.

## Build

    macos/ImeDrive/build.sh
    macos/ImeDrive/build/ImeDrive.app/Contents/MacOS/ime-drive --list   # dump input-source ids

The driver prints `accessibility trusted=true|false` at start. In Devin
sessions trust is inherited from the responsible process — no manual TCC
grant was needed on macOS 26.5.2. If it prints `false`, grant Accessibility
to the harness in System Settings → Privacy & Security.

## Install Sogou

1. Download from the official page <https://shurufa.sogou.com/mac> — the
   `下载` button serves `sogou_mac.zip` (~186 MB, from `ime.gtimg.com`,
   Sogou's CDN). Seen version 6.25.1.
2. Verify the vendor before running anything:

       unzip -o sogou_mac.zip -d sogou_pkg
       codesign -dvv sogou_pkg/sogou_mac_*/SogouInstaller.app 2>&1 | grep Authority
       # expect: Authority=Developer ID Application: Beijing Sogou
       # Technology Development Co.,Ltd. (DFD88F82SU)

3. Run the installer app and click through. Afterwards a first-run helper
   dialog may appear; AX clicks on system dialogs often don't register —
   pressing Return on the default (highlighted) button does.
4. The input source appears as `com.sogou.inputmethod.sogou.pinyin`. The
   harness enables and selects it itself; no manual enable is needed.

## Install Baidu

1. Download from the official page <https://srf.baidu.com/input/mac.html> —
   `baiduinput_mac.dmg` (~83 MB, from `imeres.baidu.com`). Seen version
   6.0.3.66.
2. Verify the vendor:

       hdiutil attach baiduinput_mac.dmg
       codesign -dvv "/Volumes/<vol>/BaiduInstaller.app" 2>&1 | grep Authority
       # expect: Authority=Developer ID Application: Baidu (China) Co., Ltd
       # (738UU3Y57V)

3. Run the installer.
4. Enabling the source is the one GUI step: the first `ime-drive` run calls
   `TISEnableInputSource("com.baidu.inputmethod.BaiduIM")`, which pops
   "Allow ImeDrive to enable Baidu输入法?" — approve it (Return on the
   default button). Until approved, enabling is a silent no-op and
   `TISSelectInputSource` on the `.pinyin` mode fails with -50; the harness
   falls back to the parent source automatically once the parent is
   enabled. Expected source id afterwards:
   `com.baidu.inputmethod.BaiduIM.pinyin`.

Both installers are universal arm64. Only those two Developer IDs are
trusted vendors.

## Run

One engine, the three twins, sequentially — a GUI session has one keyboard
focus, so never run two engines on one VM at once:

    for twin in eval3 eval3-abbreviated eval3-mixed; do
      macos/ImeDrive/build/ImeDrive.app/Contents/MacOS/ime-drive \
        --engine <ENGINE> --eval-set data/run3_pool/$twin.jsonl \
        --out data/gui/<ENGINE>/$twin.jsonl --slice test
    done

`--slice test` types only the test share (dev share 0.0905, keyed on the
record hash — identical to `mlime eval rime`). The `--out` file is the
journal: kill and restart at will, records already in it are skipped. A
record over the 20 s budget is written as a failure line, never retried.
~1–2.3 s/record → each twin is well over an hour; the sweep runs
unattended.

## Score

    uv run --project python mlime eval gui --engine <ENGINE> \
      --results data/gui/<ENGINE>/$twin.jsonl \
      --eval-set data/run3_pool/$twin.jsonl \
      --out data/gui/reports/<ENGINE>-$twin.json

`eval gui` refuses a results file whose meta `slice` doesn't cover what it
scores (`all` covers any slice; files written before `--slice` existed count
as `all`).

## Per-engine notes

- apple — `com.apple.inputmethod.SCIM.ITABC`, built in. Candidate window is
  a TextInputUI panel owned by the client process; AX exposes every
  candidate (all pages).
- sogou — engine-process candidate window; clean 5-candidate first page in
  AX.
- baidu — candidate panel exposes NO AX text (custom-drawn): `first_page`
  is always `[]`; `candidate_window` is detected by window-server geometry
  instead. Recorded honestly, not skipped.
