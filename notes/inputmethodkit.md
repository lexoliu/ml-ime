# InputMethodKit findings

Split into what has been measured on these machines (a Mac mini on macOS
26.6.2 and an arm64 VM on macOS 26.5.2, Swift 6.3.3, SDK 26.5) and what is
still hearsay. Anything under "unverified" must not be allowed to constrain a
design decision.

## Verified

- `import InputMethodKit` works from a plain SwiftPM executable target under Swift
  6 language mode with strict concurrency. No Xcode project is needed.
- SwiftPM does not produce a bundle, so `build.sh` assembles `.app/Contents/{MacOS,
  Resources,Info.plist}` by hand and ad-hoc signs it. `codesign --verify --deep
  --strict` passes and the bundle "satisfies its Designated Requirement".
- A second instance cannot bind the same `InputMethodConnectionName`; it dies with
  `[IMKServer _createConnection]: *Failed* to register NSConnection name=...`.
  `build.sh --install` therefore kills the running copy before replacing it.
- `TISCreateInputSourceList(nil, true)` does include *disabled* input sources:
  installed=318 against enabled=8 on this machine. So a missing entry means the
  system has not indexed the input method, not that it is merely switched off.
- **Registered is not selectable.** After `TISRegisterInputSource`, the probe's
  *mode* source reads `enabled=1 selectable=1`, but `TISSelectInputSource` on it
  returns **-50** while the parent *bundle* source stays `enabled=0`.
  `TISEnableInputSource` on either source returns 0 and changes nothing;
  `TISDisableInputSource` afterwards is likewise a no-op. The step that first
  makes select return 0 is putting the bundle source into
  `com.apple.inputsources` `AppleEnabledThirdPartyInputSources`:

      defaults write com.apple.inputsources AppleEnabledThirdPartyInputSources \
        -array-add '{"Bundle ID"="cool.lexo.inputmethod.ContextProbe"; InputSourceKind="Keyboard Input Method";}'

  The parent entry alone suffices (select returns 0 immediately; verified by
  removing the entry, watching -50 come back, and re-adding it). Adding the
  input source in System Settings -> Keyboard -> Input Sources -> Edit -> +
  writes the same parent entry plus a second `"Input Mode"` entry for the mode
  source, and also suffices -- the two routes are equivalent.

## Falsified

- **`InputMethodConnectionName` does not have to be
  `$(PRODUCT_BUNDLE_IDENTIFIER)_Connection`.** Squirrel, a shipping input method,
  uses `Squirrel_Connection` -- the executable name plus `_Connection`, with the
  bundle identifier `im.rime.inputmethod.Squirrel` nowhere in it. The probe now
  follows Squirrel.
- **Ad-hoc signing is enough.** No Developer ID is needed for Text Input Sources
  to accept an input method.
- **No logout is needed to install or select one.** The `defaults write` above
  takes effect immediately; the "log out and back in" suggestion in
  qingjian-team/qingjian#209 (the public report of the same -50 symptom) is not
  required.
- **Electron does not return nothing.** Chrome 153, VS Code 1.139, Discord
  0.0.413 and Obsidian 1.13.7 all answer `selectedRange()` and serve the full
  64-character window (quirks below). The second-hand claim that
  `attributedSubstring`/`selectedRange` only work in AppKit views is wrong on
  current versions.

## Host coverage

Measured on macOS 26.5.2 (arm64 VM), Swift 6.3.3, by `Tools/sweep.sh`. Each host
got a focused field holding a 135-character seed; "64 before caret" is whether
`attributedSubstring(from:)` returned 64 characters for the request. Content is
compared under `MLIME_PROBE_SAMPLE` only for the fixture page and the scratch
files; everywhere else only lengths were recorded.

| Host | Version | `selectedRange()` answers | 64 before caret | Content right (sampled) |
|---|---|---|---|---|
| TextEdit | 1.20 | yes | yes | yes |
| Notes | 4.13 | yes | yes | not sampled |
| Mail | 16.0 | yes (compose body) | yes | not sampled |
| Messages | 26.0 | main window declines; sign-in field (AuthKit remote view) answers | yes, in the sign-in field | not sampled |
| Safari | 26.5.2 | yes (fixture textarea and address bar) | yes | yes |
| Google Chrome | 153.0.8010.37 | yes (fixture textarea and address bar) | yes | yes |
| Terminal | 2.15 | yes (the tty buffer is the document) | yes | not sampled |
| Xcode | 26.6 | yes | yes | yes |
| Visual Studio Code | 1.139.1 | yes (editor) | yes | yes |
| Discord | 0.0.413 | yes | yes | not sampled |
| Obsidian | 1.13.7 | yes | yes | not sampled |
| WeChat | 4.1.15 | no text field on the login UI (QR-code sign-in only) | n/a | n/a |
| Pages / Slack | not installed | -- | -- | -- |

Quirks the table does not show:

- `documentLength` lies broadly. Chrome, Safari, Mail, Xcode, VS Code and
  Discord all report 0 or `INT_MAX` while still serving the substring; the
  probe's `substring` already requests a fixed range regardless, so treat
  `length()` as unusable for bounds.
- Some Chromium fields pin `selectedRange` at `{0,1}` while text is typed (seen
  in Discord and Obsidian) but report the real position once the caret settles
  (on click) or serve `attributedSubstring` correctly anyway. The VS Code
  editor and the Chrome textarea report real positions throughout.
- Records taken inside `didCommand`/`inputText` describe the *pre-move* caret;
  an arrow key probes the position before it executes.
- The JSONL log can interleave partial writes (5 truncated lines across ~500
  records when two probe instances ran at once); readers should skip
  unparseable lines.
- Clients that already hold a probe session may not re-fire `activateServer`
  when the sweep re-selects the input method, so an immediate second sweep can
  skip hosts the first one covered; reopen the host to re-probe it.

`Tools/sweep.sh` runs this table unattended once its prerequisites are granted:
the probe installed and registered (`../build.sh --install`), and Automation
consent for the process running the script (every `osascript` call is bounded
by `with timeout`, so a missing grant skips that host instead of stalling, and
Accessibility is needed only for the optional arrow-key re-measurement). The
script writes the `AppleEnabledThirdPartyInputSources` entries itself when they
are missing, and restores the previous input source on exit.

## The one thing that actually blocks installation

A bundle dropped into `~/Library/Input Methods` is invisible to Text Input Sources
until something calls **`TISRegisterInputSource(CFURLRef)`** on it. Until then the
input method installs, launches, holds its Mach connection and passes
`codesign --verify --deep --strict`, yet never appears in System Settings -- and
**nothing anywhere in the system log says why**.

None of the obvious things substitute for it. `lsregister -f`, killing
`TextInputMenuAgent` and `TextInputSwitcher`, adding every `Info.plist` key
Squirrel carries (`NSPrincipalClass`, `InputMethodServerDelegateClass`,
`CFBundleSignature`, all four icon keys), and renaming the bundle identifier into
the `.inputmethod.` convention each changed nothing. `TISRegisterInputSource`
returned `noErr` and the installed-source count went from 318 to 320 immediately.

`main.swift` therefore registers its own bundle on every launch, before starting
`IMKServer`. The call is idempotent, so this costs nothing and removes the
failure mode entirely.

Diagnostic worth keeping: `TISCreateInputSourceList(nil, true)` counts *installed*
sources and `TISCreateInputSourceList(nil, false)` counts *enabled* ones -- 320
against 8 here. A missing entry in the first list means unregistered, not merely
switched off, and that distinction is what separated this bug from a red herring.

## Unverified (second-hand, milestone 5)

Relayed from a 2026 write-up by the vChewing author; not read directly, and the
one claim above that could be checked turned out to be wrong, so treat the rest
accordingly.

- `IMKCandidates` is said to be unusable; draw candidates in a reused `NSPanel`.
- Sandboxing needs
  `com.apple.security.temporary-exception.mach-register.global-name` set to the
  connection name.
- Attaching a debugger to an input method freezes the host application, so all
  logic must live in a library testable without a running IME -- which is the
  layout this repository already uses, for independent reasons.
- `IMKInputController` should hold no state; keep per-client sessions in a cache
  keyed by a weak reference to the client.

## What the probe is for

The probe passes every keystroke through untouched and records only
availability and lengths -- content only under `MLIME_PROBE_SAMPLE`, since the
text being probed is whatever the user happens to be writing. The claim it was
built to test -- `attributedSubstring`/`selectedRange` working in AppKit text
views and returning nothing in Electron -- has now been measured; see "Host
coverage" (short version: AppKit answers, and so does Electron).

Note that the input method's own commit history is always available as context
regardless of the host, and covers the dominant case of typing a long passage in
one place. Host surrounding text adds editing in place and replying below a quote.
