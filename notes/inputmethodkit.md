# InputMethodKit findings

Split into what has been measured on this machine (macOS 26.6.2, Swift 6.3.3,
SDK 26.5) and what is still hearsay. Anything under "unverified" must not be
allowed to constrain a design decision.

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

## Falsified

- **`InputMethodConnectionName` does not have to be
  `$(PRODUCT_BUNDLE_IDENTIFIER)_Connection`.** Squirrel, a shipping input method,
  uses `Squirrel_Connection` -- the executable name plus `_Connection`, with the
  bundle identifier `im.rime.inputmethod.Squirrel` nowhere in it. The probe now
  follows Squirrel.

## Open: the probe is installed but not indexed

The bundle is installed at `~/Library/Input Methods/ContextProbe.app`, runs, is
validly signed, and its `Info.plist` now carries every input-method key Squirrel's
does: `NSPrincipalClass`, `CFBundleSignature`, `CFBundleIconFile`,
`InputMethodServerControllerClass`, `InputMethodServerDelegateClass`,
`ComponentInputModeDict` with `tsInputModeCharacterRepertoireKey` inside the mode
dict, and all four icon keys pointing at a generated `probe.pdf`.

It still does not appear in `TISCreateInputSourceList`, and nothing in the system
log mentions it. `lsregister -f` and killing `TextInputMenuAgent` /
`TextInputSwitcher` did not change that.

The untested hypothesis is that the input-source database is rebuilt at login, so
a newly installed input method is invisible until the user logs out and back in.
This is widely repeated in input-method projects' installation instructions but
has not been confirmed here. Next step is to test it.

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

`IMKTextInput.attributedSubstring(from:)` and `selectedRange()` are said to work in
AppKit text views and to return nothing in Electron. That is a claim, not a
measurement, and the whole product thesis rests on it. The probe passes every
keystroke through untouched and records only availability and lengths -- content
only under `MLIME_PROBE_SAMPLE`, since the text being probed is whatever the user
happens to be writing.

Note that the input method's own commit history is always available as context
regardless of the host, and covers the dominant case of typing a long passage in
one place. Host surrounding text adds editing in place and replying below a quote.
