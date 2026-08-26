# InputMethodKit notes (parked, unverified)

Collected while planning; **none of this has been verified against a running input
method**. It belongs to milestone 5 and must not be allowed to constrain design
decisions before then. Verify each line before acting on it.

Source: a 2026 write-up by the vChewing author, relayed second-hand. Not read directly.

- `IMKCandidates` is reportedly unusable in current macOS; draw candidates in an
  `NSPanel` instead, and reuse a single panel rather than creating one per update.
- `Info.plist` key `InputMethodConnectionName` must equal
  `$(PRODUCT_BUNDLE_IDENTIFIER)_Connection`, or loading fails once sandboxed.
- Sandboxing needs `com.apple.security.temporary-exception.mach-register.global-name`
  set to that connection name. Shipping the model inside the bundle avoids the
  home-relative read/write exception.
- Attaching a debugger to the input method freezes the host application. All logic
  therefore has to live in a library that is testable without a running IME --
  which is the layout this repository already uses, for independent reasons.
- `IMKInputController` should not hold state; keep per-client sessions in a cache
  keyed by a weak reference to the client.
- Install to `~/Library/Input Methods/`, register once, add in System Settings.
  `killall` the process to reload after a rebuild.

## The one thing that actually needs measuring first

Milestone 0.5 exists because the product thesis depends on reading the host
application's surrounding text, and availability is per-application:

- `IMKTextInput.attributedSubstring(from:)` and `selectedRange()` are reported to
  work in AppKit text views and to return `NSNotFound` or nothing in Electron and
  some custom editors.
- This is a claim, not a measurement. The probe app exists to turn it into a table.

Note that the IME's own commit history is always available as context regardless of
what the host provides, and covers the dominant case of typing a long passage in
one place. Host surrounding text adds editing-in-place and replying-below-a-quote.
