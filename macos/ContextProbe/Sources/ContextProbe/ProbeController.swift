import Foundation
import InputMethodKit
import os

/// An input method that types nothing and measures everything.
///
/// Every keystroke is passed straight through to the host application, so the
/// probe can be left selected while doing ordinary work. What it does on the way
/// is ask the client for the text around the caret and record whether the client
/// answered.
///
/// This exists because the whole point of the project -- conditioning the model
/// on text already on screen -- rests on an assumption that is documented
/// nowhere authoritative: that `IMKTextInput.attributedSubstring(from:)` works.
/// It reportedly works in AppKit text views and fails in Electron. Reportedly is
/// not a number.
@objc(ProbeController)
final class ProbeController: IMKInputController {
    /// How many characters of context to ask for on each side of the caret.
    private static let contextWindow = 64
    /// Longest prefix recorded when sampling is switched on.
    private static let sampleLimit = 16
    private static let logger = Logger(
        subsystem: "cool.lexo.inputmethod.ContextProbe", category: "controller")
    /// Set `MLIME_PROBE_SAMPLE` to record short text prefixes as well as counts.
    private static let samplingEnabled = ProcessInfo.processInfo.environment["MLIME_PROBE_SAMPLE"] != nil

    private let log = ProbeLog()
    private let started = Date()
    private var keystrokes: [String: Int] = [:]

    override func inputText(_ string: String!, client sender: Any!) -> Bool {
        probe(sender)
        // Never consume. The probe must not get in the way of the typing it is
        // measuring, or nobody will leave it selected long enough to be useful.
        return false
    }

    override func didCommand(by selector: Selector!, client sender: Any!) -> Bool {
        probe(sender)
        return false
    }

    /// Interrogate one client and append what it said.
    private func probe(_ sender: Any?) {
        guard let client = sender as? IMKTextInput else {
            Self.logger.error("client does not conform to IMKTextInput")
            return
        }
        let bundleIdentifier = client.bundleIdentifier()
        let key = bundleIdentifier ?? "<unknown>"
        let keystroke = (keystrokes[key] ?? 0) + 1
        keystrokes[key] = keystroke

        let documentLength = client.length()
        let selected = client.selectedRange()
        let record = ProbeRecord(
            elapsed: Date().timeIntervalSince(started),
            keystroke: keystroke,
            clientBundleIdentifier: bundleIdentifier,
            // A client that declines to say reports NSNotFound here too.
            documentLength: documentLength == NSNotFound ? nil : documentLength,
            selectedRange: RangeReport(selected),
            markedRange: RangeReport(client.markedRange()),
            contextBefore: Self.substring(before: selected, from: client),
            contextAfter: Self.substring(after: selected, from: client, documentLength: documentLength)
        )
        log.append(record)
    }

    /// Ask for up to `contextWindow` characters ending at the caret.
    private static func substring(before selected: NSRange, from client: IMKTextInput) -> SubstringReport? {
        guard selected.location != NSNotFound else { return nil }
        let length = min(contextWindow, selected.location)
        guard length > 0 else { return nil }
        return read(NSRange(location: selected.location - length, length: length), from: client)
    }

    /// Ask for up to `contextWindow` characters starting after the selection.
    ///
    /// Clamped against the reported document length, because asking beyond the end
    /// is the kind of thing that makes a host misbehave, and a probe that crashes
    /// its host teaches nothing.
    private static func substring(
        after selected: NSRange, from client: IMKTextInput, documentLength: Int
    ) -> SubstringReport? {
        guard selected.location != NSNotFound, documentLength != NSNotFound else { return nil }
        let start = selected.location + selected.length
        let length = min(contextWindow, max(0, documentLength - start))
        guard length > 0 else { return nil }
        return read(NSRange(location: start, length: length), from: client)
    }

    private static func read(_ range: NSRange, from client: IMKTextInput) -> SubstringReport {
        let returned = client.attributedSubstring(from: range)?.string
        return SubstringReport(
            requested: RangeReport(range),
            returnedCharacters: returned?.count,
            sample: samplingEnabled ? returned.map { String($0.prefix(sampleLimit)) } : nil
        )
    }
}
