import Foundation

/// What one client reported about the text around the caret.
///
/// Everything here is a measurement of *availability*, not of content: the whole
/// point of the probe is to find out which applications answer
/// `attributedSubstring(from:)` at all. Content is recorded only when
/// `MLIME_PROBE_SAMPLE` is set, and even then only a short prefix, because the
/// text being probed is whatever the user happens to be writing.
struct ProbeRecord: Codable {
    /// Seconds since the probe process started. Relative, so the log carries no
    /// wall-clock record of when someone was typing.
    let elapsed: Double
    /// How many keystrokes this client has been probed with so far. Lets the
    /// report tell a genuinely static value apart from one that never moves.
    let keystroke: Int
    let clientBundleIdentifier: String?
    let documentLength: Int?
    let selectedRange: RangeReport
    let markedRange: RangeReport
    let contextBefore: SubstringReport?
    let contextAfter: SubstringReport?
}

/// An `NSRange` as the API actually returned it, including the not-found case
/// that means "this client declines to say".
struct RangeReport: Codable {
    let location: Int?
    let length: Int?

    init(_ range: NSRange) {
        if range.location == NSNotFound {
            location = nil
            length = nil
        } else {
            location = range.location
            length = range.length
        }
    }

    /// Whether the client answered at all.
    var isAvailable: Bool { location != nil }
}

/// The outcome of one `attributedSubstring(from:)` call.
struct SubstringReport: Codable {
    let requested: RangeReport
    /// Characters actually returned. `nil` means the call returned nothing.
    let returnedCharacters: Int?
    /// A short prefix, present only under `MLIME_PROBE_SAMPLE`.
    let sample: String?
}
