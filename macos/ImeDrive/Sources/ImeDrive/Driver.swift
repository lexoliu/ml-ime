import Cocoa
import Foundation

/// The record loop: type each eval record's keystrokes into the text view the
/// process owns, accept the default candidate until the composition is
/// consumed, and append one JSON line per record to the results file.
///
/// The run is resumable: an existing results file is opened for append and
/// every record already in it is skipped, so an interrupted run is continued
/// by repeating the same command.
final class Driver {
    /// One line of the eval set: the keystrokes are all the driver needs.
    struct EvalRow {
        let index: Int
        let pinyin: String
    }

    /// ANSI virtual key codes for the characters an eval record can hold
    /// (letters, digits, the apostrophe some sets use as a syllable break).
    private static let keycodes: [Character: CGKeyCode] = [
        "a": 0x00, "s": 0x01, "d": 0x02, "f": 0x03, "h": 0x04, "g": 0x05, "z": 0x06,
        "x": 0x07, "c": 0x08, "v": 0x09, "b": 0x0B, "q": 0x0C, "w": 0x0D, "e": 0x0E,
        "r": 0x0F, "y": 0x10, "t": 0x11, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15,
        "6": 0x16, "5": 0x17, "9": 0x19, "7": 0x1A, "8": 0x1C, "0": 0x1D, "o": 0x1F,
        "u": 0x20, "i": 0x22, "p": 0x23, "l": 0x25, "j": 0x26, "'": 0x27, "k": 0x28,
        "n": 0x2D, "m": 0x2E, "-": 0x1B, "=": 0x18, "[": 0x21, "]": 0x1E, ";": 0x29,
        ",": 0x2B, "/": 0x2C, ".": 0x2F, "\\": 0x2A, " ": 0x31,
    ]

    private let engine: String
    private let textView: NSTextView
    private var watchPids: Set<pid_t>
    private let bundleHints: [String]
    private let sourceID: String
    /// The client window's bounds in window-server coordinates, for
    /// presence detection of panels without accessibility text.
    private let clientBounds: CGRect
    private let eventSource = CGEventSource(stateID: .hidSystemState)
    /// Cap on how long a single keystroke's settle wait can run; the wait
    /// ends as soon as the view state changes.
    private let keyWaitCapMs: Int
    /// Per-record wall budget; exceeding it writes a failure line.
    private let recordTimeoutMs: Int

    private let out: FileHandle
    private let done: Set<Int>

    init(
        engine: String, evalSet: URL, out outURL: URL, textView: NSTextView,
        watchPids: Set<pid_t>, bundleHints: [String], sourceID: String,
        clientBounds: CGRect, keyWaitCapMs: Int, recordTimeoutMs: Int
    ) throws {
        self.engine = engine
        self.textView = textView
        self.watchPids = watchPids
        self.bundleHints = bundleHints
        self.sourceID = sourceID
        self.clientBounds = clientBounds
        self.keyWaitCapMs = keyWaitCapMs
        self.recordTimeoutMs = recordTimeoutMs

        var rows: [EvalRow] = []
        var index = 0
        for line in try String(contentsOf: evalSet, encoding: .utf8).split(
            whereSeparator: \.isNewline)
        {
            guard !line.trimmingCharacters(in: .whitespaces).isEmpty,
                let data = line.data(using: .utf8),
                let row = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                let pinyin = row["pinyin"] as? String
            else {
                throw NSError(
                    domain: "ImeDrive", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "bad eval-set line \(index + 1)"])
            }
            rows.append(EvalRow(index: index, pinyin: pinyin))
            index += 1
        }
        self.rows = rows

        var done: Set<Int> = []
        if FileManager.default.fileExists(atPath: outURL.path),
            let existing = try? String(contentsOf: outURL, encoding: .utf8)
        {
            for line in existing.split(whereSeparator: \.isNewline) {
                guard let data = line.data(using: .utf8),
                    let row = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                    let record = row["record"] as? Int
                else { continue }
                done.insert(record)
            }
        }
        self.done = done
        // The results file is the journal: never truncate it — a line
        // already written is a record already done.
        if !FileManager.default.fileExists(atPath: outURL.path) {
            FileManager.default.createFile(atPath: outURL.path, contents: nil)
        }
        out = try FileHandle(forWritingTo: outURL)
        out.seekToEndOfFile()
    }

    private let rows: [EvalRow]

    /// JSON-serializable meta line, written once at the top of a fresh file.
    func writeMeta() {
        let meta: [String: Any] = [
            "type": "meta",
            "engine": engine,
            "input_source_id": sourceID,
            "macos": ProcessInfo.processInfo.operatingSystemVersionString,
            "started": ISO8601DateFormatter().string(from: Date()),
            "pid": Int(ProcessInfo.processInfo.processIdentifier),
        ]
        if out.offsetInFile == 0 {
            write(meta)
        }
    }

    /// Type every record that is not already in the results file.
    func run(limit: Int?) {
        writeMeta()
        var typed = 0
        for row in rows {
            if done.contains(row.index) { continue }
            let result = typeOne(row)
            write(result)
            if result["failed"] as? Bool == true {
                FileHandle.standardError.write(
                    "ime-drive: record \(row.index) failed: \(result["error"] ?? "")\n"
                        .data(using: .utf8)!)
            }
            typed += 1
            if typed % 50 == 0 {
                FileHandle.standardError.write(
                    "ime-drive: \(typed) records (index \(row.index))\n".data(using: .utf8)!)
            }
            if let limit, typed >= limit { break }
        }
        FileHandle.standardError.write(
            "ime-drive: done, \(typed) records this run\n".data(using: .utf8)!)
    }

    /// Type one record and return its JSON line fields. Every wait is a
    /// bounded poll on the view state or the candidate window — a record
    /// costs what the engine needs and no more. Exceeding the per-record
    /// budget writes a failure line instead of hanging.
    private func typeOne(_ row: EvalRow) -> [String: Any] {
        let start = Date()
        let deadline = start.addingTimeInterval(Double(recordTimeoutMs) / 1000)
        clearView()
        var sawWindow = false
        var firstPage: [String] = []
        var presses = 0
        var error: String?

        typing: do {
            for char in row.pinyin {
                guard let code = Self.keycodes[char] else {
                    FileHandle.standardError.write(
                        "ime-drive: no keycode for \(char.debugDescription), skipping record \(row.index)\n"
                            .data(using: .utf8)!)
                    continue
                }
                if Date() > deadline {
                    error = "timed out after \(recordTimeoutMs) ms while typing"
                    break typing
                }
                let before = viewState()
                post(code)
                waitStateChange(since: before, timeoutMs: keyWaitCapMs, deadline: deadline)
            }

            // The engine's process may only have spawned once typing began
            // (SCIM starts lazily), so refresh the watch set before polling.
            refreshWatchPids()
            let remaining = Int(deadline.timeIntervalSinceNow * 1000)
            if remaining <= 0 {
                error = "timed out after \(recordTimeoutMs) ms before candidates"
                break typing
            }
            if let page = waitForCandidates(timeoutMs: min(1500, remaining)) {
                sawWindow = true
                firstPage = page
            } else {
                // The panel may exist yet expose no text (Baidu): check for
                // an engine-owned floating window near the text view.
                sawWindow = CandidateWindow.present(
                    watchPids: watchPids, near: clientBounds)
            }

            // Accept the first candidate until nothing is composing. A user
            // gets the same by pressing space.
            let maxPresses = row.pinyin.count + 16
            while presses < maxPresses {
                let state = viewState()
                guard state.markedLength > 0 else { break }
                if Date() > deadline {
                    error = "timed out after \(recordTimeoutMs) ms while accepting"
                    break
                }
                post(CGKeyCode(0x31))  // space
                presses += 1
                waitStateChange(since: state, timeoutMs: 500, deadline: deadline)
            }
        }

        // Escape drops any remainder; the view may not change if nothing was
        // pending, so this wait is short.
        let beforeEscape = viewState()
        post(CGKeyCode(0x35))
        waitStateChange(since: beforeEscape, timeoutMs: 200, deadline: deadline)

        var committed = viewString()
        if let range = markedRange() {
            // Composition left over: keep only what was committed.
            let text = committed as NSString
            committed =
                text.substring(to: range.location)
                + text.substring(from: range.location + range.length)
        }
        var result: [String: Any] = [
            "record": row.index,
            "engine": engine,
            "committed": committed,
            "first_page": firstPage,
            "wall_ms": Int(Date().timeIntervalSince(start) * 1000),
            "candidate_window": sawWindow,
            "space_presses": presses,
        ]
        if let error {
            result["failed"] = true
            result["error"] = error
        }
        return result
    }

    /// Poll until a candidate window's strings stop changing, or the timeout.
    /// Returns the settled page, normalized.
    private func waitForCandidates(timeoutMs: Int) -> [String]? {
        let deadline = Date().addingTimeInterval(Double(timeoutMs) / 1000)
        var last: [String]?
        var stable = 0
        while Date() < deadline {
            if let raw = CandidateWindow.read(watchPids: watchPids) {
                let page = CandidateWindow.normalize(raw)
                stable = page == last ? stable + 1 : 0
                last = page
                if stable >= 2 { return page }
            } else {
                last = nil
                stable = 0
            }
            usleep(30_000)
        }
        return last
    }

    /// Re-scan for engine processes; input-method helpers start lazily.
    private func refreshWatchPids() {
        for app in NSWorkspace.shared.runningApplications {
            guard let bundleID = app.bundleIdentifier?.lowercased() else { continue }
            if bundleHints.contains(where: { bundleID.contains($0.lowercased()) }) {
                watchPids.insert(app.processIdentifier)
            }
        }
    }

    private func post(_ code: CGKeyCode) {
        for keyDown in [true, false] {
            let event = CGEvent(keyboardEventSource: eventSource, virtualKey: code, keyDown: keyDown)
            event?.post(tap: .cghidEventTap)
        }
        usleep(12_000)
    }

    /// What the polling waits compare against: the view's text length and
    /// the marked (composing) range. Either one changing means the engine
    /// reacted to the keystroke.
    private struct ViewState: Equatable {
        var textLength: Int
        var markedLocation: Int
        var markedLength: Int
    }

    private func viewState() -> ViewState {
        let view = textView
        return DispatchQueue.main.sync {
            let marked = view.markedRange()
            return ViewState(
                textLength: view.string.count,
                markedLocation: marked.location == NSNotFound ? -1 : marked.location,
                markedLength: marked.location == NSNotFound ? 0 : marked.length)
        }
    }

    /// Poll the view state until it differs from `before` or the timeout —
    /// the bounded wait that replaced the fixed keystroke delay.
    private func waitStateChange(since before: ViewState, timeoutMs: Int, deadline: Date) {
        let cap = Date().addingTimeInterval(Double(timeoutMs) / 1000)
        let end = min(cap, deadline)
        while Date() < end {
            if viewState() != before { return }
            usleep(5_000)
        }
    }

    /// Clear the text view for the next record, on the main thread, then
    /// wait briefly for the clear to take effect.
    private func clearView() {
        let view = textView
        DispatchQueue.main.sync {
            view.unmarkText()
            view.string = ""
        }
        let deadline = Date().addingTimeInterval(0.5)
        while Date() < deadline {
            let state = viewState()
            if state.textLength == 0, state.markedLength == 0 { return }
            usleep(10_000)
        }
    }

    private func viewString() -> String {
        let view = textView
        return DispatchQueue.main.sync { view.string }
    }

    private func markedRange() -> NSRange? {
        let view = textView
        return DispatchQueue.main.sync {
            let range = view.markedRange()
            return range.location == NSNotFound ? nil : range
        }
    }

    private func write(_ fields: [String: Any]) {
        let data = try! JSONSerialization.data(withJSONObject: fields)
        out.write(data)
        out.write("\n".data(using: .utf8)!)
    }
}
