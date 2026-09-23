import Foundation
import os

/// Append-only JSONL sink for probe records.
///
/// Serialised through `JSONEncoder` rather than assembled as text, so a field
/// renamed in `ProbeRecord` cannot silently produce a log the report tool then
/// misreads.
final class ProbeLog: @unchecked Sendable {
    private let handle: FileHandle
    private let encoder = JSONEncoder()
    private let queue = DispatchQueue(label: "cool.lexo.mlime.probe.log")
    private static let logger = Logger(subsystem: "cool.lexo.inputmethod.ContextProbe", category: "log")

    /// Where the probe writes. Fixed rather than configurable so the report tool
    /// and the reader of these instructions cannot disagree about it.
    static var url: URL {
        FileManager.default
            .homeDirectoryForCurrentUser
            .appending(path: "Library/Logs/mlime-context-probe.jsonl")
    }

    /// Open the log, creating it if needed.
    ///
    /// Failing here is fatal: a probe that cannot record has nothing to offer,
    /// and silently continuing would produce an empty report that reads like a
    /// negative result.
    init() {
        let url = Self.url
        let manager = FileManager.default
        do {
            try manager.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            if !manager.fileExists(atPath: url.path) {
                manager.createFile(atPath: url.path, contents: nil)
            }
            handle = try FileHandle(forWritingTo: url)
            try handle.seekToEnd()
        } catch {
            fatalError("context probe cannot open \(url.path): \(error)")
        }
        Self.logger.info("probe log opened at \(url.path, privacy: .public)")
    }

    /// Append one record.
    func append(_ record: ProbeRecord) {
        queue.async { [handle, encoder] in
            do {
                var line = try encoder.encode(record)
                line.append(0x0A)
                try handle.write(contentsOf: line)
            } catch {
                Self.logger.error("dropped a probe record: \(error.localizedDescription)")
            }
        }
    }
}
