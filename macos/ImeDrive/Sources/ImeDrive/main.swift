import ApplicationServices
import Carbon
import Cocoa
import Foundation

/// ime-drive --engine <apple|sogou|baidu> --eval-set <jsonl> --out <jsonl>
///
/// Types each eval record's keystrokes into a text view this process owns,
/// reads the selected input method's candidate window through the
/// accessibility tree, and accepts the default candidate until the
/// composition is consumed — what a user gets by pressing space.
///
///   --list            dump every installed input source as JSON and exit
///   --limit N         stop after N records this run (smoke tests)
///   --key-wait-cap-ms N
///                     per-key settle budget; the wait ends as soon as the
///                     view state changes (default 300)
///   --record-timeout-ms N
///                     per-record wall budget; exceeding it writes a
///                     failure line instead of hanging (default 20000)
///   --slice S         dev|test|all — which keyed split to type (default all;
///                     the split is the record hash, dev share 0.0905)
///   --watch-bundle S  extra bundle-id substring whose processes are read for
///                     the candidate window before any other app
struct Args {
    var engine = ""
    var evalSet: URL?
    var out: URL?
    var limit: Int?
    var list = false
    var keyWaitCapMs = 300
    var recordTimeoutMs = 20_000
    var slice = "all"
    var watchBundle: String?

    static func parse() -> Args {
        var args = Args()
        var iterator = CommandLine.arguments.dropFirst().makeIterator()
        while let flag = iterator.next() {
            switch flag {
            case "--engine": args.engine = iterator.next() ?? ""
            case "--eval-set": args.evalSet = iterator.next().map(URL.init(fileURLWithPath:))
            case "--out": args.out = iterator.next().map(URL.init(fileURLWithPath:))
            case "--limit": args.limit = iterator.next().flatMap(Int.init)
            case "--list": args.list = true
            case "--key-wait-cap-ms":
                args.keyWaitCapMs = iterator.next().flatMap(Int.init) ?? 300
            case "--record-timeout-ms":
                args.recordTimeoutMs = iterator.next().flatMap(Int.init) ?? 20_000
            case "--slice": args.slice = iterator.next() ?? "all"
            case "--watch-bundle": args.watchBundle = iterator.next()
            case "--help", "-h":
                let usage =
                    "usage: ime-drive --engine <apple|sogou|baidu> --eval-set f.jsonl "
                    + "--out o.jsonl [--limit N] [--key-wait-cap-ms N] [--list] "
                    + "[--watch-bundle S]\n"
                FileHandle.standardError.write(usage.data(using: .utf8)!)
                exit(0)
            default:
                FileHandle.standardError.write(
                    "ime-drive: unknown argument \(flag)\n".data(using: .utf8)!)
                exit(2)
            }
        }
        return args
    }
}

/// Which input source and which processes belong to each engine.
struct EngineSpec {
    /// Substrings matched (lowercased) against the input source id; first
    /// match wins. Order matters: "baidu" before falling back to nothing.
    let sourceIdHints: [String]
    /// Substrings matched against the input source's bundle id and against
    /// running applications' bundle ids to find candidate-window owners.
    let bundleHints: [String]
    /// Whether the engine's candidate panel exposes its candidates through
    /// the accessibility tree. When false (Baidu, whose panel is custom
    /// drawn) readiness is keyed on the panel's window-server presence and
    /// the accessibility read is skipped outright.
    let panelHasAxText: Bool
}

let engines: [String: EngineSpec] = [
    "apple": EngineSpec(
        sourceIdHints: ["com.apple.inputmethod.scim.itabc"],
        bundleHints: ["com.apple.inputmethod"], panelHasAxText: true
    ),
    "sogou": EngineSpec(
        sourceIdHints: ["com.sogou.inputmethod.sogou.pinyin", "sogou"],
        bundleHints: ["sogou"], panelHasAxText: true
    ),
    "baidu": EngineSpec(
        sourceIdHints: ["com.baidu.inputmethod.baiduim.pinyin", "baidu"],
        bundleHints: ["baidu"], panelHasAxText: false
    ),
]

func err(_ message: String) -> Never {
    FileHandle.standardError.write("ime-drive: \(message)\n".data(using: .utf8)!)
    exit(1)
}

let args = Args.parse()

if args.list {
    let sources = allInputSources().map(\.asDict)
    let data = try! JSONSerialization.data(
        withJSONObject: sources, options: [.prettyPrinted])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
    exit(0)
}

guard let spec = engines[args.engine] else {
    err("unknown engine \(args.engine); want one of \(engines.keys.sorted())")
}
guard let evalSet = args.evalSet, let out = args.out else {
    err("--eval-set and --out are required")
}
guard ["dev", "test", "all"].contains(args.slice) else {
    err("--slice must be dev, test or all, got \(args.slice)")
}

// Reading another process's candidate window needs Accessibility trust; ask
// (the OS shows its prompt once) and report what we got — a Sogou or Baidu
// run without it reads nothing.
let trusted = AXIsProcessTrustedWithOptions(
    ["AXTrustedCheckOptionPrompt" as CFString: true] as CFDictionary)
FileHandle.standardError.write(
    "ime-drive: accessibility trusted=\(trusted)\n".data(using: .utf8)!)

// App + owned text view, set up before touching TIS: selecting an input
// source needs the process to have a live text-input context, which Cocoa
// creates for the window being typed into. Keystrokes posted at the HID tap
// are delivered to the focused window, so this window must stay key.
let app = NSApplication.shared
app.setActivationPolicy(.regular)
let window = NSWindow(
    contentRect: NSRect(x: 200, y: 200, width: 640, height: 200),
    styleMask: [.titled, .closable, .miniaturizable],
    backing: .buffered, defer: false)
window.title = "ime-drive — \(args.engine)"
let textView = NSTextView(frame: NSRect(x: 0, y: 0, width: 640, height: 200))
textView.font = NSFont.systemFont(ofSize: 24)
textView.isAutomaticQuoteSubstitutionEnabled = false
textView.isAutomaticTextReplacementEnabled = false
textView.isAutomaticDashSubstitutionEnabled = false
textView.isAutomaticSpellingCorrectionEnabled = false
window.contentView = textView
window.orderFrontRegardless()
window.makeKeyAndOrderFront(nil)
app.activate(ignoringOtherApps: true)
window.makeFirstResponder(textView)

// Resolve the engine's input source and select it. The source must already be
// installed; it is enabled here if needed, and re-fetched afterwards because
// enabling can change the object the list hands back.
var sources = allInputSources()
// Hints are tried in order so a mode (".pinyin") wins over its parent
// input method when both appear in the source list.
var picked = spec.sourceIdHints
    .compactMap { hint in sources.first { $0.id.lowercased().contains(hint) } }
    .first
guard let wanted = picked else {
    err(
        "no input source for engine \(args.engine); installed ids:\n"
            + sources.map { "  \($0.id)" }.joined(separator: "\n"))
}
if !wanted.enabled {
    let status = TISEnableInputSource(wanted.source)
    if status != noErr {
        err("TISEnableInputSource(\(wanted.id)) -> OSStatus \(status)")
    }
    sources = allInputSources()
    picked = sources.first { $0.id == wanted.id }
}
guard var source = picked else {
    err("input source \(wanted.id) vanished after enabling")
}
// Input modes (trailing ".pinyin" etc.) can't always be selected directly —
// TIS wants the parent input method, whose default mode then applies.
if let failure = select(source) {
    let parentID = source.id.components(separatedBy: ".").dropLast().joined(separator: ".")
    guard var parent = sources.first(where: { $0.id == parentID }) else {
        err(failure)
    }
    if !parent.enabled {
        let status = TISEnableInputSource(parent.source)
        if status != noErr {
            err("TISEnableInputSource(\(parent.id)) -> OSStatus \(status)")
        }
        sources = allInputSources()
        guard let fresh = sources.first(where: { $0.id == parentID }) else {
            err("input source \(parentID) vanished after enabling")
        }
        parent = fresh
    }
    if let parentFailure = select(parent) {
        err("\(failure); parent \(parent.id): \(parentFailure)")
    }
    source = parent
}
FileHandle.standardError.write(
    "ime-drive: selected input source \(source.id) (\(source.name))\n".data(using: .utf8)!)

// The client window's frame, translated to window-server coordinates
// (top-left origin) for candidate-window presence checks.
let clientBounds: CGRect = {
    let frame = window.frame
    let screenHeight = NSScreen.main?.frame.height ?? 0
    return CGRect(
        x: frame.minX, y: screenHeight - frame.maxY,
        width: frame.width, height: frame.height)
}()

// The engine's processes, to read the candidate window from: the input
// source's own bundle plus any --watch-bundle hint. They may take a moment to
// start after the source is selected.
let hints = spec.bundleHints + (args.watchBundle.map { [$0] } ?? [])
var watchPids = Set<pid_t>()
let watchDeadline = Date().addingTimeInterval(10)
while Date() < watchDeadline {
    watchPids = Set(
        NSWorkspace.shared.runningApplications.compactMap { app in
            guard let bundleID = app.bundleIdentifier?.lowercased() else { return nil }
            return hints.contains { bundleID.contains($0.lowercased()) }
                ? app.processIdentifier : nil
        })
    if !watchPids.isEmpty { break }
    usleep(200_000)
}
FileHandle.standardError.write(
    "ime-drive: watching pids \(watchPids.sorted()) for the candidate window\n"
        .data(using: .utf8)!)

// The driver runs off the main thread: the main thread must stay in the run
// loop so the input method's commits reach the text view.
nonisolated(unsafe) var driver: Driver!
do {
    driver = try Driver(
        engine: args.engine, evalSet: evalSet, out: out, textView: textView,
        watchPids: watchPids, bundleHints: hints, sourceID: source.id,
        clientBounds: clientBounds, keyWaitCapMs: args.keyWaitCapMs,
        recordTimeoutMs: args.recordTimeoutMs, slice: args.slice,
        panelHasAxText: spec.panelHasAxText)
} catch {
    err("setup failed: \(error.localizedDescription)")
}

DispatchQueue.global(qos: .userInitiated).async {
    driver.run(limit: args.limit)
    exit(0)
}

app.run()
