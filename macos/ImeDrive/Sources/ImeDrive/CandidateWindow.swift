import ApplicationServices
import Cocoa
import Foundation

/// Reading the candidate window through the accessibility tree.
///
/// Where the candidate window lives differs by engine and macOS release:
/// Apple's Pinyin draws it through TextInputUI in a floating window owned by
/// the *client* process (us), while Sogou and Baidu draw it in a process of
/// their own. So the window is found by shape, not by owner: any small
/// floating window on screen whose accessibility subtree holds Chinese text.
/// The engine's own processes are additionally scanned outright, in case an
/// engine draws the panel without a window-server entry.
enum CandidateWindow {
    /// A window larger than this is a document window, not a candidate panel.
    private static let maxWindowArea = 2_000_000.0
    /// Bounds-matching tolerance between the window server's rect and the
    /// accessibility element's frame.
    private static let frameTolerance = 40.0

    /// Whether *string* holds at least one CJK ideograph.
    private static func containsCJK(_ string: String) -> Bool {
        string.unicodeScalars.contains { scalar in
            (0x4E00...0x9FFF).contains(scalar.value)
                || (0x3400...0x4DBF).contains(scalar.value)
                || (0xF900...0xFAFF).contains(scalar.value)
        }
    }

    private static func attribute<T>(
        _ element: AXUIElement, _ name: String, as type: T.Type
    ) -> T? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success,
            let value
        else { return nil }
        return value as? T
    }

    private static func rectValue(_ element: AXUIElement, _ name: String) -> CGRect? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success,
            let value
        else { return nil }
        let axValue = value as! AXValue
        var rect = CGRect.zero
        guard AXValueGetType(axValue) == .cgRect, AXValueGetValue(axValue, .cgRect, &rect)
        else { return nil }
        return rect
    }

    private static func frame(of element: AXUIElement) -> CGRect? {
        guard let position = rectValue(element, kAXPositionAttribute),
            let size = rectValue(element, kAXSizeAttribute)
        else { return nil }
        return CGRect(origin: position.origin, size: size.size)
    }

    /// Every text-bearing leaf of *element*, in accessibility order.
    private static func collect(_ element: AXUIElement, into strings: inout [String], depth: Int) {
        if depth > 8 || strings.count > 200 { return }
        if role(of: element) == "AXMenuBar" || role(of: element) == "AXMenu" { return }
        for attributeName in [kAXValueAttribute, kAXTitleAttribute] {
            if let text = attribute(element, attributeName, as: String.self), !text.isEmpty {
                strings.append(text)
            }
        }
        let children =
            attribute(element, kAXChildrenAttribute, as: [AXUIElement].self)
            ?? attribute(element, kAXVisibleChildrenAttribute, as: [AXUIElement].self)
            ?? []
        for child in children {
            collect(child, into: &strings, depth: depth + 1)
        }
    }

    private static func role(of element: AXUIElement) -> String {
        attribute(element, kAXRoleAttribute, as: String.self) ?? ""
    }

    /// The strings of the AX window of *pid* that matches *bounds*, or of
    /// every window it owns when bounds is nil.
    private static func windowStrings(ofPid pid: pid_t, bounds: CGRect?) -> [[String]] {
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, 0.4)
        let windows =
            attribute(app, kAXWindowsAttribute, as: [AXUIElement].self)
            ?? attribute(app, kAXChildrenAttribute, as: [AXUIElement].self)
            ?? []
        var result: [[String]] = []
        for window in windows {
            if let bounds, let frame = frame(of: window) {
                if abs(frame.origin.x - bounds.origin.x) > frameTolerance
                    || abs(frame.origin.y - bounds.origin.y) > frameTolerance
                    || abs(frame.width - bounds.width) > frameTolerance
                {
                    continue
                }
            }
            var strings: [String] = []
            collect(window, into: &strings, depth: 0)
            if !strings.isEmpty {
                result.append(strings)
            }
        }
        return result
    }

    /// Small floating windows currently on screen: (owner pid, bounds).
    private static func floatingWindows() -> [(pid_t, CGRect)] {
        guard let list = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID)
            as? [[String: Any]]
        else { return [] }
        var result: [(pid_t, CGRect)] = []
        for window in list {
            guard let pid = window[kCGWindowOwnerPID as String] as? Int,
                let layer = window[kCGWindowLayer as String] as? Int, layer != 0,
                let bounds = window[kCGWindowBounds as String] as? [String: CGFloat],
                let x = bounds["X"], let y = bounds["Y"],
                let width = bounds["Width"], let height = bounds["Height"],
                width * height < maxWindowArea
            else { continue }
            result.append((pid_t(pid), CGRect(x: x, y: y, width: width, height: height)))
        }
        return result
    }

    /// The first candidate page as shown right now, or nil when no candidate
    /// window is up.
    ///
    /// Two passes: every window of *watchPids* (the engine's own processes),
    /// then every small floating window on screen regardless of owner, matched
    /// to its AX element by bounds. A window counts as a candidate window only
    /// if it holds a CJK string — the pinyin itself is ASCII, so this is what
    /// distinguishes "the IME answered" from chrome that merely floats.
    static func read(watchPids: Set<pid_t>) -> [String]? {
        var best: [String]?
        for pid in watchPids {
            for strings in windowStrings(ofPid: pid, bounds: nil) {
                if strings.contains(where: containsCJK), strings.count > (best?.count ?? 0) {
                    best = strings
                }
            }
        }
        if best == nil {
            for (pid, bounds) in floatingWindows() {
                for strings in windowStrings(ofPid: pid, bounds: bounds) {
                    if strings.contains(where: containsCJK),
                        strings.count > (best?.count ?? 0)
                    {
                        best = strings
                    }
                }
            }
        }
        return best
    }

    /// Whether an engine-owned floating panel is on screen near the client
    /// window, whose bounds are in window-server coordinates. Used when the
    /// panel exists but exposes no accessibility text (Baidu draws its
    /// candidates without any AX strings): the window still counts as a
    /// candidate window, with an empty first page.
    static func present(watchPids: Set<pid_t>, near clientBounds: CGRect) -> Bool {
        for (pid, bounds) in floatingWindows() {
            guard watchPids.contains(pid) else { continue }
            let xOverlap = bounds.minX < clientBounds.maxX + 60
                && bounds.maxX > clientBounds.minX - 60
            let nearY = bounds.minY < clientBounds.maxY + 160
                && bounds.maxY > clientBounds.minY - 160
            if xOverlap && nearY { return true }
        }
        return false
    }

    /// Strip the row label a candidate window prepends ("1.", "1 ", "2、") and
    /// whitespace; the candidates themselves remain.
    static func normalize(_ strings: [String]) -> [String] {
        var result: [String] = []
        for raw in strings {
            var text = raw
            while let first = text.unicodeScalars.first,
                CharacterSet(charactersIn: "0123456789.、:：)）]】_- ").contains(first)
            {
                text = String(text.unicodeScalars.dropFirst())
            }
            text = text.trimmingCharacters(in: .whitespaces)
            if !text.isEmpty { result.append(text) }
        }
        return result
    }
}
