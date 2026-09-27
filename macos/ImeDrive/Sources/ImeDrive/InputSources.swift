import Carbon
import Foundation

/// One entry of `TISCreateInputSourceList`, flattened to what the report needs.
struct InputSourceInfo {
    let id: String
    let name: String
    let bundleID: String?
    let category: String?
    let enabled: Bool
    let selected: Bool
    let source: TISInputSource

    var asDict: [String: Any] {
        [
            "id": id,
            "name": name,
            "bundle_id": bundleID as Any,
            "category": category as Any,
            "enabled": enabled,
            "selected": selected,
        ]
    }
}

private func property<T>(_ source: TISInputSource, _ key: CFString, as type: T.Type) -> T? {
    guard let ref = TISGetInputSourceProperty(source, key) else { return nil }
    return Unmanaged<AnyObject>.fromOpaque(ref).takeUnretainedValue() as? T
}

/// Every installed input source, enabled or not.
func allInputSources() -> [InputSourceInfo] {
    guard let list = TISCreateInputSourceList(nil, true)?.takeRetainedValue() as? [TISInputSource]
    else { return [] }
    return list.map { source in
        InputSourceInfo(
            id: property(source, kTISPropertyInputSourceID, as: String.self) ?? "",
            name: property(source, kTISPropertyLocalizedName, as: String.self) ?? "",
            bundleID: property(source, kTISPropertyBundleID, as: String.self),
            category: property(source, kTISPropertyInputSourceCategory, as: String.self),
            enabled: property(source, kTISPropertyInputSourceIsEnabled, as: Bool.self) ?? false,
            selected: property(source, kTISPropertyInputSourceIsSelected, as: Bool.self) ?? false,
            source: source
        )
    }
}

/// Enable (if needed) and select *source*. Returns an error string on failure.
@discardableResult
func select(_ source: InputSourceInfo) -> String? {
    if !source.enabled {
        let status = TISEnableInputSource(source.source)
        if status != noErr {
            return "TISEnableInputSource(\(source.id)) -> OSStatus \(status)"
        }
    }
    let status = TISSelectInputSource(source.source)
    if status != noErr {
        return "TISSelectInputSource(\(source.id)) -> OSStatus \(status)"
    }
    return nil
}
