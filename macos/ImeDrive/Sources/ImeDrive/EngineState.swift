import Foundation

/// What produced the run: the engine's preference domains (learning, cloud
/// input and the toggles the vendor exposes) and the files it learns words
/// into, recorded in the journal's meta line so a report can say what state
/// it was measured in.
///
/// The twins share their sentences: an engine that learns a committed
/// sentence sees it as a first-letter hit on the abbreviated twin. A run is
/// only a measurement when the state below was fresh — a fresh VM, or the
/// learned files removed/reset before the twin.
enum EngineState {
    /// Preference domains read wholesale (filtered to JSON-safe values).
    private static let prefDomains: [String: [String]] = [
        "apple": [
            "com.apple.inputmethod.CoreChineseEngineFramework",
            "com.apple.inputmethod.SCIM.ITABC",
        ],
        "sogou": ["com.sogou.inputmethod.sogou"],
        "baidu": ["com.baidu.inputmethod.BaiduIM"],
    ]

    /// Files each engine learns the user's words into, relative to
    /// ~/Library unless absolute. Apple writes its learned phrases to the
    /// dynamic phrase lexicon; Sogou keeps its user dictionary and learned
    /// bigrams under SogouPY; Baidu under BaiduInput/userdict.
    private static let learnedFiles: [String: [String]] = [
        "apple": [
            "Dictionaries/DynamicPhraseLexicon_zh_Hans.db",
        ],
        "sogou": [
            "Application Support/Sogou/InputMethod/SogouPY/sgim_gd_usr.bin",
            "Application Support/Sogou/InputMethod/SogouPY/sgim_gd_usr_a_bigram.bin",
            "Application Support/Sogou/InputMethod/SogouPY.users/CellRecord.dat",
            "Application Support/Sogou/InputMethod/SogouPY/UserPreferences.plist",
        ],
        "baidu": [
            "Application Support/BaiduInput/userdict/usr3user.bin",
            "Application Support/BaiduInput/userdict/usr3cell.bin",
            "Application Support/BaiduInput/userdict/black_white_dict2.bin",
        ],
    ]

    /// The engine's settings and learned-state files, JSON-serializable.
    static func snapshot(engine: String) -> [String: Any] {
        var preferences: [String: Any] = [:]
        for domain in prefDomains[engine] ?? [] {
            if let values = UserDefaults(suiteName: domain)?.dictionaryRepresentation(),
                !values.isEmpty
            {
                preferences[domain] = jsonSafe(values)
            }
        }

        var files: [String: Any] = [:]
        let home = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library")
        for relative in learnedFiles[engine] ?? [] {
            let url = home.appendingPathComponent(relative)
            guard let attrs = try? FileManager.default.attributesOfItem(
                atPath: url.path)
            else {
                files[relative] = ["exists": false]
                continue
            }
            files[relative] = [
                "exists": true,
                "size": attrs[.size] as? Int ?? 0,
                "mtime": ISO8601DateFormatter().string(
                    from: attrs[.modificationDate] as? Date ?? .distantPast),
            ]
        }
        return ["preferences": preferences, "learned_files": files]
    }

    /// Recursively keep only what JSONSerialization can write.
    private static func jsonSafe(_ value: Any) -> Any? {
        // Bool and NSNumber bridge to each other (`NSNumber(1) as? Bool`
        // succeeds), so the two are told apart by CoreFoundation type id,
        // not by cast order.
        let typeID = CFGetTypeID(value as CFTypeRef)
        if typeID == CFBooleanGetTypeID() { return value as? Bool }
        if typeID == CFNumberGetTypeID() { return value as? NSNumber }
        switch value {
        case let v as String: return v
        case let v as [Any]: return v.compactMap(jsonSafe)
        case let v as [String: Any]:
            var out: [String: Any] = [:]
            for (key, item) in v {
                if let safe = jsonSafe(item) { out[key] = safe }
            }
            return out
        case let v as Data: return "<\(v.count) bytes>"
        case let v as Date: return ISO8601DateFormatter().string(from: v)
        default: return nil
        }
    }
}
