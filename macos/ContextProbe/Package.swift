// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ContextProbe",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "ContextProbe",
            swiftSettings: [.swiftLanguageMode(.v6)]
        )
    ]
)
