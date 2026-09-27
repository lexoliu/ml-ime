// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ImeDrive",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "ImeDrive",
            swiftSettings: [.swiftLanguageMode(.v6)]
        )
    ]
)
