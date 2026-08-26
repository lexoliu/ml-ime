import Cocoa
import InputMethodKit
import os

private let logger = Logger(subsystem: "cool.lexo.inputmethod.ContextProbe", category: "main")

let bundle = Bundle.main
guard let connectionName = bundle.infoDictionary?["InputMethodConnectionName"] as? String else {
    fatalError("Info.plist is missing InputMethodConnectionName")
}
guard let identifier = bundle.bundleIdentifier else {
    fatalError("bundle has no identifier; the executable was probably run outside its .app")
}
guard let server = IMKServer(name: connectionName, bundleIdentifier: identifier) else {
    fatalError("IMKServer refused to start on connection \(connectionName)")
}
logger.info("context probe listening on \(connectionName, privacy: .public)")
logger.info("writing to \(ProbeLog.url.path, privacy: .public)")

// `server` is unused past this point but must outlive the run loop.
withExtendedLifetime(server) {
    NSApplication.shared.run()
}
