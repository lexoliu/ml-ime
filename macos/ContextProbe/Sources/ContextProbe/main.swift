import Carbon
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
// A bundle dropped into ~/Library/Input Methods is not visible to Text Input
// Sources until something registers it. Without this the input method installs,
// runs, verifies, and simply never appears in System Settings -- with nothing in
// the system log to say why. Registering is idempotent, so doing it on every
// launch costs nothing and removes a whole class of "did the install work?".
let registration = TISRegisterInputSource(bundle.bundleURL as CFURL)
guard registration == noErr else {
    fatalError("TISRegisterInputSource refused \(bundle.bundleURL.path): OSStatus \(registration)")
}
logger.info("registered input source from \(bundle.bundleURL.path, privacy: .public)")

guard let server = IMKServer(name: connectionName, bundleIdentifier: identifier) else {
    fatalError("IMKServer refused to start on connection \(connectionName)")
}
logger.info("context probe listening on \(connectionName, privacy: .public)")
logger.info("writing to \(ProbeLog.url.path, privacy: .public)")

// `server` is unused past this point but must outlive the run loop.
withExtendedLifetime(server) {
    NSApplication.shared.run()
}
