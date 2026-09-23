import Carbon
import Foundation

func source(id wanted: String) -> TISInputSource? {
    let all = TISCreateInputSourceList(nil, true)!.takeRetainedValue() as! [TISInputSource]
    return all.first { s in
        guard let p = TISGetInputSourceProperty(s, kTISPropertyInputSourceID) else { return false }
        return (Unmanaged<CFString>.fromOpaque(p).takeUnretainedValue() as String) == wanted
    }
}

let action = CommandLine.arguments[1]
if action == "current" {
    let s = TISCopyCurrentKeyboardInputSource()!.takeRetainedValue()
    let p = TISGetInputSourceProperty(s, kTISPropertyInputSourceID)!
    print(Unmanaged<CFString>.fromOpaque(p).takeUnretainedValue() as String)
    exit(0)
}
let wanted = CommandLine.arguments[2]
guard let s = source(id: wanted) else { fatalError("no input source \(wanted)") }
switch action {
case "enable":
    let r = TISEnableInputSource(s)
    print("enable \(wanted) -> \(r)")
    if r != noErr { exit(1) }
case "select":
    let r = TISSelectInputSource(s)
    print("select \(wanted) -> \(r)")
    if r != noErr { exit(1) }
case "disable":
    let r = TISDisableInputSource(s)
    print("disable \(wanted) -> \(r)")
default:
    fatalError("unknown action \(action)")
}
