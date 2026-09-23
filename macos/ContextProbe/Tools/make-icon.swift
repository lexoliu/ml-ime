// Generates Resources/probe.pdf, the menu-bar icon.
//
// Input methods reference their icon from Info.plist, and a vector PDF is what
// the platform's own input methods ship. Drawn in code rather than committed as
// an opaque binary so the shape can be changed without a design tool.
import CoreGraphics
import Foundation

let side: CGFloat = 32
let url = URL(fileURLWithPath: "Resources/probe.pdf") as CFURL
var box = CGRect(x: 0, y: 0, width: side, height: side)

guard let context = CGContext(url, mediaBox: &box, nil) else {
    fatalError("could not open Resources/probe.pdf for writing")
}
context.beginPDFPage(nil)

// A filled ring: legible at menu-bar size in both light and dark menus, and
// distinct from any shipping input method's glyph.
context.setFillColor(gray: 0, alpha: 1)
context.addEllipse(in: box.insetBy(dx: 3, dy: 3))
context.addEllipse(in: box.insetBy(dx: 10, dy: 10))
context.setShouldAntialias(true)
context.fillPath(using: .evenOdd)

context.endPDFPage()
context.closePDF()
