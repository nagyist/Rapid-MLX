import AppKit
import SwiftUI
import Testing

/// SPIKE (golden-flow sink pilot): prove that a SwiftUI view mounted in a
/// never-shown ``NSWindow`` exposes a walkable accessibility hierarchy and
/// honors ``accessibilityPerformPress()`` — the two capabilities the
/// AX-driven golden flows rely on via the out-of-process driver. Both
/// holding in-process means chat-shard journeys can run inside
/// `swift test` without launching the app or taking the OS foreground.
///
/// Two non-obvious ingredients, discovered empirically (macOS 26):
///
/// 1. A bare ``NSHostingView`` — even inside a window — reports zero AX
///    children. SwiftUI materializes its ``AccessibilityNode`` tree only
///    when it believes an assistive client is attached, which the
///    `accessibilityEnhancedUserInterface` KVC toggle simulates (the same
///    flag VoiceOver sets over the AX wire; snapshot-testing tools use the
///    UIKit twin for the same purpose).
/// 2. The materialized children are `SwiftUI.AccessibilityNode` instances,
///    which implement the AX getters/actions but are NOT KVC-compliant and
///    do not conform to ``NSAccessibilityProtocol``. They must be driven
///    through `responds(to:)`/`perform(_:)`.
@MainActor
private final class SpikeModel: ObservableObject {
    @Published var label = "before-press"
    var pressed = false
}

private struct SpikeView: View {
    @ObservedObject var model: SpikeModel

    var body: some View {
        VStack {
            Text(model.label)
                .accessibilityIdentifier("Spike.Label")
            Button("Press Me") {
                model.pressed = true
                model.label = "after-press"
            }
            .accessibilityIdentifier("Spike.Button")
        }
        .frame(width: 300, height: 200)
    }
}

/// Selector-based AX surface reader for SwiftUI's private node class.
@MainActor
private enum AXProbe {
    static func string(_ obj: NSObject, _ selector: String) -> String {
        let sel = NSSelectorFromString(selector)
        guard obj.responds(to: sel), let result = obj.perform(sel) else { return "" }
        return (result.takeUnretainedValue() as? String) ?? ""
    }

    static func children(_ obj: NSObject) -> [NSObject] {
        let sel = NSSelectorFromString("accessibilityChildren")
        guard obj.responds(to: sel), let result = obj.perform(sel) else { return [] }
        return (result.takeUnretainedValue() as? [NSObject]) ?? []
    }

    static func find(_ obj: NSObject, id target: String) -> NSObject? {
        if string(obj, "accessibilityIdentifier") == target { return obj }
        for child in children(obj) {
            if let hit = find(child, id: target) { return hit }
        }
        return nil
    }

    /// (identifier, role, text) triples in depth-first order — the same
    /// currency the bash flows' `see_main` JSON captures.
    static func walk(_ obj: NSObject, into out: inout [(id: String, role: String, text: String)]) {
        let id = string(obj, "accessibilityIdentifier")
        let role = string(obj, "accessibilityRole")
        var text = ""
        let valueSel = NSSelectorFromString("accessibilityValue")
        if obj.responds(to: valueSel), let value = obj.perform(valueSel) {
            text = (value.takeUnretainedValue() as? String) ?? ""
        }
        if text.isEmpty { text = string(obj, "accessibilityLabel") }
        if !id.isEmpty || !text.isEmpty {
            out.append((id: id, role: role, text: text))
        }
        for child in children(obj) {
            walk(child, into: &out)
        }
    }
}

@MainActor
@Suite("In-process AX spike")
struct InProcessAXSpikeTests {
    @Test("Never-shown window exposes AX identifiers and presses in-process")
    func offscreenHostExposesAXAndPress() {
        let model = SpikeModel()
        let host = NSHostingView(rootView: SpikeView(model: model))
        host.frame = CGRect(x: 0, y: 0, width: 300, height: 200)
        // The window is created but never ordered in: no screen presence,
        // no foreground, no Dock churn — the property the sink depends on.
        let window = NSWindow(
            contentRect: host.frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        window.contentView = host
        host.layoutSubtreeIfNeeded()

        // Ingredient 1: simulate an attached assistive client, then give
        // SwiftUI one runloop turn to build its AccessibilityNode tree.
        NSApplication.shared.setValue(true, forKey: "accessibilityEnhancedUserInterface")
        RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.1))

        var tree: [(id: String, role: String, text: String)] = []
        AXProbe.walk(host, into: &tree)
        let ids = tree.map(\.id)
        #expect(ids.contains("Spike.Label"), "label identifier missing from AX tree: \(tree)")
        #expect(ids.contains("Spike.Button"), "button identifier missing from AX tree: \(tree)")
        #expect(tree.contains { $0.text.contains("before-press") },
                "label text missing from AX tree: \(tree)")

        guard let button = AXProbe.find(host, id: "Spike.Button") else {
            Issue.record("button not findable by identifier")
            return
        }
        let pressSel = NSSelectorFromString("accessibilityPerformPress")
        #expect(button.responds(to: pressSel), "AccessibilityNode lost accessibilityPerformPress")
        _ = button.perform(pressSel)
        RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.1))
        #expect(model.pressed, "press action did not reach the SwiftUI action closure")

        host.layoutSubtreeIfNeeded()
        var after: [(id: String, role: String, text: String)] = []
        AXProbe.walk(host, into: &after)
        #expect(after.contains { $0.text.contains("after-press") },
                "state change did not surface in the re-walked AX tree: \(after)")
    }
}
