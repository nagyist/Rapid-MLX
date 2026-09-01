import AppKit

// Xcode requires a target application for an UI-testing bundle. The tests
// deliberately launch the already-built production bundle by identifier; this
// host normally remains inert. A native attachment journey can opt into its
// small, deterministic file-drag surface so the production app receives a real
// macOS file-URL drag without depending on Finder's ambient window geometry.
final class FileDragView: NSView, NSDraggingSource {
    private enum DragOutcome: String {
        case pending
        case copy
        case none
    }

    let fileURL: URL
    private var startedDragging = false
    private var dragOutcome = DragOutcome.pending {
        didSet {
            NSAccessibility.post(element: self, notification: .valueChanged)
        }
    }

    init(fileURL: URL) {
        self.fileURL = fileURL
        super.init(frame: .zero)
        setAccessibilityElement(true)
        setAccessibilityRole(.button)
        setAccessibilityLabel(fileURL.lastPathComponent)
        setAccessibilityIdentifier("RapidUITests.FileDragSource")
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { nil }

    override func accessibilityValue() -> Any? { dragOutcome.rawValue }

    override func draw(_ dirtyRect: NSRect) {
        NSColor.windowBackgroundColor.setFill()
        dirtyRect.fill()
        NSWorkspace.shared.icon(forFile: fileURL.path).draw(
            in: bounds.insetBy(dx: 24, dy: 12),
            from: .zero,
            operation: .sourceOver,
            fraction: 1
        )
    }

    override func mouseDown(with event: NSEvent) {
        startedDragging = false
        dragOutcome = .pending
    }

    override func mouseDragged(with event: NSEvent) {
        guard !startedDragging else { return }
        startedDragging = true
        let item = NSDraggingItem(pasteboardWriter: fileURL as NSURL)
        item.setDraggingFrame(bounds, contents: NSWorkspace.shared.icon(forFile: fileURL.path))
        beginDraggingSession(with: [item], event: event, source: self)
    }

    func draggingSession(
        _ session: NSDraggingSession,
        sourceOperationMaskFor context: NSDraggingContext
    ) -> NSDragOperation { .copy }

    func draggingSession(
        _ session: NSDraggingSession,
        endedAt screenPoint: NSPoint,
        operation: NSDragOperation
    ) {
        dragOutcome = operation.contains(.copy) ? .copy : .none
    }
}

let app = NSApplication.shared
if let path = ProcessInfo.processInfo.environment["RAPID_XCUI_DRAG_FILE"] {
    app.setActivationPolicy(.regular)
    let size = NSSize(width: 140, height: 110)
    let visible = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1_440, height: 900)
    let panel = NSPanel(
        contentRect: NSRect(
            x: visible.minX + 20,
            y: visible.maxY - size.height - 20,
            width: size.width,
            height: size.height
        ),
        styleMask: [.titled],
        backing: .buffered,
        defer: false
    )
    panel.title = "Drag attachment"
    panel.contentView = FileDragView(fileURL: URL(fileURLWithPath: path))
    panel.level = .floating
    panel.makeKeyAndOrderFront(nil)
    app.activate(ignoringOtherApps: true)
    app.run()
} else {
    app.setActivationPolicy(.accessory)
    app.terminate(nil)
}
