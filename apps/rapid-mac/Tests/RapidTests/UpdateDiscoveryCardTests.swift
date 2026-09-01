import Testing
@testable import Rapid

@MainActor
@Suite("Update discovery card presentation")
struct UpdateDiscoveryCardTests {
    @Test("A new actionable release is presented")
    func presentsNewRelease() {
        #expect(ContentView.shouldPresentUpdateCard(
            releaseVersion: "0.13.4",
            dismissedVersion: "",
            handedOffVersion: nil,
            onboardingVisible: false,
            blockingOverlayVisible: false,
            hasAction: true
        ))
    }

    @Test("Dismissal applies only to the dismissed version")
    func dismissalIsVersionScoped() {
        #expect(!ContentView.shouldPresentUpdateCard(
            releaseVersion: "0.13.4",
            dismissedVersion: "0.13.4",
            handedOffVersion: nil,
            onboardingVisible: false,
            blockingOverlayVisible: false,
            hasAction: true
        ))
        #expect(ContentView.shouldPresentUpdateCard(
            releaseVersion: "0.13.5",
            dismissedVersion: "0.13.4",
            handedOffVersion: nil,
            onboardingVisible: false,
            blockingOverlayVisible: false,
            hasAction: true
        ))
    }

    @Test("Sparkle hand-off suppresses the duplicate card for this session")
    func handoffSuppressesDuplicateSurface() {
        #expect(!ContentView.shouldPresentUpdateCard(
            releaseVersion: "0.13.4",
            dismissedVersion: "",
            handedOffVersion: "0.13.4",
            onboardingVisible: false,
            blockingOverlayVisible: false,
            hasAction: true
        ))
    }

    @Test("Onboarding and blocking overlays defer presentation")
    func defersForHigherPrioritySurfaces() {
        for state in [(true, false), (false, true)] {
            #expect(!ContentView.shouldPresentUpdateCard(
                releaseVersion: "0.13.4",
                dismissedVersion: "",
                handedOffVersion: nil,
                onboardingVisible: state.0,
                blockingOverlayVisible: state.1,
                hasAction: true
            ))
        }
    }

    @Test("A card without a safe action stays hidden")
    func hidesWithoutAction() {
        #expect(!ContentView.shouldPresentUpdateCard(
            releaseVersion: "0.13.4",
            dismissedVersion: "",
            handedOffVersion: nil,
            onboardingVisible: false,
            blockingOverlayVisible: false,
            hasAction: false
        ))
    }
}
