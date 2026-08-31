import Foundation
import Observation

@MainActor
@Observable
final class VideoGenViewModel {
    enum Mode: String, CaseIterable, Identifiable {
        case text
        case image

        var id: String { rawValue }
        var title: String { self == .text ? "Text" : "Image" }
        var capability: VideoModelCapability {
            self == .text ? .textToVideo : .imageToVideo
        }
    }

    struct ReferenceImage: Equatable {
        let data: Data
        let fileName: String
        let mimeType: String
    }

    var videoModels: [ModelEntry] = []
    var catalogLoaded = false
    var selectedAlias = ""
    var mode: Mode = .text

    var prompt = ""
    /// Zero is an internal "capabilities not loaded" sentinel. The first
    /// successful capabilities response replaces it with the shortest safe
    /// preset before the control or Generate action becomes available.
    var seconds = 0
    var size = ""
    var seed = Int.random(in: 1...999_999)
    var referenceImage: ReferenceImage?

    var capabilities: VideoCapabilities?
    var jobs: [VideoJob] = []
    var selectedJobID: String?
    var previewURL: URL?
    var isPreparing = false
    var isSubmitting = false
    var isRefreshing = false
    var isLoadingPreview = false
    var errorMessage: String?

    private let server: ServerManager
    private let physicalRAMGB: Double
    @ObservationIgnored private let client: any VideoClientProtocol
    @ObservationIgnored private let catalogLoader: (URL) async -> [ModelEntry]
    @ObservationIgnored private var catalogRefreshGeneration: UInt = 0
    @ObservationIgnored private var previewGeneration: UInt = 0

    init(
        server: ServerManager,
        client: any VideoClientProtocol = VideoClient(),
        physicalRAMGB: Double = MacHardware.detect().physicalRAMGB,
        catalogLoader: @escaping (URL) async -> [ModelEntry] = {
            await ModelCatalog.videoEntries(binary: $0)
        }
    ) {
        self.server = server
        self.client = client
        self.physicalRAMGB = physicalRAMGB
        self.catalogLoader = catalogLoader
    }

    var selectedModel: ModelEntry? {
        videoModels.first { $0.alias == selectedAlias }
    }

    var selectedJob: VideoJob? {
        if let selectedJobID, let job = jobs.first(where: { $0.id == selectedJobID }) {
            return job
        }
        return jobs.first
    }

    var supportedModes: [Mode] {
        guard let model = selectedModel else { return [] }
        return Mode.allCases.filter { candidate in
            model.videoCapabilities.contains(candidate.capability)
                && (capabilities?.modes.contains(candidate.capability) ?? true)
        }
    }

    var sizePresets: [String] { capabilities?.sizePresets ?? [] }
    var durationPresets: [Int] { capabilities?.durationPresets ?? [] }

    var isSelectedModelEligible: Bool {
        selectedModel.map(isModelEligible) ?? false
    }

    var memoryRequirementText: String? {
        guard let minimum = selectedModel?.minimumMemoryGB else { return nil }
        return "Needs at least \(Int(minimum.rounded())) GB unified memory; this Mac has \(Int(physicalRAMGB.rounded())) GB."
    }

    var isServerReady: Bool {
        server.servingAlias == selectedAlias && server.activeBearer != nil
    }

    var canSubmit: Bool {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return isServerReady
            && capabilities != nil
            && supportedModes.contains(mode)
            && !trimmed.isEmpty
            && !size.isEmpty
            && !isSubmitting
            && (mode == .text || referenceImage != nil)
    }

    var hasActiveJobs: Bool {
        jobs.contains { $0.status == .queued || $0.status == .inProgress }
    }

    func isModelEligible(_ model: ModelEntry) -> Bool {
        guard let minimum = model.minimumMemoryGB,
              minimum.isFinite, minimum > 0, physicalRAMGB > 0 else { return false }
        return physicalRAMGB >= minimum
    }

    func refreshCatalog() async {
        catalogRefreshGeneration &+= 1
        let generation = catalogRefreshGeneration
        guard let binary = server.binaryPath else {
            catalogLoaded = true
            videoModels = []
            selectedAlias = ""
            return
        }
        let loaded = await catalogLoader(binary)
        guard !Task.isCancelled, generation == catalogRefreshGeneration else { return }
        let filtered = loaded.filter {
            $0.kind == .video && !$0.videoCapabilities.isEmpty
        }
        let previousAlias = selectedAlias
        videoModels = filtered
        catalogLoaded = true
        let stillValid = filtered.contains { $0.alias == selectedAlias }
        if selectedAlias.isEmpty || !stillValid {
            selectedAlias = (filtered.first { $0.cached && isModelEligible($0) }
                ?? filtered.first(where: isModelEligible)
                ?? filtered.first)?.alias ?? ""
        }
        if previousAlias != selectedAlias {
            selectedModelDidChange()
        } else if !supportedModes.contains(mode) {
            mode = supportedModes.first ?? .text
        }
    }

    func selectModel(_ alias: String) {
        guard selectedAlias != alias else { return }
        selectedAlias = alias
        selectedModelDidChange()
    }

    func selectMode(_ next: Mode) {
        guard supportedModes.contains(next) else { return }
        mode = next
        if next == .text { referenceImage = nil }
    }

    func setReference(_ reference: ReferenceImage?) {
        referenceImage = reference
    }

    func prepareSelectedModel() async {
        guard !isPreparing, let model = selectedModel, isSelectedModelEligible else { return }
        isPreparing = true
        errorMessage = nil
        defer { isPreparing = false }
        let ready = await server.ensureVideoServing(
            alias: model.alias,
            hfPath: model.hfRepo,
            minimumMemoryGB: model.minimumMemoryGB
        )
        guard ready else {
            errorMessage = "Rapid couldn't start this video model. Check the memory notice or server log, then try again."
            return
        }
        await refreshServerData()
    }

    func serverStateDidChange() async {
        if isServerReady {
            if capabilities == nil { await refreshServerData() }
        } else {
            capabilities = nil
        }
    }

    func refreshServerData() async {
        guard isServerReady else { return }
        isRefreshing = true
        defer { isRefreshing = false }
        do {
            let newCapabilities = try await client.capabilities(
                port: server.activePort, bearer: server.activeBearer
            )
            capabilities = newCapabilities
            reconcileControls()
            errorMessage = nil
            do {
                jobs = try await client.list(
                    port: server.activePort, bearer: server.activeBearer, limit: 30
                )
                reconcileSelection()
            } catch {
                // Controls remain usable when history alone is unavailable.
                errorMessage = "Video controls are ready, but recent videos couldn't be loaded."
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func pollJobs() async {
        guard isServerReady, !isRefreshing else { return }
        do {
            let previous = selectedJob?.status
            jobs = try await client.list(
                port: server.activePort, bearer: server.activeBearer, limit: 30
            )
            reconcileSelection()
            if selectedJob?.status == .completed, previous != .completed {
                await loadSelectedPreview()
            }
        } catch {
            // Poll failures are transient during a model stop/restart. The
            // explicit refresh/start actions surface actionable errors.
        }
    }

    func submit() async {
        guard canSubmit else { return }
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let reference = mode == .image ? referenceImage : nil
        isSubmitting = true
        errorMessage = nil
        defer { isSubmitting = false }
        do {
            let job = try await client.create(
                VideoCreateRequest(
                    prompt: trimmed,
                    model: selectedAlias,
                    seconds: seconds,
                    size: size,
                    seed: seed,
                    reference: reference?.data,
                    referenceFileName: reference?.fileName,
                    referenceMIMEType: reference?.mimeType
                ),
                port: server.activePort,
                bearer: server.activeBearer
            )
            jobs.removeAll { $0.id == job.id }
            jobs.insert(job, at: 0)
            selectedJobID = job.id
            previewURL = nil
            prompt = ""
            seed = Int.random(in: 1...999_999)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func selectJob(_ id: String) async {
        selectedJobID = id
        previewURL = nil
        await loadSelectedPreview()
    }

    func loadSelectedPreview() async {
        guard let job = selectedJob, job.status == .completed, isServerReady else {
            previewURL = nil
            return
        }
        previewGeneration &+= 1
        let generation = previewGeneration
        isLoadingPreview = true
        defer { if generation == previewGeneration { isLoadingPreview = false } }
        do {
            let url = try await client.content(
                id: job.id, port: server.activePort, bearer: server.activeBearer
            )
            guard generation == previewGeneration else { return }
            previewURL = url
        } catch {
            guard generation == previewGeneration else { return }
            errorMessage = error.localizedDescription
        }
    }

    func delete(_ job: VideoJob) async {
        guard isServerReady, job.status != .inProgress else { return }
        do {
            try await client.delete(
                id: job.id, port: server.activePort, bearer: server.activeBearer
            )
            jobs.removeAll { $0.id == job.id }
            if selectedJobID == job.id {
                selectedJobID = jobs.first?.id
                previewURL = nil
                await loadSelectedPreview()
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func selectedModelDidChange() {
        capabilities = nil
        jobs = []
        selectedJobID = nil
        previewURL = nil
        referenceImage = nil
        errorMessage = nil
        if !supportedModes.contains(mode) { mode = supportedModes.first ?? .text }
    }

    private func reconcileControls() {
        let modes = supportedModes.filter {
            capabilities?.modes.contains($0.capability) == true
        }
        if !modes.contains(mode) {
            mode = modes.first ?? supportedModes.first ?? .text
            if mode == .text { referenceImage = nil }
        }
        if !durationPresets.contains(seconds) {
            seconds = durationPresets.first ?? capabilities?.limits.seconds.default ?? 1
        }
        if !sizePresets.contains(size) { size = sizePresets.first ?? "" }
    }

    private func reconcileSelection() {
        guard !jobs.isEmpty else {
            selectedJobID = nil
            previewURL = nil
            return
        }
        if selectedJobID == nil || !jobs.contains(where: { $0.id == selectedJobID }) {
            selectedJobID = jobs.first?.id
        }
    }
}
