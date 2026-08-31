import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("VideoGenViewModel")
struct VideoGenViewModelTests {
    @Test("Catalog fails closed to explicit Video rows and chooses a model that fits")
    func catalogFilteringAndMemoryChoice() async {
        let chat = ModelEntry(alias: "chat", hfRepo: nil, sizeOnDisk: nil, cached: true)
        let tooLarge = ModelEntry(
            alias: "video-32", hfRepo: "org/large", sizeOnDisk: nil, cached: true,
            kind: .video, videoCapabilities: [.textToVideo], minimumMemoryGB: 32
        )
        let fitting = ModelEntry(
            alias: "video-24", hfRepo: "org/fitting", sizeOnDisk: nil, cached: true,
            kind: .video,
            videoCapabilities: [.textToVideo, .imageToVideo], minimumMemoryGB: 24
        )
        let server = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        let viewModel = VideoGenViewModel(
            server: server,
            client: VideoFakeClient(),
            physicalRAMGB: 24,
            catalogLoader: { _ in [chat, tooLarge, fitting] }
        )

        await viewModel.refreshCatalog()

        #expect(viewModel.videoModels == [tooLarge, fitting])
        #expect(viewModel.selectedAlias == "video-24")
        #expect(viewModel.isSelectedModelEligible)
        #expect(!viewModel.isModelEligible(tooLarge))
        #expect(viewModel.supportedModes == [.text, .image])
    }

    @Test("Live capabilities gate Image mode and submission carries the reference")
    func capabilityDrivenSubmission() async throws {
        let model = ModelEntry(
            alias: "ltx-2.3-mlx-q4", hfRepo: "org/ltx", sizeOnDisk: "9 GB", cached: true,
            kind: .video,
            videoCapabilities: [.textToVideo, .imageToVideo], minimumMemoryGB: 24
        )
        let client = VideoFakeClient()
        let server = ServerManager(
            testingState: .ready(alias: model.alias),
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            activeBearer: "test-bearer"
        )
        let viewModel = VideoGenViewModel(
            server: server,
            client: client,
            physicalRAMGB: 32,
            catalogLoader: { _ in [model] }
        )

        await viewModel.refreshCatalog()
        await viewModel.refreshServerData()
        #expect(viewModel.size == "512x512")
        #expect(viewModel.seconds == 1)

        viewModel.selectMode(.image)
        viewModel.prompt = "Ocean waves moving around a black rock"
        #expect(!viewModel.canSubmit)
        viewModel.setReference(.init(
            data: Data("png".utf8), fileName: "rock.png", mimeType: "image/png"
        ))
        #expect(viewModel.canSubmit)

        await viewModel.submit()

        let requests = await client.recordedRequests()
        let request = try #require(requests.first)
        #expect(request.model == model.alias)
        #expect(request.reference == Data("png".utf8))
        #expect(request.referenceFileName == "rock.png")
        #expect(viewModel.jobs.first?.status == .queued)
        #expect(viewModel.prompt.isEmpty)
    }
}

private actor VideoFakeClient: VideoClientProtocol {
    private var requests: [VideoCreateRequest] = []

    func capabilities(port: Int, bearer: String?) async throws -> VideoCapabilities {
        try JSONDecoder().decode(VideoCapabilities.self, from: Data(Self.capabilitiesJSON.utf8))
    }

    func create(
        _ request: VideoCreateRequest,
        port: Int,
        bearer: String?
    ) async throws -> VideoJob {
        requests.append(request)
        return VideoJob(
            id: "video_0123456789abcdef0123456789abcdef",
            model: request.model,
            prompt: request.prompt,
            seconds: String(request.seconds),
            size: request.size,
            status: .queued,
            progress: 0,
            createdAt: 123,
            completedAt: nil,
            error: nil
        )
    }

    func list(port: Int, bearer: String?, limit: Int) async throws -> [VideoJob] { [] }
    func delete(id: String, port: Int, bearer: String?) async throws {}
    func content(id: String, port: Int, bearer: String?) async throws -> URL {
        URL(fileURLWithPath: "/tmp/\(id).mp4")
    }

    func recordedRequests() -> [VideoCreateRequest] { requests }

    private static let capabilitiesJSON = #"""
    {
      "model":"org/ltx","family":"ltx-2.3",
      "modes":["text-to-video","image-to-video"],
      "limits":{
        "size":{"type":"range","width":{"minimum":256,"maximum":1920,"multiple_of":64},"height":{"minimum":256,"maximum":1920,"multiple_of":64}},
        "seconds":{"minimum":1,"maximum":20,"default":4}
      }
    }
    """#
}
