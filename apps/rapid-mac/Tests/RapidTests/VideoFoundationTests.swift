import Foundation
import Testing
@testable import Rapid

@Suite("Video foundation")
struct VideoFoundationTests {
    @Test("Video JSON exposes exact request modes and memory floor")
    func parsesMachineReadableVideoCapabilities() throws {
        let output = """
        {"text":[],"audio":[],"image":[],"video":[
          {"alias":"wan-ti2v","hf_path":"org/wan","video_modes":["text-to-video","image-to-video"],"min_memory_gb":32},
          {"alias":"ltx-t2v","hf_path":"org/ltx","video_modes":["text-to-video"],"min_memory_gb":24},
          {"alias":"unknown-mode","hf_path":"org/bad","video_modes":["video-to-video"],"min_memory_gb":24},
          {"alias":"duplicate-mode","hf_path":"org/bad","video_modes":["text-to-video","text-to-video"],"min_memory_gb":24},
          {"alias":"boolean-floor","hf_path":"org/bad","video_modes":["text-to-video"],"min_memory_gb":true},
          {"alias":"missing-floor","hf_path":"org/bad","video_modes":["text-to-video"]}
        ]}
        """

        let rows = ModelCatalog.parseVideoRowsJSON(output)
        #expect(rows.count == 2)
        let wan = try #require(rows.first { $0.alias == "wan-ti2v" })
        #expect(wan.hfRepo == "org/wan")
        #expect(wan.capabilities == [.textToVideo, .imageToVideo])
        #expect(wan.minimumMemoryGB == 32)
        #expect(rows.first { $0.alias == "ltx-t2v" }?.capabilities == [.textToVideo])
    }

    @Test("Video entries join machine-readable catalog to complete cache rows")
    func videoEntriesResolveCacheByRepository() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-video-catalog-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let binary = directory.appendingPathComponent("rapid-mlx")
        let script = """
        #!/bin/sh
        if [ "$1" = "models" ]; then
          printf '%s' '{"text":[],"audio":[],"image":[],"video":[{"alias":"wan-ti2v","hf_path":"org/wan","video_modes":["text-to-video","image-to-video"],"min_memory_gb":32}]}'
        else
          cat <<'EOF'
        Cached models (1 on disk)
        Alias       HF repo   Size
        (unmapped)  org/wan   9.5 GiB
        EOF
        fi
        """
        try script.write(to: binary, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binary.path)

        let entries = await ModelCatalog.videoEntries(binary: binary, hubCacheOverride: nil)
        let entry = try #require(entries.first)
        #expect(entry.kind == .video)
        #expect(entry.cached)
        #expect(entry.sizeOnDisk == "9.5 GiB")
        #expect(entry.videoCapabilities == [.textToVideo, .imageToVideo])
        #expect(entry.minimumMemoryGB == 32)
    }

    @Test("Video artifacts honor HOME isolation")
    func videoArtifactDirectoryHonorsHome() {
        let directory = ApplicationSupportLocator.videoArtifactsDirectory(
            environment: ["HOME": "/tmp/rapid-video-test"]
        )
        #expect(directory.path == "/tmp/rapid-video-test/Library/Application Support/Rapid/VideoArtifacts")
        #expect(ApplicationSupportLocator.videoArtifactsFolderName == "VideoArtifacts")
    }
}
