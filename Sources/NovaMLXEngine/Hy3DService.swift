import CoreGraphics
import Foundation
import Hy3DMLX
import HunyuanPaintMLX
import ImageIO
import MLX
import NovaMLXCore
import NovaMLXUtils

/// Live progress for one 3D job. Same field set as image/video progress so
/// all modalities poll as one object. Phase strings are stable:
/// `queued | loading | shape | paint | export | done`.
public struct Hy3DProgress: Sendable, Equatable {
    public var phase: String
    public var completed: Int
    public var total: Int
    public var percent: Int

    public init(phase: String, completed: Int, total: Int, percent: Int) {
        self.phase = phase
        self.completed = completed
        self.total = total
        self.percent = percent
    }
}

/// One image-to-3D job accepted onto the serial 3D queue.
///
/// Status names are `H3VideoJobStatus` — the same strings image and video
/// jobs answer with on `GET /v1/tasks/{id}`.
public struct Hy3DJob: Sendable {
    public var id: String
    public var status: H3VideoJobStatus
    public var model: String
    /// Optional paint model id (`texture: true` picks one if nil).
    public var paintModel: String?
    public var texture: Bool
    public var seed: UInt64?
    public var octreeResolution: Int
    public var steps: Int?
    public var guidance: Float?
    public var imageData: Data?
    public var error: String?
    public var createdAt: Date
    public var startedAt: Date?
    public var finishedAt: Date?
    /// Live phase/percent, updated by the runner as stages complete.
    public var phase: String
    public var phasePercent: Int
    /// GLB written under `~/.nova/3d/<id>/`.
    public var outputURL: URL?
    public var verts: Int?
    public var faces: Int?
    /// `shape` or `shape+texture` — the stable size tier for usage metering.
    public var sizeTier: String?

    public init(
        id: String,
        status: H3VideoJobStatus,
        model: String,
        paintModel: String? = nil,
        texture: Bool = false,
        seed: UInt64? = nil,
        octreeResolution: Int = 256,
        steps: Int? = nil,
        guidance: Float? = nil,
        imageData: Data? = nil,
        error: String? = nil,
        createdAt: Date,
        startedAt: Date? = nil,
        finishedAt: Date? = nil,
        phase: String = "queued",
        phasePercent: Int = 0,
        outputURL: URL? = nil,
        verts: Int? = nil,
        faces: Int? = nil,
        sizeTier: String? = nil
    ) {
        self.id = id
        self.status = status
        self.model = model
        self.paintModel = paintModel
        self.texture = texture
        self.seed = seed
        self.octreeResolution = octreeResolution
        self.steps = steps
        self.guidance = guidance
        self.imageData = imageData
        self.error = error
        self.createdAt = createdAt
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.phase = phase
        self.phasePercent = phasePercent
        self.outputURL = outputURL
        self.verts = verts
        self.faces = faces
        self.sizeTier = sizeTier
    }
}

/// One loaded Hunyuan3D slot.
///
/// Shape slots keep a resident `ShapeGenerator` (≈1.4–3.6 GB depending on
/// `NOVA_HY3D_QUANT`) until `unload`, mirroring `ImageGenerationContainer`.
/// Paint slots only remember their directory — a paint pipeline costs tens of
/// GB peak, so it is built per job and freed when the job finishes.
final class Hy3DContainer: @unchecked Sendable {
    let modelId: String
    let directory: URL
    let isPaint: Bool
    let generator: ShapeGenerator?

    init(modelId: String, directory: URL, isPaint: Bool, generator: ShapeGenerator?) {
        self.modelId = modelId
        self.directory = directory
        self.isPaint = isPaint
        self.generator = generator
    }

    var isPBR: Bool {
        // paintpbr (paint-large) runs the PBR path; paint-small is rgb.
        modelId.lowercased().contains("pbr") || modelId.lowercased().contains("paint-large")
    }
}

/// Serial image-to-3D queue (Hunyuan3D-2.1 via vendors/hunyuan3d-swift).
/// One sampler at a time, same philosophy as `ImageTaskService`. Status names
/// are `H3VideoJobStatus`; jobs answer on `GET /v1/tasks/{id}` as
/// `task = "3d.generation"` plus the `/v1/3d/generations` aliases.
public final class Hy3DService: @unchecked Sendable {
    /// Finished jobs stay queryable for a day (same TTL family as image tasks).
    public static let jobTTL: TimeInterval = 24 * 60 * 60

    /// Where GLB artifacts land: `<baseDir>/3d/<job id>/output.glb`.
    public static var jobsDir: URL {
        NovaMLXPaths.baseDir.appendingPathComponent("3d", isDirectory: true)
    }

    /// DiT weight quantization applied at load: 0 = fp16, 4/8 = weights-only
    /// (fp16 resident ≈3.6 GB, 4-bit ≈1.4 GB). Overridable via NOVA_HY3D_QUANT.
    public static var quantize: Int {
        if let raw = ProcessInfo.processInfo.environment["NOVA_HY3D_QUANT"], let q = Int(raw), q == 0 || q == 4 || q == 8 {
            return q
        }
        return 0
    }

    private var containers: [String: Hy3DContainer] = [:]
    private let lock = NSLock()
    private var jobs: [String: Hy3DJob] = [:]
    private var runner: Task<Void, Never>?
    private let autostart: Bool

    /// `autostart` is false only in tests that need a queued job to stay pending.
    public init(autostart: Bool = true) {
        self.autostart = autostart
    }

    public static func isValidId(_ id: String) -> Bool {
        id.range(of: "^h3d-[a-zA-Z0-9-]+$", options: .regularExpression) != nil
    }

    // MARK: - Model lifecycle

    /// Load a Hunyuan3D slot directory: shape = single `.safetensors` +
    /// sibling `config.yaml`; paint = diffusers layout (`unet/`, `dinov2/`).
    public func loadModel(from url: URL, config: ModelConfig) async throws {
        let modelId = config.identifier.id
        guard config.identifier.family == .hunyuan3d || config.modelType == .threed else {
            throw NovaMLXError.unsupportedModel("'\(modelId)' is not a Hunyuan3D model")
        }
        if isLoaded(modelId) { return }
        try FileManager.default.createDirectory(at: Self.jobsDir, withIntermediateDirectories: true)

        let isPaint = Self.isPaintDirectory(url)
        let container: Hy3DContainer
        if isPaint {
            NovaMLXLog.info("[Hy3D] Registering paint model \(modelId) at \(url.path) (weights load per job)")
            try Self.ensurePaintPackageLayout(url)
            container = Hy3DContainer(modelId: modelId, directory: url, isPaint: true, generator: nil)
        } else {
            guard let weights = Self.shapeWeightsURL(in: url) else {
                throw NovaMLXError.modelLoadFailed(modelId, underlying: NovaMLXError.apiError("no .safetensors checkpoint under \(url.path)"))
            }
            NovaMLXLog.info("[Hy3D] Loading shape model \(modelId) from \(weights.lastPathComponent) (quantize=\(Self.quantize))")
            let generator = try ShapeGenerator(weightsURL: weights, quantize: Self.quantize)
            container = Hy3DContainer(modelId: modelId, directory: url, isPaint: false, generator: generator)
        }
        lock.withLock {
            containers[modelId] = container
        }
        NovaMLXLog.info("[Hy3D] Model loaded: \(modelId) (\(isPaint ? "paint" : "shape"))")
    }

    public func unload(modelId: String) {
        let removed: Hy3DContainer? = lock.withLock {
            containers.removeValue(forKey: modelId)
        }
        if removed?.isPaint == false {
            MLX.Memory.clearCache()
        }
        NovaMLXLog.info("[Hy3D] Model unloaded: \(modelId)")
    }

    public func isLoaded(_ modelId: String) -> Bool {
        lock.withLock { containers[modelId] != nil }
    }

    public func listLoadedModels() -> [String] {
        lock.withLock { Array(containers.keys) }
    }

    /// Directory of a loaded model (nil if not loaded).
    public func loadedDirectory(modelId: String) -> URL? {
        lock.withLock { containers[modelId]?.directory }
    }

    /// True when the directory is a paint slot (`unet/` + `dinov2/`, no root config.json).
    public static func isPaintDirectory(_ url: URL) -> Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: url.appendingPathComponent("unet/config.json").path)
            && fm.fileExists(atPath: url.appendingPathComponent("dinov2").appendingPathComponent("config.json").path)
            && !fm.fileExists(atPath: url.appendingPathComponent("config.json").path)
    }

    /// The MLX paint repos are flat (`unet/`, `vae/`, `dinov2/`, `realesrgan/`),
    /// but `PaintPipeline` expects the upstream weights-package layout
    /// (`hunyuan3d-paintpbr-v2-1/unet/…`, `hunyuan3d-paint-v2-0/vae/…`,
    /// `dinov2-giant/…`). Bridge the two with idempotent symlinks inside the
    /// model directory so a plain `nova download` works without manual layout
    /// surgery.
    static func ensurePaintPackageLayout(_ dir: URL) throws {
        let fm = FileManager.default
        func bridge(_ packaged: String, to existing: String) throws {
            let link = dir.appendingPathComponent(packaged)
            guard !fm.fileExists(atPath: link.path) else { return }
            try fm.createDirectory(
                at: link.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fm.createSymbolicLink(
                at: link, withDestinationURL: dir.appendingPathComponent(existing))
        }
        try bridge("hunyuan3d-paintpbr-v2-1/unet", to: "unet")
        try bridge("hunyuan3d-paint-v2-0/unet", to: "unet")
        try bridge("hunyuan3d-paint-v2-0/vae", to: "vae")
        try bridge("dinov2-giant", to: "dinov2")
    }

    private static func shapeWeightsURL(in directory: URL) -> URL? {
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: directory.path) else { return nil }
        return files
            .filter { $0.lowercased().hasSuffix(".safetensors") }
            .sorted()
            .map { directory.appendingPathComponent($0) }
            .first
    }

    // MARK: - Job queue

    @discardableResult
    public func submit(
        model: String,
        paintModel: String? = nil,
        texture: Bool = false,
        image: Data,
        seed: UInt64? = nil,
        octreeResolution: Int = 256,
        steps: Int? = nil,
        guidance: Float? = nil
    ) throws -> Hy3DJob {
        let loaded = lock.withLock { containers }
        guard let shape = loaded[model], !shape.isPaint else {
            throw NovaMLXError.modelNotFound("3D shape model '\(model)' is not loaded")
        }
        if texture {
            let resolved = paintModel ?? Self.defaultPaintModel(containers: loaded)
            guard let resolved, loaded[resolved]?.isPaint == true else {
                throw NovaMLXError.apiError(
                    "texture=true needs a loaded paint model; load one (e.g. zimengxiong/hunyuan3d-mlx-paint-large) or pass paint_model"
                )
            }
        }
        let job = Hy3DJob(
            id: "h3d-" + UUID().uuidString.lowercased(),
            status: .pending,
            model: model,
            paintModel: paintModel,
            texture: texture,
            seed: seed,
            octreeResolution: octreeResolution,
            steps: steps,
            guidance: guidance,
            imageData: image,
            createdAt: Date()
        )
        lock.withLock {
            expireLocked(now: Date())
            jobs[job.id] = job
            if autostart { ensureRunnerLocked() }
        }
        return job
    }

    /// Prefer an explicitly loaded paint slot when the request did not name one.
    private static func defaultPaintModel(containers: [String: Hy3DContainer]) -> String? {
        containers
            .filter { $0.value.isPaint }
            .keys
            .sorted()
            .first
    }

    public func job(id: String) -> Hy3DJob? {
        guard Self.isValidId(id) else { return nil }
        return lock.withLock {
            guard var job = jobs[id] else { return nil }
            if job.status.isTerminal, let finished = job.finishedAt,
               Date().timeIntervalSince(finished) >= Self.jobTTL {
                job.status = .expired
                job.imageData = nil
                jobs[id] = job
            }
            return job
        }
    }

    public func progress(id: String) -> Hy3DProgress? {
        guard Self.isValidId(id) else { return nil }
        return lock.withLock {
            guard let job = jobs[id], job.status == .inProgress else { return nil }
            return Hy3DProgress(phase: job.phase, completed: job.phasePercent, total: 100, percent: min(99, job.phasePercent))
        }
    }

    @discardableResult
    public func cancel(id: String) throws -> Hy3DJob {
        guard Self.isValidId(id) else {
            throw NovaMLXError.modelNotFound("3D job '\(id)'")
        }
        return try lock.withLock {
            guard var job = jobs[id] else {
                throw NovaMLXError.modelNotFound("3D job '\(id)'")
            }
            switch job.status {
            case .pending:
                job.status = .cancelled
                job.finishedAt = Date()
                jobs[id] = job
                return job
            case .inProgress:
                // The runner polls job status between sampler steps and aborts.
                job.error = "cancelled"
                jobs[id] = job
                return job
            case .completed, .failed, .cancelled, .expired:
                return job
            }
        }
    }

    /// GLB for a completed job (nil if absent/expired).
    public func contentURL(id: String) -> URL? {
        guard Self.isValidId(id) else { return nil }
        return lock.withLock { jobs[id]?.outputURL }.flatMap { url in
            FileManager.default.fileExists(atPath: url.path) ? url : nil
        }
    }

    private func ensureRunnerLocked() {
        guard runner == nil else { return }
        runner = Task { [weak self] in
            await self?.runQueue()
        }
    }

    private func runQueue() async {
        while true {
            let next: Hy3DJob? = lock.withLock {
                if let job = jobs.values
                    .filter({ $0.status == .pending })
                    .sorted(by: { $0.createdAt < $1.createdAt })
                    .first {
                    return job
                }
                runner = nil
                return nil
            }
            guard let job = next else { return }
            await run(job)
        }
    }

    private func updatePhase(_ id: String, phase: String, percent: Int) {
        lock.withLock {
            guard var job = jobs[id], job.status == .inProgress else { return }
            job.phase = phase
            job.phasePercent = max(job.phasePercent, min(percent, 99))
            jobs[id] = job
        }
    }

    private func isCancelledJob(_ id: String) -> Bool {
        lock.withLock { jobs[id]?.status == .cancelled || jobs[id]?.error == "cancelled" }
    }

    private func run(_ job: Hy3DJob) async {
        let started = lock.withLock { () -> Bool in
            guard var current = jobs[job.id], current.status == .pending else { return false }
            current.status = .inProgress
            current.startedAt = Date()
            current.phase = "loading"
            jobs[job.id] = current
            return true
        }
        guard started else { return }
        let jobSelf = self
        let id = job.id

        do {
            let container: Hy3DContainer? = lock.withLock { containers[job.model] }
            guard let container, !container.isPaint, let generator = container.generator else {
                throw NovaMLXError.modelNotFound("3D shape model '\(job.model)' is not loaded")
            }
            guard let imageData = job.imageData, let cg = Self.cgImage(from: imageData) else {
                throw NovaMLXError.apiError("Failed to decode input image")
            }

            let jobDir = Self.jobsDir.appendingPathComponent(job.id, isDirectory: true)
            try FileManager.default.createDirectory(at: jobDir, withIntermediateDirectories: true)
            let shapeGLB = jobDir.appendingPathComponent("shape.glb")
            let finalGLB = jobDir.appendingPathComponent("output.glb")

            // ---- shape ----
            let paintModel = job.texture
                ? (job.paintModel ?? Self.defaultPaintModel(containers: lock.withLock { containers }))
                : nil
            let shapeSpan: ClosedRange<Float> = paintModel != nil ? 0.05...0.55 : 0.05...0.95
            let seed = job.seed ?? UInt64.random(in: UInt64.min...UInt64.max)
            let steps = job.steps ?? 30
            let guidance = job.guidance ?? 5.0
            updatePhase(id, phase: "shape", percent: 5)

            // generate() (not generateGLB) so cancellation is polled per denoise step.
            guard let mesh = generator.generate(
                image: cg,
                steps: steps,
                guidance: guidance,
                seed: seed,
                resolution: job.octreeResolution,
                octree: true,
                isCancelled: { jobSelf.isCancelledJob(id) },
                onProgress: { p in
                    let pct = Int((shapeSpan.lowerBound + (shapeSpan.upperBound - shapeSpan.lowerBound) * p.fraction) * 100)
                    jobSelf.updatePhase(id, phase: "shape", percent: pct)
                }
            ) else {
                throw NovaMLXError.apiError("cancelled" as String)
            }
            if isCancelledJob(id) { throw NovaMLXError.apiError("cancelled" as String) }
            try GLB.write(mesh, to: shapeGLB)
            let verts = mesh.vertices.count
            let faces = mesh.faces.count

            // ---- optional paint ----
            if let paintModel {
                let paintContainer: Hy3DContainer? = lock.withLock { containers[paintModel] }
                guard let paintContainer, paintContainer.isPaint else {
                    throw NovaMLXError.modelNotFound("paint model '\(paintModel)' is not loaded")
                }
                updatePhase(id, phase: "paint", percent: 60)
                let inputURL = writeTempImage(imageData)
                let paintDir = paintContainer.directory.path
                let isPBR = paintContainer.isPBR
                let meshPath = shapeGLB.path
                let outPath = finalGLB.path
                try await Task.detached(priority: .userInitiated) {
                    try jobSelf.runPaint(
                        paintDirectory: paintDir,
                        isPBR: isPBR,
                        meshPath: meshPath,
                        imagePath: inputURL.path,
                        outGLB: outPath,
                        onProgress: { fraction in
                            jobSelf.updatePhase(id, phase: "paint", percent: 60 + Int(fraction * 35))
                        }
                    )
                }.value
                try? FileManager.default.removeItem(at: inputURL)
            } else {
                updatePhase(id, phase: "export", percent: 95)
                if FileManager.default.fileExists(atPath: finalGLB.path) {
                    try FileManager.default.removeItem(at: finalGLB)
                }
                try FileManager.default.moveItem(at: shapeGLB, to: finalGLB)
            }
            try? FileManager.default.removeItem(at: shapeGLB)

            lock.withLock {
                guard var current = jobs[id], current.status == .inProgress else { return }
                current.status = .completed
                current.finishedAt = Date()
                current.phase = "done"
                current.phasePercent = 100
                current.outputURL = finalGLB
                current.verts = verts
                current.faces = faces
                current.sizeTier = paintModel != nil ? "shape+texture" : "shape"
                current.imageData = nil
                jobs[id] = current
            }
            NovaMLXLog.info("[Hy3D] Job \(id) completed: \(verts) verts, \(faces) faces -> \(finalGLB.path)")
        } catch {
            let message = String(describing: error)
            let cancelled = message.localizedCaseInsensitiveContains("cancelled")
                || isCancelledJob(id)
            lock.withLock {
                guard var current = jobs[id] else { return }
                if current.status == .cancelled { return }
                current.status = cancelled ? .cancelled : .failed
                current.error = cancelled ? nil : message
                current.finishedAt = Date()
                current.imageData = nil
                jobs[id] = current
            }
            if !cancelled {
                NovaMLXLog.error("[Hy3D] Job \(id) failed: \(message)")
            }
        }
    }

    /// Paint stage. Runs on a detached task; a paint pipeline peaks around
    /// 25–39 GB, so it is built per job and freed when this returns.
    private func runPaint(
        paintDirectory: String,
        isPBR: Bool,
        meshPath: String,
        imagePath: String,
        outGLB: String,
        onProgress: @escaping (Float) -> Void
    ) throws {
        let pipe = PaintPipeline(weightsRoot: paintDirectory)
        if isPBR {
            try pipe.run(meshPath: meshPath, imagePath: imagePath, outGLB: outGLB)
        } else {
            let mesh = loadMesh(meshPath)
            guard let r = try pipe.paintRGB(mesh: mesh, imagePath: imagePath, onProgress: { _, f in onProgress(f) }) else {
                throw NovaMLXError.apiError("paint stage returned no result (UV unwrap failed?)")
            }
            try writeGLB(path: outGLB, vertices: r.vertices, faces: r.faces, uvs: r.uvs,
                     baseColorPNG: r.albedoPNG, metallicRoughnessPNG: nil)
        }
    }

    private func writeTempImage(_ data: Data) -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("hy3d_input_\(UUID().uuidString).png")
        try? data.write(to: url)
        return url
    }

    /// Must be called with `lock` held.
    private func expireLocked(now: Date) {
        for (id, var job) in jobs {
            guard job.status.isTerminal, let finished = job.finishedAt else { continue }
            guard now.timeIntervalSince(finished) >= Self.jobTTL else { continue }
            job.status = .expired
            job.imageData = nil
            jobs[id] = job
        }
    }

    private static func cgImage(from data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
}
