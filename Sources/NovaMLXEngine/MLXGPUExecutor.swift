import Darwin
import Foundation
import NovaMLXCore
import NovaMLXUtils

/// One pthread for MLX load and prefill.
///
/// `mlx-swift` 0.31's `Stream.gpu` is `mlx_default_gpu_stream_new()`, which
/// registers a Metal command encoder in that thread's map
/// (`new_stream`, not `new_thread_unsafe_stream`). `get_command_encoder`
/// throws `There is no Stream(gpu, N) in current thread` when eval runs
/// elsewhere. A serial `DispatchQueue` does not stay on one pthread: after
/// load the queue goes idle, and the chat job can resume on a different
/// thread. `86f265b` plus a local `UnownedJob` stash still died that way
/// (`Stream(gpu, 17)`) after the weights had been paged in.
///
/// `ExecutorJob` is noncopyable and cannot be captured by an escaping
/// closure. `UnownedJob` is the handoff that compiles on Mac Swift 6.
/// Visible to `NovaMLXWorker` (another target). Not a client API:
/// `NovaMLXEngine` is a library product, so this stays `package`.
package final class MLXGPUExecutor: TaskExecutor, @unchecked Sendable {
    package static let shared: MLXGPUExecutor = {
        let executor = MLXGPUExecutor()
        executor.start()
        return executor
    }()

    private let cond = NSCondition()
    private var blocks: [() -> Void] = []
    private var owner: pthread_t?
    private var started = false

    private init() {}

    private func start() {
        let thread = Thread { [self] in
            self.threadMain()
        }
        thread.name = "novamlx.worker.mlx"
        // Layer eval walks a deep Swift stack. The 512 KiB default is tight.
        thread.stackSize = 8 * 1024 * 1024
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    private func threadMain() {
        cond.lock()
        owner = pthread_self()
        started = true
        cond.broadcast()
        cond.unlock()
        NovaMLXLog.info("[MLXGPU] thread started pthread=\(MLXGPUExecutor.pthreadLabel())")
        cond.lock()
        while true {
            while blocks.isEmpty {
                cond.wait()
            }
            let block = blocks.removeFirst()
            cond.unlock()
            block()
            cond.lock()
        }
    }

    package func enqueue(_ job: consuming ExecutorJob) {
        let unowned = UnownedJob(job)
        let executor = asUnownedTaskExecutor()
        schedule {
            unowned.runSynchronously(on: executor)
        }
    }

    /// Run `body` on the MLX pthread. Re-enters inline when the caller
    /// already is that pthread, so a job cannot deadlock on itself.
    package func sync<T>(_ body: @escaping () -> T) -> T {
        if ownsCurrentThread() {
            return body()
        }
        let box = SyncBox<T>()
        let done = NSCondition()
        var finished = false
        schedule {
            box.value = body()
            done.lock()
            finished = true
            done.signal()
            done.unlock()
        }
        done.lock()
        while !finished {
            done.wait()
        }
        done.unlock()
        return box.value!
    }

    /// Suspend once so the task resumes on this executor.
    ///
    /// `withTaskExecutorPreference` and `Task(executorPreference:)` keep
    /// running on the caller until the first await. MLX before that await
    /// would bind `Stream.gpu` to the wrong pthread.
    package func resumeOnPinnedThread() async {
        // Already on the MLX pthread: returning without suspending keeps
        // the caller there. A second hop would queue a resume behind the
        // job that is running.
        if ownsCurrentThread() {
            return
        }
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            schedule {
                cont.resume()
            }
        }
    }

    package static func pthreadLabel() -> String {
        String(describing: pthread_self())
    }

    private func ownsCurrentThread() -> Bool {
        cond.lock()
        let owner = self.owner
        let ready = started
        cond.unlock()
        guard ready, let owner else { return false }
        return pthread_equal(owner, pthread_self()) != 0
    }

    private func schedule(_ block: @escaping () -> Void) {
        cond.lock()
        while !started {
            cond.wait()
        }
        blocks.append(block)
        cond.signal()
        cond.unlock()
    }
}

private final class SyncBox<T>: @unchecked Sendable {
    var value: T?
}
