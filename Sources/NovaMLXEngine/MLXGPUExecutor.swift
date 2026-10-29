import Darwin
import Dispatch
import Foundation

/// One serial queue for MLX load and prefill.
///
/// Metal command encoders live in a thread-local map. A stream created while
/// loading is invisible to `async_eval` on any other thread
/// (`There is no Stream(gpu, N) in current thread`). Task-executor preference
/// on the worker's load/generate handlers does not cover the unstructured
/// `Task` inside `stream()`, so chat was still leaving that thread.
final class MLXGPUExecutor: TaskExecutor, @unchecked Sendable {
    static let shared = MLXGPUExecutor()

    private let queue = DispatchQueue(label: "novamlx.worker.mlx")
    private let key = DispatchSpecificKey<UInt8>()

    private init() {
        queue.setSpecific(key: key, value: 1)
    }

    func enqueue(_ job: consuming ExecutorJob) {
        // ExecutorJob is noncopyable and cannot be captured by queue.async.
        let unowned = UnownedJob(job)
        let executor = asUnownedTaskExecutor()
        queue.async {
            unowned.runSynchronously(on: executor)
        }
    }

    /// Run GPU work on the load queue. Re-enters inline when the caller is
    /// already that queue, so a task-executor job cannot deadlock on sync.
    func sync<T>(_ body: () throws -> T) rethrows -> T {
        if DispatchQueue.getSpecific(key: key) != nil {
            return try body()
        }
        return try queue.sync(execute: body)
    }

    static func pthreadLabel() -> String {
        String(describing: pthread_self())
    }
}
