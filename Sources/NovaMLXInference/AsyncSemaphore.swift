import Foundation

/// Simple async-safe semaphore for serializing async work.
final class AsyncSemaphore: @unchecked Sendable {
    private var count: Int
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private let lock = NSLock()

    init(value: Int = 1) {
        self.count = value
    }

    func wait() async {
        // Try to acquire synchronously first
        if tryAcquire() { return }

        // Must wait — suspend until signal() wakes us
        await withCheckedContinuation { continuation in
            lock.lock()
            // Double-check after acquiring lock
            if count > 0 {
                count -= 1
                lock.unlock()
                continuation.resume()
                return
            }
            waiters.append(continuation)
            lock.unlock()
        }
    }

    func signal() {
        lock.lock()
        if let waiter = waiters.isEmpty ? nil : waiters.removeFirst() {
            lock.unlock()
            waiter.resume()
        } else {
            count += 1
            lock.unlock()
        }
    }

    /// Sync fast-path acquisition. Returns true if permit acquired.
    private func tryAcquire() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if count > 0 {
            count -= 1
            return true
        }
        return false
    }
}
