import Foundation

// The official benchmark runner is primarily for low-level kernel benchmarks (Fused SDPA, etc.).
// For the distributed speculative decoding verification, use the standalone simulator:
//
//   swift DistributedSpeculativeSimulation.swift
//
// The DistributedSpeculativeBench.swift in this module is kept as reference implementation
// of the simulation logic (can be expanded later into a proper --distributed-spec mode
// once the runner is refactored to support lightweight modes).
