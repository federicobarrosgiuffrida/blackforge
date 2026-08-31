#pragma once

#include <array>
#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "blackforge/blackbit/benchmark.hpp"
#include "blackforge/blackbit/cuda_memory.hpp"

namespace blackforge::blackbit::cuda {

struct BenchmarkResult {
    BlackBitConfig config;
    BenchmarkOptions options;
    MemoryEstimate estimate;
    DeviceMemorySnapshot memory;
    std::array<std::size_t, kMemoryArenaCount> arenaPeakBytes{};
    std::size_t totalParameters = 0;
    std::size_t activeParameters = 0;
    std::size_t packedBytes = 0;
    std::size_t scaleBytes = 0;
    std::size_t denseParameterBytes = 0;
    std::size_t optimizerBytes = 0;
    std::size_t gradientPeakBytes = 0;
    std::size_t cumulativeGradientBytes = 0;
    double forwardBackwardMs = 0.0;
    double optimizerMs = 0.0;
    double tokensPerSecond = 0.0;
    float finalLoss = 0.0F;
    double routingEntropy = 0.0;
    double maxExpertUtilization = 0.0;
    std::size_t droppedAssignments = 0;
    std::size_t ternaryFlips = 0;
    std::size_t nanInfCount = 0;
    bool fullPrecisionMasterCopy = false;
    bool fullModelGradientBuffer = false;
    bool withinBudget = false;
    bool instantiateOnly = false;
    bool milestoneH = false;
    // Attribuzione del tempo GPU per fase dell'ULTIMO passo misurato,
    // vuota se il profiler era spento. Un solo passo e non la media:
    // gli eventi vengono azzerati a ogni passo, e mediare richiederebbe
    // di risolverli (quindi sincronizzare) dentro il ciclo cronometrato.
    std::string gpuProfile;

    [[nodiscard]] std::string report() const;
};

BenchmarkResult runBenchmark(const BlackBitConfig& config, const BenchmarkOptions& options,
                             int device = 0, std::ostream* progress = nullptr);

std::vector<BenchmarkResult> runBenchmarkLadder(const BlackBitConfig& config,
                                                const BenchmarkOptions& baseOptions,
                                                const std::vector<std::size_t>& sequenceLengths,
                                                int device = 0, std::ostream* progress = nullptr);

}  // namespace blackforge::blackbit::cuda
