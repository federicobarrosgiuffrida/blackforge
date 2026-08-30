#pragma once

#include <cstddef>
#include <ostream>
#include <string>

#include "blackforge/blackbit/benchmark.hpp"
#include "blackforge/blackbit/cuda_memory.hpp"

namespace blackforge::blackbit::cuda {

struct BenchmarkResult {
    BlackBitConfig config;
    BenchmarkOptions options;
    MemoryEstimate estimate;
    DeviceMemorySnapshot memory;
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

    [[nodiscard]] std::string report() const;
};

BenchmarkResult runBenchmark(const BlackBitConfig& config, const BenchmarkOptions& options,
                             int device = 0, std::ostream* progress = nullptr);

}  // namespace blackforge::blackbit::cuda
