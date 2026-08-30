#include "blackforge/blackbit/cuda_benchmark.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/blackbit/cuda_low_rank_optimizer.hpp"
#include "blackforge/blackbit/cuda_model.hpp"

namespace blackforge::blackbit::cuda {

namespace {

std::string mib(std::size_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MiB";
    return out.str();
}

std::string gib(std::size_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)
        << " GiB";
    return out.str();
}

}  // namespace

std::string BenchmarkResult::report() const {
    std::ostringstream out;
    out << "BlackBit CUDA benchmark — " << config.name << "\n";
    out << "  GPU fisica cuda:" << MemoryTelemetry::instance().device() << ", seq " << options.seqLen
        << ", micro-batch " << options.microBatch << ", passi " << options.steps;
    if (instantiateOnly) out << " (INSTANTIATE ONLY: nessun forward/update)";
    out << "\n\n";
    out << "Parametri\n";
    out << "  totali                         " << totalParameters << "\n";
    out << "  attivi per token               " << activeParameters << "\n\n";
    out << "Memoria GPU persistente (MISURATO)\n";
    out << "  packed ternary weights         " << mib(packedBytes) << "\n";
    out << "  scales                         " << mib(scaleBytes) << "\n";
    out << "  router/norm                    " << mib(denseParameterBytes) << "\n";
    out << "  optimizer                      " << mib(optimizerBytes) << "\n";
    out << "  CUDA context/external baseline " << mib(memory.baselineDeviceUsedBytes) << "\n\n";
    out << "Picchi GPU (MISURATO)\n";
    for (std::size_t index = 0; index < kMemoryArenaCount; ++index) {
        const auto arena = static_cast<MemoryArena>(index);
        out << "  " << std::left << std::setw(29) << memoryArenaName(arena) << std::right
            << mib(arenaPeakBytes[index]) << "\n";
    }
    out << "  BlackForge accounted peak      " << gib(memory.blackForgePeakBytes) << "\n";
    out << "  ACTUAL NVIDIA DEVICE PEAK      " << gib(memory.devicePeakUsedBytes) << "\n";
    out << "  actual NVIDIA device idle/last " << gib(memory.usedBytes) << "\n";
    out << "  actual device free             " << gib(memory.freeBytes) << "\n";
    out << "  prediction for same shape      " << gib(estimate.total()) << " (PREVISTO)\n";
    out << "  hard max_vram_mb               " << options.maxVramMb << " MiB\n\n";
    out << "Tempi (MISURATO)\n";
    out << "  forward + loss + backward      " << std::fixed << std::setprecision(2) << forwardBackwardMs
        << " ms/step\n";
    out << "  optimizer                      " << optimizerMs << " ms/step\n";
    out << "  total                          " << (forwardBackwardMs + optimizerMs) << " ms/step\n";
    out << "  throughput                     " << std::setprecision(3) << tokensPerSecond << " tokens/s\n\n";
    out << "Addestramento (MISURATO)\n";
    out << "  loss finale                    " << finalLoss << "\n";
    out << "  routing entropy                " << routingEntropy << " nat\n";
    out << "  max expert utilization         " << (100.0 * maxExpertUtilization) << " %\n";
    out << "  dropped assignments            " << droppedAssignments << "\n";
    out << "  trit flips                     " << ternaryFlips << "\n";
    out << "  gradient peak                  " << mib(gradientPeakBytes) << "\n";
    out << "  gradient produced/reused       " << mib(cumulativeGradientBytes) << "\n";
    out << "  NaN/Inf count                  " << nanInfCount << "\n\n";
    out << "Verifiche da oggetti/allocazioni reali\n";
    out << "  TERNARY PARAMETERS CHANGED: "
        << (instantiateOnly ? "NOT TESTED (instantiate-only)" : (ternaryFlips > 0 ? "YES" : "NO")) << "\n";
    out << "  FULL PRECISION MASTER COPY: " << (fullPrecisionMasterCopy ? "YES" : "NO") << "\n";
    out << "  FULL MODEL GRADIENT BUFFER: " << (fullModelGradientBuffer ? "YES" : "NO") << "\n";
    out << "  PEAK GPU MEMORY < " << options.maxVramMb << " MiB: " << (withinBudget ? "YES" : "NO") << "\n";
    if (milestoneH) {
        out << "\n================================\n"
            << "  BLACKBIT MILESTONE H PASSED\n"
            << "================================\n";
    }
    return out.str();
}

BenchmarkResult runBenchmark(const BlackBitConfig& config, const BenchmarkOptions& options, int device,
                             std::ostream* progress) {
    config.validate();
    if (options.seqLen == 0 || options.microBatch == 0 ||
        options.seqLen > config.maxSeqLen) {
        throw std::invalid_argument("CUDA BlackBit benchmark: invalid training shape or step count");
    }
    BenchmarkResult result;
    result.config = config;
    result.options = options;
    result.instantiateOnly = options.steps == 0;
    const ParameterCount parameters = countParameters(config);
    result.totalParameters = parameters.total();
    result.activeParameters = countActiveParameters(config);
    LowMemoryOptions estimateOptions;
    estimateOptions.optimizerRank = options.optimizer.rank;
    estimateOptions.activationCheckpointing = options.runtime.recompute != ActivationRecompute::None;
    result.estimate = estimateTrainingMemory(config, {options.microBatch, options.seqLen}, estimateOptions);
    if (options.dryRun) throw std::invalid_argument("CUDA BlackBit benchmark: --dry-run is CPU estimator-only");

    MemoryTelemetry& telemetry = MemoryTelemetry::instance();
    const std::size_t limit = options.maxVramMb == 0 ? 0 : options.maxVramMb * 1024ULL * 1024ULL;
    telemetry.initialize(device, limit);
    if (telemetry.currentTotal() == 0) telemetry.resetAccounting();

    blackforge::blackbit::BlackBitModel cpuReference(config, options.seed);
    cpuReference.setComputeDType(ComputeDType::BF16);
    cpuReference.setRuntimeOptions(options.runtime);
    {
        cuda::BlackBitModel model(cpuReference);
        LowRankProjectedOptimizer optimizer(options.optimizer);
        model.registerParameters(optimizer);
        result.packedBytes = telemetry.current(MemoryArena::PackedWeights);
        result.scaleBytes = telemetry.current(MemoryArena::Scales);
        result.denseParameterBytes = telemetry.current(MemoryArena::DenseParameters);
        result.optimizerBytes = optimizer.stateBytes();

        const std::size_t tokens = options.microBatch * options.seqLen;
        std::mt19937 random(options.seed);
        std::vector<int> tokenIds(tokens);
        std::vector<int> targets(tokens);
        for (std::size_t index = 0; index < tokens; ++index) {
            tokenIds[index] = static_cast<int>(random() % config.vocabSize);
            targets[index] = static_cast<int>(random() % config.vocabSize);
        }
        for (std::size_t step = 0; step < options.warmupSteps && options.steps != 0; ++step) {
            (void)model.trainStep(tokenIds, targets, options.microBatch, options.seqLen, &optimizer);
            optimizer.endStep();
        }
        resetGradientLifetimeStats();
        telemetry.resetPeaks();
        double forwardBackwardTotal = 0.0;
        double optimizerTotal = 0.0;
        for (std::size_t step = 0; step < options.steps; ++step) {
            BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());
            const auto before = std::chrono::steady_clock::now();
            const BlackBitStepResult stepResult =
                model.trainStep(tokenIds, targets, options.microBatch, options.seqLen, &optimizer);
            BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());
            const auto afterBackward = std::chrono::steady_clock::now();
            optimizer.endStep();
            BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());
            const auto afterOptimizer = std::chrono::steady_clock::now();
            forwardBackwardTotal += std::chrono::duration<double, std::milli>(afterBackward - before).count();
            optimizerTotal += std::chrono::duration<double, std::milli>(afterOptimizer - afterBackward).count();
            result.finalLoss = stepResult.loss;
            result.routingEntropy = stepResult.meanRoutingEntropy();
            result.maxExpertUtilization = stepResult.maxExpertUtilization();
            result.droppedAssignments = stepResult.droppedAssignments();
            result.nanInfCount += optimizer.stats().lastStep.nanInfCount +
                                  (stepResult.sawNaN || stepResult.sawInf ? 1U : 0U);
            result.memory = telemetry.snapshot();
            if (progress != nullptr) {
                *progress << "  CUDA step " << (step + 1) << "/" << options.steps << ": loss "
                          << stepResult.loss << ", actual peak " << gib(result.memory.devicePeakUsedBytes)
                          << ", flips " << optimizer.stats().lastStep.flips << "\n";
            }
        }
        if (options.steps != 0) {
            result.forwardBackwardMs = forwardBackwardTotal / options.steps;
            result.optimizerMs = optimizerTotal / options.steps;
            result.tokensPerSecond = static_cast<double>(tokens * options.steps) /
                                     ((forwardBackwardTotal + optimizerTotal) / 1000.0);
        }
        result.ternaryFlips = optimizer.stats().totalFlips;
        result.gradientPeakBytes = gradientLifetimeStats().peakLiveBytes;
        result.cumulativeGradientBytes = gradientLifetimeStats().cumulativeBytes;
        result.memory = telemetry.snapshot();
        for (std::size_t index = 0; index < kMemoryArenaCount; ++index) {
            result.arenaPeakBytes[index] = telemetry.peak(static_cast<MemoryArena>(index));
        }
        result.fullPrecisionMasterCopy =
            telemetry.current(MemoryArena::PackedWeights) + telemetry.current(MemoryArena::Scales) >=
            parameters.ternary() * sizeof(float);
        result.fullModelGradientBuffer = result.gradientPeakBytes >= parameters.total() * sizeof(float) / 2;
        result.withinBudget = limit == 0 || result.memory.devicePeakUsedBytes < limit;
        result.milestoneH = parameters.total() >= 9000000000ULL && options.seqLen >= 16 &&
                            options.steps > 0 && result.ternaryFlips > 0 && result.nanInfCount == 0 &&
                            !result.fullPrecisionMasterCopy && !result.fullModelGradientBuffer &&
                            result.withinBudget;
    }
    return result;
}

std::vector<BenchmarkResult> runBenchmarkLadder(const BlackBitConfig& config,
                                                const BenchmarkOptions& baseOptions,
                                                const std::vector<std::size_t>& sequenceLengths,
                                                int device, std::ostream* progress) {
    config.validate();
    if (sequenceLengths.empty() || baseOptions.microBatch == 0 || baseOptions.steps == 0 ||
        baseOptions.dryRun) {
        throw std::invalid_argument("CUDA BlackBit ladder: lengths and measured steps are required");
    }
    for (const std::size_t seq : sequenceLengths) {
        if (seq == 0 || seq > config.maxSeqLen) {
            throw std::invalid_argument("CUDA BlackBit ladder: sequence length exceeds model limit");
        }
    }
    MemoryTelemetry& telemetry = MemoryTelemetry::instance();
    const std::size_t limit = baseOptions.maxVramMb == 0 ? 0 : baseOptions.maxVramMb * 1024ULL * 1024ULL;
    telemetry.initialize(device, limit);
    if (telemetry.currentTotal() == 0) telemetry.resetAccounting();
    blackforge::blackbit::BlackBitModel cpuReference(config, baseOptions.seed);
    cpuReference.setComputeDType(ComputeDType::BF16);
    cpuReference.setRuntimeOptions(baseOptions.runtime);
    cuda::BlackBitModel model(cpuReference);
    LowRankProjectedOptimizer optimizer(baseOptions.optimizer);
    model.registerParameters(optimizer);
    const ParameterCount parameters = countParameters(config);
    std::vector<BenchmarkResult> results;
    results.reserve(sequenceLengths.size());
    for (const std::size_t seq : sequenceLengths) {
        BenchmarkOptions options = baseOptions;
        options.seqLen = seq;
        BenchmarkResult result;
        result.config = config;
        result.options = options;
        result.totalParameters = parameters.total();
        result.activeParameters = countActiveParameters(config);
        LowMemoryOptions estimateOptions;
        estimateOptions.optimizerRank = options.optimizer.rank;
        estimateOptions.activationCheckpointing = options.runtime.recompute != ActivationRecompute::None;
        result.estimate = estimateTrainingMemory(config, {options.microBatch, seq}, estimateOptions);
        result.packedBytes = telemetry.current(MemoryArena::PackedWeights);
        result.scaleBytes = telemetry.current(MemoryArena::Scales);
        result.denseParameterBytes = telemetry.current(MemoryArena::DenseParameters);
        result.optimizerBytes = optimizer.stateBytes();
        const std::size_t tokens = options.microBatch * seq;
        std::mt19937 random(options.seed + static_cast<unsigned int>(seq));
        std::vector<int> tokenIds(tokens);
        std::vector<int> targets(tokens);
        for (std::size_t index = 0; index < tokens; ++index) {
            tokenIds[index] = static_cast<int>(random() % config.vocabSize);
            targets[index] = static_cast<int>(random() % config.vocabSize);
        }
        for (std::size_t warmup = 0; warmup < options.warmupSteps; ++warmup) {
            (void)model.trainStep(tokenIds, targets, options.microBatch, seq, &optimizer);
            optimizer.endStep();
        }
        const std::size_t flipsBefore = optimizer.stats().totalFlips;
        resetGradientLifetimeStats();
        telemetry.resetPeaks();
        double forwardBackwardTotal = 0.0;
        double optimizerTotal = 0.0;
        for (std::size_t step = 0; step < options.steps; ++step) {
            BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());
            const auto before = std::chrono::steady_clock::now();
            const BlackBitStepResult stepResult =
                model.trainStep(tokenIds, targets, options.microBatch, seq, &optimizer);
            BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());
            const auto afterBackward = std::chrono::steady_clock::now();
            optimizer.endStep();
            BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());
            const auto afterOptimizer = std::chrono::steady_clock::now();
            forwardBackwardTotal += std::chrono::duration<double, std::milli>(afterBackward - before).count();
            optimizerTotal += std::chrono::duration<double, std::milli>(afterOptimizer - afterBackward).count();
            result.finalLoss = stepResult.loss;
            result.routingEntropy = stepResult.meanRoutingEntropy();
            result.maxExpertUtilization = stepResult.maxExpertUtilization();
            result.droppedAssignments = stepResult.droppedAssignments();
            result.nanInfCount += optimizer.stats().lastStep.nanInfCount +
                                  (stepResult.sawNaN || stepResult.sawInf ? 1U : 0U);
        }
        result.forwardBackwardMs = forwardBackwardTotal / options.steps;
        result.optimizerMs = optimizerTotal / options.steps;
        result.tokensPerSecond = static_cast<double>(tokens * options.steps) /
                                 ((forwardBackwardTotal + optimizerTotal) / 1000.0);
        result.ternaryFlips = optimizer.stats().totalFlips - flipsBefore;
        result.gradientPeakBytes = gradientLifetimeStats().peakLiveBytes;
        result.cumulativeGradientBytes = gradientLifetimeStats().cumulativeBytes;
        result.memory = telemetry.snapshot();
        for (std::size_t index = 0; index < kMemoryArenaCount; ++index) {
            result.arenaPeakBytes[index] = telemetry.peak(static_cast<MemoryArena>(index));
        }
        result.fullPrecisionMasterCopy =
            telemetry.current(MemoryArena::PackedWeights) + telemetry.current(MemoryArena::Scales) >=
            parameters.ternary() * sizeof(float);
        result.fullModelGradientBuffer = result.gradientPeakBytes >= parameters.total() * sizeof(float) / 2;
        result.withinBudget = limit == 0 || result.memory.devicePeakUsedBytes < limit;
        result.milestoneH = parameters.total() >= 9000000000ULL && seq >= 16 &&
                            result.ternaryFlips > 0 && result.nanInfCount == 0 &&
                            !result.fullPrecisionMasterCopy && !result.fullModelGradientBuffer &&
                            result.withinBudget;
        results.push_back(result);
        if (progress != nullptr) *progress << "\n" << results.back().report() << std::flush;
    }
    return results;
}

}  // namespace blackforge::blackbit::cuda
