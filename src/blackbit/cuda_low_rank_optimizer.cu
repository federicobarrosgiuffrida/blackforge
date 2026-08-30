#include "blackforge/blackbit/cuda_low_rank_optimizer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/blackbit/stochastic_round.hpp"
#include "blackforge/blackbit/ternary_update.hpp"

namespace blackforge::blackbit::cuda {

namespace {

constexpr int kBlockSize = 256;

struct DeviceStepStats {
    unsigned long long elements;
    unsigned long long flips;
    unsigned long long positiveFlips;
    unsigned long long negativeFlips;
    unsigned long long saturated;
    unsigned long long nanInf;
    unsigned long long gradientCount;
    unsigned long long optimizerCount;
    unsigned long long updateCount;
    double gradientSquared;
    double optimizerSquared;
    double updateSquared;
};

unsigned int gridFor(std::size_t count) {
    return static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
}

__device__ float projection(std::uint64_t seed, std::uint64_t epoch, std::size_t row, std::size_t component,
                            std::size_t rank) {
    const std::uint64_t mixed = counterRandom(seed ^ (epoch * 0x9E3779B97F4A7C15ULL),
                                               static_cast<std::uint64_t>(row) * rank + component);
    return ((mixed & 1ULL) != 0 ? 1.0F : -1.0F) / sqrtf(static_cast<float>(rank));
}

__global__ void gradientStatsKernel(const float* gradient, std::size_t count, DeviceStepStats* stats) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = gradient[index];
    if (!isfinite(value)) {
        atomicAdd(&stats->nanInf, 1ULL);
        return;
    }
    atomicAdd(&stats->gradientSquared, static_cast<double>(value) * static_cast<double>(value));
    atomicAdd(&stats->gradientCount, 1ULL);
}

__global__ void projectGradientKernel(float* accumulator, const float* gradient, std::size_t firstRow,
                                      std::size_t rowCount, std::size_t cols, std::size_t rank,
                                      std::uint64_t seed, std::uint64_t epoch) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = rank * cols;
    if (index >= count) return;
    const std::size_t component = index / cols;
    const std::size_t col = index % cols;
    float sum = accumulator[index];
    for (std::size_t localRow = 0; localRow < rowCount; ++localRow) {
        sum += projection(seed, epoch, firstRow + localRow, component, rank) *
               gradient[localRow * cols + col];
    }
    accumulator[index] = sum;
}

__global__ void adamDirectionKernel(float* firstMoment, float* secondMoment, float* accumulator,
                                    std::size_t count, float beta1, float beta2, float bias1, float bias2,
                                    float epsilon, DeviceStepStats* stats) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float gradient = accumulator[index];
    const float first = beta1 * firstMoment[index] + (1.0F - beta1) * gradient;
    const float second = beta2 * secondMoment[index] + (1.0F - beta2) * gradient * gradient;
    firstMoment[index] = first;
    secondMoment[index] = second;
    const float direction = (first / bias1) / (sqrtf(second / bias2) + epsilon);
    accumulator[index] = direction;
    if (!isfinite(direction)) {
        atomicAdd(&stats->nanInf, 1ULL);
        return;
    }
    atomicAdd(&stats->optimizerSquared, static_cast<double>(direction) * static_cast<double>(direction));
    atomicAdd(&stats->optimizerCount, 1ULL);
}

__global__ void updatePackedKernel(std::uint32_t* packed, std::size_t rows, std::size_t cols,
                                   std::size_t wordsPerRow, const float* direction, std::size_t rank,
                                   float learningRate, std::uint64_t seed, std::uint64_t epoch,
                                   std::uint64_t step, DeviceStepStats* stats) {
    const std::size_t wordIndex = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t wordCount = rows * wordsPerRow;
    if (wordIndex >= wordCount) return;
    const std::size_t row = wordIndex / wordsPerRow;
    const std::size_t firstCol = (wordIndex % wordsPerRow) * kTritsPerWord;
    std::uint32_t word = packed[wordIndex];
    const std::uint64_t stepSeed = splitMix64(seed ^ (step * 0xD1B54A32D192ED03ULL));

    for (int byte = 0; byte < 4; ++byte) {
        int trits[kTritsPerByte];
        decodeTritByte(wordByte(word, byte), trits);
        for (std::size_t slot = 0; slot < kTritsPerByte; ++slot) {
            const std::size_t col = firstCol + static_cast<std::size_t>(byte) * kTritsPerByte + slot;
            if (col >= cols) continue;
            float update = 0.0F;
            for (std::size_t component = 0; component < rank; ++component) {
                update += -learningRate * projection(seed, epoch, row, component, rank) *
                          direction[component * cols + col];
            }
            if (!isfinite(update)) {
                atomicAdd(&stats->nanInf, 1ULL);
                continue;
            }
            const int oldTrit = trits[slot];
            const float target = static_cast<float>(oldTrit) + update;
            const int newTrit = stochasticRoundToTrit(target, stepSeed, row * cols + col);
            trits[slot] = newTrit;
            atomicAdd(&stats->elements, 1ULL);
            atomicAdd(&stats->updateSquared, static_cast<double>(update) * static_cast<double>(update));
            atomicAdd(&stats->updateCount, 1ULL);
            if (target <= -1.0F || target >= 1.0F) atomicAdd(&stats->saturated, 1ULL);
            if (newTrit != oldTrit) {
                atomicAdd(&stats->flips, 1ULL);
                if (newTrit > oldTrit) {
                    atomicAdd(&stats->positiveFlips, 1ULL);
                } else {
                    atomicAdd(&stats->negativeFlips, 1ULL);
                }
            }
        }
        word = setWordByte(word, byte, encodeTritByte(trits[0], trits[1], trits[2], trits[3], trits[4]));
    }
    packed[wordIndex] = word;
}

}  // namespace

LowRankProjectedOptimizer::LowRankProjectedOptimizer(LowRankOptimizerOptions options)
    : options_(options), deviceStats_(sizeof(DeviceStepStats), MemoryArena::Optimizer) {
    if (options_.rank == 0 || options_.projectionInterval == 0) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: rank and projection interval must be positive");
    }
    BLACKFORGE_CUDA_CHECK(cudaMemset(deviceStats_.data(), 0, deviceStats_.bytes()));
}

void LowRankProjectedOptimizer::allocateState(const std::string& name, State& state,
                                               cuda::TernaryTensor& weight) {
    state.weight = &weight;
    state.rows = weight.rows();
    state.cols = weight.rowLength();
    const auto override = rankOverrides_.find(name);
    state.rank = std::min(override == rankOverrides_.end() ? options_.rank : override->second, state.rows);
    state.seed = options_.seed ^ parameterNameHash(name);
    const std::size_t bytes = state.rank * state.cols * sizeof(float);
    state.firstMoment = Buffer(bytes, MemoryArena::Optimizer);
    state.secondMoment = Buffer(bytes, MemoryArena::Optimizer);
    state.accumulator = Buffer(bytes, MemoryArena::Optimizer);
    BLACKFORGE_CUDA_CHECK(cudaMemset(state.firstMoment.data(), 0, bytes));
    BLACKFORGE_CUDA_CHECK(cudaMemset(state.secondMoment.data(), 0, bytes));
    BLACKFORGE_CUDA_CHECK(cudaMemset(state.accumulator.data(), 0, bytes));
}

void LowRankProjectedOptimizer::registerTernary(const std::string& name, cuda::TernaryTensor& weight) {
    auto [iterator, inserted] = states_.try_emplace(name);
    if (!inserted) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: parameter '" + name + "' already registered");
    }
    allocateState(name, iterator->second, weight);
}

void LowRankProjectedOptimizer::setRankFor(const std::string& name, std::size_t rank) {
    if (rank == 0) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: rank must be positive");
    }
    if (states_.count(name) != 0) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: setRankFor must be called before registration");
    }
    rankOverrides_[name] = rank;
}

LowRankProjectedOptimizer::State& LowRankProjectedOptimizer::stateFor(const ParameterId& id) {
    auto found = states_.find(id.name);
    if (found == states_.end()) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: unregistered parameter '" + id.name + "'");
    }
    if (found->second.rows != id.rows || found->second.cols != id.cols) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: shape mismatch for parameter '" + id.name + "'");
    }
    return found->second;
}

void LowRankProjectedOptimizer::consumeWeightGradientBlock(const ParameterId& id, std::size_t firstRow,
                                                            std::size_t rowCount, const float* deviceBlock) {
    State& state = stateFor(id);
    if (firstRow + rowCount > state.rows || (deviceBlock == nullptr && rowCount != 0)) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: invalid gradient block");
    }
    state.touched = true;
    const std::size_t gradientValues = rowCount * state.cols;
    gradientStatsKernel<<<gridFor(gradientValues), kBlockSize>>>(deviceBlock, gradientValues,
                                                                 deviceStats_.as<DeviceStepStats>());
    const std::size_t projectedValues = state.rank * state.cols;
    projectGradientKernel<<<gridFor(projectedValues), kBlockSize>>>(
        state.accumulator.as<float>(), deviceBlock, firstRow, rowCount, state.cols, state.rank, state.seed,
        state.projectionEpoch);
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
}

void LowRankProjectedOptimizer::endStep() {
    const float bias1 = 1.0F - std::pow(options_.beta1, static_cast<float>(step_ + 1));
    const float bias2 = 1.0F - std::pow(options_.beta2, static_cast<float>(step_ + 1));
    for (auto& [name, state] : states_) {
        (void)name;
        if (!state.touched) continue;
        const std::size_t projectedValues = state.rank * state.cols;
        adamDirectionKernel<<<gridFor(projectedValues), kBlockSize>>>(
            state.firstMoment.as<float>(), state.secondMoment.as<float>(), state.accumulator.as<float>(),
            projectedValues, options_.beta1, options_.beta2, bias1, bias2, options_.eps,
            deviceStats_.as<DeviceStepStats>());
        updatePackedKernel<<<gridFor(state.rows * state.weight->wordsPerRow()), kBlockSize>>>(
            state.weight->packedWords(), state.rows, state.cols, state.weight->wordsPerRow(),
            state.accumulator.as<float>(), state.rank, options_.learningRate, state.seed, state.projectionEpoch,
            step_, deviceStats_.as<DeviceStepStats>());
        BLACKFORGE_CUDA_CHECK(cudaMemset(state.accumulator.data(), 0, state.accumulator.bytes()));
        state.touched = false;
    }
    BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());

    DeviceStepStats device{};
    BLACKFORGE_CUDA_CHECK(cudaMemcpy(&device, deviceStats_.data(), sizeof(device), cudaMemcpyDeviceToHost));
    LowRankOptimizerStepStats last;
    last.elements = static_cast<std::size_t>(device.elements);
    last.flips = static_cast<std::size_t>(device.flips);
    last.positiveFlips = static_cast<std::size_t>(device.positiveFlips);
    last.negativeFlips = static_cast<std::size_t>(device.negativeFlips);
    last.saturatedTrits = static_cast<std::size_t>(device.saturated);
    last.nanInfCount = static_cast<std::size_t>(device.nanInf);
    last.gradientRms = device.gradientCount == 0 ? 0.0 : std::sqrt(device.gradientSquared / device.gradientCount);
    last.optimizerNorm = device.optimizerCount == 0 ? 0.0 : std::sqrt(device.optimizerSquared / device.optimizerCount);
    last.updateRms = device.updateCount == 0 ? 0.0 : std::sqrt(device.updateSquared / device.updateCount);
    stats_.lastStep = last;
    stats_.totalFlips += last.flips;
    stats_.totalElements += last.elements;
    ++step_;
    stats_.stepCount = step_;

    if (step_ % options_.projectionInterval == 0) {
        for (auto& [name, state] : states_) {
            (void)name;
            ++state.projectionEpoch;
            BLACKFORGE_CUDA_CHECK(cudaMemset(state.firstMoment.data(), 0, state.firstMoment.bytes()));
            BLACKFORGE_CUDA_CHECK(cudaMemset(state.secondMoment.data(), 0, state.secondMoment.bytes()));
        }
        ++stats_.projectionReseeds;
    }
    BLACKFORGE_CUDA_CHECK(cudaMemset(deviceStats_.data(), 0, deviceStats_.bytes()));
}

std::size_t LowRankProjectedOptimizer::stateBytes() const {
    std::size_t result = deviceStats_.bytes();
    for (const auto& [name, state] : states_) {
        (void)name;
        result += state.firstMoment.bytes() + state.secondMoment.bytes() + state.accumulator.bytes();
    }
    return result;
}

std::size_t LowRankProjectedOptimizer::conventionalStateBytes() const {
    std::size_t result = 0;
    for (const auto& [name, state] : states_) {
        (void)name;
        result += 2 * state.rows * state.cols * sizeof(float);
    }
    return result;
}

}  // namespace blackforge::blackbit::cuda
