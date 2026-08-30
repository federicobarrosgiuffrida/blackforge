#include "blackforge/blackbit/cuda_low_rank_optimizer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

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
    __shared__ double squared[kBlockSize];
    __shared__ unsigned int valid[kBlockSize];
    __shared__ unsigned int invalid[kBlockSize];
    double localSquared = 0.0;
    unsigned int localValid = 0;
    unsigned int localInvalid = 0;
    if (index < count) {
        const float value = gradient[index];
        if (isfinite(value)) {
            localSquared = static_cast<double>(value) * static_cast<double>(value);
            localValid = 1;
        } else {
            localInvalid = 1;
        }
    }
    squared[threadIdx.x] = localSquared;
    valid[threadIdx.x] = localValid;
    invalid[threadIdx.x] = localInvalid;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            squared[threadIdx.x] += squared[threadIdx.x + stride];
            valid[threadIdx.x] += valid[threadIdx.x + stride];
            invalid[threadIdx.x] += invalid[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&stats->gradientSquared, squared[0]);
        atomicAdd(&stats->gradientCount, static_cast<unsigned long long>(valid[0]));
        if (invalid[0] != 0) atomicAdd(&stats->nanInf, static_cast<unsigned long long>(invalid[0]));
    }
}

// Una componente per blocco: cosi' tutti i thread del blocco condividono
// lo stesso 'component' per costruzione, anche quando cols non e' un
// multiplo della dimensione del blocco.
//
// projection() dipende da (riga, componente) e non dalla colonna: nel
// ciclo originale ogni thread la ricalcolava per ogni riga, quindi i 256
// thread di un blocco valutavano 256 volte gli stessi hash. Qui il
// blocco li calcola una volta in memoria condivisa e li riusa.
//
// Le somme restano per riga crescente, con gli stessi raggruppamenti,
// quindi l'accumulatore e' bit-identico a prima.
__global__ void projectGradientKernel(float* accumulator, const float* gradient, std::size_t firstRow,
                                      std::size_t rowCount, std::size_t cols, std::size_t rank,
                                      std::uint64_t seed, std::uint64_t epoch) {
    const std::size_t component = blockIdx.y;
    if (component >= rank) return;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = col < cols;
    __shared__ float sharedProjection[kBlockSize];
    float sum = active ? accumulator[component * cols + col] : 0.0F;
    for (std::size_t base = 0; base < rowCount; base += kBlockSize) {
        const std::size_t chunk = min(static_cast<std::size_t>(kBlockSize), rowCount - base);
        if (threadIdx.x < chunk) {
            sharedProjection[threadIdx.x] =
                projection(seed, epoch, firstRow + base + threadIdx.x, component, rank);
        }
        __syncthreads();
        if (active) {
            for (std::size_t k = 0; k < chunk; ++k) {
                sum += sharedProjection[k] * gradient[(base + k) * cols + col];
            }
        }
        __syncthreads();
    }
    if (active) accumulator[component * cols + col] = sum;
}

__global__ void accumulateDenseKernel(float* accumulator, const float* gradient, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) accumulator[index] += gradient[index];
}

__global__ void squaredNormKernel(const float* values, std::size_t count, double* squared) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const double value = values[index];
        atomicAdd(squared, value * value);
    }
}

__global__ void denseAdamKernel(float* values, float* firstMoment, float* secondMoment, float* accumulator,
                                std::size_t count, float beta1, float beta2, float bias1, float bias2,
                                float epsilon, float learningRate, float weightDecay, float parameterScale,
                                DeviceStepStats* stats) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float gradient = accumulator[index];
    const float first = beta1 * firstMoment[index] + (1.0F - beta1) * gradient;
    const float second = beta2 * secondMoment[index] + (1.0F - beta2) * gradient * gradient;
    firstMoment[index] = first;
    secondMoment[index] = second;
    const float direction = (first / bias1) / (sqrtf(second / bias2) + epsilon);
    float parameter = values[index];
    const float update = -learningRate * parameterScale * direction;
    parameter += update;
    parameter -= learningRate * weightDecay * parameter;
    values[index] = parameter;
    accumulator[index] = 0.0F;
    if (!isfinite(parameter) || !isfinite(direction)) {
        atomicAdd(&stats->nanInf, 1ULL);
        return;
    }
    atomicAdd(&stats->optimizerSquared, static_cast<double>(direction) * static_cast<double>(direction));
    atomicAdd(&stats->optimizerCount, 1ULL);
    atomicAdd(&stats->updateSquared, static_cast<double>(update) * static_cast<double>(update));
    atomicAdd(&stats->updateCount, 1ULL);
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
    unsigned long long elements = 0;
    unsigned long long flips = 0;
    unsigned long long positiveFlips = 0;
    unsigned long long negativeFlips = 0;
    unsigned long long saturated = 0;
    unsigned long long nanInf = 0;
    double updateSquared = 0.0;

    // projection() dipende da (riga, componente) e NON dalla colonna,
    // ma il ciclo originale la ricalcolava per ogni trit: 20 * rank
    // hash per thread invece di rank. Scambiando i due cicli la
    // proiezione viene valutata una volta per componente e riusata sui
    // 20 trit della word. L'ordine di accumulazione di 'update' resta
    // per componente crescente, esattamente come prima, quindi il
    // risultato e' bit-identico (stessa somma, stessi raggruppamenti).
    int trits[kTritsPerWord];
    for (int byte = 0; byte < 4; ++byte) {
        decodeTritByte(wordByte(word, byte), trits + static_cast<std::size_t>(byte) * kTritsPerByte);
    }

    float updates[kTritsPerWord];
    for (std::size_t slot = 0; slot < kTritsPerWord; ++slot) updates[slot] = 0.0F;
    for (std::size_t component = 0; component < rank; ++component) {
        const float scaled = -learningRate * projection(seed, epoch, row, component, rank);
        const float* directionRow = direction + component * cols;
        for (std::size_t slot = 0; slot < kTritsPerWord; ++slot) {
            const std::size_t col = firstCol + slot;
            if (col < cols) updates[slot] += scaled * directionRow[col];
        }
    }

    for (std::size_t slot = 0; slot < kTritsPerWord; ++slot) {
        const std::size_t col = firstCol + slot;
        if (col >= cols) continue;
        const float update = updates[slot];
        if (!isfinite(update)) {
            ++nanInf;
            continue;
        }
        const int oldTrit = trits[slot];
        const float target = static_cast<float>(oldTrit) + update;
        const int newTrit = stochasticRoundToTrit(target, stepSeed, row * cols + col);
        trits[slot] = newTrit;
        ++elements;
        updateSquared += static_cast<double>(update) * static_cast<double>(update);
        if (target <= -1.0F || target >= 1.0F) ++saturated;
        if (newTrit != oldTrit) {
            ++flips;
            if (newTrit > oldTrit) {
                ++positiveFlips;
            } else {
                ++negativeFlips;
            }
        }
    }
    for (int byte = 0; byte < 4; ++byte) {
        const int* byteTrits = trits + static_cast<std::size_t>(byte) * kTritsPerByte;
        word = setWordByte(word, byte,
                           encodeTritByte(byteTrits[0], byteTrits[1], byteTrits[2], byteTrits[3], byteTrits[4]));
    }
    packed[wordIndex] = word;
    atomicAdd(&stats->elements, elements);
    atomicAdd(&stats->updateSquared, updateSquared);
    atomicAdd(&stats->updateCount, elements);
    if (flips != 0) atomicAdd(&stats->flips, flips);
    if (positiveFlips != 0) atomicAdd(&stats->positiveFlips, positiveFlips);
    if (negativeFlips != 0) atomicAdd(&stats->negativeFlips, negativeFlips);
    if (saturated != 0) atomicAdd(&stats->saturated, saturated);
    if (nanInf != 0) atomicAdd(&stats->nanInf, nanInf);
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
    if (denseStates_.count(name) != 0) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: parameter '" + name + "' already registered");
    }
    auto [iterator, inserted] = states_.try_emplace(name);
    if (!inserted) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: parameter '" + name + "' already registered");
    }
    allocateState(name, iterator->second, weight);
}

void LowRankProjectedOptimizer::registerDense(const std::string& name, Tensor& values) {
    if (states_.count(name) != 0 || values.elementCount() == 0) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: invalid or duplicate dense parameter '" +
                                    name + "'");
    }
    auto [iterator, inserted] = denseStates_.try_emplace(name);
    if (!inserted) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: parameter '" + name + "' already registered");
    }
    DenseState& state = iterator->second;
    state.values = &values;
    const std::size_t bytes = values.bytes();
    state.firstMoment = Buffer(bytes, MemoryArena::Optimizer);
    state.secondMoment = Buffer(bytes, MemoryArena::Optimizer);
    state.accumulator = Buffer(bytes, MemoryArena::Optimizer);
    BLACKFORGE_CUDA_CHECK(cudaMemset(state.firstMoment.data(), 0, bytes));
    BLACKFORGE_CUDA_CHECK(cudaMemset(state.secondMoment.data(), 0, bytes));
    BLACKFORGE_CUDA_CHECK(cudaMemset(state.accumulator.data(), 0, bytes));
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
    const dim3 projectionGrid(gridFor(state.cols), static_cast<unsigned int>(state.rank));
    projectGradientKernel<<<projectionGrid, kBlockSize>>>(
        state.accumulator.as<float>(), deviceBlock, firstRow, rowCount, state.cols, state.rank, state.seed,
        state.projectionEpoch);
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
}

void LowRankProjectedOptimizer::consumeDenseGradient(const ParameterId& id, const float* deviceValues,
                                                      std::size_t count) {
    auto found = denseStates_.find(id.name);
    if (found == denseStates_.end()) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: unregistered dense parameter '" +
                                    id.name + "'");
    }
    DenseState& state = found->second;
    if (count != state.values->elementCount() || id.rows * id.cols != count ||
        (deviceValues == nullptr && count != 0)) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: dense gradient shape mismatch for '" +
                                    id.name + "'");
    }
    gradientStatsKernel<<<gridFor(count), kBlockSize>>>(deviceValues, count, deviceStats_.as<DeviceStepStats>());
    accumulateDenseKernel<<<gridFor(count), kBlockSize>>>(state.accumulator.as<float>(), deviceValues, count);
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    state.touched = true;
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
    for (auto& [name, state] : denseStates_) {
        (void)name;
        if (!state.touched) continue;
        Buffer squared(sizeof(double), MemoryArena::Temporary);
        BLACKFORGE_CUDA_CHECK(cudaMemset(squared.data(), 0, squared.bytes()));
        const std::size_t count = state.values->elementCount();
        squaredNormKernel<<<gridFor(count), kBlockSize>>>(state.values->data(), count, squared.as<double>());
        double hostSquared = 0.0;
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(&hostSquared, squared.data(), sizeof(hostSquared), cudaMemcpyDeviceToHost));
        const float parameterRms = static_cast<float>(std::sqrt(hostSquared / static_cast<double>(count)));
        const float scale = parameterRms > 0.0F ? parameterRms : 1.0F;
        denseAdamKernel<<<gridFor(count), kBlockSize>>>(
            state.values->data(), state.firstMoment.as<float>(), state.secondMoment.as<float>(),
            state.accumulator.as<float>(), count, options_.beta1, options_.beta2, bias1, bias2, options_.eps,
            options_.learningRate, options_.weightDecay, scale, deviceStats_.as<DeviceStepStats>());
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
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
    for (const auto& [name, state] : denseStates_) {
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
    for (const auto& [name, state] : denseStates_) {
        (void)name;
        result += 2 * state.values->bytes();
    }
    return result;
}

namespace {

template <typename T>
void writeScalar(std::ostream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T readScalar(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("CUDA optimizer state: truncated stream");
    return value;
}

void writeName(std::ostream& out, const std::string& name) {
    writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(name.size()));
    out.write(name.data(), static_cast<std::streamsize>(name.size()));
}

std::string readName(std::istream& in) {
    const std::size_t length = readScalar<std::uint32_t>(in);
    std::string name(length, '\0');
    in.read(name.data(), static_cast<std::streamsize>(length));
    if (!in) throw std::runtime_error("CUDA optimizer state: truncated parameter name");
    return name;
}

void writeDeviceVector(std::ostream& out, const Buffer& buffer) {
    const std::size_t count = buffer.bytes() / sizeof(float);
    writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(count));
    std::vector<float> host(count);
    if (count != 0) {
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(host.data(), buffer.data(), buffer.bytes(), cudaMemcpyDeviceToHost));
        out.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(buffer.bytes()));
    }
}

void readDeviceVector(std::istream& in, Buffer& buffer) {
    const std::size_t count = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
    if (count * sizeof(float) != buffer.bytes()) {
        throw std::runtime_error("CUDA optimizer state: buffer size differs from registered parameter");
    }
    std::vector<float> host(count);
    if (count != 0) {
        in.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(buffer.bytes()));
        if (!in) throw std::runtime_error("CUDA optimizer state: truncated moment buffer");
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(buffer.data(), host.data(), buffer.bytes(), cudaMemcpyHostToDevice));
    }
}

}  // namespace

void LowRankProjectedOptimizer::serializeState(std::ostream& out) const {
    for (const auto& [name, state] : states_) {
        if (state.touched) throw std::runtime_error("CUDA optimizer state: checkpoint requested mid-step for '" + name + "'");
    }
    for (const auto& [name, state] : denseStates_) {
        if (state.touched) throw std::runtime_error("CUDA optimizer state: checkpoint requested mid-step for '" + name + "'");
    }
    writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(step_));
    writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(states_.size()));
    for (const auto& [name, state] : states_) {
        writeName(out, name);
        writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(state.rank));
        writeScalar<std::uint64_t>(out, state.projectionEpoch);
        writeDeviceVector(out, state.firstMoment);
        writeDeviceVector(out, state.secondMoment);
        writeScalar<std::uint64_t>(out, 0);  // CPU-compatible empty consolidation residual.
    }
    writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(denseStates_.size()));
    for (const auto& [name, state] : denseStates_) {
        writeName(out, name);
        writeDeviceVector(out, state.firstMoment);
        writeDeviceVector(out, state.secondMoment);
    }
    if (!out) throw std::runtime_error("CUDA optimizer state: write failed");
}

void LowRankProjectedOptimizer::deserializeState(std::istream& in) {
    step_ = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
    stats_.stepCount = step_;
    const std::size_t ternaryCount = readScalar<std::uint32_t>(in);
    if (ternaryCount != states_.size()) throw std::runtime_error("CUDA optimizer state: ternary state count mismatch");
    for (std::size_t index = 0; index < ternaryCount; ++index) {
        const std::string name = readName(in);
        auto found = states_.find(name);
        if (found == states_.end()) throw std::runtime_error("CUDA optimizer state: unknown ternary parameter '" + name + "'");
        State& state = found->second;
        const std::size_t rank = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
        if (rank != state.rank) throw std::runtime_error("CUDA optimizer state: rank mismatch for '" + name + "'");
        state.projectionEpoch = readScalar<std::uint64_t>(in);
        readDeviceVector(in, state.firstMoment);
        readDeviceVector(in, state.secondMoment);
        const std::size_t residualCount = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
        if (residualCount != 0) throw std::runtime_error("CUDA optimizer state: consolidation residual is unsupported");
        BLACKFORGE_CUDA_CHECK(cudaMemset(state.accumulator.data(), 0, state.accumulator.bytes()));
        state.touched = false;
    }
    const std::size_t denseCount = readScalar<std::uint32_t>(in);
    if (denseCount != denseStates_.size()) throw std::runtime_error("CUDA optimizer state: dense state count mismatch");
    for (std::size_t index = 0; index < denseCount; ++index) {
        const std::string name = readName(in);
        auto found = denseStates_.find(name);
        if (found == denseStates_.end()) throw std::runtime_error("CUDA optimizer state: unknown dense parameter '" + name + "'");
        readDeviceVector(in, found->second.firstMoment);
        readDeviceVector(in, found->second.secondMoment);
        BLACKFORGE_CUDA_CHECK(cudaMemset(found->second.accumulator.data(), 0, found->second.accumulator.bytes()));
        found->second.touched = false;
    }
}

}  // namespace blackforge::blackbit::cuda
