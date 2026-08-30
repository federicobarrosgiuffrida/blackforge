#include "blackforge/blackbit/cuda_attention.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

#include "blackforge/backend/cuda/cuda_check.hpp"

namespace blackforge::blackbit::cuda {

namespace {

constexpr int kAttentionThreads = 256;

__global__ void gqaForwardKernel(float* output, float* rowMax, float* rowSum, const float* query,
                                 const float* key, const float* value, std::size_t batch,
                                 std::size_t seq, std::size_t heads, std::size_t kvHeads,
                                 std::size_t headDim, std::size_t headsPerGroup, float scale) {
    const std::size_t row = blockIdx.x;
    const std::size_t queryPosition = row % seq;
    const std::size_t head = (row / seq) % heads;
    const std::size_t batchIndex = row / (seq * heads);
    if (batchIndex >= batch) return;
    const std::size_t kvHead = head / headsPerGroup;
    __shared__ double reduction[kAttentionThreads];
    __shared__ float correction;
    __shared__ float weight;
    __shared__ float runningMaximum;
    __shared__ float runningTotal;
    if (threadIdx.x == 0) {
        runningMaximum = -3.402823466e+38F;
        runningTotal = 0.0F;
    }
    __syncthreads();
    const float* queryRow = query + ((batchIndex * seq + queryPosition) * heads + head) * headDim;
    float accumulator = 0.0F;
    for (std::size_t keyPosition = 0; keyPosition <= queryPosition; ++keyPosition) {
        const float* keyRow = key + ((batchIndex * seq + keyPosition) * kvHeads + kvHead) * headDim;
        double local = 0.0;
        for (std::size_t dimension = threadIdx.x; dimension < headDim; dimension += blockDim.x) {
            local += static_cast<double>(queryRow[dimension]) * static_cast<double>(keyRow[dimension]);
        }
        reduction[threadIdx.x] = local;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score = static_cast<float>(reduction[0]) * scale;
            const float nextMaximum = fmaxf(runningMaximum, score);
            correction = runningMaximum == -3.402823466e+38F ? 0.0F : expf(runningMaximum - nextMaximum);
            weight = expf(score - nextMaximum);
            runningTotal = runningTotal * correction + weight;
            runningMaximum = nextMaximum;
        }
        __syncthreads();
        if (threadIdx.x < headDim) {
            const float* valueRow = value + ((batchIndex * seq + keyPosition) * kvHeads + kvHead) * headDim;
            accumulator = accumulator * correction + weight * valueRow[threadIdx.x];
        }
        __syncthreads();
    }
    if (threadIdx.x < headDim) {
        output[((batchIndex * seq + queryPosition) * heads + head) * headDim + threadIdx.x] =
            accumulator / runningTotal;
    }
    if (threadIdx.x == 0) {
        rowMax[row] = runningMaximum;
        rowSum[row] = runningTotal;
    }
}

__global__ void gqaBackwardKernel(float* gradQuery, float* gradKey, float* gradValue,
                                  const float* query, const float* key, const float* value,
                                  const float* output, const float* gradOutput, const float* rowMax,
                                  const float* rowSum, std::size_t batch, std::size_t seq,
                                  std::size_t heads, std::size_t kvHeads, std::size_t headDim,
                                  std::size_t headsPerGroup, float scale) {
    const std::size_t row = blockIdx.x;
    const std::size_t queryPosition = row % seq;
    const std::size_t head = (row / seq) % heads;
    const std::size_t batchIndex = row / (seq * heads);
    if (batchIndex >= batch) return;
    const std::size_t kvHead = head / headsPerGroup;
    __shared__ double dotReduction[kAttentionThreads];
    __shared__ double probabilityReduction[kAttentionThreads];
    __shared__ float probability;
    __shared__ float gradScore;
    const std::size_t queryOffset = ((batchIndex * seq + queryPosition) * heads + head) * headDim;
    const float* queryRow = query + queryOffset;
    const float* outputRow = output + queryOffset;
    const float* gradOutputRow = gradOutput + queryOffset;
    double localCorrection = 0.0;
    for (std::size_t dimension = threadIdx.x; dimension < headDim; dimension += blockDim.x) {
        localCorrection += static_cast<double>(outputRow[dimension]) * gradOutputRow[dimension];
    }
    probabilityReduction[threadIdx.x] = localCorrection;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) probabilityReduction[threadIdx.x] += probabilityReduction[threadIdx.x + stride];
        __syncthreads();
    }
    const double correctionValue = probabilityReduction[0];
    float queryGradient = 0.0F;
    for (std::size_t keyPosition = 0; keyPosition <= queryPosition; ++keyPosition) {
        const std::size_t kvOffset = ((batchIndex * seq + keyPosition) * kvHeads + kvHead) * headDim;
        const float* keyRow = key + kvOffset;
        const float* valueRow = value + kvOffset;
        double localDot = 0.0;
        double localProbabilityGradient = 0.0;
        for (std::size_t dimension = threadIdx.x; dimension < headDim; dimension += blockDim.x) {
            localDot += static_cast<double>(queryRow[dimension]) * keyRow[dimension];
            localProbabilityGradient += static_cast<double>(gradOutputRow[dimension]) * valueRow[dimension];
        }
        dotReduction[threadIdx.x] = localDot;
        probabilityReduction[threadIdx.x] = localProbabilityGradient;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                dotReduction[threadIdx.x] += dotReduction[threadIdx.x + stride];
                probabilityReduction[threadIdx.x] += probabilityReduction[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            probability = expf(static_cast<float>(dotReduction[0]) * scale - rowMax[row]) / rowSum[row];
            gradScore = probability * static_cast<float>(probabilityReduction[0] - correctionValue) * scale;
        }
        __syncthreads();
        if (threadIdx.x < headDim) {
            queryGradient += gradScore * keyRow[threadIdx.x];
            atomicAdd(gradKey + kvOffset + threadIdx.x, gradScore * queryRow[threadIdx.x]);
            atomicAdd(gradValue + kvOffset + threadIdx.x, probability * gradOutputRow[threadIdx.x]);
        }
        __syncthreads();
    }
    if (threadIdx.x < headDim) gradQuery[queryOffset + threadIdx.x] = queryGradient;
}

}  // namespace

GqaAttention::GqaAttention(blackforge::blackbit::GqaAttention& cpuReference, const BlackBitConfig& config)
    : config_(config),
      q_(cpuReference.queryProjection()),
      k_(cpuReference.keyProjection()),
      v_(cpuReference.valueProjection()),
      o_(cpuReference.outputProjection()) {
    config_.validate();
    if (config_.headDim > kAttentionThreads || config_.headDim % 2 != 0) {
        throw std::invalid_argument("CUDA GqaAttention: head_dim must be even and at most 256");
    }
}

Tensor GqaAttention::forward(const Tensor& input, AttentionCache& cache) const {
    if (input.rank() != 3 || input.dim(2) != config_.hiddenSize) {
        throw std::invalid_argument("CUDA GqaAttention: expected [batch, seq, hidden]");
    }
    const std::size_t batch = input.dim(0);
    const std::size_t seq = input.dim(1);
    if (seq > config_.maxSeqLen) throw std::invalid_argument("CUDA GqaAttention: sequence exceeds max_seq_len");
    cache.query = q_.forward(input);
    cache.key = k_.forward(input);
    cache.value = v_.forward(input);
    applyRopeInPlace(cache.query, batch, seq, config_.numHeads, config_.headDim);
    applyRopeInPlace(cache.key, batch, seq, config_.numKvHeads, config_.headDim);
    cache.attnOutput = Tensor({batch, seq, config_.queryDim()}, MemoryArena::Attention);
    cache.rowMax = Tensor({batch, config_.numHeads, seq}, MemoryArena::Attention);
    cache.rowSum = Tensor({batch, config_.numHeads, seq}, MemoryArena::Attention);
    const std::size_t rows = batch * config_.numHeads * seq;
    const float scale = 1.0F / std::sqrt(static_cast<float>(config_.headDim));
    gqaForwardKernel<<<static_cast<unsigned int>(rows), kAttentionThreads>>>(
        cache.attnOutput.data(), cache.rowMax.data(), cache.rowSum.data(), cache.query.data(), cache.key.data(),
        cache.value.data(), batch, seq, config_.numHeads, config_.numKvHeads, config_.headDim,
        config_.headsPerKvGroup(), scale);
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    return o_.forward(cache.attnOutput);
}

Tensor GqaAttention::backward(const Tensor& input, const Tensor& gradOutput, const AttentionCache& cache,
                              GradientSink* sink) const {
    const std::size_t batch = input.dim(0);
    const std::size_t seq = input.dim(1);
    Tensor gradAttnOutput = o_.backward(cache.attnOutput, gradOutput, sink);
    Tensor gradQuery = Tensor::zeros(cache.query.shape(), MemoryArena::Attention);
    Tensor gradKey = Tensor::zeros(cache.key.shape(), MemoryArena::Attention);
    Tensor gradValue = Tensor::zeros(cache.value.shape(), MemoryArena::Attention);
    const std::size_t rows = batch * config_.numHeads * seq;
    const float scale = 1.0F / std::sqrt(static_cast<float>(config_.headDim));
    gqaBackwardKernel<<<static_cast<unsigned int>(rows), kAttentionThreads>>>(
        gradQuery.data(), gradKey.data(), gradValue.data(), cache.query.data(), cache.key.data(),
        cache.value.data(), cache.attnOutput.data(), gradAttnOutput.data(), cache.rowMax.data(),
        cache.rowSum.data(), batch, seq, config_.numHeads, config_.numKvHeads, config_.headDim,
        config_.headsPerKvGroup(), scale);
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    applyRopeInPlace(gradQuery, batch, seq, config_.numHeads, config_.headDim, true);
    applyRopeInPlace(gradKey, batch, seq, config_.numKvHeads, config_.headDim, true);
    Tensor fromQuery = q_.backward(input, gradQuery, sink);
    Tensor fromKey = k_.backward(input, gradKey, sink);
    Tensor fromValue = v_.backward(input, gradValue, sink);
    Tensor result = add(fromQuery, fromKey, MemoryArena::Activations);
    addInPlace(result, fromValue);
    return result;
}

}  // namespace blackforge::blackbit::cuda
