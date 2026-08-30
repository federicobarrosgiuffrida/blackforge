#pragma once

#include <cstddef>

#include "blackforge/blackbit/cuda_tensor.hpp"
#include "blackforge/blackbit/cuda_ternary.hpp"

namespace blackforge::blackbit::cuda {

Tensor add(const Tensor& a, const Tensor& b, MemoryArena arena = MemoryArena::Activations);
void addInPlace(Tensor& target, const Tensor& source);

struct RmsNormCache {
    Tensor inverseRms;
};

Tensor rmsNormForward(const Tensor& input, const Tensor& gamma, RmsNormCache& cache);

struct RmsNormGrad {
    Tensor input;
    Tensor gamma;
};

RmsNormGrad rmsNormBackward(const Tensor& input, const Tensor& gamma, const Tensor& gradOutput,
                            const RmsNormCache& cache);

void applyRopeInPlace(Tensor& tensor, std::size_t batch, std::size_t seq, std::size_t heads,
                      std::size_t headDim, bool transpose = false);

Tensor siluMultiply(const Tensor& gate, const Tensor& up);

struct SiluMultiplyGrad {
    Tensor gate;
    Tensor up;
};

SiluMultiplyGrad siluMultiplyBackward(const Tensor& gate, const Tensor& up, const Tensor& gradOutput);

Tensor embeddingLookup(const IndexTensor& tokenIds, const cuda::TernaryTensor& table, std::size_t hidden);

struct CrossEntropyResult {
    float loss = 0.0F;
    std::size_t scoredTokens = 0;
    std::size_t nanInfCount = 0;
    Tensor gradient;
};

CrossEntropyResult crossEntropySparse(const Tensor& logits, const IndexTensor& targets);

// Dense row-major weight layout [outFeatures, inFeatures]. Used only for
// router/norm-sized parameters, never for the large ternary matrices.
Tensor denseLinear(const Tensor& input, const Tensor& weight);

struct DenseLinearGrad {
    Tensor input;
    Tensor weight;
};

DenseLinearGrad denseLinearBackward(const Tensor& input, const Tensor& weight, const Tensor& gradOutput);

}  // namespace blackforge::blackbit::cuda
