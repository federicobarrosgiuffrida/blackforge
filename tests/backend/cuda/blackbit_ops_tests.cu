#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "blackforge/blackbit/cuda_ops.hpp"
#include "blackforge/blackbit/gradient.hpp"
#include "blackforge/blackbit/model.hpp"
#include "blackforge/blackbit/rope.hpp"
#include "blackforge/blackbit/ternary.hpp"
#include "blackforge/runtime/tensor.hpp"

namespace bb = blackforge::blackbit;
namespace bbcuda = blackforge::blackbit::cuda;
using blackforge::runtime::Tensor;

namespace {

Tensor values(std::vector<std::size_t> shape, float scale, float phase = 0.0F) {
    std::size_t count = 1;
    for (const std::size_t dimension : shape) count *= dimension;
    std::vector<float> data(count);
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.31F + phase) * scale;
    }
    return Tensor(std::move(shape), std::move(data));
}

void expectNear(const Tensor& actual, const Tensor& expected, float tolerance = 2.0e-5F) {
    ASSERT_EQ(actual.shape(), expected.shape());
    for (std::size_t i = 0; i < actual.elementCount(); ++i) {
        EXPECT_NEAR(actual.at(i), expected.at(i), tolerance) << "index " << i;
    }
}

}  // namespace

TEST(CudaBlackBitOpsTest, RmsNormForwardAndBackwardMatchCpuReference) {
    constexpr std::size_t features = 13;
    bb::RmsNorm cpu("norm", features);
    for (std::size_t i = 0; i < features; ++i) cpu.gamma()[i] = 0.7F + 0.03F * static_cast<float>(i);
    const Tensor input = values({2, 3, features}, 0.8F, 0.2F);
    const Tensor gradOutput = values({2, 3, features}, 0.3F, 0.7F);
    const Tensor gamma({features}, cpu.gamma());

    bb::DenseGradientCollector collector;
    const Tensor expectedOutput = cpu.forward(input);
    const Tensor expectedGradInput = cpu.backward(input, gradOutput, &collector);

    const auto deviceInput = bbcuda::Tensor::fromHost(input);
    const auto deviceGamma = bbcuda::Tensor::fromHost(gamma, bbcuda::MemoryArena::DenseParameters);
    const auto deviceGradOutput = bbcuda::Tensor::fromHost(gradOutput);
    bbcuda::RmsNormCache cache;
    const Tensor actualOutput = bbcuda::rmsNormForward(deviceInput, deviceGamma, cache).toHost();
    auto gradients = bbcuda::rmsNormBackward(deviceInput, deviceGamma, deviceGradOutput, cache);

    expectNear(actualOutput, expectedOutput, 2.0e-5F);
    expectNear(gradients.input.toHost(), expectedGradInput, 2.0e-5F);
    const Tensor expectedGamma({features}, collector.gradient("norm"));
    expectNear(gradients.gamma.toHost(), expectedGamma, 3.0e-5F);
}

TEST(CudaBlackBitOpsTest, RopeForwardAndTransposeMatchCpuFormula) {
    constexpr std::size_t batch = 2;
    constexpr std::size_t seq = 5;
    constexpr std::size_t heads = 3;
    constexpr std::size_t headDim = 8;
    Tensor expected = values({batch, seq, heads, headDim}, 0.6F, 0.4F);
    Tensor original = expected;
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t s = 0; s < seq; ++s) {
            for (std::size_t h = 0; h < heads; ++h) {
                bb::applyRope(expected.data().data() + ((b * seq + s) * heads + h) * headDim, headDim, s);
            }
        }
    }
    auto device = bbcuda::Tensor::fromHost(original);
    bbcuda::applyRopeInPlace(device, batch, seq, heads, headDim);
    expectNear(device.toHost(), expected, 2.0e-5F);
    bbcuda::applyRopeInPlace(device, batch, seq, heads, headDim, true);
    expectNear(device.toHost(), original, 3.0e-5F);
}

TEST(CudaBlackBitOpsTest, SwiGluForwardAndBackwardMatchAnalyticReference) {
    const Tensor gate = values({4, 11}, 1.2F, 0.1F);
    const Tensor up = values({4, 11}, 0.9F, 0.8F);
    const Tensor gradOutput = values({4, 11}, 0.4F, 0.3F);
    Tensor output = Tensor::zeros(gate.shape());
    Tensor gradGate = Tensor::zeros(gate.shape());
    Tensor gradUp = Tensor::zeros(gate.shape());
    for (std::size_t i = 0; i < gate.elementCount(); ++i) {
        const float sigmoid = 1.0F / (1.0F + std::exp(-gate.at(i)));
        const float silu = gate.at(i) * sigmoid;
        output.at(i) = silu * up.at(i);
        gradUp.at(i) = gradOutput.at(i) * silu;
        gradGate.at(i) = gradOutput.at(i) * up.at(i) * sigmoid *
                         (1.0F + gate.at(i) * (1.0F - sigmoid));
    }
    const auto deviceGate = bbcuda::Tensor::fromHost(gate);
    const auto deviceUp = bbcuda::Tensor::fromHost(up);
    const auto deviceGradOutput = bbcuda::Tensor::fromHost(gradOutput);
    expectNear(bbcuda::siluMultiply(deviceGate, deviceUp).toHost(), output);
    auto gradients = bbcuda::siluMultiplyBackward(deviceGate, deviceUp, deviceGradOutput);
    expectNear(gradients.gate.toHost(), gradGate);
    expectNear(gradients.up.toHost(), gradUp);
}

TEST(CudaBlackBitOpsTest, PackedEmbeddingLookupMatchesCpuLogicalRows) {
    constexpr std::size_t vocab = 7;
    constexpr std::size_t hidden = 23;
    std::vector<std::int8_t> trits(vocab * hidden);
    for (std::size_t i = 0; i < trits.size(); ++i) {
        trits[i] = static_cast<std::int8_t>(static_cast<int>((i * 5 + 2) % 3) - 1);
    }
    std::vector<float> scales(vocab * 2);
    for (std::size_t i = 0; i < scales.size(); ++i) scales[i] = 0.04F + 0.003F * static_cast<float>(i);
    const bb::TernaryTensor host = bb::TernaryTensor::fromTrits({vocab, hidden}, trits, scales, 20);
    const bbcuda::TernaryTensor device(host);
    const std::vector<int> ids{6, 0, 3, 3, 1, 5};
    const auto deviceIds = bbcuda::IndexTensor::fromHost({2, 3}, ids);
    const Tensor actual = bbcuda::embeddingLookup(deviceIds, device, hidden).toHost();
    Tensor expected = Tensor::zeros({2, 3, hidden});
    for (std::size_t token = 0; token < ids.size(); ++token) {
        for (std::size_t col = 0; col < hidden; ++col) {
            expected.at(token * hidden + col) = host.at(static_cast<std::size_t>(ids[token]) * hidden + col);
        }
    }
    expectNear(actual, expected, 1.0e-7F);
}

TEST(CudaBlackBitOpsTest, SparseCrossEntropyMatchesStableCpuReference) {
    constexpr std::size_t rows = 5;
    constexpr std::size_t vocab = 17;
    const Tensor logits = values({rows, vocab}, 3.0F, 0.4F);
    const std::vector<int> targets{3, -1, 16, 0, 9};
    Tensor expectedGradient = Tensor::zeros({rows, vocab});
    double expectedLoss = 0.0;
    std::size_t scored = 0;
    for (std::size_t row = 0; row < rows; ++row) {
        if (targets[row] < 0) continue;
        ++scored;
        float maximum = logits.at(row * vocab);
        for (std::size_t col = 1; col < vocab; ++col) maximum = std::max(maximum, logits.at(row * vocab + col));
        double denominator = 0.0;
        for (std::size_t col = 0; col < vocab; ++col) denominator += std::exp(logits.at(row * vocab + col) - maximum);
        expectedLoss += maximum + std::log(denominator) - logits.at(row * vocab + static_cast<std::size_t>(targets[row]));
        for (std::size_t col = 0; col < vocab; ++col) {
            float gradient = static_cast<float>(std::exp(logits.at(row * vocab + col) - maximum) / denominator);
            if (col == static_cast<std::size_t>(targets[row])) gradient -= 1.0F;
            expectedGradient.at(row * vocab + col) = gradient;
        }
    }
    for (std::size_t row = 0; row < rows; ++row) {
        if (targets[row] < 0) continue;
        for (std::size_t col = 0; col < vocab; ++col) {
            expectedGradient.at(row * vocab + col) /= static_cast<float>(scored);
        }
    }
    const auto deviceLogits = bbcuda::Tensor::fromHost(logits);
    const auto deviceTargets = bbcuda::IndexTensor::fromHost({rows}, targets);
    auto result = bbcuda::crossEntropySparse(deviceLogits, deviceTargets);
    EXPECT_EQ(result.scoredTokens, scored);
    EXPECT_EQ(result.nanInfCount, 0U);
    EXPECT_NEAR(result.loss, static_cast<float>(expectedLoss / scored), 2.0e-5F);
    expectNear(result.gradient.toHost(), expectedGradient, 2.0e-6F);
}

TEST(CudaBlackBitOpsTest, DenseRouterLinearForwardAndBackwardMatchCpuReference) {
    constexpr std::size_t rows = 7;
    constexpr std::size_t inFeatures = 9;
    constexpr std::size_t outFeatures = 4;
    const Tensor input = values({rows, inFeatures}, 0.7F, 0.2F);
    const Tensor weight = values({outFeatures, inFeatures}, 0.3F, 0.8F);
    const Tensor gradOutput = values({rows, outFeatures}, 0.4F, 0.5F);
    Tensor expectedOutput = Tensor::zeros({rows, outFeatures});
    Tensor expectedGradInput = Tensor::zeros({rows, inFeatures});
    Tensor expectedGradWeight = Tensor::zeros({outFeatures, inFeatures});
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t out = 0; out < outFeatures; ++out) {
            for (std::size_t col = 0; col < inFeatures; ++col) {
                expectedOutput.at(row * outFeatures + out) += input.at(row * inFeatures + col) * weight.at(out * inFeatures + col);
                expectedGradInput.at(row * inFeatures + col) += gradOutput.at(row * outFeatures + out) * weight.at(out * inFeatures + col);
                expectedGradWeight.at(out * inFeatures + col) += gradOutput.at(row * outFeatures + out) * input.at(row * inFeatures + col);
            }
        }
    }
    const auto deviceInput = bbcuda::Tensor::fromHost(input);
    const auto deviceWeight = bbcuda::Tensor::fromHost(weight, bbcuda::MemoryArena::DenseParameters);
    const auto deviceGradOutput = bbcuda::Tensor::fromHost(gradOutput);
    expectNear(bbcuda::denseLinear(deviceInput, deviceWeight).toHost(), expectedOutput);
    auto gradients = bbcuda::denseLinearBackward(deviceInput, deviceWeight, deviceGradOutput);
    expectNear(gradients.input.toHost(), expectedGradInput);
    expectNear(gradients.weight.toHost(), expectedGradWeight);
}
