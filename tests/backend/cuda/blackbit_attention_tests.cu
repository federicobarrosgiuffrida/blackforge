#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/blackbit/attention.hpp"
#include "blackforge/blackbit/cuda_attention.hpp"
#include "blackforge/blackbit/gradient.hpp"
#include "blackforge/runtime/tensor.hpp"

namespace bb = blackforge::blackbit;
namespace bbcuda = blackforge::blackbit::cuda;
using blackforge::runtime::Tensor;

namespace {

bb::BlackBitConfig attentionConfig() {
    bb::BlackBitConfig config;
    config.name = "cuda-attention-test";
    config.vocabSize = 64;
    config.hiddenSize = 16;
    config.numLayers = 1;
    config.numHeads = 4;
    config.numKvHeads = 2;
    config.headDim = 4;
    config.numExperts = 2;
    config.expertsPerToken = 2;
    config.expertHidden = 24;
    config.maxSeqLen = 16;
    config.ternaryGroupSize = 20;
    config.validate();
    return config;
}

bb::BlackBitConfig attention128Config() {
    bb::BlackBitConfig config;
    config.name = "cuda-attention-128-test";
    config.vocabSize = 256;
    config.hiddenSize = 128;
    config.numLayers = 1;
    config.numHeads = 1;
    config.numKvHeads = 1;
    config.headDim = 128;
    config.numExperts = 2;
    config.expertsPerToken = 2;
    config.expertHidden = 192;
    config.maxSeqLen = 8;
    config.ternaryGroupSize = 20;
    config.validate();
    return config;
}

Tensor values(std::vector<std::size_t> shape, float scale, float phase = 0.0F) {
    std::size_t count = 1;
    for (const std::size_t dimension : shape) count *= dimension;
    std::vector<float> data(count);
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.19F + phase) * scale;
    }
    return Tensor(std::move(shape), std::move(data));
}

void expectNear(const Tensor& actual, const Tensor& expected, float tolerance) {
    ASSERT_EQ(actual.shape(), expected.shape());
    for (std::size_t i = 0; i < actual.elementCount(); ++i) {
        EXPECT_NEAR(actual.at(i), expected.at(i), tolerance) << "index " << i;
    }
}

class DeviceGradientCollector final : public bbcuda::GradientSink {
public:
    void consumeWeightGradientBlock(const bb::ParameterId& id, std::size_t firstRow,
                                    std::size_t rowCount, const float* block) override {
        auto& values = gradients[id.name];
        if (values.empty()) values.resize(id.rows * id.cols, 0.0F);
        std::vector<float> host(rowCount * id.cols);
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(host.data(), block, host.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::copy(host.begin(), host.end(), values.begin() + static_cast<std::ptrdiff_t>(firstRow * id.cols));
    }

    std::unordered_map<std::string, std::vector<float>> gradients;
};

}  // namespace

TEST(CudaBlackBitAttentionTest, GqaForwardMatchesCpuWithoutDuplicatingKvOrScores) {
    const bb::BlackBitConfig config = attentionConfig();
    bb::GqaAttention cpu("attention", config);
    cpu.initialize(71);
    cpu.setComputeDType(bb::ComputeDType::BF16);
    bbcuda::GqaAttention gpu(cpu, config);
    const Tensor input = values({2, 5, config.hiddenSize}, 0.5F, 0.3F);
    bb::AttentionCache cpuCache;
    bbcuda::AttentionCache gpuCache;
    const Tensor expected = cpu.forward(input, cpuCache);
    const auto deviceInput = bbcuda::Tensor::fromHost(input);
    const Tensor actual = gpu.forward(deviceInput, gpuCache).toHost();

    expectNear(actual, expected, 2.5e-2F);
    EXPECT_EQ(gpuCache.key.elementCount(), 2 * 5 * config.numKvHeads * config.headDim);
    EXPECT_LT(gpuCache.key.elementCount(), 2 * 5 * config.numHeads * config.headDim);
    EXPECT_EQ(gpuCache.rowMax.elementCount(), 2 * config.numHeads * 5);
    EXPECT_EQ(gpuCache.rowSum.elementCount(), 2 * config.numHeads * 5);
    EXPECT_LT(gpuCache.rowSum.elementCount(), 2 * config.numHeads * 5 * 5);
    const Tensor rowSum = gpuCache.rowSum.toHost();
    for (const float value : rowSum.data()) EXPECT_GT(value, 0.0F);
}

TEST(CudaBlackBitAttentionTest, GqaBackwardMatchesCpuAndStreamsAllProjectionGradients) {
    const bb::BlackBitConfig config = attentionConfig();
    bb::GqaAttention cpu("attention", config);
    cpu.initialize(91);
    cpu.setComputeDType(bb::ComputeDType::BF16);
    bbcuda::GqaAttention gpu(cpu, config);
    const Tensor input = values({1, 4, config.hiddenSize}, 0.45F, 0.1F);
    const Tensor gradOutput = values({1, 4, config.hiddenSize}, 0.2F, 0.7F);
    bb::AttentionCache cpuCache;
    (void)cpu.forward(input, cpuCache);
    bb::DenseGradientCollector cpuGradients;
    const Tensor expectedGradInput = cpu.backward(input, gradOutput, cpuCache, &cpuGradients);

    const auto deviceInput = bbcuda::Tensor::fromHost(input);
    const auto deviceGradOutput = bbcuda::Tensor::fromHost(gradOutput);
    bbcuda::AttentionCache gpuCache;
    (void)gpu.forward(deviceInput, gpuCache);
    DeviceGradientCollector gpuGradients;
    bbcuda::resetGradientLifetimeStats();
    const Tensor actualGradInput = gpu.backward(deviceInput, deviceGradOutput, gpuCache, &gpuGradients).toHost();

    expectNear(actualGradInput, expectedGradInput, 3.5e-2F);
    for (const std::string name : {"attention.q", "attention.k", "attention.v", "attention.o"}) {
        const auto& expected = cpuGradients.gradient(name);
        const auto& actual = gpuGradients.gradients.at(name);
        ASSERT_EQ(actual.size(), expected.size()) << name;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], 4.0e-2F) << name << " index " << i;
        }
    }
    EXPECT_EQ(bbcuda::gradientLifetimeStats().liveBytes, 0U);
    EXPECT_GT(bbcuda::gradientLifetimeStats().blocksProduced, 0U);
}

TEST(CudaBlackBitAttentionTest, SpecializedHeadDim128ForwardAndBackwardMatchCpu) {
    const bb::BlackBitConfig config = attention128Config();
    bb::GqaAttention cpu("attention128", config);
    cpu.initialize(117);
    cpu.setComputeDType(bb::ComputeDType::BF16);
    bbcuda::GqaAttention gpu(cpu, config);
    const Tensor input = values({1, 3, config.hiddenSize}, 0.25F, 0.2F);
    const Tensor gradOutput = values({1, 3, config.hiddenSize}, 0.08F, 0.6F);
    bb::AttentionCache cpuCache;
    const Tensor expectedOutput = cpu.forward(input, cpuCache);
    bb::DenseGradientCollector cpuGradients;
    const Tensor expectedGradInput = cpu.backward(input, gradOutput, cpuCache, &cpuGradients);

    const auto deviceInput = bbcuda::Tensor::fromHost(input);
    const auto deviceGradOutput = bbcuda::Tensor::fromHost(gradOutput);
    bbcuda::AttentionCache gpuCache;
    const Tensor actualOutput = gpu.forward(deviceInput, gpuCache).toHost();
    DeviceGradientCollector gpuGradients;
    const Tensor actualGradInput = gpu.backward(deviceInput, deviceGradOutput, gpuCache, &gpuGradients).toHost();

    expectNear(actualOutput, expectedOutput, 2.5e-2F);
    expectNear(actualGradInput, expectedGradInput, 3.5e-2F);
    for (const std::string name : {"attention128.q", "attention128.k", "attention128.v", "attention128.o"}) {
        const auto& expected = cpuGradients.gradient(name);
        const auto& actual = gpuGradients.gradients.at(name);
        ASSERT_EQ(actual.size(), expected.size()) << name;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], 4.0e-2F) << name << " index " << i;
        }
    }
}
