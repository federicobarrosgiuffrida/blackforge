#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/blackbit/cuda_memory.hpp"
#include "blackforge/blackbit/cuda_model.hpp"
#include "blackforge/blackbit/gradient.hpp"
#include "blackforge/blackbit/model.hpp"
#include "blackforge/runtime/tensor.hpp"

namespace bb = blackforge::blackbit;
namespace bbcuda = blackforge::blackbit::cuda;
using blackforge::runtime::Tensor;

namespace {

bb::BlackBitConfig modelConfig() {
    bb::BlackBitConfig config;
    config.name = "cuda-model-test";
    config.vocabSize = 32;
    config.hiddenSize = 12;
    config.numLayers = 2;
    config.numHeads = 3;
    config.numKvHeads = 1;
    config.headDim = 4;
    config.numExperts = 4;
    config.expertsPerToken = 2;
    config.expertHidden = 20;
    config.maxSeqLen = 16;
    config.ternaryGroupSize = 20;
    config.expertCapacityFactor = 2.0F;
    config.validate();
    return config;
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

    void consumeDenseGradient(const bb::ParameterId& id, const float* values, std::size_t count) override {
        auto& host = gradients[id.name];
        if (host.empty()) host.resize(count, 0.0F);
        std::vector<float> contribution(count);
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(contribution.data(), values, count * sizeof(float), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < count; ++i) host[i] += contribution[i];
    }

    std::unordered_map<std::string, std::vector<float>> gradients;
};

}  // namespace

TEST(CudaBlackBitModelTest, FullForwardMatchesCpuThroughAttentionMoeAndFinalNorm) {
    const bb::BlackBitConfig config = modelConfig();
    bb::BlackBitModel cpu(config, 123);
    cpu.setComputeDType(bb::ComputeDType::BF16);
    bbcuda::BlackBitModel gpu(cpu);
    const std::vector<int> ids{1, 7, 3, 9, 2, 5, 11, 4};
    bb::BlackBitForwardCache cpuCache;
    bbcuda::ForwardCache gpuCache;
    const Tensor expected = cpu.forwardHidden(ids, 2, 4, cpuCache);
    const Tensor actual = gpu.forwardHidden(ids, 2, 4, gpuCache).toHost();
    expectNear(actual, expected, 6.0e-2F);
    EXPECT_EQ(gpuCache.blocks.size(), config.numLayers);
    EXPECT_EQ(gpuCache.blockInputs.size(), config.numLayers);
}

TEST(CudaBlackBitModelTest, ThreePassTiedHeadLossAndAllGradientsMatchCpu) {
    const bb::BlackBitConfig config = modelConfig();
    bb::BlackBitModel cpu(config, 77);
    cpu.setComputeDType(bb::ComputeDType::BF16);
    bb::BlackBitRuntimeOptions options;
    options.recompute = bb::ActivationRecompute::PerLayer;
    options.vocabChunk = 7;
    cpu.setRuntimeOptions(options);
    bbcuda::BlackBitModel gpu(cpu);
    const std::vector<int> ids{1, 4, 7, 2, 9};
    const std::vector<int> targets{4, 7, 2, 9, 3};
    bb::DenseGradientCollector cpuGradients;
    const bb::BlackBitStepResult cpuResult = cpu.trainStep(ids, targets, 1, 5, &cpuGradients);
    DeviceGradientCollector gpuGradients;
    bbcuda::resetGradientLifetimeStats();
    const bb::BlackBitStepResult gpuResult = gpu.trainStep(ids, targets, 1, 5, &gpuGradients);

    EXPECT_NEAR(gpuResult.loss, cpuResult.loss, 3.0e-2F);
    EXPECT_EQ(gpuResult.scoredTokens, cpuResult.scoredTokens);
    EXPECT_EQ(gpuResult.routing.size(), cpuResult.routing.size());
    EXPECT_TRUE(gpuGradients.gradients.contains("embedding"));
    EXPECT_TRUE(gpuGradients.gradients.contains("final_norm"));
    for (const auto& [name, actual] : gpuGradients.gradients) {
        ASSERT_TRUE(cpuGradients.has(name)) << name;
        const auto& expected = cpuGradients.gradient(name);
        ASSERT_EQ(actual.size(), expected.size()) << name;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], 1.2e-1F) << name << " index " << i;
        }
    }
    EXPECT_EQ(bbcuda::gradientLifetimeStats().liveBytes, 0U);
    EXPECT_LT(bbcuda::gradientLifetimeStats().peakLiveBytes,
              config.vocabSize * config.hiddenSize * sizeof(float));
}

TEST(CudaBlackBitModelTest, RealGpuTrainingStepChangesPackedTernaryParametersUnderHardLimit) {
    const bb::BlackBitConfig config = modelConfig();
    bb::BlackBitModel cpu(config, 29);
    cpu.setComputeDType(bb::ComputeDType::BF16);
    bbcuda::MemoryTelemetry::instance().setMaxDeviceBytes(7800ULL * 1024ULL * 1024ULL);
    bbcuda::BlackBitModel gpu(cpu);
    bb::LowRankOptimizerOptions optimizerOptions;
    optimizerOptions.learningRate = 0.3F;
    optimizerOptions.rank = 4;
    bbcuda::LowRankProjectedOptimizer optimizer(optimizerOptions);
    gpu.registerParameters(optimizer);
    const auto before = gpu.embedding().weight().download().packedWords();
    const std::vector<int> ids{1, 3, 5, 7, 9, 11};
    const std::vector<int> targets{3, 5, 7, 9, 11, 2};
    const bb::BlackBitStepResult result = gpu.trainStep(ids, targets, 1, 6, &optimizer);
    optimizer.endStep();
    const auto after = gpu.embedding().weight().download().packedWords();

    EXPECT_FALSE(result.sawNaN);
    EXPECT_FALSE(result.sawInf);
    EXPECT_GT(result.loss, 0.0F);
    EXPECT_NE(after, before);
    EXPECT_GT(optimizer.stats().lastStep.flips, 0U);
    EXPECT_EQ(optimizer.stats().lastStep.nanInfCount, 0U);
    EXPECT_EQ(bbcuda::gradientLifetimeStats().liveBytes, 0U);
    const auto memory = bbcuda::MemoryTelemetry::instance().snapshot();
    EXPECT_LT(memory.devicePeakUsedBytes, 7800ULL * 1024ULL * 1024ULL);
}
