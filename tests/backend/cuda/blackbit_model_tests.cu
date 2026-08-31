#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/blackbit/cuda_memory.hpp"
#include "blackforge/blackbit/cuda_benchmark.hpp"
#include "blackforge/blackbit/cuda_checkpoint.hpp"
#include "blackforge/blackbit/cuda_model.hpp"
#include "blackforge/blackbit/cuda_train.hpp"
#include "blackforge/data/dataset.hpp"
#include "blackforge/blackbit/gradient.hpp"
#include "blackforge/blackbit/model.hpp"
#include "blackforge/runtime/tensor.hpp"
#include "blackforge/tokenizer/tokenizer.hpp"

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

TEST(CudaBlackBitModelTest, PackedCheckpointRestoresWeightsOptimizerRngAndTokenPosition) {
    const bb::BlackBitConfig config = modelConfig();
    bb::BlackBitModel cpu(config, 41);
    cpu.setComputeDType(bb::ComputeDType::BF16);
    bbcuda::BlackBitModel original(cpu);
    bb::LowRankOptimizerOptions optimizerOptions;
    optimizerOptions.learningRate = 0.25F;
    optimizerOptions.rank = 4;
    optimizerOptions.seed = 0x99887766ULL;
    bbcuda::LowRankProjectedOptimizer originalOptimizer(optimizerOptions);
    original.registerParameters(originalOptimizer);
    const std::vector<int> ids{1, 5, 2, 8, 3};
    const std::vector<int> targets{5, 2, 8, 3, 9};
    (void)original.trainStep(ids, targets, 1, 5, &originalOptimizer);
    originalOptimizer.endStep();

    bb::BlackBitTrainingState expectedState;
    expectedState.step = 1;
    expectedState.tokensSeen = 5;
    expectedState.learningRate = optimizerOptions.learningRate;
    expectedState.rngSeed = 0x1020304050607080ULL;
    expectedState.optimizerStep = originalOptimizer.stepCount();
    const std::filesystem::path path = std::filesystem::temp_directory_path() /
                                       "blackforge_cuda_blackbit_resume_test.bfbit";
    bbcuda::saveCheckpoint(path.string(), original, expectedState, &originalOptimizer);
    EXPECT_GT(std::filesystem::file_size(path), 1024U);
    const auto savedEmbedding = original.embedding().weight().download().packedWords();

    bb::BlackBitModel freshCpu(config, 999);
    freshCpu.setComputeDType(bb::ComputeDType::BF16);
    bbcuda::BlackBitModel restored(freshCpu);
    bbcuda::LowRankProjectedOptimizer restoredOptimizer(optimizerOptions);
    restored.registerParameters(restoredOptimizer);
    const bb::BlackBitTrainingState actualState =
        bbcuda::loadCheckpoint(path.string(), restored, &restoredOptimizer);
    EXPECT_EQ(actualState.step, expectedState.step);
    EXPECT_EQ(actualState.tokensSeen, expectedState.tokensSeen);
    EXPECT_EQ(actualState.rngSeed, expectedState.rngSeed);
    EXPECT_EQ(actualState.optimizerStep, expectedState.optimizerStep);
    EXPECT_EQ(restoredOptimizer.stepCount(), originalOptimizer.stepCount());
    EXPECT_EQ(restored.embedding().weight().download().packedWords(), savedEmbedding);

    const bb::BlackBitStepResult originalNext = original.trainStep(ids, targets, 1, 5, &originalOptimizer);
    const bb::BlackBitStepResult restoredNext = restored.trainStep(ids, targets, 1, 5, &restoredOptimizer);
    originalOptimizer.endStep();
    restoredOptimizer.endStep();
    EXPECT_NEAR(restoredNext.loss, originalNext.loss, 1.0e-6F);
    EXPECT_EQ(restored.embedding().weight().download().packedWords(),
              original.embedding().weight().download().packedWords());

    // The CUDA writer deliberately emits the existing portable BFBIT v1
    // format. Loading the same file through the CPU implementation proves
    // it did not serialize a GPU-only dense/master representation.
    bb::BlackBitModel cpuLoaded(config, 17);
    bb::LowRankProjectedOptimizer cpuOptimizer(optimizerOptions);
    cpuLoaded.registerParameters(cpuOptimizer);
    const bb::BlackBitTrainingState cpuState =
        bb::loadCheckpoint(path.string(), cpuLoaded, &cpuOptimizer);
    EXPECT_EQ(cpuState.tokensSeen, expectedState.tokensSeen);
    EXPECT_EQ(cpuLoaded.embedding().weight().packedWords(), savedEmbedding);
    EXPECT_EQ(cpuOptimizer.stepCount(), expectedState.optimizerStep);
    std::filesystem::remove(path);
}

TEST(CudaBlackBitModelTest, RealTextTrainerWritesPortableCheckpointTokenizerAndMetadata) {
    bb::BlackBitConfig config = modelConfig();
    config.vocabSize = 512;
    config.validate();
    const std::filesystem::path base = std::filesystem::temp_directory_path() /
                                       "blackforge_cuda_blackbit_text_trainer_test";
    const std::string datasetPath = base.string() + ".bfdata";
    const std::string tokenizerPath = base.string() + ".bftok";
    const std::string checkpointPath = base.string() + ".bfbit";
    blackforge::data::saveDataset(datasetPath, {4}, {4},
                                  {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F},
                                  {2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F}, 2);
    blackforge::tokenizer::saveTokenizer(blackforge::tokenizer::Tokenizer(), tokenizerPath);
    bbcuda::TrainingOptions options;
    options.datasetPath = datasetPath;
    options.tokenizerPath = tokenizerPath;
    options.saveCheckpoint = checkpointPath;
    options.steps = 1;
    options.optimizer.rank = 4;
    options.optimizer.learningRate = 0.25F;
    const bbcuda::TrainingResult result = bbcuda::train(config, options);
    EXPECT_EQ(result.finalStep, 1U);
    EXPECT_EQ(result.finalTokens, 4U);
    EXPECT_GT(result.ternaryFlips, 0U);
    EXPECT_TRUE(std::filesystem::exists(checkpointPath));
    EXPECT_TRUE(std::filesystem::exists(checkpointPath + ".bftok"));
    EXPECT_TRUE(std::filesystem::exists(result.manifestPath));
    std::ifstream metadata(result.manifestPath);
    const std::string contents((std::istreambuf_iterator<char>(metadata)),
                               std::istreambuf_iterator<char>());
    EXPECT_NE(contents.find("blackforge-blackbit-training-metadata-v1"), std::string::npos);
    EXPECT_NE(contents.find("\"sequence_length\": 4"), std::string::npos);
    EXPECT_NE(contents.find("\"tokenizer_vocab_size\": 259"), std::string::npos);
    metadata.close();
    std::filesystem::remove(datasetPath);
    std::filesystem::remove(tokenizerPath);
    std::filesystem::remove(checkpointPath);
    std::filesystem::remove(checkpointPath + ".bftok");
    std::filesystem::remove(result.manifestPath);
}

TEST(CudaBlackBitModelTest, SequenceLadderReusesOneModelAndMeasuresEachLengthIndependently) {
    bb::BenchmarkOptions options;
    options.microBatch = 1;
    options.steps = 1;
    options.warmupSteps = 0;
    options.optimizer.rank = 4;
    const auto results = bbcuda::runBenchmarkLadder(modelConfig(), options, {4, 8});
    ASSERT_EQ(results.size(), 2U);
    EXPECT_EQ(results[0].options.seqLen, 4U);
    EXPECT_EQ(results[1].options.seqLen, 8U);
    for (const auto& result : results) {
        EXPECT_GT(result.ternaryFlips, 0U);
        EXPECT_EQ(result.nanInfCount, 0U);
        EXPECT_GT(result.memory.devicePeakUsedBytes, 0U);
        EXPECT_GT(result.arenaPeakBytes[static_cast<std::size_t>(bbcuda::MemoryArena::GradientTiles)], 0U);
        EXPECT_TRUE(result.withinBudget);
    }
}
