#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/blackbit/cuda_moe.hpp"
#include "blackforge/blackbit/gradient.hpp"
#include "blackforge/blackbit/moe.hpp"
#include "blackforge/runtime/tensor.hpp"

namespace bb = blackforge::blackbit;
namespace bbcuda = blackforge::blackbit::cuda;
using blackforge::runtime::Tensor;

namespace {

bb::BlackBitConfig moeConfig() {
    bb::BlackBitConfig config;
    config.name = "cuda-moe-test";
    config.vocabSize = 64;
    config.hiddenSize = 12;
    config.numLayers = 1;
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

Tensor values(std::vector<std::size_t> shape, float scale, float phase = 0.0F) {
    std::size_t count = 1;
    for (const std::size_t dimension : shape) count *= dimension;
    std::vector<float> data(count);
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.23F + phase) * scale;
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

    void consumeDenseGradient(const bb::ParameterId& id, const float* values, std::size_t count) override {
        auto& host = gradients[id.name];
        host.resize(count);
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(host.data(), values, count * sizeof(float), cudaMemcpyDeviceToHost));
    }

    std::unordered_map<std::string, std::vector<float>> gradients;
};

}  // namespace

TEST(CudaBlackBitMoETest, SparseTop2ForwardMatchesCpuRoutingAndOutput) {
    const bb::BlackBitConfig config = moeConfig();
    bb::MoELayer cpu("moe", config);
    cpu.initialize(51);
    cpu.setComputeDType(bb::ComputeDType::BF16);
    bbcuda::MoELayer gpu(cpu, config);
    const Tensor input = values({2, 5, config.hiddenSize}, 0.55F, 0.3F);
    bb::MoECache cpuCache;
    bb::MoERoutingStats cpuStats;
    const Tensor expected = cpu.forward(input, cpuCache, cpuStats);
    bbcuda::MoECache gpuCache;
    bb::MoERoutingStats gpuStats;
    const auto deviceInput = bbcuda::Tensor::fromHost(input);
    const Tensor actual = gpu.forward(deviceInput, gpuCache, gpuStats).toHost();

    expectNear(actual, expected, 3.0e-2F);
    EXPECT_EQ(gpuCache.expertOfSlot.toHost(), cpuCache.expertOfSlot);
    EXPECT_EQ(gpuStats.tokensPerExpert, cpuStats.tokensPerExpert);
    EXPECT_EQ(gpuStats.droppedAssignments, cpuStats.droppedAssignments);
    EXPECT_NEAR(gpuStats.routingEntropy, cpuStats.routingEntropy, 2.0e-6);
    EXPECT_NEAR(gpuStats.loadBalancingLoss, cpuStats.loadBalancingLoss, 2.0e-6F);
    EXPECT_EQ(gpuCache.tokensOfExpert.elementCount(), config.numExperts * gpuCache.capacity);
    EXPECT_LT(gpuCache.expertOutputOfSlot.elementCount(), input.dim(0) * input.dim(1) *
                                                            config.numExperts * config.hiddenSize);
}

TEST(CudaBlackBitMoETest, BackwardMatchesCpuForExpertsRouterAndInput) {
    const bb::BlackBitConfig config = moeConfig();
    bb::MoELayer cpu("moe", config);
    cpu.initialize(81);
    cpu.setComputeDType(bb::ComputeDType::BF16);
    bbcuda::MoELayer gpu(cpu, config);
    const Tensor input = values({1, 6, config.hiddenSize}, 0.45F, 0.2F);
    const Tensor gradOutput = values({1, 6, config.hiddenSize}, 0.17F, 0.8F);
    bb::MoECache cpuCache;
    bb::MoERoutingStats cpuStats;
    (void)cpu.forward(input, cpuCache, cpuStats);
    bb::DenseGradientCollector cpuGradients;
    const Tensor expectedGradInput = cpu.backward(input, gradOutput, cpuCache, cpuStats, &cpuGradients);

    const auto deviceInput = bbcuda::Tensor::fromHost(input);
    const auto deviceGradOutput = bbcuda::Tensor::fromHost(gradOutput);
    bbcuda::MoECache gpuCache;
    bb::MoERoutingStats gpuStats;
    (void)gpu.forward(deviceInput, gpuCache, gpuStats);
    EXPECT_EQ(gpuStats.droppedAssignments, 0U);
    DeviceGradientCollector gpuGradients;
    bbcuda::resetGradientLifetimeStats();
    const Tensor actualGradInput = gpu.backward(deviceInput, deviceGradOutput, gpuCache, gpuStats,
                                                &gpuGradients).toHost();

    expectNear(actualGradInput, expectedGradInput, 5.0e-2F);
    for (const auto& [name, actual] : gpuGradients.gradients) {
        ASSERT_TRUE(cpuGradients.has(name)) << name;
        const auto& expected = cpuGradients.gradient(name);
        ASSERT_EQ(actual.size(), expected.size()) << name;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], 5.0e-2F) << name << " index " << i;
        }
    }
    EXPECT_TRUE(gpuGradients.gradients.contains("moe.router"));
    EXPECT_EQ(bbcuda::gradientLifetimeStats().liveBytes, 0U);
}

TEST(CudaBlackBitMoETest, CapacityOverflowIsReportedRatherThanAllocatingDenseExpertTensor) {
    bb::BlackBitConfig config = moeConfig();
    config.expertCapacityFactor = 0.25F;
    bb::MoELayer cpu("moe", config);
    cpu.initialize(19);
    cpu.setComputeDType(bb::ComputeDType::BF16);
    bbcuda::MoELayer gpu(cpu, config);
    const Tensor input = values({1, 8, config.hiddenSize}, 0.4F, 0.6F);
    const auto deviceInput = bbcuda::Tensor::fromHost(input);
    bbcuda::MoECache cache;
    bb::MoERoutingStats stats;
    (void)gpu.forward(deviceInput, cache, stats);
    EXPECT_GT(stats.droppedAssignments, 0U);
    EXPECT_EQ(stats.assignments, 8 * config.expertsPerToken);
    EXPECT_EQ(cache.tokensOfExpert.elementCount(), config.numExperts * gpu.capacityFor(8));
    EXPECT_LT(cache.tokensOfExpert.elementCount(), 8 * config.numExperts);
}

// Semantica corretta del Top-2: quando la capacita' TOTALE basta, il
// router non deve MAI scartare un'assegnazione, nemmeno se ogni token
// preferisce gli stessi due esperti. Il vecchio percorso, che scartava
// lo slot quando l'esperto preferito era pieno, falliva esattamente qui
// e faceva sembrare il modello piu' veloce perche' eseguiva meno MoE.
TEST(CudaBlackBitMoETest, SkewedRouterKeepsTwoDistinctExpertsPerTokenWhenTotalCapacityIsSufficient) {
    bb::BlackBitConfig config = moeConfig();
    config.expertCapacityFactor = 1.0F;  // capacita' totale ESATTAMENTE pari alle assegnazioni
    config.validate();

    bb::MoELayer cpu("moe", config);
    cpu.initialize(37);
    cpu.setComputeDType(bb::ComputeDType::BF16);

    // Router deliberatamente collassato: con un input tutto positivo il
    // logit dell'esperto e vale k_e * somma(x), quindi OGNI token ordina
    // gli esperti allo stesso modo (0 > 1 > 2 > 3). E' il caso peggiore
    // per la capacita' ed e' quello che il test deve coprire.
    const std::array<float, 4> skew{2.0F, 1.0F, 0.25F, 0.05F};
    ASSERT_EQ(config.numExperts, skew.size());
    auto& routerWeight = cpu.router().weight();
    for (std::size_t expert = 0; expert < config.numExperts; ++expert) {
        for (std::size_t h = 0; h < config.hiddenSize; ++h) {
            routerWeight[expert * config.hiddenSize + h] = skew[expert];
        }
    }

    bbcuda::MoELayer gpu(cpu, config);

    const std::size_t tokens = 16;
    std::vector<float> data(tokens * config.hiddenSize);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = 0.5F + 0.25F * std::abs(std::sin(static_cast<float>(i) * 0.31F));
    }
    const Tensor input({1, tokens, config.hiddenSize}, std::move(data));

    const std::size_t capacity = cpu.capacityFor(tokens);
    const std::size_t assignments = tokens * config.expertsPerToken;
    ASSERT_EQ(capacity * config.numExperts, assignments)
        << "il test deve girare al limite esatto di capacita' totale";

    bb::MoECache cpuCache;
    bb::MoERoutingStats cpuStats;
    const Tensor expected = cpu.forward(input, cpuCache, cpuStats);

    bbcuda::MoECache gpuCache;
    bb::MoERoutingStats gpuStats;
    const auto deviceInput = bbcuda::Tensor::fromHost(input);
    const Tensor actual = gpu.forward(deviceInput, gpuCache, gpuStats).toHost();

    // (2) e (4): nessuno scarto artificiale, la capacita' totale basta.
    EXPECT_EQ(cpuStats.droppedAssignments, 0U);
    EXPECT_EQ(gpuStats.droppedAssignments, 0U);
    EXPECT_EQ(gpuStats.assignments, assignments);

    // (3): CPU e CUDA scelgono ESATTAMENTE gli stessi esperti.
    const std::vector<int> gpuExpertOfSlot = gpuCache.expertOfSlot.toHost();
    EXPECT_EQ(gpuExpertOfSlot, cpuCache.expertOfSlot);
    expectNear(actual, expected, 3.0e-2F);

    // (1): due assegnazioni valide e DISTINTE per ogni token.
    for (std::size_t token = 0; token < tokens; ++token) {
        const int first = gpuExpertOfSlot[token * config.expertsPerToken];
        const int second = gpuExpertOfSlot[token * config.expertsPerToken + 1];
        EXPECT_GE(first, 0) << "token " << token;
        EXPECT_GE(second, 0) << "token " << token;
        EXPECT_NE(first, second) << "token " << token;
    }

    // (8): al limite di capacita' l'utilizzo risulta perfettamente
    // bilanciato, perche' ogni esperto viene riempito fino alla quota.
    for (std::size_t expert = 0; expert < config.numExperts; ++expert) {
        EXPECT_EQ(gpuStats.tokensPerExpert[expert], capacity) << "esperto " << expert;
    }
    EXPECT_EQ(gpuStats.tokensPerExpert, cpuStats.tokensPerExpert);

    // (5) e (6): il router resta addestrabile e OGNI esperto scelto
    // riceve gradiente, non solo il preferito.
    const Tensor gradOutput = values({1, tokens, config.hiddenSize}, 0.19F, 0.4F);
    const auto deviceGradOutput = bbcuda::Tensor::fromHost(gradOutput);
    DeviceGradientCollector gpuGradients;
    (void)gpu.backward(deviceInput, deviceGradOutput, gpuCache, gpuStats, &gpuGradients);
    EXPECT_TRUE(gpuGradients.gradients.contains("moe.router"));
    const auto& routerGradient = gpuGradients.gradients.at("moe.router");
    EXPECT_TRUE(std::any_of(routerGradient.begin(), routerGradient.end(),
                            [](float v) { return v != 0.0F; }))
        << "il gradiente del router deve essere attivo";
    for (std::size_t expert = 0; expert < config.numExperts; ++expert) {
        const std::string prefix = "moe.expert" + std::to_string(expert);
        for (const std::string suffix : {".gate", ".up", ".down"}) {
            const std::string name = prefix + suffix;
            ASSERT_TRUE(gpuGradients.gradients.contains(name)) << name;
            const auto& gradient = gpuGradients.gradients.at(name);
            EXPECT_TRUE(std::any_of(gradient.begin(), gradient.end(), [](float v) { return v != 0.0F; }))
                << name << " deve ricevere un gradiente non nullo";
        }
    }
}
