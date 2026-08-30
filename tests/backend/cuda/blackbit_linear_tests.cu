#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/blackbit/cuda_ternary_linear.hpp"
#include "blackforge/blackbit/gradient.hpp"
#include "blackforge/blackbit/ternary_linear.hpp"
#include "blackforge/runtime/tensor.hpp"

namespace bb = blackforge::blackbit;
namespace bbcuda = blackforge::blackbit::cuda;
using blackforge::runtime::Tensor;

namespace {

Tensor values(std::vector<std::size_t> shape, float scale, float phase = 0.0F) {
    std::size_t count = 1;
    for (std::size_t dimension : shape) count *= dimension;
    std::vector<float> data(count);
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.37F + phase) * scale;
    }
    return Tensor(std::move(shape), std::move(data));
}

void expectNear(const Tensor& actual, const Tensor& expected, float tolerance) {
    ASSERT_EQ(actual.shape(), expected.shape());
    for (std::size_t i = 0; i < actual.elementCount(); ++i) {
        EXPECT_NEAR(actual.data()[i], expected.data()[i], tolerance) << "index " << i;
    }
}

class DeviceGradientCollector final : public bbcuda::GradientSink {
public:
    explicit DeviceGradientCollector(std::size_t values) : gradient(values, 0.0F) {}

    void consumeWeightGradientBlock(const bb::ParameterId& id, std::size_t firstRow, std::size_t rowCount,
                                     const float* deviceBlock) override {
        std::vector<float> host(rowCount * id.cols);
        BLACKFORGE_CUDA_CHECK(
            cudaMemcpy(host.data(), deviceBlock, host.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::copy(host.begin(), host.end(), gradient.begin() + static_cast<std::ptrdiff_t>(firstRow * id.cols));
        peakBlockBytes = std::max(peakBlockBytes, host.size() * sizeof(float));
        ++blocks;
    }

    std::vector<float> gradient;
    std::size_t peakBlockBytes = 0;
    std::size_t blocks = 0;
};

bb::TernaryLinear referenceLinear(std::size_t tileRows) {
    bb::TernaryLinear linear("test.linear", 13, 11, 20, tileRows);
    linear.loadDense(values({11, 13}, 0.35F, 0.2F));
    linear.setComputeDType(bb::ComputeDType::BF16);
    return linear;
}

}  // namespace

TEST(CudaBlackBitLinearTest, ForwardMatchesCpuReferenceWithBf16Tolerance) {
    auto cpu = referenceLinear(3);
    bbcuda::TernaryLinear gpu(cpu);
    const Tensor input = values({2, 4, 13}, 0.7F, 0.4F);

    const Tensor expected = cpu.forward(input);
    const auto deviceInput = bbcuda::Tensor::fromHost(input);
    const Tensor actual = gpu.forward(deviceInput).toHost();

    expectNear(actual, expected, 1.5e-2F);
    EXPECT_GT(gpu.metrics().decodeMs, 0.0);
    EXPECT_GT(gpu.metrics().gemmMs, 0.0);
    EXPECT_GT(gpu.metrics().decodeGigabytesPerSecond(), 0.0);
}

TEST(CudaBlackBitLinearTest, ForwardDoesNotDependOnTileRows) {
    auto cpuSmallTile = referenceLinear(2);
    auto cpuLargeTile = referenceLinear(11);
    bbcuda::TernaryLinear small(cpuSmallTile);
    bbcuda::TernaryLinear large(cpuLargeTile);
    const Tensor input = values({5, 13}, 0.4F, 0.7F);
    const auto deviceInput = bbcuda::Tensor::fromHost(input);
    const auto deviceInput2 = bbcuda::Tensor::fromHost(input);

    expectNear(small.forward(deviceInput).toHost(), large.forward(deviceInput2).toHost(), 1.0e-5F);
    EXPECT_EQ(small.metrics().dequantWorkspaceBytes, 2 * 13 * sizeof(std::uint16_t));
    EXPECT_LT(small.metrics().dequantWorkspaceBytes, 11 * 13 * sizeof(std::uint16_t));
}

TEST(CudaBlackBitLinearTest, BackwardMatchesCpuAndStreamsWeightGradient) {
    auto cpu = referenceLinear(3);
    bbcuda::TernaryLinear gpu(cpu);
    const Tensor input = values({6, 13}, 0.5F, 0.1F);
    const Tensor gradOutput = values({6, 11}, 0.2F, 0.8F);

    bb::DenseGradientCollector cpuCollector;
    const Tensor expectedInputGradient = cpu.backward(input, gradOutput, &cpuCollector);

    const auto deviceInput = bbcuda::Tensor::fromHost(input);
    const auto deviceGradOutput = bbcuda::Tensor::fromHost(gradOutput);
    DeviceGradientCollector gpuCollector(11 * 13);
    bbcuda::resetGradientLifetimeStats();
    const Tensor actualInputGradient = gpu.backward(deviceInput, deviceGradOutput, &gpuCollector).toHost();

    expectNear(actualInputGradient, expectedInputGradient, 2.5e-2F);
    const auto& expectedWeightGradient = cpuCollector.gradient("test.linear");
    ASSERT_EQ(gpuCollector.gradient.size(), expectedWeightGradient.size());
    for (std::size_t i = 0; i < expectedWeightGradient.size(); ++i) {
        EXPECT_NEAR(gpuCollector.gradient[i], expectedWeightGradient[i], 2.5e-2F) << "weight index " << i;
    }

    const auto& lifetime = bbcuda::gradientLifetimeStats();
    EXPECT_EQ(lifetime.liveBytes, 0U);
    EXPECT_EQ(lifetime.blocksProduced, 4U);
    EXPECT_EQ(lifetime.blocksProduced, lifetime.blocksReleased);
    EXPECT_EQ(lifetime.peakLiveBytes, 3 * 13 * sizeof(float));
    EXPECT_EQ(gpuCollector.peakBlockBytes, lifetime.peakLiveBytes);
    EXPECT_GT(lifetime.reuseRatio(), 3.0);
    EXPECT_LT(lifetime.peakLiveBytes, 11 * 13 * sizeof(float));
}

TEST(CudaBlackBitLinearTest, RejectsIncompatibleShapes) {
    auto cpu = referenceLinear(3);
    bbcuda::TernaryLinear gpu(cpu);
    const auto wrongInput = bbcuda::Tensor::fromHost(values({2, 12}, 1.0F));
    EXPECT_THROW(
        {
            auto ignored = gpu.forward(wrongInput);
            (void)ignored;
        },
        std::invalid_argument);
}
