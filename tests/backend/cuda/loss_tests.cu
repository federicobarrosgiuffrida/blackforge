#include "blackforge/backend/cuda/loss.hpp"

#include <gtest/gtest.h>

#include "blackforge/backend/cpu/loss.hpp"
#include "blackforge/backend/cuda/device_tensor.hpp"

using blackforge::runtime::Tensor;
namespace cpu = blackforge::backend::cpu;
namespace cuda = blackforge::backend::cuda;

TEST(CudaLossTest, MseCorrispondeAllaVersioneCpu) {
    Tensor pred({3}, {0.3F, -0.7F, 1.2F});
    Tensor target({3}, {0.5F, 0.1F, -0.4F});

    cpu::LossResult cpuResult = cpu::meanSquaredError(pred, target);
    cuda::LossResult gpuResult =
        cuda::meanSquaredError(cuda::DeviceTensor::fromHost(pred), cuda::DeviceTensor::fromHost(target));
    Tensor gpuGrad = gpuResult.grad.toHost();

    EXPECT_NEAR(gpuResult.value, cpuResult.value, 1e-5F);
    ASSERT_EQ(gpuGrad.elementCount(), cpuResult.grad.elementCount());
    for (std::size_t i = 0; i < gpuGrad.elementCount(); ++i) {
        EXPECT_NEAR(gpuGrad.at(i), cpuResult.grad.at(i), 1e-5F) << "indice " << i;
    }
}

TEST(CudaLossTest, MseValeZeroQuandoPredizioneETargetCoincidono) {
    Tensor pred({4}, {0.1F, -0.2F, 0.3F, 5.0F});
    cuda::LossResult result =
        cuda::meanSquaredError(cuda::DeviceTensor::fromHost(pred), cuda::DeviceTensor::fromHost(pred));
    EXPECT_NEAR(result.value, 0.0F, 1e-6F);
}

TEST(CudaLossTest, MseSuTensoreGrandeCorrispondeAllaVersioneCpu) {
    // Piu' grande del block size (256) del kernel di riduzione: verifica
    // che il grid-stride loop dentro sumReduceKernel sommi davvero tutti
    // gli elementi, non solo i primi 256.
    std::vector<float> predValues(500);
    std::vector<float> targetValues(500);
    for (std::size_t i = 0; i < 500; ++i) {
        predValues[i] = static_cast<float>(i % 11) - 5.0F;
        targetValues[i] = static_cast<float>(i % 7) - 3.0F;
    }
    Tensor pred({500}, predValues);
    Tensor target({500}, targetValues);

    cpu::LossResult cpuResult = cpu::meanSquaredError(pred, target);
    cuda::LossResult gpuResult =
        cuda::meanSquaredError(cuda::DeviceTensor::fromHost(pred), cuda::DeviceTensor::fromHost(target));

    EXPECT_NEAR(gpuResult.value, cpuResult.value, 1e-3F);
}

TEST(CudaLossTest, MseLanciaSuFormeIncompatibili) {
    Tensor pred({2}, {1.0F, 2.0F});
    Tensor target({3}, {1.0F, 2.0F, 3.0F});
    EXPECT_THROW((void)cuda::meanSquaredError(cuda::DeviceTensor::fromHost(pred), cuda::DeviceTensor::fromHost(target)),
                 std::invalid_argument);
}
