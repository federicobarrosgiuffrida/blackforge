#include "blackforge/backend/cuda/ops.hpp"

#include <gtest/gtest.h>

#include "blackforge/backend/cpu/ops.hpp"
#include "blackforge/backend/cuda/device_tensor.hpp"

using blackforge::runtime::Tensor;
namespace cpu = blackforge::backend::cpu;
namespace cuda = blackforge::backend::cuda;

TEST(CudaDeviceTensorTest, RoundTripHostDeviceHostPreservaIValori) {
    Tensor host({2, 3}, {1.0F, -2.0F, 3.5F, 0.0F, -4.5F, 6.0F});

    cuda::DeviceTensor device = cuda::DeviceTensor::fromHost(host);
    EXPECT_EQ(device.shape(), host.shape());

    Tensor roundTripped = device.toHost();
    ASSERT_EQ(roundTripped.elementCount(), host.elementCount());
    for (std::size_t i = 0; i < host.elementCount(); ++i) {
        EXPECT_FLOAT_EQ(roundTripped.at(i), host.at(i));
    }
}

TEST(CudaOpsTest, AddCorrispondeAllaVersioneCpu) {
    Tensor a({4}, {1.0F, 2.0F, -3.0F, 4.5F});
    Tensor b({4}, {10.0F, -20.0F, 30.0F, 0.5F});

    Tensor cpuResult = cpu::add(a, b);
    Tensor gpuResult = cuda::add(cuda::DeviceTensor::fromHost(a), cuda::DeviceTensor::fromHost(b)).toHost();

    for (std::size_t i = 0; i < cpuResult.elementCount(); ++i) {
        EXPECT_NEAR(gpuResult.at(i), cpuResult.at(i), 1e-5F) << "indice " << i;
    }
}

TEST(CudaOpsTest, AddBiasCorrispondeAllaVersioneCpu) {
    Tensor input({3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    Tensor bias({2}, {10.0F, -10.0F});

    Tensor cpuResult = cpu::addBias(input, bias);
    Tensor gpuResult =
        cuda::addBias(cuda::DeviceTensor::fromHost(input), cuda::DeviceTensor::fromHost(bias)).toHost();

    for (std::size_t i = 0; i < cpuResult.elementCount(); ++i) {
        EXPECT_NEAR(gpuResult.at(i), cpuResult.at(i), 1e-5F) << "indice " << i;
    }
}

TEST(CudaOpsTest, MatmulCalcolaIlProdottoCorretto) {
    // Stessi valori del test CPU corrispondente: verifica sia la
    // correttezza assoluta (valori noti) sia, indirettamente, che la
    // conversione row-major/column-major per cuBLAS sia quella giusta.
    Tensor a({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    Tensor b({2, 2}, {5.0F, 6.0F, 7.0F, 8.0F});

    Tensor result = cuda::matmul(cuda::DeviceTensor::fromHost(a), cuda::DeviceTensor::fromHost(b)).toHost();

    ASSERT_EQ(result.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_NEAR(result.at(0), 19.0F, 1e-4F);  // 1*5 + 2*7
    EXPECT_NEAR(result.at(1), 22.0F, 1e-4F);  // 1*6 + 2*8
    EXPECT_NEAR(result.at(2), 43.0F, 1e-4F);  // 3*5 + 4*7
    EXPECT_NEAR(result.at(3), 50.0F, 1e-4F);  // 3*6 + 4*8
}

TEST(CudaOpsTest, MatmulNonQuadratoCorrispondeAllaVersioneCpu) {
    // [3,4] x [4,2] -> [3,2]: verifica la formula cuBLAS anche quando
    // M, N, K sono tutti diversi tra loro (un caso quadrato potrebbe
    // nascondere uno scambio errato di m/n).
    Tensor a({3, 4}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
    Tensor b({4, 2}, {0.5F, -0.5F, 1.0F, 2.0F, -1.0F, 0.0F, 3.0F, 1.0F});

    Tensor cpuResult = cpu::matmul(a, b);
    Tensor gpuResult = cuda::matmul(cuda::DeviceTensor::fromHost(a), cuda::DeviceTensor::fromHost(b)).toHost();

    ASSERT_EQ(gpuResult.shape(), cpuResult.shape());
    for (std::size_t i = 0; i < cpuResult.elementCount(); ++i) {
        EXPECT_NEAR(gpuResult.at(i), cpuResult.at(i), 1e-3F) << "indice " << i;
    }
}

TEST(CudaOpsTest, LinearCorrispondeAllaVersioneCpu) {
    Tensor input({2, 3}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F});
    Tensor weight({3, 4}, {0.1F, 0.2F, -0.1F, 0.3F, 0.05F, -0.2F, 0.4F, 0.1F, -0.3F, 0.1F, 0.2F, -0.05F});
    Tensor bias({4}, {1.0F, -1.0F, 0.5F, 0.0F});

    Tensor cpuResult = cpu::linear(input, weight, bias);
    Tensor gpuResult = cuda::linear(cuda::DeviceTensor::fromHost(input), cuda::DeviceTensor::fromHost(weight),
                                     cuda::DeviceTensor::fromHost(bias))
                            .toHost();

    ASSERT_EQ(gpuResult.shape(), cpuResult.shape());
    for (std::size_t i = 0; i < cpuResult.elementCount(); ++i) {
        EXPECT_NEAR(gpuResult.at(i), cpuResult.at(i), 1e-3F) << "indice " << i;
    }
}

TEST(CudaOpsTest, SiluCorrispondeAllaVersioneCpu) {
    Tensor input({5}, {-3.0F, -0.5F, 0.0F, 0.5F, 3.0F});
    Tensor cpuResult = cpu::silu(input);
    Tensor gpuResult = cuda::silu(cuda::DeviceTensor::fromHost(input)).toHost();

    for (std::size_t i = 0; i < cpuResult.elementCount(); ++i) {
        EXPECT_NEAR(gpuResult.at(i), cpuResult.at(i), 1e-5F) << "indice " << i;
    }
}

TEST(CudaOpsTest, ReluCorrispondeAllaVersioneCpu) {
    Tensor input({4}, {-2.0F, 0.0F, 0.5F, 3.0F});
    Tensor cpuResult = cpu::relu(input);
    Tensor gpuResult = cuda::relu(cuda::DeviceTensor::fromHost(input)).toHost();

    for (std::size_t i = 0; i < cpuResult.elementCount(); ++i) {
        EXPECT_NEAR(gpuResult.at(i), cpuResult.at(i), 1e-5F) << "indice " << i;
    }
}

TEST(CudaOpsTest, GeluCorrispondeAllaVersioneCpu) {
    Tensor input({5}, {-3.0F, -0.5F, 0.0F, 0.5F, 3.0F});
    Tensor cpuResult = cpu::gelu(input);
    Tensor gpuResult = cuda::gelu(cuda::DeviceTensor::fromHost(input)).toHost();

    for (std::size_t i = 0; i < cpuResult.elementCount(); ++i) {
        EXPECT_NEAR(gpuResult.at(i), cpuResult.at(i), 1e-5F) << "indice " << i;
    }
}

TEST(CudaOpsTest, RmsnormCorrispondeAllaVersioneCpu) {
    Tensor input({2, 4}, {1.0F, 2.0F, 3.0F, 4.0F, -0.5F, 2.5F, -1.5F, 0.5F});

    Tensor cpuResult = cpu::rmsnorm(input);
    Tensor gpuResult = cuda::rmsnorm(cuda::DeviceTensor::fromHost(input)).toHost();

    ASSERT_EQ(gpuResult.shape(), cpuResult.shape());
    for (std::size_t i = 0; i < cpuResult.elementCount(); ++i) {
        EXPECT_NEAR(gpuResult.at(i), cpuResult.at(i), 1e-4F) << "indice " << i;
    }
}

TEST(CudaOpsTest, RmsnormConFeatureMaggioriDelBlockSizeCorrispondeAllaVersioneCpu) {
    // features (300) supera il numero di thread per blocco (256): serve
    // a esercitare davvero il ciclo grid-stride dentro il kernel, non
    // solo il caso in cui un thread copre un solo elemento a testa.
    std::vector<float> values(300);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i % 7) - 3.0F;
    }
    Tensor input({1, 300}, values);

    Tensor cpuResult = cpu::rmsnorm(input);
    Tensor gpuResult = cuda::rmsnorm(cuda::DeviceTensor::fromHost(input)).toHost();

    ASSERT_EQ(gpuResult.shape(), cpuResult.shape());
    for (std::size_t i = 0; i < cpuResult.elementCount(); ++i) {
        EXPECT_NEAR(gpuResult.at(i), cpuResult.at(i), 1e-4F) << "indice " << i;
    }
}

TEST(CudaOpsTest, RmsnormLanciaSeNonERango2) {
    Tensor input({4}, {1.0F, 2.0F, 3.0F, 4.0F});
    EXPECT_THROW((void)cuda::rmsnorm(cuda::DeviceTensor::fromHost(input)), std::invalid_argument);
}

TEST(CudaOpsTest, SoftmaxCorrispondeAllaVersioneCpu) {
    Tensor input({2, 4}, {1.0F, 2.0F, 3.0F, 0.5F, -1.0F, 0.0F, 2.5F, 1.5F});

    Tensor cpuResult = cpu::softmax(input);
    Tensor gpuResult = cuda::softmax(cuda::DeviceTensor::fromHost(input)).toHost();

    ASSERT_EQ(gpuResult.shape(), cpuResult.shape());
    for (std::size_t i = 0; i < cpuResult.elementCount(); ++i) {
        EXPECT_NEAR(gpuResult.at(i), cpuResult.at(i), 1e-5F) << "indice " << i;
    }
}

TEST(CudaOpsTest, SoftmaxConFeatureMaggioriDelBlockSizeCorrispondeAllaVersioneCpu) {
    std::vector<float> values(300);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i % 13) - 6.0F;
    }
    Tensor input({1, 300}, values);

    Tensor cpuResult = cpu::softmax(input);
    Tensor gpuResult = cuda::softmax(cuda::DeviceTensor::fromHost(input)).toHost();

    for (std::size_t i = 0; i < cpuResult.elementCount(); ++i) {
        EXPECT_NEAR(gpuResult.at(i), cpuResult.at(i), 1e-5F) << "indice " << i;
    }
}

TEST(CudaOpsTest, SoftmaxLanciaSeNonERango2) {
    Tensor input({4}, {1.0F, 2.0F, 3.0F, 4.0F});
    EXPECT_THROW((void)cuda::softmax(cuda::DeviceTensor::fromHost(input)), std::invalid_argument);
}

TEST(CudaOpsTest, AddLanciaSuFormeIncompatibili) {
    Tensor a({2}, {1.0F, 2.0F});
    Tensor b({3}, {1.0F, 2.0F, 3.0F});
    EXPECT_THROW((void)cuda::add(cuda::DeviceTensor::fromHost(a), cuda::DeviceTensor::fromHost(b)),
                 std::invalid_argument);
}
