#include "blackforge/backend/cuda/autodiff.hpp"

#include <functional>

#include <gtest/gtest.h>

#include "blackforge/backend/cpu/autodiff.hpp"
#include "blackforge/backend/cpu/ops.hpp"
#include "blackforge/backend/cuda/device_tensor.hpp"
#include "blackforge/backend/cuda/ops.hpp"

using blackforge::runtime::Tensor;
namespace cpu = blackforge::backend::cpu;
namespace cuda = blackforge::backend::cuda;

namespace {

// Derivata numerica (differenze centrali) via round-trip host<->device:
// perturba l'input su host, esegue il forward CUDA, riporta il
// risultato su host per calcolare il prodotto scalare con gradOutput.
// Stesso principio dei test di gradient checking del backend CPU
// (tests/backend/cpu/autodiff_tests.cpp), applicato al backend CUDA.
float numericalDerivative(const std::function<float(const Tensor&)>& f, Tensor t, std::size_t index,
                           float eps = 1e-3F) {
    float original = t.at(index);
    t.at(index) = original + eps;
    float plus = f(t);
    t.at(index) = original - eps;
    float minus = f(t);
    return (plus - minus) / (2.0F * eps);
}

float dot(const Tensor& a, const Tensor& b) {
    float sum = 0.0F;
    for (std::size_t i = 0; i < a.elementCount(); ++i) {
        sum += a.at(i) * b.at(i);
    }
    return sum;
}

}  // namespace

TEST(CudaAutodiffTest, MatmulBackwardCorrispondeAllaVersioneCpu) {
    Tensor a({2, 3}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F});
    Tensor b({3, 2}, {0.7F, -0.1F, 0.2F, 0.3F, -0.4F, 0.5F});
    Tensor gradOutput({2, 2}, {1.0F, -1.0F, 0.5F, 2.0F});

    cpu::MatmulGrad cpuGrad = cpu::matmulBackward(a, b, gradOutput);
    cuda::MatmulGrad gpuGrad = cuda::matmulBackward(cuda::DeviceTensor::fromHost(a), cuda::DeviceTensor::fromHost(b),
                                                     cuda::DeviceTensor::fromHost(gradOutput));
    Tensor gpuDA = gpuGrad.dA.toHost();
    Tensor gpuDB = gpuGrad.dB.toHost();

    for (std::size_t i = 0; i < cpuGrad.dA.elementCount(); ++i) {
        EXPECT_NEAR(gpuDA.at(i), cpuGrad.dA.at(i), 1e-4F) << "dA indice " << i;
    }
    for (std::size_t i = 0; i < cpuGrad.dB.elementCount(); ++i) {
        EXPECT_NEAR(gpuDB.at(i), cpuGrad.dB.at(i), 1e-4F) << "dB indice " << i;
    }
}

TEST(CudaAutodiffTest, MatmulBackwardNonQuadratoCorrispondeAllaDerivataNumerica) {
    // M, N, K tutti diversi: un errore di indicizzazione (es. scambio
    // tra m/k/n dentro i kernel dA/dB) qui produrrebbe valori
    // visibilmente sbagliati o addirittura un crash per accesso fuori
    // dai limiti del buffer.
    Tensor a({3, 4}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
    Tensor b({4, 2}, {0.5F, -0.5F, 1.0F, 2.0F, -1.0F, 0.0F, 3.0F, 1.0F});
    Tensor gradOutput({3, 2}, {1.0F, -0.5F, 0.5F, 2.0F, -1.0F, 0.3F});

    cuda::MatmulGrad gpuGrad = cuda::matmulBackward(cuda::DeviceTensor::fromHost(a), cuda::DeviceTensor::fromHost(b),
                                                     cuda::DeviceTensor::fromHost(gradOutput));
    Tensor gpuDA = gpuGrad.dA.toHost();
    Tensor gpuDB = gpuGrad.dB.toHost();

    auto fA = [&](const Tensor& aVar) {
        Tensor out = cuda::matmul(cuda::DeviceTensor::fromHost(aVar), cuda::DeviceTensor::fromHost(b)).toHost();
        return dot(out, gradOutput);
    };
    for (std::size_t i = 0; i < a.elementCount(); ++i) {
        EXPECT_NEAR(gpuDA.at(i), numericalDerivative(fA, a, i), 1e-2F) << "dA indice " << i;
    }

    auto fB = [&](const Tensor& bVar) {
        Tensor out = cuda::matmul(cuda::DeviceTensor::fromHost(a), cuda::DeviceTensor::fromHost(bVar)).toHost();
        return dot(out, gradOutput);
    };
    for (std::size_t i = 0; i < b.elementCount(); ++i) {
        EXPECT_NEAR(gpuDB.at(i), numericalDerivative(fB, b, i), 1e-2F) << "dB indice " << i;
    }
}

TEST(CudaAutodiffTest, AddBiasBackwardCorrispondeAllaVersioneCpu) {
    Tensor gradOutput({3, 2}, {1.0F, 0.5F, -1.0F, 2.0F, 0.3F, -0.2F});

    cpu::AddBiasGrad cpuGrad = cpu::addBiasBackward(gradOutput);
    cuda::AddBiasGrad gpuGrad = cuda::addBiasBackward(cuda::DeviceTensor::fromHost(gradOutput));
    Tensor gpuDInput = gpuGrad.dInput.toHost();
    Tensor gpuDBias = gpuGrad.dBias.toHost();

    for (std::size_t i = 0; i < cpuGrad.dInput.elementCount(); ++i) {
        EXPECT_NEAR(gpuDInput.at(i), cpuGrad.dInput.at(i), 1e-5F) << "dInput indice " << i;
    }
    for (std::size_t i = 0; i < cpuGrad.dBias.elementCount(); ++i) {
        EXPECT_NEAR(gpuDBias.at(i), cpuGrad.dBias.at(i), 1e-4F) << "dBias indice " << i;
    }
}

TEST(CudaAutodiffTest, SiluBackwardCorrispondeAllaVersioneCpu) {
    Tensor input({4}, {-1.5F, -0.3F, 0.7F, 2.0F});
    Tensor gradOutput({4}, {1.0F, -0.5F, 2.0F, 0.3F});

    Tensor cpuGrad = cpu::siluBackward(input, gradOutput);
    Tensor gpuGrad =
        cuda::siluBackward(cuda::DeviceTensor::fromHost(input), cuda::DeviceTensor::fromHost(gradOutput)).toHost();

    for (std::size_t i = 0; i < cpuGrad.elementCount(); ++i) {
        EXPECT_NEAR(gpuGrad.at(i), cpuGrad.at(i), 1e-5F) << "indice " << i;
    }
}

TEST(CudaAutodiffTest, ReluBackwardCorrispondeAllaVersioneCpu) {
    Tensor input({4}, {-2.0F, -0.4F, 0.6F, 3.0F});
    Tensor gradOutput({4}, {1.0F, -0.5F, 2.0F, 0.3F});

    Tensor cpuGrad = cpu::reluBackward(input, gradOutput);
    Tensor gpuGrad =
        cuda::reluBackward(cuda::DeviceTensor::fromHost(input), cuda::DeviceTensor::fromHost(gradOutput)).toHost();

    for (std::size_t i = 0; i < cpuGrad.elementCount(); ++i) {
        EXPECT_NEAR(gpuGrad.at(i), cpuGrad.at(i), 1e-5F) << "indice " << i;
    }
}

TEST(CudaAutodiffTest, GeluBackwardCorrispondeAllaVersioneCpu) {
    Tensor input({4}, {-1.5F, -0.3F, 0.7F, 2.0F});
    Tensor gradOutput({4}, {1.0F, -0.5F, 2.0F, 0.3F});

    Tensor cpuGrad = cpu::geluBackward(input, gradOutput);
    Tensor gpuGrad =
        cuda::geluBackward(cuda::DeviceTensor::fromHost(input), cuda::DeviceTensor::fromHost(gradOutput)).toHost();

    for (std::size_t i = 0; i < cpuGrad.elementCount(); ++i) {
        EXPECT_NEAR(gpuGrad.at(i), cpuGrad.at(i), 1e-5F) << "indice " << i;
    }
}

TEST(CudaAutodiffTest, RmsnormBackwardCorrispondeAllaVersioneCpu) {
    Tensor input({2, 3}, {0.5F, -1.2F, 0.3F, 2.0F, 0.1F, -0.5F});
    Tensor gradOutput({2, 3}, {1.0F, -0.5F, 2.0F, 0.3F, -1.0F, 0.7F});

    Tensor cpuGrad = cpu::rmsnormBackward(input, gradOutput);
    Tensor gpuGrad =
        cuda::rmsnormBackward(cuda::DeviceTensor::fromHost(input), cuda::DeviceTensor::fromHost(gradOutput))
            .toHost();

    for (std::size_t i = 0; i < cpuGrad.elementCount(); ++i) {
        EXPECT_NEAR(gpuGrad.at(i), cpuGrad.at(i), 1e-4F) << "indice " << i;
    }
}

TEST(CudaAutodiffTest, RmsnormBackwardCorrispondeAllaDerivataNumerica) {
    Tensor input({2, 3}, {0.5F, -1.2F, 0.3F, 2.0F, 0.1F, -0.5F});
    Tensor gradOutput({2, 3}, {1.0F, -0.5F, 2.0F, 0.3F, -1.0F, 0.7F});

    Tensor gpuGrad =
        cuda::rmsnormBackward(cuda::DeviceTensor::fromHost(input), cuda::DeviceTensor::fromHost(gradOutput))
            .toHost();

    auto f = [&](const Tensor& x) {
        Tensor out = cuda::rmsnorm(cuda::DeviceTensor::fromHost(x)).toHost();
        return dot(out, gradOutput);
    };
    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(gpuGrad.at(i), numericalDerivative(f, input, i), 1e-2F) << "indice " << i;
    }
}
