#include "blackforge/backend/cuda/autodiff.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

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

TEST(CudaAutodiffTest, MatmulBf16BackwardCorrispondeApprossimativamenteAMatmulBackwardFp32) {
    // Stessa formula analitica di matmulBackward (dA = gradOutput @ B^T,
    // dB = A^T @ gradOutput), ma via Tensor Core BF16: tolleranza
    // deliberatamente piu' larga della controparte FP32 (vedi il
    // commento equivalente in ops_tests.cu/MatmulBf16Corrisponde...).
    Tensor a({3, 4}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
    Tensor b({4, 2}, {0.5F, -0.5F, 1.0F, 2.0F, -1.0F, 0.0F, 3.0F, 1.0F});
    Tensor gradOutput({3, 2}, {1.0F, -0.5F, 0.5F, 2.0F, -1.0F, 0.3F});

    cuda::MatmulGrad fp32Grad = cuda::matmulBackward(cuda::DeviceTensor::fromHost(a), cuda::DeviceTensor::fromHost(b),
                                                       cuda::DeviceTensor::fromHost(gradOutput));
    cuda::MatmulGrad bf16Grad = cuda::matmulBf16Backward(
        cuda::DeviceTensor::fromHost(a), cuda::DeviceTensor::fromHost(b), cuda::DeviceTensor::fromHost(gradOutput));

    Tensor fp32DA = fp32Grad.dA.toHost();
    Tensor bf16DA = bf16Grad.dA.toHost();
    Tensor fp32DB = fp32Grad.dB.toHost();
    Tensor bf16DB = bf16Grad.dB.toHost();

    for (std::size_t i = 0; i < fp32DA.elementCount(); ++i) {
        float tolerance = std::max(0.1F, std::abs(fp32DA.at(i)) * 0.05F);
        EXPECT_NEAR(bf16DA.at(i), fp32DA.at(i), tolerance) << "dA indice " << i;
    }
    for (std::size_t i = 0; i < fp32DB.elementCount(); ++i) {
        float tolerance = std::max(0.1F, std::abs(fp32DB.at(i)) * 0.05F);
        EXPECT_NEAR(bf16DB.at(i), fp32DB.at(i), tolerance) << "dB indice " << i;
    }
}

TEST(CudaAutodiffTest, MatmulBf16BackwardCorrispondeAllaDerivataNumerica) {
    // Gradient check numerico diretto (non solo confronto con la
    // versione FP32): usa matmulBf16 stesso come funzione forward per
    // la derivata numerica, cosi' la formula di backward viene
    // verificata contro la stessa funzione a cui si applica davvero,
    // non contro un proxy FP32.
    //
    // eps deliberatamente molto piu' grande del default (1e-3): l'input
    // perturbato viene convertito in BF16 PRIMA del prodotto (7 bit di
    // mantissa, passo di quantizzazione ~0.3*2^-7 =~ 0.0023 per valori
    // dell'ordine di 0.3), quindi una perturbazione piu' piccola del
    // passo di quantizzazione puo' venire riassorbita dall'arrotondamento
    // e produrre una derivata numerica che misura solo rumore di
    // quantizzazione, non il vero gradiente. matmul() e' lineare in
    // ciascun operando, quindi un eps piu' grande non introduce errore
    // di curvatura (a differenza di una funzione non lineare) — qui
    // serve solo a superare il "gradino" di quantizzazione BF16.
    Tensor a({2, 3}, {0.3F, -0.6F, 0.2F, -0.4F, 0.5F, 0.1F});
    Tensor b({3, 2}, {0.2F, -0.1F, 0.4F, 0.1F, -0.3F, 0.2F});
    Tensor gradOutput({2, 2}, {1.0F, -0.5F, 0.3F, 0.7F});

    cuda::MatmulGrad analytic = cuda::matmulBf16Backward(
        cuda::DeviceTensor::fromHost(a), cuda::DeviceTensor::fromHost(b), cuda::DeviceTensor::fromHost(gradOutput));
    Tensor analyticDA = analytic.dA.toHost();

    auto fA = [&](const Tensor& aVar) {
        Tensor out = cuda::matmulBf16(cuda::DeviceTensor::fromHost(aVar), cuda::DeviceTensor::fromHost(b)).toHost();
        return dot(out, gradOutput);
    };
    for (std::size_t i = 0; i < a.elementCount(); ++i) {
        float numeric = numericalDerivative(fA, a, i, /*eps=*/0.1F);
        float tolerance = std::max(0.1F, std::abs(numeric) * 0.1F);
        EXPECT_NEAR(analyticDA.at(i), numeric, tolerance) << "dA indice " << i;
    }
}

TEST(CudaAutodiffTest, MatmulBf16BackwardLanciaSuFormeIncompatibili) {
    Tensor a({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    Tensor b({4, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    Tensor gradOutput({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    EXPECT_THROW((void)cuda::matmulBf16Backward(cuda::DeviceTensor::fromHost(a), cuda::DeviceTensor::fromHost(b),
                                                 cuda::DeviceTensor::fromHost(gradOutput)),
                 std::invalid_argument);
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

TEST(CudaAutodiffTest, SoftmaxBackwardCorrispondeAllaVersioneCpu) {
    Tensor input({2, 3}, {0.5F, -1.2F, 0.3F, 2.0F, 0.1F, -0.5F});
    Tensor gradOutput({2, 3}, {1.0F, -0.5F, 2.0F, 0.3F, -1.0F, 0.7F});

    Tensor cpuGrad = cpu::softmaxBackward(input, gradOutput);
    Tensor gpuGrad =
        cuda::softmaxBackward(cuda::DeviceTensor::fromHost(input), cuda::DeviceTensor::fromHost(gradOutput))
            .toHost();

    for (std::size_t i = 0; i < cpuGrad.elementCount(); ++i) {
        EXPECT_NEAR(gpuGrad.at(i), cpuGrad.at(i), 1e-4F) << "indice " << i;
    }
}

TEST(CudaAutodiffTest, SoftmaxBackwardCorrispondeAllaDerivataNumerica) {
    Tensor input({2, 3}, {0.5F, -1.2F, 0.3F, 2.0F, 0.1F, -0.5F});
    Tensor gradOutput({2, 3}, {1.0F, -0.5F, 2.0F, 0.3F, -1.0F, 0.7F});

    Tensor gpuGrad =
        cuda::softmaxBackward(cuda::DeviceTensor::fromHost(input), cuda::DeviceTensor::fromHost(gradOutput))
            .toHost();

    auto f = [&](const Tensor& x) {
        Tensor out = cuda::softmax(cuda::DeviceTensor::fromHost(x)).toHost();
        return dot(out, gradOutput);
    };
    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(gpuGrad.at(i), numericalDerivative(f, input, i), 1e-2F) << "indice " << i;
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

TEST(CudaAutodiffTest, MatmulTransposeBBackwardCorrispondeAllaVersioneCpu) {
    Tensor a({2, 3}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F});
    Tensor b({4, 3}, {0.7F, -0.1F, 0.2F, 0.3F, -0.4F, 0.5F, 0.2F, 0.1F, -0.3F, -0.6F, 0.4F, 0.1F});
    Tensor gradOutput({2, 4}, {1.0F, -1.0F, 0.5F, 2.0F, -0.3F, 0.8F, -1.2F, 0.4F});

    cpu::MatmulTransposeBGrad cpuGrad = cpu::matmulTransposeBBackward(a, b, gradOutput);
    cuda::MatmulTransposeBGrad gpuGrad = cuda::matmulTransposeBBackward(
        cuda::DeviceTensor::fromHost(a), cuda::DeviceTensor::fromHost(b), cuda::DeviceTensor::fromHost(gradOutput));
    Tensor gpuDA = gpuGrad.dA.toHost();
    Tensor gpuDB = gpuGrad.dB.toHost();

    for (std::size_t i = 0; i < cpuGrad.dA.elementCount(); ++i) {
        EXPECT_NEAR(gpuDA.at(i), cpuGrad.dA.at(i), 1e-4F) << "dA indice " << i;
    }
    for (std::size_t i = 0; i < cpuGrad.dB.elementCount(); ++i) {
        EXPECT_NEAR(gpuDB.at(i), cpuGrad.dB.at(i), 1e-4F) << "dB indice " << i;
    }
}

TEST(CudaAutodiffTest, EmbeddingLookupBackwardCorrispondeAllaVersioneCpuEAccumulaSuTokenRipetuti) {
    // vocabolario di 3, dim=2; token id [0, 1, 0] (il token 0 compare due
    // volte: verifica sia la corrispondenza CPU/GPU sia lo scatter-add
    // via atomicAdd sulla stessa riga della tabella).
    Tensor tokenIds({1, 3}, {0.0F, 1.0F, 0.0F});
    Tensor gradOutput({1, 3, 2}, {1.0F, 2.0F, 10.0F, 20.0F, 3.0F, 4.0F});

    Tensor cpuGrad = cpu::embeddingLookupBackward(tokenIds, gradOutput, /*vocabSize=*/3);
    Tensor gpuGrad = cuda::embeddingLookupBackward(cuda::DeviceTensor::fromHost(tokenIds),
                                                    cuda::DeviceTensor::fromHost(gradOutput), /*vocabSize=*/3)
                          .toHost();

    ASSERT_EQ(gpuGrad.shape(), (std::vector<std::size_t>{3, 2}));
    for (std::size_t i = 0; i < cpuGrad.elementCount(); ++i) {
        EXPECT_NEAR(gpuGrad.at(i), cpuGrad.at(i), 1e-5F) << "indice " << i;
    }
}

TEST(CudaAutodiffTest, EmbeddingLookupBackwardLanciaSeIlTokenIdEFuoriVocabolario) {
    Tensor tokenIds({1, 2}, {0.0F, 5.0F});
    Tensor gradOutput({1, 2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

    EXPECT_THROW(cuda::embeddingLookupBackward(cuda::DeviceTensor::fromHost(tokenIds),
                                                cuda::DeviceTensor::fromHost(gradOutput), /*vocabSize=*/3),
                 std::invalid_argument);
}

TEST(CudaAutodiffTest, AddPositionalEmbeddingBackwardCorrispondeAllaVersioneCpu) {
    Tensor gradOutput({2, 2, 2}, {1.0F, -0.5F, 2.0F, 0.3F, -1.0F, 0.7F, 0.6F, 0.2F});

    cpu::PositionalEmbeddingGrad cpuGrad = cpu::addPositionalEmbeddingBackward(gradOutput, /*maxSeqLen=*/2);
    cuda::PositionalEmbeddingGrad gpuGrad =
        cuda::addPositionalEmbeddingBackward(cuda::DeviceTensor::fromHost(gradOutput), /*maxSeqLen=*/2);
    Tensor gpuDInput = gpuGrad.dInput.toHost();
    Tensor gpuDTable = gpuGrad.dTable.toHost();

    for (std::size_t i = 0; i < cpuGrad.dInput.elementCount(); ++i) {
        EXPECT_NEAR(gpuDInput.at(i), cpuGrad.dInput.at(i), 1e-5F) << "dInput indice " << i;
    }
    for (std::size_t i = 0; i < cpuGrad.dTable.elementCount(); ++i) {
        EXPECT_NEAR(gpuDTable.at(i), cpuGrad.dTable.at(i), 1e-5F) << "dTable indice " << i;
    }
}

TEST(CudaAutodiffTest, FeedForwardBackwardCorrispondeAllaVersioneCpu) {
    Tensor input({1, 2, 3}, {0.3F, -0.6F, 0.2F, -0.4F, 0.5F, 0.1F});
    Tensor w1({3, 4}, {0.2F, -0.1F, 0.3F, 0.1F, -0.2F, 0.4F, 0.1F, -0.3F, 0.3F, 0.2F, -0.1F, 0.2F});
    Tensor b1({4}, {0.1F, -0.1F, 0.05F, 0.0F});
    Tensor w2({4, 3}, {0.1F, -0.2F, 0.3F, 0.2F, 0.1F, -0.1F, -0.3F, 0.2F, 0.1F, 0.1F, -0.1F, 0.2F});
    Tensor b2({3}, {0.05F, -0.05F, 0.1F});
    Tensor gradOutput({1, 2, 3}, {1.0F, -0.5F, 0.3F, -0.2F, 0.7F, -0.4F});

    cpu::FeedForwardGrad cpuGrad = cpu::feedForwardBackward(input, w1, b1, w2, b2, gradOutput);
    cuda::FeedForwardGrad gpuGrad = cuda::feedForwardBackward(
        cuda::DeviceTensor::fromHost(input), cuda::DeviceTensor::fromHost(w1), cuda::DeviceTensor::fromHost(b1),
        cuda::DeviceTensor::fromHost(w2), cuda::DeviceTensor::fromHost(b2), cuda::DeviceTensor::fromHost(gradOutput));

    Tensor gpuDInput = gpuGrad.dInput.toHost();
    Tensor gpuDW1 = gpuGrad.dW1.toHost();
    Tensor gpuDB1 = gpuGrad.dB1.toHost();
    Tensor gpuDW2 = gpuGrad.dW2.toHost();
    Tensor gpuDB2 = gpuGrad.dB2.toHost();

    for (std::size_t i = 0; i < cpuGrad.dInput.elementCount(); ++i) {
        EXPECT_NEAR(gpuDInput.at(i), cpuGrad.dInput.at(i), 1e-4F) << "dInput indice " << i;
    }
    for (std::size_t i = 0; i < cpuGrad.dW1.elementCount(); ++i) {
        EXPECT_NEAR(gpuDW1.at(i), cpuGrad.dW1.at(i), 1e-4F) << "dW1 indice " << i;
    }
    for (std::size_t i = 0; i < cpuGrad.dB1.elementCount(); ++i) {
        EXPECT_NEAR(gpuDB1.at(i), cpuGrad.dB1.at(i), 1e-4F) << "dB1 indice " << i;
    }
    for (std::size_t i = 0; i < cpuGrad.dW2.elementCount(); ++i) {
        EXPECT_NEAR(gpuDW2.at(i), cpuGrad.dW2.at(i), 1e-4F) << "dW2 indice " << i;
    }
    for (std::size_t i = 0; i < cpuGrad.dB2.elementCount(); ++i) {
        EXPECT_NEAR(gpuDB2.at(i), cpuGrad.dB2.at(i), 1e-4F) << "dB2 indice " << i;
    }
}

TEST(CudaAutodiffTest, SelfAttentionBackwardCorrispondeAllaVersioneCpu) {
    // batch=1, seq=3, dim=4, numHeads=2 (headDim=2): stessi dati del
    // test CPU equivalente (tests/backend/cpu/autodiff_tests.cpp), cosi'
    // da poter confrontare direttamente i due risultati analitici.
    Tensor input({1, 3, 4}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F, 0.2F, -0.1F, 0.3F, 0.2F, -0.4F, 0.5F});
    Tensor wq({4, 4}, {0.3F, -0.1F, 0.2F, 0.05F, 0.1F, 0.4F, -0.2F, 0.15F, -0.3F, 0.2F, 0.1F, -0.05F, 0.2F, -0.1F,
                        0.3F, 0.1F});
    Tensor wk({4, 4}, {0.1F, 0.2F, -0.1F, 0.3F, -0.2F, 0.1F, 0.4F, -0.1F, 0.3F, -0.3F, 0.1F, 0.2F, -0.1F, 0.2F,
                        -0.2F, 0.1F});
    Tensor wv({4, 4}, {0.2F, 0.1F, -0.1F, 0.3F, 0.1F, -0.2F, 0.3F, 0.1F, -0.1F, 0.3F, 0.2F, -0.1F, 0.3F, 0.1F, -0.2F,
                        0.2F});
    Tensor wout({4, 4}, {0.1F, -0.1F, 0.2F, 0.1F, 0.2F, 0.1F, -0.1F, 0.2F, -0.1F, 0.2F, 0.1F, -0.1F, 0.1F, 0.2F,
                          -0.1F, 0.2F});
    Tensor gradOutput({1, 3, 4}, {1.0F, -0.5F, 0.3F, -0.2F, 0.7F, -0.4F, 0.2F, -0.1F, -0.3F, 0.6F, 0.1F, -0.5F});

    cpu::SelfAttentionGrad cpuGrad = cpu::selfAttentionBackward(input, wq, wk, wv, wout, /*numHeads=*/2, gradOutput);
    cuda::SelfAttentionGrad gpuGrad = cuda::selfAttentionBackward(
        cuda::DeviceTensor::fromHost(input), cuda::DeviceTensor::fromHost(wq), cuda::DeviceTensor::fromHost(wk),
        cuda::DeviceTensor::fromHost(wv), cuda::DeviceTensor::fromHost(wout), /*numHeads=*/2,
        cuda::DeviceTensor::fromHost(gradOutput));

    Tensor gpuDInput = gpuGrad.dInput.toHost();
    Tensor gpuDWq = gpuGrad.dWq.toHost();
    Tensor gpuDWk = gpuGrad.dWk.toHost();
    Tensor gpuDWv = gpuGrad.dWv.toHost();
    Tensor gpuDWout = gpuGrad.dWout.toHost();

    for (std::size_t i = 0; i < cpuGrad.dInput.elementCount(); ++i) {
        EXPECT_NEAR(gpuDInput.at(i), cpuGrad.dInput.at(i), 1e-3F) << "dInput indice " << i;
    }
    for (std::size_t i = 0; i < cpuGrad.dWq.elementCount(); ++i) {
        EXPECT_NEAR(gpuDWq.at(i), cpuGrad.dWq.at(i), 1e-3F) << "dWq indice " << i;
    }
    for (std::size_t i = 0; i < cpuGrad.dWk.elementCount(); ++i) {
        EXPECT_NEAR(gpuDWk.at(i), cpuGrad.dWk.at(i), 1e-3F) << "dWk indice " << i;
    }
    for (std::size_t i = 0; i < cpuGrad.dWv.elementCount(); ++i) {
        EXPECT_NEAR(gpuDWv.at(i), cpuGrad.dWv.at(i), 1e-3F) << "dWv indice " << i;
    }
    for (std::size_t i = 0; i < cpuGrad.dWout.elementCount(); ++i) {
        EXPECT_NEAR(gpuDWout.at(i), cpuGrad.dWout.at(i), 1e-3F) << "dWout indice " << i;
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
