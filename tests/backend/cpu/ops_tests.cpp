#include "blackforge/backend/cpu/ops.hpp"

#include <cmath>

#include <gtest/gtest.h>

using blackforge::runtime::Tensor;
namespace cpu = blackforge::backend::cpu;

TEST(CpuOpsTest, AddSommaElementwise) {
    Tensor a({2}, {1.0F, 2.0F});
    Tensor b({2}, {10.0F, 20.0F});
    Tensor result = cpu::add(a, b);

    EXPECT_FLOAT_EQ(result.at(0), 11.0F);
    EXPECT_FLOAT_EQ(result.at(1), 22.0F);
}

TEST(CpuOpsTest, AddLanciaSuFormeIncompatibili) {
    Tensor a({2}, {1.0F, 2.0F});
    Tensor b({3}, {1.0F, 2.0F, 3.0F});
    EXPECT_THROW(cpu::add(a, b), std::invalid_argument);
}

TEST(CpuOpsTest, AddBiasTrasmetteSuOgniRiga) {
    Tensor input({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    Tensor bias({2}, {10.0F, 20.0F});
    Tensor result = cpu::addBias(input, bias);

    EXPECT_FLOAT_EQ(result.at(0), 11.0F);
    EXPECT_FLOAT_EQ(result.at(1), 22.0F);
    EXPECT_FLOAT_EQ(result.at(2), 13.0F);
    EXPECT_FLOAT_EQ(result.at(3), 24.0F);
}

TEST(CpuOpsTest, MatmulCalcolaIlProdottoCorretto) {
    Tensor a({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    Tensor b({2, 2}, {5.0F, 6.0F, 7.0F, 8.0F});
    Tensor result = cpu::matmul(a, b);

    ASSERT_EQ(result.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ(result.at(0), 19.0F);  // 1*5 + 2*7
    EXPECT_FLOAT_EQ(result.at(1), 22.0F);  // 1*6 + 2*8
    EXPECT_FLOAT_EQ(result.at(2), 43.0F);  // 3*5 + 4*7
    EXPECT_FLOAT_EQ(result.at(3), 50.0F);  // 3*6 + 4*8
}

TEST(CpuOpsTest, MatmulLanciaSuDimensioniIncompatibili) {
    Tensor a({2, 3}, std::vector<float>(6, 1.0F));
    Tensor b({2, 2}, std::vector<float>(4, 1.0F));
    EXPECT_THROW(cpu::matmul(a, b), std::invalid_argument);
}

TEST(CpuOpsTest, LinearCombinaMatmulEBias) {
    // input [1,2] (1x2), weight identita' (2x2), bias [10, 20]
    Tensor input({1, 2}, {3.0F, 4.0F});
    Tensor weight({2, 2}, {1.0F, 0.0F, 0.0F, 1.0F});
    Tensor bias({2}, {10.0F, 20.0F});

    Tensor result = cpu::linear(input, weight, bias);
    EXPECT_FLOAT_EQ(result.at(0), 13.0F);
    EXPECT_FLOAT_EQ(result.at(1), 24.0F);
}

TEST(CpuOpsTest, ReluAzzeraIValoriNegativi) {
    Tensor input({3}, {-2.0F, 0.0F, 3.0F});
    Tensor result = cpu::relu(input);

    EXPECT_FLOAT_EQ(result.at(0), 0.0F);
    EXPECT_FLOAT_EQ(result.at(1), 0.0F);
    EXPECT_FLOAT_EQ(result.at(2), 3.0F);
}

TEST(CpuOpsTest, SiluValeZeroInZeroEConvergeAXPerValoriGrandi) {
    Tensor input({3}, {0.0F, 20.0F, -20.0F});
    Tensor result = cpu::silu(input);

    EXPECT_FLOAT_EQ(result.at(0), 0.0F);
    EXPECT_NEAR(result.at(1), 20.0F, 0.01F);  // sigmoid(20) ~= 1
    EXPECT_NEAR(result.at(2), 0.0F, 0.01F);   // sigmoid(-20) ~= 0
}

TEST(CpuOpsTest, GeluValeZeroInZeroEConvergeAXPerValoriGrandi) {
    Tensor input({3}, {0.0F, 20.0F, -20.0F});
    Tensor result = cpu::gelu(input);

    EXPECT_FLOAT_EQ(result.at(0), 0.0F);
    EXPECT_NEAR(result.at(1), 20.0F, 0.01F);
    EXPECT_NEAR(result.at(2), 0.0F, 0.01F);
}

TEST(CpuOpsTest, RmsnormNormalizzaOgniRigaARadiceMediaQuadraticaUnitaria) {
    Tensor input({2, 4}, {1.0F, 2.0F, 3.0F, 4.0F, -1.0F, -1.0F, -1.0F, -1.0F});
    Tensor result = cpu::rmsnorm(input);

    ASSERT_EQ(result.shape(), (std::vector<std::size_t>{2, 4}));
    for (std::size_t row = 0; row < 2; ++row) {
        double sumSquares = 0.0;
        for (std::size_t col = 0; col < 4; ++col) {
            float v = result.at(row * 4 + col);
            sumSquares += static_cast<double>(v) * static_cast<double>(v);
        }
        // Con eps trascurabile rispetto ai valori usati, la RMS
        // dell'uscita normalizzata deve essere vicina a 1.
        double rms = std::sqrt(sumSquares / 4.0);
        EXPECT_NEAR(rms, 1.0, 1e-3) << "riga " << row;
    }
}

TEST(CpuOpsTest, RmsnormPreservaIlSegnoEIlRapportoTraElementi) {
    Tensor input({1, 2}, {2.0F, -4.0F});
    Tensor result = cpu::rmsnorm(input);

    EXPECT_GT(result.at(0), 0.0F);
    EXPECT_LT(result.at(1), 0.0F);
    // Il rapporto tra i due elementi (entrambi divisi per la stessa
    // costante rms) deve restare invariato: -2.
    EXPECT_NEAR(result.at(1) / result.at(0), -2.0F, 1e-4F);
}

TEST(CpuOpsTest, RmsnormLanciaSeNonERango2) {
    Tensor input({4}, {1.0F, 2.0F, 3.0F, 4.0F});
    EXPECT_THROW((void)cpu::rmsnorm(input), std::invalid_argument);
}
