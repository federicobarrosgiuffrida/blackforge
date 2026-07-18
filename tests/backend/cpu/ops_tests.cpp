#include "blackforge/backend/cpu/ops.hpp"

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
