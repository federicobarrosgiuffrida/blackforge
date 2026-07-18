#include "blackforge/backend/cpu/loss.hpp"

#include <gtest/gtest.h>

using blackforge::runtime::Tensor;
namespace cpu = blackforge::backend::cpu;

TEST(LossTest, MseCalcolaIlValoreCorretto) {
    Tensor pred({2}, {1.0F, 2.0F});
    Tensor target({2}, {0.0F, 0.0F});

    cpu::LossResult result = cpu::meanSquaredError(pred, target);
    EXPECT_FLOAT_EQ(result.value, 2.5F);  // (1^2 + 2^2) / 2
}

TEST(LossTest, MseValeZeroQuandoPredizioneETargetCoincidono) {
    Tensor pred({3}, {0.1F, -0.2F, 0.3F});
    cpu::LossResult result = cpu::meanSquaredError(pred, pred);
    EXPECT_FLOAT_EQ(result.value, 0.0F);
}

TEST(LossTest, MseLanciaSuFormeIncompatibili) {
    Tensor pred({2}, {1.0F, 2.0F});
    Tensor target({3}, {1.0F, 2.0F, 3.0F});
    EXPECT_THROW(cpu::meanSquaredError(pred, target), std::invalid_argument);
}

TEST(LossTest, MseGradienteCorrispondeAllaDerivataNumerica) {
    Tensor pred({3}, {0.3F, -0.7F, 1.2F});
    Tensor target({3}, {0.5F, 0.1F, -0.4F});

    cpu::LossResult result = cpu::meanSquaredError(pred, target);

    float eps = 1e-3F;
    for (std::size_t i = 0; i < pred.elementCount(); ++i) {
        Tensor plusPred = pred;
        plusPred.at(i) += eps;
        Tensor minusPred = pred;
        minusPred.at(i) -= eps;

        float plus = cpu::meanSquaredError(plusPred, target).value;
        float minus = cpu::meanSquaredError(minusPred, target).value;
        float numeric = (plus - minus) / (2.0F * eps);

        EXPECT_NEAR(result.grad.at(i), numeric, 1e-3F) << "indice " << i;
    }
}
