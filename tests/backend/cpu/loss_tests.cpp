#include "blackforge/backend/cpu/loss.hpp"

#include <cmath>

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

TEST(LossTest, CrossEntropyCalcolaIlValoreCorrettoSuLogitsUniformi) {
    // 2 classi, logit uguali -> softmax = [0.5, 0.5]; target one-hot
    // sulla classe 0 -> loss = -log(0.5).
    Tensor logits({1, 2}, {0.0F, 0.0F});
    Tensor target({1, 2}, {1.0F, 0.0F});

    cpu::LossResult result = cpu::softmaxCrossEntropy(logits, target);
    EXPECT_NEAR(result.value, -std::log(0.5), 1e-5);
}

TEST(LossTest, CrossEntropyGradienteEProbabilitaMenoTargetDivisoBatch) {
    Tensor logits({1, 2}, {0.0F, 0.0F});
    Tensor target({1, 2}, {1.0F, 0.0F});

    cpu::LossResult result = cpu::softmaxCrossEntropy(logits, target);
    // softmax([0,0]) = [0.5, 0.5]; grad = (softmax - target) / batch(=1).
    EXPECT_NEAR(result.grad.at(0), 0.5F - 1.0F, 1e-5F);
    EXPECT_NEAR(result.grad.at(1), 0.5F - 0.0F, 1e-5F);
}

TEST(LossTest, CrossEntropyValeQuasiZeroQuandoIlModelloEMoltoSicuroEGiusto) {
    Tensor logits({1, 2}, {20.0F, -20.0F});
    Tensor target({1, 2}, {1.0F, 0.0F});

    cpu::LossResult result = cpu::softmaxCrossEntropy(logits, target);
    EXPECT_NEAR(result.value, 0.0F, 1e-4F);
}

TEST(LossTest, CrossEntropyLanciaSuFormeIncompatibili) {
    Tensor logits({1, 2}, {1.0F, 2.0F});
    Tensor target({1, 3}, {1.0F, 0.0F, 0.0F});
    EXPECT_THROW((void)cpu::softmaxCrossEntropy(logits, target), std::invalid_argument);
}

TEST(LossTest, CrossEntropyLanciaSeNonERango2) {
    Tensor logits({2}, {1.0F, 2.0F});
    Tensor target({2}, {1.0F, 0.0F});
    EXPECT_THROW((void)cpu::softmaxCrossEntropy(logits, target), std::invalid_argument);
}

TEST(LossTest, CrossEntropyGradienteCorrispondeAllaDerivataNumerica) {
    // Batch di 2 esempi, 3 classi; logits non banali e target one-hot
    // diversi per esempio, per esercitare davvero la normalizzazione
    // per riga della softmax.
    Tensor logits({2, 3}, {0.5F, -1.2F, 0.3F, 2.0F, 0.1F, -0.5F});
    Tensor target({2, 3}, {0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F});

    cpu::LossResult result = cpu::softmaxCrossEntropy(logits, target);

    float eps = 1e-3F;
    for (std::size_t i = 0; i < logits.elementCount(); ++i) {
        Tensor plusLogits = logits;
        plusLogits.at(i) += eps;
        Tensor minusLogits = logits;
        minusLogits.at(i) -= eps;

        float plus = cpu::softmaxCrossEntropy(plusLogits, target).value;
        float minus = cpu::softmaxCrossEntropy(minusLogits, target).value;
        float numeric = (plus - minus) / (2.0F * eps);

        EXPECT_NEAR(result.grad.at(i), numeric, 1e-3F) << "indice " << i;
    }
}
