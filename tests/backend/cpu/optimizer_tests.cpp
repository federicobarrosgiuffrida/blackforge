#include "blackforge/backend/cpu/optimizer.hpp"

#include <cmath>

#include <gtest/gtest.h>

using blackforge::runtime::Tensor;
namespace cpu = blackforge::backend::cpu;

TEST(OptimizerTest, SgdAggiornaIlParametroSecondoLaFormulaEsatta) {
    cpu::Parameter param{"p", Tensor({2}, {1.0F, -2.0F}), Tensor({2}, {0.5F, 0.5F})};
    cpu::SGD optimizer(0.1F);

    optimizer.step({&param});

    EXPECT_FLOAT_EQ(param.value.at(0), 1.0F - 0.1F * 0.5F);
    EXPECT_FLOAT_EQ(param.value.at(1), -2.0F - 0.1F * 0.5F);
}

TEST(OptimizerTest, AdamWPrimoStepCorrispondeAllaFormulaConBiasCorrection) {
    cpu::Parameter param{"p", Tensor({1}, {1.0F}), Tensor({1}, {0.2F})};
    cpu::AdamW optimizer(/*learningRate=*/0.1F, /*beta1=*/0.9F, /*beta2=*/0.999F, /*eps=*/1e-8F,
                          /*weightDecay=*/0.0F);

    optimizer.step({&param});

    float expected = 1.0F - 0.1F * (0.2F / (std::sqrt(0.04F) + 1e-8F));
    EXPECT_NEAR(param.value.at(0), expected, 1e-5F);
}

TEST(OptimizerTest, AdamWMantieneStatoTraUnoStepEIlSuccessivo) {
    cpu::Parameter param{"p", Tensor({1}, {1.0F}), Tensor({1}, {0.2F})};
    cpu::AdamW optimizer(0.1F);

    optimizer.step({&param});
    float afterFirst = param.value.at(0);

    param.grad = Tensor({1}, {0.2F});  // stesso gradiente al secondo step
    optimizer.step({&param});
    float afterSecond = param.value.at(0);

    // La bias correction e i momenti accumulati cambiano con lo step:
    // l'ampiezza del secondo aggiornamento non deve essere identica a
    // quella del primo.
    EXPECT_NE(afterFirst - 1.0F, afterSecond - afterFirst);
}

TEST(OptimizerTest, WeightDecayDisaccoppiatoAgisceAncheAGradienteNullo) {
    cpu::Parameter param{"p", Tensor({1}, {1.0F}), Tensor({1}, {0.0F})};
    cpu::AdamW optimizer(/*learningRate=*/0.1F, /*beta1=*/0.9F, /*beta2=*/0.999F, /*eps=*/1e-8F,
                          /*weightDecay=*/0.1F);

    optimizer.step({&param});

    // Con gradiente zero mHat = vHat = 0: l'unico contributo
    // all'aggiornamento e' il weight decay, applicato direttamente al
    // parametro (non al gradiente, come nel classico AdamW).
    EXPECT_FLOAT_EQ(param.value.at(0), 1.0F - 0.1F * 0.1F * 1.0F);
}
