#include "blackforge/backend/lr_schedule.hpp"

#include <gtest/gtest.h>

using blackforge::backend::cosineAnnealingLearningRate;

TEST(LrScheduleTest, CosineVaLeBaseLearningRateAllaPrimaEpoca) {
    EXPECT_NEAR(cosineAnnealingLearningRate(1, 100, 0.1F), 0.1F, 1e-6F);
}

TEST(LrScheduleTest, CosineVaAZeroAllUltimaEpoca) {
    EXPECT_NEAR(cosineAnnealingLearningRate(100, 100, 0.1F), 0.0F, 1e-5F);
}

TEST(LrScheduleTest, CosineValeMetaACircaMetaPercorso) {
    // A meta' del percorso (progress = 0.5), cos(pi/2) = 0: lr = baseLr * 0.5.
    EXPECT_NEAR(cosineAnnealingLearningRate(51, 101, 0.1F), 0.05F, 1e-4F);
}

TEST(LrScheduleTest, CosineDecresceMonotonamente) {
    float previous = cosineAnnealingLearningRate(1, 20, 0.1F);
    for (long long epoch = 2; epoch <= 20; ++epoch) {
        float current = cosineAnnealingLearningRate(epoch, 20, 0.1F);
        EXPECT_LE(current, previous) << "epoca " << epoch;
        previous = current;
    }
}

TEST(LrScheduleTest, ConUnaSolaEpocaRestituisceSempreLaBaseLearningRate) {
    // Con totalEpochs <= 1 la formula del progresso (epoch-1)/(totalEpochs-1)
    // dividerebbe per zero: gestito esplicitamente restituendo baseLr.
    EXPECT_NEAR(cosineAnnealingLearningRate(1, 1, 0.05F), 0.05F, 1e-6F);
}
