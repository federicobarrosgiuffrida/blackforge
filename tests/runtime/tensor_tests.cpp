#include "blackforge/runtime/tensor.hpp"

#include <gtest/gtest.h>

using blackforge::runtime::Tensor;

TEST(TensorTest, CostruisceEValidaLaFormaRispettoAiDati) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_EQ(t.rank(), 2u);
    EXPECT_EQ(t.dim(0), 2u);
    EXPECT_EQ(t.dim(1), 3u);
    EXPECT_EQ(t.elementCount(), 6u);
}

TEST(TensorTest, LanciaSeIDatiNonCorrispondonoAllaForma) {
    EXPECT_THROW(Tensor({2, 3}, {1, 2, 3}), std::invalid_argument);
}

TEST(TensorTest, ZerosProduceTuttiZeri) {
    Tensor t = Tensor::zeros({2, 2});
    for (std::size_t i = 0; i < t.elementCount(); ++i) {
        EXPECT_FLOAT_EQ(t.at(i), 0.0F);
    }
}

TEST(TensorTest, FilledProduceIlValoreRichiesto) {
    Tensor t = Tensor::filled({3}, 2.5F);
    EXPECT_FLOAT_EQ(t.at(0), 2.5F);
    EXPECT_FLOAT_EQ(t.at(1), 2.5F);
    EXPECT_FLOAT_EQ(t.at(2), 2.5F);
}

TEST(TensorTest, StatisticheMinMaxMean) {
    Tensor t({4}, {1.0F, 2.0F, 3.0F, 4.0F});
    EXPECT_FLOAT_EQ(t.min(), 1.0F);
    EXPECT_FLOAT_EQ(t.max(), 4.0F);
    EXPECT_FLOAT_EQ(t.mean(), 2.5F);
}

TEST(TensorTest, ShapeToStringFormattaCorrettamente) {
    Tensor t({2, 3}, {0, 0, 0, 0, 0, 0});
    EXPECT_EQ(t.shapeToString(), "[2, 3]");
}
