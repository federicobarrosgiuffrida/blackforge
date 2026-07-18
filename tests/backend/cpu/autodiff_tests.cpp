#include "blackforge/backend/cpu/autodiff.hpp"

#include <functional>

#include <gtest/gtest.h>

#include "blackforge/backend/cpu/ops.hpp"

using blackforge::runtime::Tensor;
namespace cpu = blackforge::backend::cpu;

namespace {

// Derivata numerica (differenze centrali) di f rispetto a t.at(index).
// E' il modo standard per verificare che una formula di backward sia
// corretta: si confronta col prodotto vettore-Jacobiano analitico.
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

TEST(AutodiffTest, MatmulBackwardCorrispondeAllaDerivataNumerica) {
    Tensor a({2, 3}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F});
    Tensor b({3, 2}, {0.7F, -0.1F, 0.2F, 0.3F, -0.4F, 0.5F});
    Tensor gradOutput({2, 2}, {1.0F, -1.0F, 0.5F, 2.0F});

    cpu::MatmulGrad analytic = cpu::matmulBackward(a, b, gradOutput);

    auto fA = [&](const Tensor& aVar) { return dot(cpu::matmul(aVar, b), gradOutput); };
    for (std::size_t i = 0; i < a.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dA.at(i), numericalDerivative(fA, a, i), 1e-2F) << "dA indice " << i;
    }

    auto fB = [&](const Tensor& bVar) { return dot(cpu::matmul(a, bVar), gradOutput); };
    for (std::size_t i = 0; i < b.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dB.at(i), numericalDerivative(fB, b, i), 1e-2F) << "dB indice " << i;
    }
}

TEST(AutodiffTest, AddBiasBackwardCorrispondeAllaDerivataNumerica) {
    Tensor input({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    Tensor bias({2}, {0.5F, -0.5F});
    Tensor gradOutput({2, 2}, {1.0F, 0.5F, -1.0F, 2.0F});

    cpu::AddBiasGrad analytic = cpu::addBiasBackward(gradOutput);

    auto fInput = [&](const Tensor& inVar) { return dot(cpu::addBias(inVar, bias), gradOutput); };
    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dInput.at(i), numericalDerivative(fInput, input, i), 1e-2F) << "dInput indice " << i;
    }

    auto fBias = [&](const Tensor& biasVar) { return dot(cpu::addBias(input, biasVar), gradOutput); };
    for (std::size_t i = 0; i < bias.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dBias.at(i), numericalDerivative(fBias, bias, i), 1e-2F) << "dBias indice " << i;
    }
}

TEST(AutodiffTest, SiluBackwardCorrispondeAllaDerivataNumerica) {
    Tensor input({4}, {-1.5F, -0.3F, 0.7F, 2.0F});
    Tensor gradOutput({4}, {1.0F, -0.5F, 2.0F, 0.3F});

    Tensor analytic = cpu::siluBackward(input, gradOutput);
    auto f = [&](const Tensor& x) { return dot(cpu::silu(x), gradOutput); };

    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(analytic.at(i), numericalDerivative(f, input, i), 1e-2F) << "indice " << i;
    }
}

TEST(AutodiffTest, ReluBackwardCorrispondeAllaDerivataNumerica) {
    // Valori lontani da zero: relu non e' differenziabile esattamente in 0.
    Tensor input({4}, {-2.0F, -0.4F, 0.6F, 3.0F});
    Tensor gradOutput({4}, {1.0F, -0.5F, 2.0F, 0.3F});

    Tensor analytic = cpu::reluBackward(input, gradOutput);
    auto f = [&](const Tensor& x) { return dot(cpu::relu(x), gradOutput); };

    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(analytic.at(i), numericalDerivative(f, input, i), 1e-2F) << "indice " << i;
    }
}

TEST(AutodiffTest, GeluBackwardCorrispondeAllaDerivataNumerica) {
    Tensor input({4}, {-1.5F, -0.3F, 0.7F, 2.0F});
    Tensor gradOutput({4}, {1.0F, -0.5F, 2.0F, 0.3F});

    Tensor analytic = cpu::geluBackward(input, gradOutput);
    auto f = [&](const Tensor& x) { return dot(cpu::gelu(x), gradOutput); };

    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(analytic.at(i), numericalDerivative(f, input, i), 1e-2F) << "indice " << i;
    }
}
