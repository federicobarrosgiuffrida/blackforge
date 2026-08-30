#include "blackforge/blackbit/ternary_linear.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "blackforge/backend/cpu/autodiff.hpp"
#include "blackforge/backend/cpu/loss.hpp"
#include "blackforge/backend/cpu/ops.hpp"
#include "blackforge/blackbit/stochastic_round.hpp"
#include "blackforge/blackbit/telemetry.hpp"
#include "blackforge/blackbit/ternary_update.hpp"

using namespace blackforge;
using blackforge::blackbit::TernaryLinear;

namespace {

runtime::Tensor randomTensor(std::vector<std::size_t> shape, unsigned int seed, float amplitude = 1.0F) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-amplitude, amplitude);
    std::size_t count = 1;
    for (std::size_t dim : shape) {
        count *= dim;
    }
    std::vector<float> data(count);
    for (float& value : data) {
        value = uniform(rng);
    }
    return runtime::Tensor(std::move(shape), std::move(data));
}

// Riferimento denso: dequantizza l'INTERA matrice e usa i primitivi
// gia' verificati del backend CPU. E' il termine di paragone contro cui
// il percorso a tile deve coincidere.
runtime::Tensor denseReferenceForward(const TernaryLinear& layer, const runtime::Tensor& input) {
    const runtime::Tensor weight = layer.weight().dequantize();  // [out, in]
    return backend::cpu::matmulTransposeB(input, weight);
}

}  // namespace

TEST(TernaryLinearTest, ForwardCoincideConIlRiferimentoDenso) {
    TernaryLinear layer("proj", 37, 23, 20, 8);
    layer.initialize(4242);

    const runtime::Tensor input = randomTensor({5, 37}, 7);
    const runtime::Tensor actual = layer.forward(input);
    const runtime::Tensor expected = denseReferenceForward(layer, input);

    ASSERT_EQ(actual.shape(), expected.shape());
    for (std::size_t i = 0; i < actual.elementCount(); ++i) {
        ASSERT_NEAR(actual.at(i), expected.at(i), 1e-4F) << "elemento " << i;
    }
}

TEST(TernaryLinearTest, IlRisultatoNonDipendeDallaDimensioneDelTile) {
    // E' la garanzia centrale del requisito 2: il tiling e'
    // un'ottimizzazione di memoria, non un'approssimazione.
    TernaryLinear layer("proj", 40, 31, 20, 31);
    layer.initialize(11);

    const runtime::Tensor input = randomTensor({3, 4, 40}, 13);
    const runtime::Tensor reference = layer.forward(input);

    for (std::size_t tile : {1U, 2U, 7U, 16U, 31U}) {
        layer.setTileRows(tile);
        const runtime::Tensor actual = layer.forward(input);
        ASSERT_EQ(actual.shape(), reference.shape());
        for (std::size_t i = 0; i < actual.elementCount(); ++i) {
            ASSERT_FLOAT_EQ(actual.at(i), reference.at(i)) << "tile " << tile << ", elemento " << i;
        }
    }
}

TEST(TernaryLinearTest, IlTileNonMaterializzaMaiLInteraMatrice) {
    blackbit::MemoryTelemetry::instance().reset();

    // 400 x 300 = 120 000 pesi: densi sarebbero 480 000 byte.
    TernaryLinear layer("grande", 300, 400, 20, 16);
    layer.initialize(5);

    blackbit::MemoryTelemetry::instance().resetPeaks();
    const runtime::Tensor input = randomTensor({2, 300}, 6);
    const runtime::Tensor output = layer.forward(input);
    ASSERT_EQ(output.dim(1), 400U);

    const std::size_t workspacePeak = blackbit::MemoryTelemetry::instance().peak(blackbit::MemoryArena::Workspace);
    EXPECT_EQ(workspacePeak, 16U * 300U * sizeof(float));
    EXPECT_LT(workspacePeak, 400U * 300U * sizeof(float) / 10)
        << "il buffer temporaneo deve restare una frazione della matrice densa";

    // E il parametro deve occupare molto meno di quanto occuperebbe
    // denso: e' l'intero punto del formato.
    const std::size_t parameterBytes = blackbit::MemoryTelemetry::instance().current(blackbit::MemoryArena::Parameter);
    EXPECT_EQ(parameterBytes, layer.parameterBytes());
    EXPECT_LT(parameterBytes, 400U * 300U * sizeof(float) / 8);
}

TEST(TernaryLinearTest, IlGradienteDellIngressoCoincideConQuelloNumerico) {
    TernaryLinear layer("proj", 6, 4, 20, 2);
    layer.initialize(31);

    const runtime::Tensor input = randomTensor({2, 6}, 17);
    const runtime::Tensor target = randomTensor({2, 4}, 19);

    auto loss = [&](const runtime::Tensor& x) {
        const runtime::Tensor y = layer.forward(x);
        return backend::cpu::meanSquaredError(y, target).value;
    };

    const runtime::Tensor y = layer.forward(input);
    const runtime::Tensor gradOutput = backend::cpu::meanSquaredError(y, target).grad;
    const runtime::Tensor gradInput = layer.backward(input, gradOutput, nullptr);

    const float epsilon = 1e-2F;
    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        runtime::Tensor plus = input;
        runtime::Tensor minus = input;
        plus.at(i) += epsilon;
        minus.at(i) -= epsilon;
        const float numeric = (loss(plus) - loss(minus)) / (2.0F * epsilon);
        ASSERT_NEAR(gradInput.at(i), numeric, 1e-3F) << "elemento " << i;
    }
}

TEST(TernaryLinearTest, IBlocchiDiGradienteDelPesoRicompongonoIlGradienteAnalitico) {
    TernaryLinear layer("proj", 9, 7, 20, 3);
    layer.initialize(77);

    const runtime::Tensor input = randomTensor({4, 9}, 23);
    const runtime::Tensor gradOutput = randomTensor({4, 7}, 29);

    blackbit::DenseGradientCollector collector;
    (void)layer.backward(input, gradOutput, &collector);

    ASSERT_TRUE(collector.has("proj"));
    const std::vector<float>& actual = collector.gradient("proj");
    ASSERT_EQ(actual.size(), 7U * 9U);

    // dW[n, k] = somma_m dY[m, n] * X[m, k]
    for (std::size_t n = 0; n < 7; ++n) {
        for (std::size_t k = 0; k < 9; ++k) {
            float expected = 0.0F;
            for (std::size_t m = 0; m < 4; ++m) {
                expected += gradOutput.at(m * 7 + n) * input.at(m * 9 + k);
            }
            ASSERT_NEAR(actual[n * 9 + k], expected, 1e-4F) << "n=" << n << " k=" << k;
        }
    }
}

TEST(TernaryLinearTest, IlPiccoDiGradienteVivoEQuelloDiUnBloccoNonDellaMatrice) {
    // Milestone D in miniatura: la prova e' un numero, non una
    // dichiarazione.
    blackbit::resetGradientLifetimeStats();

    TernaryLinear layer("grande", 200, 300, 20, 25);
    layer.initialize(3);

    const runtime::Tensor input = randomTensor({2, 200}, 4);
    const runtime::Tensor gradOutput = randomTensor({2, 300}, 5);
    (void)layer.backward(input, gradOutput, nullptr);

    const blackbit::GradientLifetimeStats& stats = blackbit::gradientLifetimeStats();
    const std::size_t blockBytes = 25U * 200U * sizeof(float);
    const std::size_t wholeMatrixBytes = 300U * 200U * sizeof(float);

    EXPECT_EQ(stats.peakLiveBytes, blockBytes);
    EXPECT_EQ(stats.liveBytes, 0U) << "ogni blocco deve essere rilasciato prima del successivo";
    EXPECT_EQ(stats.blocksProduced, stats.blocksReleased);
    EXPECT_EQ(stats.blocksProduced, 12U);  // 300 righe / 25
    EXPECT_EQ(stats.cumulativeBytes, wholeMatrixBytes);
    EXPECT_LT(stats.peakLiveBytes, wholeMatrixBytes / 10);
}

TEST(TernaryLinearTest, UnaReteMinusculaImparaDavveroConPesiTernari) {
    // Milestone B: due proiezioni ternarie, nessun peso continuo da
    // nessuna parte, aggiornate solo con arrotondamento stocastico.
    // Se la loss scende, i trit si sono davvero mossi nella direzione
    // giusta.
    constexpr std::size_t kIn = 24;
    constexpr std::size_t kHidden = 32;
    constexpr std::size_t kOut = 4;
    constexpr std::size_t kBatch = 16;

    TernaryLinear first("l1", kIn, kHidden, 20, 8);
    TernaryLinear second("l2", kHidden, kOut, 20, 8);
    first.initialize(1);
    second.initialize(2);

    const runtime::Tensor input = randomTensor({kBatch, kIn}, 101, 0.5F);
    const runtime::Tensor target = randomTensor({kBatch, kOut}, 202, 0.5F);

    blackbit::TernarySgdSink optimizer(0.05F);
    optimizer.registerTernary("l1", first.weight());
    optimizer.registerTernary("l2", second.weight());

    auto step = [&]() {
        const runtime::Tensor hidden = first.forward(input);
        const runtime::Tensor activated = backend::cpu::silu(hidden);
        const runtime::Tensor output = second.forward(activated);

        const backend::cpu::LossResult loss = backend::cpu::meanSquaredError(output, target);

        const runtime::Tensor dActivated = second.backward(activated, loss.grad, &optimizer);
        const runtime::Tensor dHidden = backend::cpu::siluBackward(hidden, dActivated);
        (void)first.backward(input, dHidden, &optimizer);
        optimizer.endStep();
        return loss.value;
    };

    const float initialLoss = step();
    float finalLoss = initialLoss;
    for (int i = 0; i < 120; ++i) {
        finalLoss = step();
    }

    EXPECT_LT(finalLoss, initialLoss * 0.85F)
        << "loss iniziale " << initialLoss << ", finale " << finalLoss;

    // E i pesi ternari devono essersi mossi: senza flip, una loss che
    // scende potrebbe venire solo dal rumore.
    EXPECT_GT(optimizer.stats().flips, 0U);
    EXPECT_GT(optimizer.stats().flipFraction(), 0.0);

    // Nessuno stato di ottimizzatore: la memoria in piu' e' zero.
    EXPECT_EQ(blackbit::TernarySgdSink::stateBytes(), 0U);
}

TEST(TernaryLinearTest, IlPercorsoBf16RestaVicinoAQuelloFp32) {
    // Storage e compute sono indipendenti: cambiare il formato di
    // calcolo non deve cambiare il risultato oltre l'errore del
    // formato stesso.
    TernaryLinear layer("proj", 64, 48, 20, 16);
    layer.initialize(9);

    const runtime::Tensor input = randomTensor({4, 64}, 91);
    const runtime::Tensor fp32 = layer.forward(input);

    layer.setComputeDType(blackbit::ComputeDType::BF16);
    const runtime::Tensor bf16 = layer.forward(input);

    for (std::size_t i = 0; i < fp32.elementCount(); ++i) {
        // BF16 ha 8 bit di mantissa: ~0,4 % di errore relativo per
        // valore, che su una somma di 64 termini resta ben sotto l'1 %.
        ASSERT_NEAR(bf16.at(i), fp32.at(i), 0.01F * (std::fabs(fp32.at(i)) + 1.0F)) << "elemento " << i;
    }
}

TEST(TernaryLinearTest, RifiutaFormeIncoerenti) {
    TernaryLinear layer("proj", 8, 4, 20, 4);
    const runtime::Tensor wrong = randomTensor({2, 5}, 1);
    EXPECT_THROW((void)layer.forward(wrong), std::invalid_argument);

    const runtime::Tensor input = randomTensor({2, 8}, 1);
    const runtime::Tensor wrongGrad = randomTensor({2, 3}, 1);
    EXPECT_THROW((void)layer.backward(input, wrongGrad, nullptr), std::invalid_argument);

    EXPECT_THROW(layer.setTileRows(0), std::invalid_argument);
    EXPECT_THROW(layer.loadDense(randomTensor({4, 7}, 1)), std::invalid_argument);
}
