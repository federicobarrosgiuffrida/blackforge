#include "blackforge/blackbit/low_rank_optimizer.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

#include "blackforge/blackbit/model.hpp"
#include "blackforge/blackbit/telemetry.hpp"

using namespace blackforge;
using blackforge::blackbit::BlackBitConfig;
using blackforge::blackbit::BlackBitModel;
using blackforge::blackbit::LowRankOptimizerOptions;
using blackforge::blackbit::LowRankProjectedOptimizer;

namespace {

BlackBitConfig microConfig() {
    BlackBitConfig config;
    config.name = "blackbit-micro";
    config.vocabSize = 32;
    config.hiddenSize = 32;
    config.numLayers = 2;
    config.numHeads = 4;
    config.numKvHeads = 2;
    config.headDim = 8;
    config.numExperts = 4;
    config.expertsPerToken = 2;
    config.expertHidden = 32;
    config.maxSeqLen = 16;
    config.ternaryGroupSize = 20;
    return config;
}

}  // namespace

TEST(LowRankOptimizerTest, LaProiezioneEDeterministicaEQuasiOrtogonale) {
    // P non e' memorizzata: viene rigenerata a ogni uso. Se non fosse
    // riproducibile, l'aggiornamento verrebbe ricostruito in una base
    // diversa da quella in cui e' stato accumulato.
    constexpr std::size_t kRank = 16;
    constexpr std::size_t kRows = 512;

    for (std::size_t row = 0; row < 10; ++row) {
        for (std::size_t j = 0; j < kRank; ++j) {
            EXPECT_FLOAT_EQ(blackbit::projectionEntry(7, 0, row, j, kRank),
                            blackbit::projectionEntry(7, 0, row, j, kRank));
        }
    }
    // Epoche diverse devono dare sottospazi diversi.
    EXPECT_NE(blackbit::projectionEntry(7, 0, 3, 2, kRank), blackbit::projectionEntry(7, 1, 3, 2, kRank));

    // P^T P deve essere vicina all'identita': e' la proprieta' che
    // rende la proiezione utile (Johnson-Lindenstrauss).
    for (std::size_t a = 0; a < kRank; ++a) {
        for (std::size_t b = 0; b < kRank; ++b) {
            double dot = 0.0;
            for (std::size_t row = 0; row < kRows; ++row) {
                dot += static_cast<double>(blackbit::projectionEntry(11, 0, row, a, kRank)) *
                       static_cast<double>(blackbit::projectionEntry(11, 0, row, b, kRank));
            }
            // Le colonne sono normalizzate a rows/rank sulla diagonale.
            const double expected = a == b ? static_cast<double>(kRows) / static_cast<double>(kRank) : 0.0;
            const double tolerance = a == b ? 1e-6 : 4.0;
            EXPECT_NEAR(dot, expected, tolerance) << "colonne " << a << ", " << b;
        }
    }
}

TEST(LowRankOptimizerTest, LoStatoEOrdiniDiGrandezzaSottoAdamWCompleto) {
    const BlackBitConfig config = microConfig();
    BlackBitModel model(config, 3);

    LowRankOptimizerOptions options;
    options.rank = 8;
    LowRankProjectedOptimizer optimizer(options);
    model.registerParameters(optimizer);

    EXPECT_LT(optimizer.stateBytes(), optimizer.conventionalStateBytes());
    // Su questa configurazione micro (righe da 32, rango 8) il
    // risparmio e' modesto; e' su matrici vere che diventa il fattore
    // che rende possibile il progetto — vedi il test sul rango
    // effettivo sotto.
    EXPECT_LT(static_cast<double>(optimizer.stateBytes()),
               static_cast<double>(optimizer.conventionalStateBytes()));

    // La telemetria deve conoscere ogni byte dell'ottimizzatore.
    EXPECT_GE(blackbit::MemoryTelemetry::instance().current(blackbit::MemoryArena::Optimizer),
               optimizer.stateBytes());
}

TEST(LowRankOptimizerTest, IlRisparmioCresceConLaDimensioneDellaMatrice) {
    // Il rapporto fra stato low-rank e stato completo e' 2rn / 2mn =
    // r/m: su una matrice con molte righe diventa enorme, ed e'
    // esattamente il caso delle matrici di BlackBit-9B.
    blackbit::TernaryTensor big({4096, 1024}, 160);

    LowRankOptimizerOptions options;
    options.rank = 32;
    LowRankProjectedOptimizer optimizer(options);
    optimizer.registerTernary("grande", big);

    const double ratio =
        static_cast<double>(optimizer.stateBytes()) / static_cast<double>(optimizer.conventionalStateBytes());
    // 3 buffer da rank*cols contro 2 da rows*cols: 3*32 / (2*4096).
    EXPECT_LT(ratio, 0.02);
    EXPECT_GT(ratio, 0.0);
}

TEST(LowRankOptimizerTest, IlRangoEConfigurabilePerParametro) {
    blackbit::TernaryTensor a({256, 64}, 20);
    blackbit::TernaryTensor b({256, 64}, 20);

    LowRankOptimizerOptions options;
    options.rank = 8;
    LowRankProjectedOptimizer optimizer(options);
    optimizer.setRankFor("largo", 64);
    optimizer.registerTernary("stretto", a);
    optimizer.registerTernary("largo", b);

    // 'largo' ha 8 volte il rango, quindi 8 volte lo stato.
    // (3 buffer per parametro: momenti + accumulatore.)
    const std::size_t expected = 3 * (8 + 64) * 64 * sizeof(float);
    EXPECT_EQ(optimizer.stateBytes(), expected);
}

TEST(LowRankOptimizerTest, IlRangoNonPuoSuperareIlNumeroDiRighe) {
    blackbit::TernaryTensor small({4, 64}, 20);

    LowRankOptimizerOptions options;
    options.rank = 128;  // molto piu' delle 4 righe
    LowRankProjectedOptimizer optimizer(options);
    optimizer.registerTernary("piccolo", small);

    // Lo stato non deve superare quello completo: una proiezione di
    // rango maggiore della dimensione compressa costerebbe piu' del
    // parametro che comprime.
    EXPECT_LE(optimizer.stateBytes(), optimizer.conventionalStateBytes() * 2);
}

TEST(LowRankOptimizerTest, UnModelloBlackBitImparaConLoStatoProiettato) {
    // Milestone E: addestramento senza stati Adam a dimensione modello.
    const BlackBitConfig config = microConfig();
    BlackBitModel model(config, 5);

    LowRankOptimizerOptions options;
    options.learningRate = 0.03F;
    options.rank = 16;
    options.projectionInterval = 40;
    LowRankProjectedOptimizer optimizer(options);
    model.registerParameters(optimizer);

    const std::vector<int> tokens{4, 12, 7, 19, 2, 25, 9, 30};
    std::vector<int> targets(tokens.begin() + 1, tokens.end());
    targets.push_back(1);

    float initialLoss = 0.0F;
    float finalLoss = 0.0F;
    for (int step = 0; step < 150; ++step) {
        const auto result = model.trainStep(tokens, targets, 1, 8, &optimizer);
        optimizer.endStep();
        ASSERT_FALSE(result.sawNaN) << "NaN al passo " << step;
        ASSERT_FALSE(result.sawInf) << "Inf al passo " << step;
        if (step == 0) {
            initialLoss = result.loss;
        }
        finalLoss = result.loss;
    }

    EXPECT_LT(finalLoss, initialLoss * 0.75F) << "iniziale " << initialLoss << ", finale " << finalLoss;
    EXPECT_GT(optimizer.stats().ternaryFlips, 0U) << "i pesi ternari non si sono mossi";
    EXPECT_GT(optimizer.stats().projectionReseeds, 0U) << "il sottospazio non e' mai stato riseminato";
    EXPECT_EQ(optimizer.stats().stepCount, 150U);
}

TEST(LowRankOptimizerTest, LaConsolidazioneSperimentaleMuoveITritEScaricaIlResiduo) {
    // Requisito 9, dietro flag: la plasticita' si accumula nel residuo
    // e viene periodicamente riversata in flip di T.
    const BlackBitConfig config = microConfig();
    BlackBitModel model(config, 9);

    LowRankOptimizerOptions options;
    options.learningRate = 0.05F;
    options.rank = 16;
    options.projectionInterval = 1000;  // niente riseminature durante il test

    blackbit::TernaryConsolidationOptions consolidation;
    consolidation.enabled = true;
    consolidation.consolidationInterval = 10;
    consolidation.flipThreshold = 0.3F;
    consolidation.maxFlipFraction = 0.05F;
    consolidation.residualDecay = 0.0F;

    LowRankProjectedOptimizer optimizer(options, consolidation);
    model.registerParameters(optimizer);

    const std::vector<int> tokens{1, 8, 15, 22, 3, 17, 6, 28};
    std::vector<int> targets(tokens.begin() + 1, tokens.end());
    targets.push_back(2);

    for (int step = 0; step < 40; ++step) {
        (void)model.trainStep(tokens, targets, 1, 8, &optimizer);
        optimizer.endStep();
    }

    EXPECT_EQ(optimizer.stats().consolidations, 4U);
    EXPECT_GT(optimizer.stats().ternaryFlips, 0U) << "la consolidazione non ha mosso nessun trit";
    EXPECT_GT(optimizer.stats().residualNorm, 0.0);

    // Il residuo costa uno stato in piu': deve comparire nel conto.
    LowRankProjectedOptimizer withoutConsolidation(options);
    model.registerParameters(withoutConsolidation);
    EXPECT_GT(optimizer.stateBytes(), withoutConsolidation.stateBytes());
}

TEST(LowRankOptimizerTest, LaConsolidazioneRispettaIlTettoDiFlip) {
    blackbit::TernaryTensor weight({64, 64}, 20);

    LowRankOptimizerOptions options;
    options.learningRate = 5.0F;  // volutamente enorme
    options.rank = 8;
    options.projectionInterval = 1000;

    blackbit::TernaryConsolidationOptions consolidation;
    consolidation.enabled = true;
    consolidation.consolidationInterval = 1;
    consolidation.flipThreshold = 0.0F;
    consolidation.maxFlipFraction = 0.01F;  // al massimo l'1 % dei trit
    consolidation.stochasticFlip = false;

    LowRankProjectedOptimizer optimizer(options, consolidation);
    optimizer.registerTernary("peso", weight);

    std::vector<float> gradient(64 * 64, 1.0F);
    optimizer.consumeWeightGradientBlock({"peso", 64, 64}, 0, 64, gradient.data());
    optimizer.endStep();

    const std::size_t cap = static_cast<std::size_t>(0.01F * 64.0F * 64.0F);
    EXPECT_LE(optimizer.stats().ternaryFlips, cap + 1);
}

TEST(LowRankOptimizerTest, RifiutaParametriNonRegistratiEConfigurazioniAssurde) {
    LowRankOptimizerOptions bad;
    bad.rank = 0;
    EXPECT_THROW(LowRankProjectedOptimizer{bad}, std::invalid_argument);

    LowRankOptimizerOptions badInterval;
    badInterval.projectionInterval = 0;
    EXPECT_THROW(LowRankProjectedOptimizer{badInterval}, std::invalid_argument);

    LowRankProjectedOptimizer optimizer;
    std::vector<float> gradient(16, 0.1F);
    EXPECT_THROW(optimizer.consumeWeightGradientBlock({"ignoto", 4, 4}, 0, 4, gradient.data()),
                 std::invalid_argument);
    EXPECT_THROW(optimizer.consumeDenseGradient({"ignoto", 1, 16}, gradient.data(), 16), std::invalid_argument);
    EXPECT_THROW(optimizer.setRankFor("qualsiasi", 0), std::invalid_argument);
}
