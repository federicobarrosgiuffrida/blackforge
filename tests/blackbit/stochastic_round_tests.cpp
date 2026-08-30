#include "blackforge/blackbit/stochastic_round.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace blackforge;

TEST(StochasticRoundTest, EDeterministicoRispettoASemeEIndice) {
    // La riproducibilita' non e' un dettaglio: senza, un addestramento
    // non e' ripetibile e nessuna regressione numerica e' diagnosticabile.
    for (std::uint64_t index = 0; index < 100; ++index) {
        EXPECT_FLOAT_EQ(blackbit::stochasticRound(0.37F, 12345, index),
                        blackbit::stochasticRound(0.37F, 12345, index));
    }
    EXPECT_NE(blackbit::counterRandom(1, 0), blackbit::counterRandom(2, 0));
    EXPECT_NE(blackbit::counterRandom(1, 0), blackbit::counterRandom(1, 1));
}

TEST(StochasticRoundTest, LUniformeStaInZeroUnoEDistribuisceBene) {
    constexpr std::size_t kSamples = 200000;
    double sum = 0.0;
    std::vector<std::size_t> buckets(10, 0);

    for (std::size_t i = 0; i < kSamples; ++i) {
        const float u = blackbit::counterUniform(0xABCDEF, i);
        ASSERT_GE(u, 0.0F);
        ASSERT_LT(u, 1.0F);
        sum += u;
        buckets[static_cast<std::size_t>(u * 10.0F)] += 1;
    }

    EXPECT_NEAR(sum / static_cast<double>(kSamples), 0.5, 0.01);
    for (std::size_t count : buckets) {
        // Ogni decimo dovrebbe raccogliere ~10 % dei campioni.
        EXPECT_NEAR(static_cast<double>(count) / static_cast<double>(kSamples), 0.1, 0.01);
    }
}

TEST(StochasticRoundTest, ENonDistortoInValoreAtteso) {
    // La proprieta' che rende utile l'arrotondamento stocastico: la
    // media dei risultati deve tendere al valore esatto, altrimenti gli
    // aggiornamenti piccoli non sparirebbero ma verrebbero sistemati
    // male, che e' peggio.
    constexpr std::size_t kSamples = 200000;

    for (float value : {0.1F, 0.25F, 0.5F, 0.9F, -0.3F, 2.75F, -1.6F}) {
        double sum = 0.0;
        for (std::size_t i = 0; i < kSamples; ++i) {
            sum += blackbit::stochasticRound(value, 7, i);
        }
        const double mean = sum / static_cast<double>(kSamples);
        // Errore standard di una Bernoulli su 200 000 campioni: < 0,002.
        EXPECT_NEAR(mean, static_cast<double>(value), 0.005) << "valore " << value;
    }
}

TEST(StochasticRoundTest, UnValoreGiaInteroNonSiMuoveMai) {
    for (float value : {-2.0F, -1.0F, 0.0F, 1.0F, 3.0F}) {
        for (std::uint64_t i = 0; i < 1000; ++i) {
            ASSERT_FLOAT_EQ(blackbit::stochasticRound(value, 99, i), value) << "valore " << value;
        }
    }
}

TEST(StochasticRoundTest, UnAggiornamentoMinusculoDiventaUnFlipRaroMaEsatto) {
    // Il caso concreto che motiva tutto: aggiornamento 300 volte piu'
    // piccolo del passo della griglia. Arrotondato al piu' vicino
    // sparirebbe; qui deve produrre un flip con probabilita' 1/300 e
    // restare esatto in media.
    constexpr std::size_t kSamples = 400000;
    const float target = 1.0F / 300.0F;  // partendo dal trit 0

    std::size_t flips = 0;
    double sum = 0.0;
    for (std::size_t i = 0; i < kSamples; ++i) {
        const int trit = blackbit::stochasticRoundToTrit(target, 4242, i);
        flips += trit != 0 ? 1 : 0;
        sum += trit;
    }

    const double flipRate = static_cast<double>(flips) / static_cast<double>(kSamples);
    EXPECT_NEAR(flipRate, 1.0 / 300.0, 0.001);
    EXPECT_NEAR(sum / static_cast<double>(kSamples), static_cast<double>(target), 0.001);
    EXPECT_GT(flips, 0U) << "con l'arrotondamento al piu' vicino sarebbero zero: e' esattamente il modo in cui un "
                             "addestramento a bassa precisione puo' fingere di funzionare";
}

TEST(StochasticRoundTest, IlTritSatuaraFuoriDaMenoUnoPiuUno) {
    for (std::uint64_t i = 0; i < 500; ++i) {
        EXPECT_LE(blackbit::stochasticRoundToTrit(5.0F, 1, i), 1);
        EXPECT_GE(blackbit::stochasticRoundToTrit(-5.0F, 1, i), -1);
    }
    // Dentro l'intervallo, invece, ogni valore ternario e' raggiungibile.
    bool sawMinus = false;
    bool sawZero = false;
    bool sawPlus = false;
    for (std::uint64_t i = 0; i < 500; ++i) {
        const int trit = blackbit::stochasticRoundToTrit(0.5F, 2, i);
        sawZero = sawZero || trit == 0;
        sawPlus = sawPlus || trit == 1;
        sawMinus = sawMinus || blackbit::stochasticRoundToTrit(-0.5F, 3, i) == -1;
    }
    EXPECT_TRUE(sawZero);
    EXPECT_TRUE(sawPlus);
    EXPECT_TRUE(sawMinus);
}
