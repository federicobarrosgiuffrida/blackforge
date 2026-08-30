#include "blackforge/blackbit/moe.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <set>

#include "blackforge/backend/cpu/loss.hpp"
#include "blackforge/blackbit/telemetry.hpp"
#include "blackforge/blackbit/ternary_update.hpp"

using namespace blackforge;
using blackforge::blackbit::BlackBitConfig;
using blackforge::blackbit::MoECache;
using blackforge::blackbit::MoELayer;
using blackforge::blackbit::MoERoutingStats;

namespace {

BlackBitConfig tinyMoEConfig() {
    BlackBitConfig config = blackbit::blackBitTiny();
    config.hiddenSize = 20;
    config.expertHidden = 20;
    config.numExperts = 4;
    config.expertsPerToken = 2;
    config.ternaryGroupSize = 20;
    config.numHeads = 2;
    config.numKvHeads = 1;
    config.headDim = 10;
    return config;
}

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

}  // namespace

TEST(MoETest, IlRouterProduceUnaDistribuzioneDiProbabilita) {
    const BlackBitConfig config = tinyMoEConfig();
    blackbit::MoERouter router("router", config.hiddenSize, config.numExperts);
    router.initialize(7);

    const runtime::Tensor input = randomTensor({12, config.hiddenSize}, 3);
    const runtime::Tensor probs = router.probabilities(input);

    ASSERT_EQ(probs.dim(0), 12U);
    ASSERT_EQ(probs.dim(1), config.numExperts);
    for (std::size_t t = 0; t < 12; ++t) {
        double sum = 0.0;
        for (std::size_t e = 0; e < config.numExperts; ++e) {
            const float p = probs.at(t * config.numExperts + e);
            ASSERT_GT(p, 0.0F);
            ASSERT_LT(p, 1.0F);
            sum += p;
        }
        ASSERT_NEAR(sum, 1.0, 1e-5);
    }
}

TEST(MoETest, ATuttiGliEspertiArrivanoTokenEIPesiSommanoAUno) {
    const BlackBitConfig config = tinyMoEConfig();
    MoELayer layer("moe", config);
    layer.initialize(11);

    MoECache cache;
    MoERoutingStats stats;
    const runtime::Tensor input = randomTensor({64, config.hiddenSize}, 5);
    const runtime::Tensor output = layer.forward(input, cache, stats);

    ASSERT_EQ(output.shape(), input.shape());
    EXPECT_EQ(stats.tokens, 64U);
    EXPECT_EQ(stats.assignments, 64U * config.expertsPerToken);

    // Con un router inizializzato quasi uniforme nessun esperto deve
    // restare a bocca asciutta: se succedesse, quell'esperto non
    // riceverebbe mai un gradiente e sarebbe parametri sprecati.
    for (std::size_t e = 0; e < config.numExperts; ++e) {
        EXPECT_GT(stats.tokensPerExpert[e], 0U) << "esperto " << e << " senza token";
    }

    // I pesi dei soli esperti scelti sommano a 1 per token (Mixtral).
    for (std::size_t t = 0; t < 64; ++t) {
        float sum = 0.0F;
        for (std::size_t slot = 0; slot < config.expertsPerToken; ++slot) {
            sum += cache.weightOfSlot[t * config.expertsPerToken + slot];
        }
        ASSERT_NEAR(sum, 1.0F, 1e-4F) << "token " << t;
    }
}

TEST(MoETest, IlRoutingEDeterministico) {
    const BlackBitConfig config = tinyMoEConfig();
    MoELayer layer("moe", config);
    layer.initialize(13);

    const runtime::Tensor input = randomTensor({32, config.hiddenSize}, 17);

    MoECache first;
    MoERoutingStats firstStats;
    const runtime::Tensor a = layer.forward(input, first, firstStats);

    MoECache second;
    MoERoutingStats secondStats;
    const runtime::Tensor b = layer.forward(input, second, secondStats);

    EXPECT_EQ(first.expertOfSlot, second.expertOfSlot);
    EXPECT_EQ(firstStats.tokensPerExpert, secondStats.tokensPerExpert);
    for (std::size_t i = 0; i < a.elementCount(); ++i) {
        ASSERT_FLOAT_EQ(a.at(i), b.at(i));
    }
}

TEST(MoETest, LEntropiaELaLossDiBilanciamentoRiflettonoIlCarico) {
    const BlackBitConfig config = tinyMoEConfig();
    MoELayer layer("moe", config);
    layer.initialize(19);

    MoECache cache;
    MoERoutingStats stats;
    const runtime::Tensor input = randomTensor({128, config.hiddenSize}, 23);
    (void)layer.forward(input, cache, stats);

    // Router quasi uniforme a inizializzazione: entropia vicina al
    // massimo log(numExperts) e loss di bilanciamento vicina a 1.
    const double maxEntropy = std::log(static_cast<double>(config.numExperts));
    EXPECT_GT(stats.routingEntropy, maxEntropy * 0.95);
    EXPECT_LE(stats.routingEntropy, maxEntropy + 1e-6);
    EXPECT_NEAR(stats.loadBalancingLoss, 1.0F, 0.25F);

    // Le utilizzazioni sommano a 1 su tutti gli esperti.
    double totalUtilization = 0.0;
    for (std::size_t e = 0; e < config.numExperts; ++e) {
        totalUtilization += stats.utilization(e);
    }
    EXPECT_NEAR(totalUtilization, 1.0, 1e-9);
    EXPECT_FALSE(stats.toString().empty());
}

TEST(MoETest, LaCapacitaScartaLeAssegnazioniInEccessoELeConta) {
    BlackBitConfig config = tinyMoEConfig();
    config.expertCapacityFactor = 0.25F;  // capacita' molto sotto la quota equa
    MoELayer layer("moe", config);
    layer.initialize(29);

    MoECache cache;
    MoERoutingStats stats;
    const runtime::Tensor input = randomTensor({64, config.hiddenSize}, 31);
    (void)layer.forward(input, cache, stats);

    EXPECT_GT(stats.droppedAssignments, 0U);
    EXPECT_GT(stats.dropRate(), 0.0);
    for (std::size_t e = 0; e < config.numExperts; ++e) {
        EXPECT_LE(stats.tokensPerExpert[e], layer.capacityFor(64));
    }

    // Uno slot scartato deve avere esperto -1 e non contribuire.
    std::size_t dropped = 0;
    for (int expert : cache.expertOfSlot) {
        dropped += expert < 0 ? 1 : 0;
    }
    EXPECT_EQ(dropped, stats.droppedAssignments);
}

TEST(MoETest, NonAllocaMaiUnTensoreDensoTokenPerEsperti) {
    // Il controllo che distingue un MoE sparso da uno che finge:
    // l'attivazione viva non deve mai avvicinarsi a
    // token * esperti * hidden.
    BlackBitConfig config = tinyMoEConfig();
    config.hiddenSize = 40;
    config.expertHidden = 40;
    config.numExperts = 8;
    MoELayer layer("moe", config);
    layer.initialize(37);

    // resetPeaks() e non reset(): il layer e' gia' costruito e ha
    // registrato i propri parametri (vedi la nota su reset() in
    // telemetry.hpp).
    blackbit::MemoryTelemetry::instance().resetPeaks();
    MoECache cache;
    MoERoutingStats stats;
    const runtime::Tensor input = randomTensor({100, config.hiddenSize}, 41);
    (void)layer.forward(input, cache, stats);

    const std::size_t densePerExpertBytes = 100U * config.numExperts * config.hiddenSize * sizeof(float);
    const std::size_t peak = blackbit::MemoryTelemetry::instance().peak(blackbit::MemoryArena::Activation);
    EXPECT_LT(peak, densePerExpertBytes / 4)
        << "picco attivazioni " << peak << " byte, un tensore denso [token, esperti, hidden] ne userebbe "
        << densePerExpertBytes;
    EXPECT_EQ(blackbit::MemoryTelemetry::instance().inconsistencies(), 0U);
}

TEST(MoETest, IlGradienteDellIngressoCoincideConQuelloNumerico) {
    BlackBitConfig config = tinyMoEConfig();
    config.routerAuxLossWeight = 0.0F;  // la loss ausiliaria non e' nella loss confrontata
    MoELayer layer("moe", config);
    layer.initialize(43);

    const runtime::Tensor input = randomTensor({3, config.hiddenSize}, 47, 0.5F);
    const runtime::Tensor target = randomTensor({3, config.hiddenSize}, 53, 0.5F);

    MoECache cache;
    MoERoutingStats stats;
    const runtime::Tensor output = layer.forward(input, cache, stats);
    const backend::cpu::LossResult loss = backend::cpu::meanSquaredError(output, target);
    const runtime::Tensor gradInput = layer.backward(input, loss.grad, cache, stats, nullptr);

    const float epsilon = 1e-2F;
    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        runtime::Tensor plus = input;
        runtime::Tensor minus = input;
        plus.at(i) += epsilon;
        minus.at(i) -= epsilon;

        MoECache scratch;
        MoERoutingStats scratchStats;
        const float lossPlus =
            backend::cpu::meanSquaredError(layer.forward(plus, scratch, scratchStats), target).value;
        const float lossMinus =
            backend::cpu::meanSquaredError(layer.forward(minus, scratch, scratchStats), target).value;

        // Il gradiente numerico e' valido solo se la perturbazione non
        // cambia la SELEZIONE degli esperti (il routing top-k e' una
        // funzione a gradini): si salta il confronto quando succede.
        MoECache checkPlus;
        MoERoutingStats checkPlusStats;
        (void)layer.forward(plus, checkPlus, checkPlusStats);
        if (checkPlus.expertOfSlot != cache.expertOfSlot) {
            continue;
        }

        const float numeric = (lossPlus - lossMinus) / (2.0F * epsilon);
        ASSERT_NEAR(gradInput.at(i), numeric, 2e-3F) << "elemento " << i;
    }
}

TEST(MoETest, TuttiGliEspertiRicevonoGradienteNelTempo) {
    // Requisito 14, test 5: se un esperto non riceve mai gradiente, i
    // suoi parametri sono peso morto.
    const BlackBitConfig config = tinyMoEConfig();
    MoELayer layer("moe", config);
    layer.initialize(59);

    blackbit::DenseGradientCollector collector;
    for (int step = 0; step < 5; ++step) {
        const runtime::Tensor input = randomTensor({48, config.hiddenSize}, 61 + step, 0.5F);
        const runtime::Tensor target = randomTensor({48, config.hiddenSize}, 71 + step, 0.5F);

        MoECache cache;
        MoERoutingStats stats;
        const runtime::Tensor output = layer.forward(input, cache, stats);
        const backend::cpu::LossResult loss = backend::cpu::meanSquaredError(output, target);
        (void)layer.backward(input, loss.grad, cache, stats, &collector);
    }

    for (std::size_t e = 0; e < config.numExperts; ++e) {
        const std::string prefix = "moe.expert" + std::to_string(e);
        for (const char* suffix : {".gate", ".up", ".down"}) {
            const std::string name = prefix + suffix;
            ASSERT_TRUE(collector.has(name)) << name << " non ha mai ricevuto gradiente";
            const std::vector<float>& grad = collector.gradient(name);
            const bool nonZero = std::any_of(grad.begin(), grad.end(), [](float v) { return v != 0.0F; });
            EXPECT_TRUE(nonZero) << name << " ha ricevuto solo gradienti nulli";
        }
    }
    EXPECT_TRUE(collector.has("moe.router"));
}

TEST(MoETest, UnBloccoMoeImparaConPesiSoloTernari) {
    const BlackBitConfig config = tinyMoEConfig();
    MoELayer layer("moe", config);
    layer.initialize(83);

    blackbit::TernarySgdSink optimizer(0.02F);  // in unita' di griglia: ~2 % di passo ternario atteso
    for (std::size_t e = 0; e < config.numExperts; ++e) {
        blackbit::MoEExpert& expert = layer.experts()[e];
        optimizer.registerTernary(expert.gate().name(), expert.gate().weight());
        optimizer.registerTernary(expert.up().name(), expert.up().weight());
        optimizer.registerTernary(expert.down().name(), expert.down().weight());
    }
    optimizer.registerDense(layer.router().name(), layer.router().weight());

    const runtime::Tensor input = randomTensor({32, config.hiddenSize}, 89, 0.5F);
    const runtime::Tensor target = randomTensor({32, config.hiddenSize}, 97, 0.3F);

    auto step = [&]() {
        MoECache cache;
        MoERoutingStats stats;
        const runtime::Tensor output = layer.forward(input, cache, stats);
        const backend::cpu::LossResult loss = backend::cpu::meanSquaredError(output, target);
        (void)layer.backward(input, loss.grad, cache, stats, &optimizer);
        optimizer.endStep();
        return loss.value;
    };

    const float initialLoss = step();
    float finalLoss = initialLoss;
    for (int i = 0; i < 250; ++i) {
        finalLoss = step();
    }

    EXPECT_LT(finalLoss, initialLoss * 0.5F) << "iniziale " << initialLoss << ", finale " << finalLoss;
    EXPECT_GT(optimizer.stats().flips, 100U);
}
