#include "blackforge/blackbit/model.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <random>

#include "blackforge/blackbit/telemetry.hpp"
#include "blackforge/blackbit/ternary_update.hpp"

using namespace blackforge;
using blackforge::blackbit::BlackBitConfig;
using blackforge::blackbit::BlackBitModel;

namespace {

// Il piu' piccolo BlackBit che sia ancora BlackBit: GQA con
// raggruppamento reale (4 teste di query su 2 di K/V), MoE top-2 su 4
// esperti, embedding legate. Stessa classe del modello da 9 miliardi.
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

TEST(BlackBitModelTest, IlForwardProduceLogitPerOgniTokenEVocabolo) {
    const BlackBitConfig config = microConfig();
    BlackBitModel model(config, 7);

    const std::vector<int> tokens{1, 5, 9, 3, 0, 7};
    blackbit::BlackBitForwardCache cache;
    const runtime::Tensor hidden = model.forwardHidden(tokens, 2, 3, cache);

    ASSERT_EQ(hidden.rank(), 3U);
    EXPECT_EQ(hidden.dim(0), 2U);
    EXPECT_EQ(hidden.dim(1), 3U);
    EXPECT_EQ(hidden.dim(2), config.hiddenSize);

    const runtime::Tensor logits = model.logits(hidden);
    EXPECT_EQ(logits.dim(0), 6U);
    EXPECT_EQ(logits.dim(1), config.vocabSize);
    for (std::size_t i = 0; i < logits.elementCount(); ++i) {
        ASSERT_TRUE(std::isfinite(logits.at(i))) << "logit non finito all'indice " << i;
    }

    EXPECT_EQ(cache.blocks.size(), config.numLayers);
    for (const auto& block : cache.blocks) {
        EXPECT_EQ(block.routing.tokens, 6U);
    }
}

TEST(BlackBitModelTest, LEmbeddingELaTestaDiUscitaSonoLoStessoParametro) {
    // Se non lo fossero, il modello avrebbe 2x i parametri
    // dell'embedding — 201 M in piu' su BlackBit-9B — senza che nulla lo
    // dichiari.
    const BlackBitConfig config = microConfig();
    BlackBitModel model(config, 11);

    EXPECT_EQ(model.embedding().outFeatures(), config.vocabSize);
    EXPECT_EQ(model.embedding().inFeatures(), config.hiddenSize);

    BlackBitConfig untied = config;
    untied.tieEmbeddings = false;
    EXPECT_THROW(BlackBitModel(untied, 11), std::invalid_argument);
}

TEST(BlackBitModelTest, LaLossPartePocoSopraIlLogaritmoDelVocabolario) {
    // Un modello appena inizializzato non deve sapere nulla: la
    // cross-entropy deve stare INTORNO a log(vocab), e il controllo e'
    // volutamente ASIMMETRICO.
    //
    // Sotto log(vocab) e' il caso pericoloso: significa che il modello
    // indovina meglio del caso senza aver imparato niente, cioe' che il
    // bersaglio sta trapelando nell'ingresso (una maschera causale
    // sbagliata, un target disallineato). Tolleranza stretta.
    //
    // Sopra e' benigno e atteso: con le embedding legate il logit del
    // vocabolo v e' <h, E_v>, e h non e' indipendente da E, quindi i
    // logit iniziali hanno una varianza non nulla che peggiora la loss
    // rispetto all'uniforme. Diventa sospetto solo se e' molto sopra,
    // il che indicherebbe logit esplosi.
    const BlackBitConfig config = microConfig();
    BlackBitModel model(config, 13);

    std::vector<int> tokens(8);
    std::vector<int> targets(8);
    std::iota(tokens.begin(), tokens.end(), 0);
    std::iota(targets.begin(), targets.end(), 1);

    const blackbit::BlackBitStepResult result = model.trainStep(tokens, targets, 1, 8, nullptr);

    const float uniform = std::log(static_cast<float>(config.vocabSize));
    EXPECT_EQ(result.scoredTokens, 8U);
    EXPECT_GT(result.loss, uniform - 0.3F) << "loss sotto l'uniforme a inizializzazione: il bersaglio trapela";
    EXPECT_LT(result.loss, uniform + 1.5F) << "logit iniziali troppo dispersi";
    EXPECT_FALSE(result.sawNaN);
    EXPECT_FALSE(result.sawInf);
}

TEST(BlackBitModelTest, LaLossACapiBloccoNonDipendeDalBloccoDiVocabolario) {
    // Il chunking del vocabolario e' memoria, non approssimazione: la
    // stessa loss deve uscire con qualunque dimensione di blocco.
    const BlackBitConfig config = microConfig();
    BlackBitModel model(config, 17);

    const std::vector<int> tokens{2, 4, 6, 8};
    const std::vector<int> targets{3, 5, 7, 9};

    model.setVocabChunk(config.vocabSize);
    const float reference = model.trainStep(tokens, targets, 1, 4, nullptr).loss;

    for (std::size_t chunk : {1U, 3U, 7U, 16U}) {
        model.setVocabChunk(chunk);
        EXPECT_NEAR(model.trainStep(tokens, targets, 1, 4, nullptr).loss, reference, 1e-5F)
            << "blocco di vocabolario " << chunk;
    }
}

TEST(BlackBitModelTest, ITargetIgnoratiNonContribuiscono) {
    const BlackBitConfig config = microConfig();
    BlackBitModel model(config, 19);

    const std::vector<int> tokens{1, 2, 3, 4};
    const std::vector<int> allScored{5, 6, 7, 8};
    const std::vector<int> someIgnored{5, -1, 7, -1};

    const auto full = model.trainStep(tokens, allScored, 1, 4, nullptr);
    const auto partial = model.trainStep(tokens, someIgnored, 1, 4, nullptr);

    EXPECT_EQ(full.scoredTokens, 4U);
    EXPECT_EQ(partial.scoredTokens, 2U);
    EXPECT_TRUE(std::isfinite(partial.loss));
}

TEST(BlackBitModelTest, GqaNonDuplicaLeTesteKV) {
    // Le proiezioni K e V producono numKvHeads * headDim, non
    // numHeads * headDim: se qualcuno "ripetesse" le teste per comodita'
    // il tensore crescerebbe qui.
    const BlackBitConfig config = microConfig();
    BlackBitModel model(config, 23);

    const std::vector<int> tokens{1, 2, 3, 4, 5, 6};
    blackbit::BlackBitForwardCache cache;
    (void)model.forwardHidden(tokens, 1, 6, cache);

    const auto& attention = cache.blocks.front().attention;
    EXPECT_EQ(attention.query.dim(2), config.numHeads * config.headDim);
    EXPECT_EQ(attention.key.dim(2), config.numKvHeads * config.headDim);
    EXPECT_EQ(attention.value.dim(2), config.numKvHeads * config.headDim);
    EXPECT_LT(attention.key.elementCount(), attention.query.elementCount());
}

TEST(BlackBitModelTest, LAttentionECausale) {
    // Cambiare un token FUTURO non deve toccare le posizioni
    // precedenti. E' il controllo che smaschera una maschera causale
    // sbagliata, che altrimenti si manifesta solo come "il modello
    // impara sospettosamente in fretta".
    const BlackBitConfig config = microConfig();
    BlackBitModel model(config, 29);

    std::vector<int> tokens{1, 2, 3, 4, 5, 6};
    blackbit::BlackBitForwardCache cacheA;
    const runtime::Tensor a = model.forwardHidden(tokens, 1, 6, cacheA);

    tokens.back() = 11;  // cambia solo l'ultima posizione
    blackbit::BlackBitForwardCache cacheB;
    const runtime::Tensor b = model.forwardHidden(tokens, 1, 6, cacheB);

    const std::size_t hidden = config.hiddenSize;
    for (std::size_t position = 0; position + 1 < 6; ++position) {
        for (std::size_t d = 0; d < hidden; ++d) {
            ASSERT_NEAR(a.at(position * hidden + d), b.at(position * hidden + d), 1e-5F)
                << "posizione " << position << " influenzata da un token futuro";
        }
    }
}

TEST(BlackBitModelTest, IlBackwardConsegnaGradientiAOgniParametro) {
    const BlackBitConfig config = microConfig();
    BlackBitModel model(config, 31);

    blackbit::DenseGradientCollector collector;
    std::mt19937 rng(3);
    for (int step = 0; step < 4; ++step) {
        std::vector<int> tokens(8);
        std::vector<int> targets(8);
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            tokens[i] = static_cast<int>(rng() % config.vocabSize);
            targets[i] = static_cast<int>(rng() % config.vocabSize);
        }
        (void)model.trainStep(tokens, targets, 1, 8, &collector);
    }

    EXPECT_TRUE(collector.has("embedding"));
    EXPECT_TRUE(collector.has("final_norm"));
    for (std::size_t layer = 0; layer < config.numLayers; ++layer) {
        const std::string prefix = "layer" + std::to_string(layer);
        EXPECT_TRUE(collector.has(prefix + ".attn_norm"));
        EXPECT_TRUE(collector.has(prefix + ".moe_norm"));
        EXPECT_TRUE(collector.has(prefix + ".attn.q"));
        EXPECT_TRUE(collector.has(prefix + ".attn.k"));
        EXPECT_TRUE(collector.has(prefix + ".attn.v"));
        EXPECT_TRUE(collector.has(prefix + ".attn.o"));
        EXPECT_TRUE(collector.has(prefix + ".moe.router"));
        for (std::size_t e = 0; e < config.numExperts; ++e) {
            EXPECT_TRUE(collector.has(prefix + ".moe.expert" + std::to_string(e) + ".down"));
        }
    }
}

TEST(BlackBitModelTest, ImparaUnCompitoDiMemorizzazioneSintetico) {
    // Milestone C: un modello linguistico BlackBit completo — GQA,
    // MoE top-2, embedding legate, pesi TUTTI ternari tranne norm e
    // router — riduce davvero la cross-entropy.
    const BlackBitConfig config = microConfig();
    BlackBitModel model(config, 37);

    blackbit::TernarySgdSink optimizer(0.03F);
    model.registerParameters(optimizer);

    // Una sequenza fissa da memorizzare: il compito piu' semplice che
    // richiede comunque che ogni parte del modello funzioni.
    const std::vector<int> tokens{3, 9, 14, 2, 7, 11, 5, 1};
    std::vector<int> targets(tokens.begin() + 1, tokens.end());
    targets.push_back(0);

    float initialLoss = 0.0F;
    float finalLoss = 0.0F;
    for (int step = 0; step < 150; ++step) {
        const auto result = model.trainStep(tokens, targets, 1, 8, &optimizer);
        optimizer.endStep();
        ASSERT_FALSE(result.sawNaN) << "NaN al passo " << step << " in " << result.firstUnstableTensor;
        ASSERT_FALSE(result.sawInf) << "Inf al passo " << step << " in " << result.firstUnstableTensor;
        if (step == 0) {
            initialLoss = result.loss;
        }
        finalLoss = result.loss;
    }

    EXPECT_LT(finalLoss, initialLoss * 0.7F) << "iniziale " << initialLoss << ", finale " << finalLoss;
    EXPECT_GT(optimizer.stats().flips, 0U) << "nessun peso ternario si e' mosso";
}

TEST(BlackBitModelTest, IlModelloOccupaMenoDiDueBitPerPesoTernario) {
    const BlackBitConfig config = microConfig();
    BlackBitModel model(config, 41);

    const blackbit::ParameterCount count = blackbit::countParameters(config);
    const double bitsPerParameter =
        static_cast<double>(model.parameterBytes()) * 8.0 / static_cast<double>(count.total());

    // Denso in float32 sarebbero 32 bit/peso; in BF16, 16.
    //
    // Su QUESTA configurazione micro il valore reale e' ~4,4 bit, non
    // 1,8: con righe da 32 elementi e gruppi da 20, ogni riga paga due
    // parole intere (40 posizioni impacchettate per 32 usate) e due
    // scale da 4 byte. E' overhead di dimensione, non di formato:
    // scompare non appena le righe hanno la lunghezza di un modello
    // vero (a hidden 3072 il conto e' 1,81 bit/peso, verificato in
    // TernaryTensorTest.LaDensitaDichiarataEQuellaReale e in
    // BlackBitConfigTest.LaStimaDiMemoriaStaSottoGliOttoGigabyte).
    EXPECT_LT(bitsPerParameter, 5.0);
    EXPECT_GT(bitsPerParameter, 1.5);
    EXPECT_LT(bitsPerParameter, 16.0 / 3.0) << "deve comunque costare meno di un terzo di BF16";

    // E la telemetria deve conoscere ogni byte: nessuna allocazione di
    // parametri nascosta.
    EXPECT_GE(blackbit::MemoryTelemetry::instance().current(blackbit::MemoryArena::Parameter),
               model.parameterBytes());
}
