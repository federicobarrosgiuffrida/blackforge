#include "blackforge/blackbit/config.hpp"

#include <gtest/gtest.h>

#include <fstream>

using namespace blackforge;
using blackforge::blackbit::BlackBitConfig;

namespace {

double giga(std::size_t value) { return static_cast<double>(value) / 1e9; }

double gibibytes(std::size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }

}  // namespace

TEST(BlackBitConfigTest, LaConfigurazione9bCentraIlBersaglioDiParametri) {
    const BlackBitConfig config = blackbit::blackBit9bA3b();
    config.validate();

    const blackbit::ParameterCount count = blackbit::countParameters(config);

    // ~9B totali: l'obiettivo dichiarato e' 8,5-9,5 G.
    EXPECT_GT(giga(count.total()), 8.5);
    EXPECT_LT(giga(count.total()), 9.5);

    // ~3B attivi per token (2 esperti su 8).
    const double active = giga(blackbit::countActiveParameters(config));
    EXPECT_GT(active, 2.5);
    EXPECT_LT(active, 3.5);

    // Gli esperti dominano: e' cio' che rende sensato il routing sparso.
    EXPECT_GT(count.experts, count.total() / 2);

    // I parametri densi (router + norm) devono restare trascurabili,
    // altrimenti tenerli fuori dalla quantizzazione costerebbe memoria.
    EXPECT_LT(static_cast<double>(count.dense()) / static_cast<double>(count.total()), 0.001);
}

TEST(BlackBitConfigTest, LeConfigurazioniDiValidazioneCresconoInScala) {
    const auto tiny = blackbit::countParameters(blackbit::blackBitTiny()).total();
    const auto small = blackbit::countParameters(blackbit::blackBitSmall()).total();
    const auto medium = blackbit::countParameters(blackbit::blackBitMedium()).total();
    const auto full = blackbit::countParameters(blackbit::blackBit9bA3b()).total();

    EXPECT_GT(tiny, 20'000'000U);
    EXPECT_LT(tiny, 50'000'000U);
    EXPECT_GT(small, 100'000'000U);
    EXPECT_LT(small, 150'000'000U);
    EXPECT_GT(medium, 300'000'000U);
    EXPECT_LT(medium, 500'000'000U);
    EXPECT_LT(tiny, small);
    EXPECT_LT(small, medium);
    EXPECT_LT(medium, full);
}

TEST(BlackBitConfigTest, GqaRiduceIParametriDiKeV) {
    BlackBitConfig gqa = blackbit::blackBit9bA3b();
    BlackBitConfig mha = gqa;
    mha.numKvHeads = mha.numHeads;  // stessa architettura senza raggruppamento

    EXPECT_LT(blackbit::countParameters(gqa).attention, blackbit::countParameters(mha).attention);
}

TEST(BlackBitConfigTest, LaStimaDiMemoriaStaSottoGliOttoGigabyte) {
    const BlackBitConfig config = blackbit::blackBit9bA3b();
    const blackbit::MemoryEstimate estimate =
        blackbit::estimateTrainingMemory(config, {1, 512}, blackbit::LowMemoryOptions{});

    // Il vincolo di progetto: tutto il passo di addestramento sotto gli
    // 8 GB (e sotto la soglia piu' stretta di 7,8 GB).
    EXPECT_LT(gibibytes(estimate.total()), 7.8);

    // I pesi impacchettati devono avvicinarsi a 1,6 bit/peso: se questo
    // controllo fallisce, da qualche parte si sta memorizzando molto
    // piu' di un trit per peso.
    const double bitsPerWeight = static_cast<double>(estimate.packedWeightBytes) * 8.0 /
                                  static_cast<double>(blackbit::countParameters(config).ternary());
    EXPECT_NEAR(bitsPerWeight, 1.6, 0.02);

    // Il picco di gradiente e' quello di UN BLOCCO DI RIGHE di una
    // matrice, non del modello: deve restare nell'ordine delle decine
    // di MB anche con una tabella di embedding da 201 M parametri.
    EXPECT_LT(estimate.gradientPeakBytes, 64U * 1024U * 1024U);
    EXPECT_LT(static_cast<double>(estimate.gradientPeakBytes) / static_cast<double>(estimate.total()), 0.01);

    // Lo stato dell'optimizer low-rank deve costare meno dei pesi
    // stessi: con AdamW ordinario costerebbe 4 volte tanto.
    EXPECT_LT(estimate.optimizerBytes, estimate.packedWeightBytes);

    // E il confronto che giustifica il progetto: l'approccio ordinario
    // non entra nemmeno lontanamente in una scheda da 8 GB.
    EXPECT_GT(gibibytes(estimate.conventionalBytes), 100.0);
}

TEST(BlackBitConfigTest, IlRicalcoloDelleAttivazioniRiduceLaMemoria) {
    const BlackBitConfig config = blackbit::blackBit9bA3b();
    blackbit::LowMemoryOptions withCheckpointing;
    blackbit::LowMemoryOptions withoutCheckpointing;
    withoutCheckpointing.activationCheckpointing = false;

    const auto a = blackbit::estimateTrainingMemory(config, {1, 512}, withCheckpointing);
    const auto b = blackbit::estimateTrainingMemory(config, {1, 512}, withoutCheckpointing);
    EXPECT_LT(a.activationBytes, b.activationBytes);
}

TEST(BlackBitConfigTest, RifiutaConfigurazioniIncoerenti) {
    BlackBitConfig config = blackbit::blackBitTiny();

    BlackBitConfig badHeads = config;
    badHeads.numKvHeads = 4;  // 6 non e' multiplo di 4
    EXPECT_THROW(badHeads.validate(), std::invalid_argument);

    BlackBitConfig badTopK = config;
    badTopK.expertsPerToken = badTopK.numExperts + 1;
    EXPECT_THROW(badTopK.validate(), std::invalid_argument);

    BlackBitConfig badGroup = config;
    badGroup.ternaryGroupSize = 30;  // non multiplo di 20
    EXPECT_THROW(badGroup.validate(), std::invalid_argument);

    BlackBitConfig badVocab = config;
    badVocab.vocabSize = 0;
    EXPECT_THROW(badVocab.validate(), std::invalid_argument);
}

TEST(BlackBitConfigTest, JsonRoundTrip) {
    const BlackBitConfig original = blackbit::blackBit9bA3b();
    const BlackBitConfig reloaded = blackbit::parseConfigJson(blackbit::toJson(original));

    EXPECT_EQ(reloaded.name, original.name);
    EXPECT_EQ(reloaded.vocabSize, original.vocabSize);
    EXPECT_EQ(reloaded.hiddenSize, original.hiddenSize);
    EXPECT_EQ(reloaded.numLayers, original.numLayers);
    EXPECT_EQ(reloaded.numHeads, original.numHeads);
    EXPECT_EQ(reloaded.numKvHeads, original.numKvHeads);
    EXPECT_EQ(reloaded.headDim, original.headDim);
    EXPECT_EQ(reloaded.numExperts, original.numExperts);
    EXPECT_EQ(reloaded.expertsPerToken, original.expertsPerToken);
    EXPECT_EQ(reloaded.expertHidden, original.expertHidden);
    EXPECT_EQ(reloaded.tieEmbeddings, original.tieEmbeddings);
    EXPECT_EQ(reloaded.weightDtype, original.weightDtype);
    EXPECT_EQ(blackbit::countParameters(reloaded).total(), blackbit::countParameters(original).total());
}

TEST(BlackBitConfigTest, IlJsonRifiutaUnaChiaveSconosciuta) {
    // Un refuso in un nome di campo deve fermare l'esecuzione, non
    // essere ignorato producendo un modello di dimensioni sbagliate.
    EXPECT_THROW(blackbit::parseConfigJson(R"({"hiden_size": 3072})"), std::runtime_error);
    EXPECT_THROW(blackbit::parseConfigJson(R"({"hidden_size": "grande"})"), std::runtime_error);
    EXPECT_THROW(blackbit::parseConfigJson(R"({"hidden_size": 3072)"), std::runtime_error);
}

TEST(BlackBitConfigTest, IlJsonAccettaCommentiEValoriParziali) {
    const BlackBitConfig config = blackbit::parseConfigJson(R"({
        // solo i campi che ci interessano: il resto resta al default
        "name": "prova",
        "num_layers": 3,
        "expert_capacity_factor": 2.0
    })");

    EXPECT_EQ(config.name, "prova");
    EXPECT_EQ(config.numLayers, 3U);
    EXPECT_FLOAT_EQ(config.expertCapacityFactor, 2.0F);
    EXPECT_EQ(config.hiddenSize, blackbit::BlackBitConfig{}.hiddenSize);
}

TEST(BlackBitConfigTest, IPresetSonoRaggiungibiliPerNome) {
    EXPECT_EQ(blackbit::blackBitPreset("tiny").name, "blackbit-tiny");
    EXPECT_EQ(blackbit::blackBitPreset("9b-a3b").name, "blackbit-9b-a3b");
    EXPECT_THROW(blackbit::blackBitPreset("gigante"), std::invalid_argument);
}

TEST(BlackBitConfigTest, IFileDiConfigurazioneDelRepositoryCoincidonoConIPreset) {
    // I file in configs/ sono quelli che la CLI riceve con --config: se
    // divergessero dai preset usati dai test, il modello misurato dal
    // benchmark non sarebbe quello validato qui.
    struct Caso {
        const char* path;
        BlackBitConfig (*preset)();
    };
    const Caso casi[] = {
        {"configs/blackbit_tiny.json", &blackbit::blackBitTiny},
        {"configs/blackbit_small.json", &blackbit::blackBitSmall},
        {"configs/blackbit_medium.json", &blackbit::blackBitMedium},
        {"configs/blackbit_9b_a3b.json", &blackbit::blackBit9bA3b},
    };

    for (const Caso& caso : casi) {
        const std::string path = std::string(BLACKFORGE_REPO_ROOT) + "/" + caso.path;
        std::ifstream probe(path);
        ASSERT_TRUE(probe.good()) << "file di configurazione mancante: " << path;
        probe.close();

        const BlackBitConfig fromFile = blackbit::loadConfigFromJson(path);
        const BlackBitConfig expected = caso.preset();
        EXPECT_EQ(fromFile.name, expected.name);
        EXPECT_EQ(blackbit::countParameters(fromFile).total(), blackbit::countParameters(expected).total());
        EXPECT_EQ(blackbit::toJson(fromFile), blackbit::toJson(expected));
    }
}
