#include "blackforge/blackbit/checkpoint.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "blackforge/blackbit/benchmark.hpp"

using namespace blackforge;
using blackforge::blackbit::BlackBitConfig;
using blackforge::blackbit::BlackBitModel;
using blackforge::blackbit::BlackBitTrainingState;

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

// File temporaneo che si cancella da solo, anche se il test fallisce.
class TempFile {
public:
    explicit TempFile(const std::string& name)
        : path_((std::filesystem::temp_directory_path() / name).string()) {}
    ~TempFile() { std::remove(path_.c_str()); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    [[nodiscard]] const std::string& path() const { return path_; }

private:
    std::string path_;
};

std::vector<int> trits(BlackBitModel& model) {
    std::vector<int> values;
    const blackbit::TernaryTensor& weight = model.embedding().weight();
    values.reserve(weight.elementCount());
    for (std::size_t i = 0; i < weight.elementCount(); ++i) {
        values.push_back(weight.tritAt(i));
    }
    return values;
}

}  // namespace

TEST(BlackBitCheckpointTest, SalvaERicaricaEsattamente) {
    const BlackBitConfig config = microConfig();
    TempFile file("blackbit_roundtrip.bfbit");

    std::vector<int> saved;
    std::vector<float> savedNorm;
    {
        BlackBitModel model(config, 5);
        saved = trits(model);
        savedNorm = model.finalNorm().gamma();

        BlackBitTrainingState state;
        state.step = 1234;
        state.tokensSeen = 987654;
        state.learningRate = 0.0123F;
        state.rngSeed = 0xDEADBEEF;
        blackbit::saveCheckpoint(file.path(), model, state);
    }

    BlackBitModel reloaded(config, 999);  // seme diverso: i pesi partono diversi
    ASSERT_NE(trits(reloaded), saved);

    const BlackBitTrainingState state = blackbit::loadCheckpoint(file.path(), reloaded);

    EXPECT_EQ(trits(reloaded), saved) << "i trit ricaricati non coincidono";
    EXPECT_EQ(reloaded.finalNorm().gamma(), savedNorm);
    EXPECT_EQ(state.step, 1234U);
    EXPECT_EQ(state.tokensSeen, 987654U);
    EXPECT_FLOAT_EQ(state.learningRate, 0.0123F);
    EXPECT_EQ(state.rngSeed, 0xDEADBEEFU);
}

TEST(BlackBitCheckpointTest, IlFileRestaImpacchettato) {
    // Il controllo che dimostra il requisito 16: un checkpoint deve
    // costare ~2 bit per peso, non 16. Se qualcuno espandesse i pesi in
    // FP16 per salvarli, questo test lo direbbe subito.
    const BlackBitConfig config = microConfig();
    TempFile file("blackbit_packed.bfbit");

    {
        BlackBitModel model(config, 7);
        blackbit::saveCheckpoint(file.path(), model, BlackBitTrainingState{});
    }

    const auto fileSize = static_cast<std::size_t>(std::filesystem::file_size(file.path()));
    const std::size_t parameters = blackbit::countParameters(config).total();
    const std::size_t fp16Size = parameters * 2;

    EXPECT_LT(fileSize, fp16Size) << "il file (" << fileSize << " byte) e' grande quanto una copia FP16 ("
                                   << fp16Size << " byte)";
}

TEST(BlackBitCheckpointTest, LAddestramentoRiprendeDaDoveEraRimasto) {
    // Requisito 14, test 10: dopo il ripristino il modello deve
    // proseguire IDENTICO, stato dell'ottimizzatore compreso.
    const BlackBitConfig config = microConfig();
    TempFile file("blackbit_resume.bfbit");

    const std::vector<int> tokens{1, 5, 9, 13, 2, 6, 10, 14};
    std::vector<int> targets(tokens.begin() + 1, tokens.end());
    targets.push_back(3);

    blackbit::LowRankOptimizerOptions options;
    options.learningRate = 0.03F;
    options.rank = 8;

    float continuousLoss = 0.0F;
    {
        BlackBitModel model(config, 11);
        blackbit::LowRankProjectedOptimizer optimizer(options);
        model.registerParameters(optimizer);

        for (int step = 0; step < 6; ++step) {
            (void)model.trainStep(tokens, targets, 1, 8, &optimizer);
            optimizer.endStep();
            if (step == 2) {
                BlackBitTrainingState state;
                state.step = 3;
                state.optimizerStep = optimizer.stepCount();
                blackbit::saveCheckpoint(file.path(), model, state, &optimizer);
            }
        }
        continuousLoss = model.trainStep(tokens, targets, 1, 8, nullptr).loss;
    }

    // Ripresa dal checkpoint: gli stessi tre passi restanti devono
    // portare esattamente alla stessa loss.
    BlackBitModel resumed(config, 999);
    blackbit::LowRankProjectedOptimizer resumedOptimizer(options);
    resumed.registerParameters(resumedOptimizer);
    const BlackBitTrainingState state = blackbit::loadCheckpoint(file.path(), resumed, &resumedOptimizer);
    EXPECT_EQ(state.step, 3U);

    for (int step = 3; step < 6; ++step) {
        (void)resumed.trainStep(tokens, targets, 1, 8, &resumedOptimizer);
        resumedOptimizer.endStep();
    }
    const float resumedLoss = resumed.trainStep(tokens, targets, 1, 8, nullptr).loss;

    EXPECT_FLOAT_EQ(resumedLoss, continuousLoss)
        << "la ripresa da checkpoint non riproduce l'addestramento continuo";
}

TEST(BlackBitCheckpointTest, RifiutaFileNonRiconoscibili) {
    const BlackBitConfig config = microConfig();
    BlackBitModel model(config, 3);

    TempFile garbage("blackbit_garbage.bfbit");
    {
        std::ofstream out(garbage.path(), std::ios::binary);
        out << "questo non e' un checkpoint";
    }
    EXPECT_THROW(blackbit::loadCheckpoint(garbage.path(), model), std::runtime_error);
    EXPECT_THROW(blackbit::loadCheckpoint("/percorso/che/non/esiste.bfbit", model), std::runtime_error);

    // Versione futura: va rifiutata dicendo quale versione ha, non
    // interpretata male.
    TempFile future("blackbit_future.bfbit");
    {
        std::ofstream out(future.path(), std::ios::binary);
        const char magic[8] = {'B', 'F', 'B', 'I', 'T', '\0', '\0', '\0'};
        out.write(magic, sizeof(magic));
        const std::uint32_t version = blackbit::kBlackBitCheckpointVersion + 1;
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    }
    EXPECT_THROW(blackbit::loadCheckpoint(future.path(), model), std::runtime_error);
}

TEST(BlackBitCheckpointTest, RifiutaUnaConfigurazioneDiversa) {
    TempFile file("blackbit_mismatch.bfbit");
    {
        BlackBitModel model(microConfig(), 5);
        blackbit::saveCheckpoint(file.path(), model, BlackBitTrainingState{});
    }

    BlackBitConfig different = microConfig();
    different.numLayers = 3;
    BlackBitModel other(different, 5);
    EXPECT_THROW(blackbit::loadCheckpoint(file.path(), other), std::runtime_error);

    // Ma l'intestazione resta leggibile senza costruire nulla: e' cosi'
    // che si sa quale modello costruire.
    const BlackBitConfig stored = blackbit::readCheckpointConfig(file.path());
    EXPECT_EQ(stored.numLayers, microConfig().numLayers);
    EXPECT_EQ(stored.vocabSize, microConfig().vocabSize);
}

TEST(BlackBitCheckpointTest, ChiedereLoStatoDellOptimizerDaUnFileCheNonLoHaEUnErrore) {
    const BlackBitConfig config = microConfig();
    TempFile file("blackbit_no_optimizer.bfbit");
    {
        BlackBitModel model(config, 5);
        blackbit::saveCheckpoint(file.path(), model, BlackBitTrainingState{});
    }

    BlackBitModel model(config, 5);
    blackbit::LowRankProjectedOptimizer optimizer;
    model.registerParameters(optimizer);
    EXPECT_THROW(blackbit::loadCheckpoint(file.path(), model, &optimizer), std::runtime_error);

    // Senza ottimizzatore lo stesso file si carica senza problemi.
    EXPECT_NO_THROW((void)blackbit::loadCheckpoint(file.path(), model));
}
