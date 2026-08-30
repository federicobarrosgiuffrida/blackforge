#include "blackforge/blackbit/benchmark.hpp"

#include <gtest/gtest.h>

#include "blackforge/blackbit/telemetry.hpp"

using namespace blackforge;
using blackforge::blackbit::BenchmarkOptions;
using blackforge::blackbit::BlackBitConfig;

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

BenchmarkOptions microOptions() {
    BenchmarkOptions options;
    options.seqLen = 8;
    options.microBatch = 1;
    options.steps = 2;
    options.warmupSteps = 1;
    options.optimizer.rank = 8;
    return options;
}

}  // namespace

TEST(BlackBitBenchmarkTest, UnEsecuzioneRealeRiportaNumeriMisurati) {
    const blackbit::BenchmarkResult result = blackbit::runBlackBitBenchmark(microConfig(), microOptions());

    EXPECT_TRUE(result.measured);
    EXPECT_GT(result.packedParameterBytes, 0U);
    EXPECT_GT(result.optimizerBytes, 0U);
    EXPECT_GT(result.peakGradientBytes, 0U);
    EXPECT_GT(result.tokensPerSecond, 0.0);
    EXPECT_GT(result.forwardBackwardMs, 0.0);
    EXPECT_GT(result.totalParameters, result.activeParameters);

    // Le due righe che il progetto deve poter stampare, dedotte dai
    // contatori e non dichiarate.
    EXPECT_FALSE(result.fullPrecisionMasterCopy);
    EXPECT_FALSE(result.fullModelGradientBuffer);
    EXPECT_TRUE(result.withinBudget);

    const std::string report = result.report();
    EXPECT_NE(report.find("FULL PRECISION MASTER COPY: NO"), std::string::npos);
    EXPECT_NE(report.find("FULL MODEL GRADIENT BUFFER: NO"), std::string::npos);
    EXPECT_NE(report.find("MISURATO"), std::string::npos);
}

TEST(BlackBitBenchmarkTest, IlPiccoDiGradienteEUnaFrazioneDiQuantoProdotto) {
    const blackbit::BenchmarkResult result = blackbit::runBlackBitBenchmark(microConfig(), microOptions());

    // Se lo stesso spazio non fosse riusato, i due numeri
    // coinciderebbero.
    EXPECT_GT(result.cumulativeGradientBytes, result.peakGradientBytes * 4);
    EXPECT_LT(result.peakGradientBytes, result.packedParameterBytes);
}

TEST(BlackBitBenchmarkTest, LOptimizerCostaMoltoMenoDiAdamWCompleto) {
    const blackbit::BenchmarkResult result = blackbit::runBlackBitBenchmark(microConfig(), microOptions());
    EXPECT_LT(result.optimizerBytes, result.conventionalOptimizerBytes);
}

TEST(BlackBitBenchmarkTest, IlDryRunNonAllocaNulla) {
    // Deve poter dimensionare BlackBit-9B su una macchina che non
    // potrebbe mai istanziarlo.
    BenchmarkOptions options;
    options.dryRun = true;
    options.seqLen = 512;
    options.microBatch = 1;
    options.maxVramMb = 0;

    const blackbit::BenchmarkResult result =
        blackbit::runBlackBitBenchmark(blackbit::blackBit9bA3b(), options);

    EXPECT_FALSE(result.measured);
    EXPECT_EQ(result.packedParameterBytes, 0U);
    EXPECT_GT(result.estimate.total(), 0U);
    EXPECT_GT(result.totalParameters, 8'500'000'000ULL);
    EXPECT_LT(result.totalParameters, 9'500'000'000ULL);
    EXPECT_GT(result.activeParameters, 2'500'000'000ULL);
    EXPECT_LT(result.activeParameters, 3'500'000'000ULL);

    const std::string report = result.report();
    EXPECT_NE(report.find("PREVISTO"), std::string::npos);
    EXPECT_EQ(report.find("MISURATO"), std::string::npos);
}

TEST(BlackBitBenchmarkTest, LaConfigurazioneDa9bStaNelBudgetDaOttoGigabyte) {
    BenchmarkOptions options;
    options.dryRun = true;
    options.seqLen = 512;
    options.microBatch = 1;
    options.maxVramMb = 7800;

    const blackbit::BenchmarkResult result =
        blackbit::runBlackBitBenchmark(blackbit::blackBit9bA3b(), options);

    EXPECT_TRUE(result.withinBudget);
    EXPECT_LT(result.estimate.total(), 7800ULL * 1024ULL * 1024ULL);
    // E l'approccio ordinario non ci starebbe di un ordine di grandezza.
    EXPECT_GT(result.estimate.conventionalBytes, 15ULL * result.estimate.total());
}

TEST(BlackBitBenchmarkTest, UnaConfigurazioneFuoriBudgetFallisceConUnaDiagnostica) {
    // BlackBit-Tiny ha 7,7 MiB di soli parametri impacchettati: con un
    // budget di 1 MiB l'esecuzione deve fermarsi con una diagnostica
    // durante la costruzione, non arrivare al primo passo.
    BenchmarkOptions options;
    options.seqLen = 8;
    options.microBatch = 1;
    options.steps = 1;
    options.maxVramMb = 1;

    try {
        (void)blackbit::runBlackBitBenchmark(blackbit::blackBitTiny(), options);
        FAIL() << "avrebbe dovuto rifiutare la configurazione prima di allocare";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("budget"), std::string::npos);
        EXPECT_NE(message.find("--seq-len"), std::string::npos)
            << "il messaggio deve dire cosa ridurre, non solo che e' fallito";
    }

    // Il budget non deve restare armato dopo un fallimento: i test
    // successivi (e qualunque altro uso nello stesso processo)
    // allocherebbero contro un limite che nessuno ha piu' chiesto.
    EXPECT_EQ(blackbit::MemoryBudget::instance().limitBytes(), 0U);
}

TEST(BlackBitBenchmarkTest, ILimitiDiFormaSonoControllatiPrimaDiAllocare) {
    BenchmarkOptions options = microOptions();
    options.seqLen = microConfig().maxSeqLen + 1;
    EXPECT_THROW((void)blackbit::runBlackBitBenchmark(microConfig(), options), std::invalid_argument);

    BenchmarkOptions noSteps = microOptions();
    noSteps.steps = 0;
    EXPECT_THROW((void)blackbit::runBlackBitBenchmark(microConfig(), noSteps), std::invalid_argument);
}

TEST(BlackBitBenchmarkTest, IlRicalcoloDelleAttivazioniRiduceIlPiccoSenzaCambiareLaLoss) {
    // Requisito 10: le modalita' di ricalcolo sono un baratto
    // memoria/tempo, non una diversa matematica.
    BenchmarkOptions withoutRecompute = microOptions();
    withoutRecompute.runtime.recompute = blackbit::ActivationRecompute::None;

    BenchmarkOptions withRecompute = microOptions();
    withRecompute.runtime.recompute = blackbit::ActivationRecompute::PerLayer;

    BenchmarkOptions fullRecompute = microOptions();
    fullRecompute.runtime.recompute = blackbit::ActivationRecompute::FullRecompute;

    const auto a = blackbit::runBlackBitBenchmark(microConfig(), withoutRecompute);
    const auto b = blackbit::runBlackBitBenchmark(microConfig(), withRecompute);
    const auto c = blackbit::runBlackBitBenchmark(microConfig(), fullRecompute);

    // La loss deve coincidere: stesso seme, stessa matematica.
    EXPECT_FLOAT_EQ(a.finalLoss, b.finalLoss);
    EXPECT_FLOAT_EQ(a.finalLoss, c.finalLoss);

    // E il picco di attivazioni non deve crescere ricalcolando.
    EXPECT_LE(b.peakActivationBytes, a.peakActivationBytes);
    EXPECT_LE(c.peakActivationBytes, a.peakActivationBytes);
}
