#pragma once

#include <cstddef>
#include <ostream>
#include <string>

#include "blackforge/blackbit/config.hpp"
#include "blackforge/blackbit/low_rank_optimizer.hpp"
#include "blackforge/blackbit/model.hpp"

// Strumento di misura di BlackBit (requisito 17).
//
// Il rapporto distingue sempre due categorie di numeri, e le etichetta:
//
//   MISURATO  — letto dalla telemetria durante l'esecuzione reale;
//   PREVISTO  — calcolato da estimateTrainingMemory() prima di allocare.
//
// Mescolarle sarebbe il modo piu' semplice per far sembrare rispettato
// un budget che non lo e'. Con --dry-run vengono riportati solo i
// numeri previsti, ed e' detto esplicitamente.
//
// Le due righe che contano davvero:
//
//   FULL PRECISION MASTER COPY: NO
//   FULL MODEL GRADIENT BUFFER: NO
//
// non sono dichiarazioni: sono dedotte dai contatori. La prima
// confronta l'arena dei parametri con quanto occuperebbe una copia
// BF16 dei pesi ternari; la seconda confronta il picco di gradiente
// vivo con il gradiente denso dell'intero modello.

namespace blackforge::blackbit {

struct BenchmarkOptions {
    std::size_t seqLen = 512;
    std::size_t microBatch = 1;
    std::size_t steps = 3;
    std::size_t warmupSteps = 1;

    // 0 = nessun limite. Superarlo interrompe con una diagnostica
    // invece di far fallire un'allocazione a meta' del secondo passo.
    std::size_t maxVramMb = 7800;

    // Se true non costruisce il modello: riporta solo la contabilita'
    // prevista. Serve a dimensionare una configurazione senza avere
    // l'hardware (o la pazienza) per istanziarla.
    bool dryRun = false;

    unsigned int seed = 42;
    LowRankOptimizerOptions optimizer;
    BlackBitRuntimeOptions runtime;
};

struct BenchmarkResult {
    BlackBitConfig config;
    BenchmarkOptions options;

    std::size_t totalParameters = 0;
    std::size_t activeParameters = 0;

    // Previsti (nessuna allocazione).
    MemoryEstimate estimate;

    // Misurati (zero se dryRun).
    bool measured = false;
    std::size_t packedParameterBytes = 0;
    std::size_t optimizerBytes = 0;
    std::size_t conventionalOptimizerBytes = 0;
    std::size_t peakActivationBytes = 0;
    std::size_t peakGradientBytes = 0;
    std::size_t peakWorkspaceBytes = 0;
    std::size_t peakTotalBytes = 0;
    std::size_t cumulativeGradientBytes = 0;

    double forwardBackwardMs = 0.0;
    double optimizerMs = 0.0;
    double tokensPerSecond = 0.0;
    float finalLoss = 0.0F;
    double routingEntropy = 0.0;
    double maxExpertUtilization = 0.0;
    std::size_t droppedAssignments = 0;
    std::size_t ternaryFlips = 0;

    bool fullPrecisionMasterCopy = true;
    bool fullModelGradientBuffer = true;
    bool withinBudget = true;

    [[nodiscard]] std::string report() const;
};

// Lancia std::runtime_error se il budget verrebbe superato (con un
// messaggio che dice di quanto e cosa ridurre) o se la configurazione
// non e' valida.
BenchmarkResult runBlackBitBenchmark(const BlackBitConfig& config, const BenchmarkOptions& options,
                                      std::ostream* progress = nullptr);

}  // namespace blackforge::blackbit
