#include "blackforge/blackbit/benchmark.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

#include "blackforge/blackbit/telemetry.hpp"

namespace blackforge::blackbit {

namespace {

std::string mib(std::size_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MiB";
    return out.str();
}

std::string gib(std::size_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0))
        << " GiB";
    return out.str();
}

std::string row(const std::string& label, const std::string& value, const char* origin) {
    std::ostringstream out;
    out << "  " << std::left << std::setw(34) << label << std::right << std::setw(14) << value << "  " << origin
        << "\n";
    return out.str();
}

}  // namespace

std::string BenchmarkResult::report() const {
    std::ostringstream out;
    out << "BlackBit benchmark — " << config.name << "\n";
    out << "  seq " << options.seqLen << ", micro-batch " << options.microBatch << ", passi " << options.steps
        << " (piu' " << options.warmupSteps << " di riscaldamento), ricalcolo attivazioni "
        << activationRecomputeName(options.runtime.recompute) << ", rango optimizer " << options.optimizer.rank
        << "\n\n";

    out << "Parametri\n";
    out << row("Totali", std::to_string(totalParameters), "");
    out << row("Attivi per token", std::to_string(activeParameters), "");
    {
        std::ostringstream fraction;
        fraction << std::fixed << std::setprecision(1)
                 << (100.0 * static_cast<double>(activeParameters) / static_cast<double>(totalParameters)) << " %";
        out << row("Frazione attiva", fraction.str(), "");
    }
    out << "\n";

    out << "Memoria\n";
    if (measured) {
        out << row("Parametri impacchettati", mib(packedParameterBytes), "MISURATO");
        out << row("Stato optimizer", mib(optimizerBytes), "MISURATO");
        out << row("  (AdamW ordinario userebbe)", mib(conventionalOptimizerBytes), "PREVISTO");
        out << row("Attivazioni (picco)", mib(peakActivationBytes), "MISURATO");
        out << row("Gradienti (picco vivo)", mib(peakGradientBytes), "MISURATO");
        out << row("  (totale prodotto e riusato)", mib(cumulativeGradientBytes), "MISURATO");
        out << row("Workspace / dequant (picco)", mib(peakWorkspaceBytes), "MISURATO");
        out << row("PICCO TOTALE", gib(peakTotalBytes), "MISURATO");
        out << "\n";
        out << row("Previsione per la stessa forma", gib(estimate.total()), "PREVISTO");
    } else {
        out << "  (esecuzione non effettuata: --dry-run, nessuna allocazione)\n";
        out << row("Pesi impacchettati", gib(estimate.packedWeightBytes), "PREVISTO");
        out << row("Scale di quantizzazione", gib(estimate.scaleBytes), "PREVISTO");
        out << row("Parametri densi (norm/router)", mib(estimate.denseParameterBytes), "PREVISTO");
        out << row("Stato optimizer", gib(estimate.optimizerBytes), "PREVISTO");
        out << row("Attivazioni", gib(estimate.activationBytes), "PREVISTO");
        out << row("Gradienti (picco di un blocco)", mib(estimate.gradientPeakBytes), "PREVISTO");
        out << row("Workspace", mib(estimate.workspaceBytes), "PREVISTO");
        out << row("TOTALE", gib(estimate.total()), "PREVISTO");
    }
    out << row("Approccio ordinario, per confronto", gib(estimate.conventionalBytes), "PREVISTO");
    if (options.maxVramMb != 0) {
        out << row("Budget dichiarato", gib(options.maxVramMb * 1024ULL * 1024ULL), "");
        out << "  " << (withinBudget ? "entro il budget" : "OLTRE IL BUDGET") << "\n";
    }
    out << "\n";

    if (measured) {
        out << "Tempi\n";
        std::ostringstream forwardBackward;
        forwardBackward << std::fixed << std::setprecision(2) << forwardBackwardMs << " ms";
        std::ostringstream optimizerTime;
        optimizerTime << std::fixed << std::setprecision(2) << optimizerMs << " ms";
        std::ostringstream throughput;
        throughput << std::fixed << std::setprecision(1) << tokensPerSecond;
        out << row("forward + backward per passo", forwardBackward.str(), "MISURATO");
        out << row("optimizer per passo", optimizerTime.str(), "MISURATO");
        out << row("token/s", throughput.str(), "MISURATO");
        out << "\n";

        out << "Addestramento\n";
        std::ostringstream loss;
        loss << std::fixed << std::setprecision(4) << finalLoss;
        std::ostringstream entropy;
        entropy << std::fixed << std::setprecision(3) << routingEntropy;
        std::ostringstream utilization;
        utilization << std::fixed << std::setprecision(1) << (maxExpertUtilization * 100.0) << " %";
        out << row("loss finale", loss.str(), "MISURATO");
        out << row("entropia di routing (nat)", entropy.str(), "MISURATO");
        out << row("utilizzo massimo di un esperto", utilization.str(), "MISURATO");
        out << row("assegnazioni scartate", std::to_string(droppedAssignments), "MISURATO");
        out << row("trit modificati", std::to_string(ternaryFlips), "MISURATO");
        out << "\n";
    }

    out << "Verifiche di progetto\n";
    out << "  FULL PRECISION MASTER COPY: " << (fullPrecisionMasterCopy ? "SI" : "NO") << "\n";
    out << "  FULL MODEL GRADIENT BUFFER: " << (fullModelGradientBuffer ? "SI" : "NO") << "\n";
    if (!measured) {
        out << "  (dedotte dalla previsione: solo un'esecuzione reale le misura)\n";
    }
    return out.str();
}

BenchmarkResult runBlackBitBenchmark(const BlackBitConfig& config, const BenchmarkOptions& options,
                                      std::ostream* progress) {
    config.validate();
    if (options.steps == 0) {
        throw std::invalid_argument("benchmark BlackBit: servono almeno 1 passo misurato");
    }
    if (options.seqLen > config.maxSeqLen) {
        throw std::invalid_argument("benchmark BlackBit: seq-len " + std::to_string(options.seqLen) +
                                     " supera max_seq_len della configurazione (" +
                                     std::to_string(config.maxSeqLen) + ")");
    }

    BenchmarkResult result;
    result.config = config;
    result.options = options;

    const ParameterCount count = countParameters(config);
    result.totalParameters = count.total();
    result.activeParameters = countActiveParameters(config);

    LowMemoryOptions lowMemory;
    lowMemory.optimizerRank = options.optimizer.rank;
    lowMemory.optimizerStateBytes = sizeof(float);
    lowMemory.activationCheckpointing = options.runtime.recompute != ActivationRecompute::None;
    result.estimate = estimateTrainingMemory(config, {options.microBatch, options.seqLen}, lowMemory);

    const std::size_t budgetBytes = options.maxVramMb * 1024ULL * 1024ULL;
    result.withinBudget = budgetBytes == 0 || result.estimate.total() <= budgetBytes;

    if (options.dryRun) {
        // Dedotte dalla previsione: nessuna voce di master copy e
        // nessun gradiente a dimensione modello compaiono nel conto.
        result.fullPrecisionMasterCopy = false;
        result.fullModelGradientBuffer = false;
        return result;
    }

    if (!result.withinBudget) {
        std::ostringstream message;
        message << "benchmark BlackBit: la configurazione richiede " << gib(result.estimate.total())
                << " previsti, oltre il budget di " << gib(budgetBytes)
                << ".\nRiduci --seq-len o --micro-batch, abbassa il rango dell'optimizer, oppure alza "
                   "--max-vram-mb se la scheda ha davvero piu' memoria.";
        throw std::runtime_error(message.str());
    }

    // Il budget viene armato PRIMA di costruire il modello: una
    // configurazione fuori misura fallisce con una diagnostica, non con
    // un'allocazione andata male a meta' del secondo passo. Il disarmo
    // e' affidato a RAII: se l'esecuzione termina con un'eccezione, un
    // budget rimasto armato farebbe fallire allocazioni successive che
    // nessuno ha piu' chiesto di limitare.
    struct BudgetGuard {
        explicit BudgetGuard(std::size_t limit) { MemoryBudget::instance().setLimitBytes(limit); }
        ~BudgetGuard() { MemoryBudget::instance().setLimitBytes(0); }
        BudgetGuard(const BudgetGuard&) = delete;
        BudgetGuard& operator=(const BudgetGuard&) = delete;
    } budgetGuard(budgetBytes);

    MemoryTelemetry::instance().reset();
    resetGradientLifetimeStats();

    {
        BlackBitModel model(config, options.seed);
        model.setRuntimeOptions(options.runtime);
        result.packedParameterBytes = model.parameterBytes();

        LowRankProjectedOptimizer optimizer(options.optimizer);
        model.registerParameters(optimizer);
        result.optimizerBytes = optimizer.stateBytes();
        result.conventionalOptimizerBytes = optimizer.conventionalStateBytes();

        // Da qui in poi si misura solo l'esecuzione: i picchi
        // dell'inizializzazione non fanno parte di un passo di
        // addestramento.
        MemoryTelemetry::instance().resetPeaks();

        const std::size_t tokens = options.microBatch * options.seqLen;
        std::mt19937 rng(options.seed);
        std::vector<int> tokenIds(tokens);
        std::vector<int> targets(tokens);
        for (std::size_t i = 0; i < tokens; ++i) {
            tokenIds[i] = static_cast<int>(rng() % config.vocabSize);
            targets[i] = static_cast<int>(rng() % config.vocabSize);
        }

        for (std::size_t step = 0; step < options.warmupSteps; ++step) {
            (void)model.trainStep(tokenIds, targets, options.microBatch, options.seqLen, &optimizer);
            optimizer.endStep();
        }

        resetGradientLifetimeStats();
        MemoryTelemetry::instance().resetPeaks();

        double forwardBackwardTotal = 0.0;
        double optimizerTotal = 0.0;
        for (std::size_t step = 0; step < options.steps; ++step) {
            const auto beforeStep = std::chrono::steady_clock::now();
            const BlackBitStepResult stepResult =
                model.trainStep(tokenIds, targets, options.microBatch, options.seqLen, &optimizer);
            const auto afterBackward = std::chrono::steady_clock::now();
            optimizer.endStep();
            const auto afterOptimizer = std::chrono::steady_clock::now();

            forwardBackwardTotal += std::chrono::duration<double, std::milli>(afterBackward - beforeStep).count();
            optimizerTotal += std::chrono::duration<double, std::milli>(afterOptimizer - afterBackward).count();

            result.finalLoss = stepResult.loss;
            result.routingEntropy = stepResult.meanRoutingEntropy();
            result.maxExpertUtilization = stepResult.maxExpertUtilization();
            result.droppedAssignments = stepResult.droppedAssignments();

            if (stepResult.sawNaN || stepResult.sawInf) {
                throw std::runtime_error(
                    std::string("benchmark BlackBit: instabilita' numerica al passo ") + std::to_string(step) +
                    " nel tensore '" + stepResult.firstUnstableTensor +
                    "'. L'esecuzione si ferma qui invece di proseguire con numeri privi di significato.");
            }

            if (progress != nullptr) {
                *progress << "  passo " << (step + 1) << "/" << options.steps << ": loss " << stepResult.loss
                          << ", picco gradiente " << mib(gradientLifetimeStats().peakLiveBytes) << "\n";
            }
        }

        result.measured = true;
        result.forwardBackwardMs = forwardBackwardTotal / static_cast<double>(options.steps);
        result.optimizerMs = optimizerTotal / static_cast<double>(options.steps);
        const double totalSeconds = (forwardBackwardTotal + optimizerTotal) / 1000.0;
        result.tokensPerSecond =
            totalSeconds > 0.0 ? static_cast<double>(tokens * options.steps) / totalSeconds : 0.0;
        result.ternaryFlips = optimizer.stats().ternaryFlips;

        const MemoryTelemetry& telemetry = MemoryTelemetry::instance();
        result.peakActivationBytes = telemetry.peak(MemoryArena::Activation);
        result.peakGradientBytes = telemetry.peak(MemoryArena::Gradient);
        result.peakWorkspaceBytes = telemetry.peak(MemoryArena::Workspace);
        result.peakTotalBytes = telemetry.peakTotal();
        result.cumulativeGradientBytes = gradientLifetimeStats().cumulativeBytes;

        // --- le due verifiche, dedotte dai contatori ---
        //
        // Una copia in piena precisione dei pesi ternari occuperebbe
        // almeno 2 byte per parametro: se l'arena dei parametri non e'
        // grande abbastanza per contenerla, non esiste.
        const std::size_t masterCopyBytes = count.ternary() * 2;
        result.fullPrecisionMasterCopy =
            telemetry.peak(MemoryArena::Parameter) >= result.packedParameterBytes + masterCopyBytes;

        // Un buffer di gradienti a dimensione modello significherebbe
        // un picco di gradiente vivo dell'ordine di 4 byte per
        // parametro. Il confronto e' con meta' di quel valore per
        // lasciare margine, e comunque il picco reale e' ordini di
        // grandezza sotto.
        const std::size_t denseGradientBytes = count.total() * sizeof(float);
        result.fullModelGradientBuffer = result.peakGradientBytes >= denseGradientBytes / 2;

        result.withinBudget = budgetBytes == 0 || result.peakTotalBytes <= budgetBytes;
    }

    return result;
}

}  // namespace blackforge::blackbit
