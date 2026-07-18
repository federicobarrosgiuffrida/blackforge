#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "blackforge/ir/module.hpp"

namespace blackforge::backend::cpu {

struct BenchmarkResult {
    std::vector<std::size_t> inputShape;
    std::size_t warmupIterations;
    std::size_t measuredIterations;
    double meanMilliseconds;
    double throughputSamplesPerSecond;

    // Stima teorica (elementi * 4 byte, dato che il backend CPU
    // memorizza sempre float32) della memoria occupata da input,
    // attivazioni intermedie e parametri: non e' memoria di processo
    // misurata (RSS), che richiederebbe API specifiche del sistema
    // operativo non ancora implementate.
    std::size_t estimatedMemoryBytes;
};

// Esegue 'warmupIterations' iterazioni di riscaldamento (scartate, per
// escludere costi una tantum come le prime allocazioni) seguite da
// 'measuredIterations' iterazioni misurate della prima pipeline del
// modello, su un input sintetico di batch 'batchSize'. Lancia
// std::invalid_argument se il modello non ha pipeline o se
// 'measuredIterations' e' zero.
//
// Se 'precision' e' presente, ogni iterazione applica la quantizzazione
// simulata dichiarata dal blocco 'precision' (vedi quantize.hpp): il
// tempo misurato riflette quindi anche il costo della quantizzazione
// stessa, non solo del calcolo in float32.
BenchmarkResult runBenchmark(const ir::ModelIR& model, std::size_t batchSize, std::size_t warmupIterations,
                              std::size_t measuredIterations,
                              const std::optional<ir::PrecisionPolicy>& precision = std::nullopt);

}  // namespace blackforge::backend::cpu
