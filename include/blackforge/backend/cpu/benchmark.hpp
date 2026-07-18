#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "blackforge/ir/module.hpp"

namespace blackforge::backend::cpu {

// Tempo medio di una singola operazione della pipeline, misurato in
// isolamento (vedi il campo perOperation di BenchmarkResult).
struct OperationBenchmark {
    std::string operationName;  // es. "linear", "softmax" (vedi ir::opKindName)
    std::size_t operationIndex;  // posizione nella pipeline (0-based)
    double meanMilliseconds;
};

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

    // Tempo medio di ciascuna operazione della pipeline, misurato
    // separatamente (stesso input reale che l'operazione riceve
    // nell'esecuzione completa, stesso numero di warmup/iterazioni
    // misurate del timing aggregato). Utile per capire quale operazione
    // domina il tempo totale, non solo il totale stesso. Solo sul
    // backend CPU per ora: un breakdown per singola operazione su CUDA
    // richiederebbe timer basati su cudaEvent (i lanci di kernel sono
    // asincroni, un semplice steady_clock attorno a ogni chiamata non
    // misurerebbe il tempo reale sul device) — lavoro futuro.
    std::vector<OperationBenchmark> perOperation;
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
