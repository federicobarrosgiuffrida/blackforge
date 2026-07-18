#include "blackforge/backend/cpu/benchmark.hpp"

#include <chrono>
#include <stdexcept>

#include "blackforge/backend/cpu/executor.hpp"
#include "blackforge/backend/cpu/ops.hpp"
#include "blackforge/backend/cpu/quantize.hpp"
#include "blackforge/backend/cpu/random_init.hpp"

namespace blackforge::backend::cpu {

namespace {

constexpr unsigned int kWeightSalt = 0x2545F491U;
constexpr unsigned int kBiasSalt = 0x27D4EB2FU;
// Stesso seme di default di Executor: runBenchmark() non ne accetta uno
// personalizzato, quindi il breakdown per operazione usa lo stesso
// valore per restare coerente con i pesi usati nel timing aggregato.
constexpr unsigned int kDefaultSeed = 42U;

// Applica una singola operazione della pipeline al suo input reale
// (catturato durante una singola esecuzione completa, vedi
// runBenchmark), rigenerando pesi/bias deterministici identici a
// quelli di Executor::run() per un layer 'linear'. Non e' pensata per
// essere corretta in generale (non gestisce LoRA, non esiste ancora su
// CUDA): serve solo al breakdown per operazione del benchmark.
runtime::Tensor applyOperationForTiming(const ir::Operation& op, const runtime::Tensor& opInput,
                                         const std::optional<ir::PrecisionPolicy>& precision) {
    switch (op.kind) {
        case ir::OpKind::Linear: {
            std::size_t inFeatures = opInput.dim(1);
            auto outFeatures = static_cast<std::size_t>(op.linearOutFeatures);
            runtime::Tensor weight =
                randomTensor({inFeatures, outFeatures}, seedFor(kDefaultSeed, op.output, kWeightSalt));
            runtime::Tensor bias = randomTensor({outFeatures}, seedFor(kDefaultSeed, op.output, kBiasSalt));
            if (precision.has_value()) {
                return linear(quantize(opInput, precision->compute), quantize(weight, precision->compute), bias);
            }
            return linear(opInput, weight, bias);
        }
        case ir::OpKind::Silu: return silu(opInput);
        case ir::OpKind::Relu: return relu(opInput);
        case ir::OpKind::Gelu: return gelu(opInput);
        case ir::OpKind::RmsNorm: return rmsnorm(opInput);
        case ir::OpKind::Softmax: return softmax(opInput);
    }
    return opInput;
}

// Esegue l'intera pipeline una volta (come Executor::run(), qui
// reimplementato per catturare l'input REALE che ogni operazione
// riceve, non solo l'uscita finale) e poi misura ogni operazione
// separatamente su quell'input, con lo stesso schema warmup/misurate
// del timing aggregato.
std::vector<OperationBenchmark> measurePerOperation(const ir::ModelIR& model, const runtime::Tensor& input,
                                                     std::size_t warmupIterations, std::size_t measuredIterations,
                                                     const std::optional<ir::PrecisionPolicy>& precision) {
    const ir::Pipeline& pipeline = model.pipelines.front();
    std::vector<OperationBenchmark> results;
    results.reserve(pipeline.operations.size());

    runtime::Tensor current = input;
    if (precision.has_value()) {
        current = quantize(current, precision->storage);
    }

    std::size_t index = 0;
    for (const auto& op : pipeline.operations) {
        runtime::Tensor opInput = current;

        for (std::size_t i = 0; i < warmupIterations; ++i) {
            runtime::Tensor warm = applyOperationForTiming(op, opInput, precision);
            (void)warm;
        }

        auto start = std::chrono::steady_clock::now();
        runtime::Tensor opOutput;
        for (std::size_t i = 0; i < measuredIterations; ++i) {
            opOutput = applyOperationForTiming(op, opInput, precision);
        }
        auto end = std::chrono::steady_clock::now();

        double totalMillis = std::chrono::duration<double, std::milli>(end - start).count();
        results.push_back(
            OperationBenchmark{ir::opKindName(op.kind), index, totalMillis / static_cast<double>(measuredIterations)});

        current = precision.has_value() ? quantize(opOutput, precision->storage) : opOutput;
        ++index;
    }

    return results;
}

std::size_t shapeElementCount(const std::vector<ir::Dim>& shape, std::size_t batchSize) {
    std::size_t count = 1;
    for (const auto& dim : shape) {
        count *= dim.isSymbolic ? batchSize : static_cast<std::size_t>(dim.literalValue);
    }
    return count;
}

// Somma gli elementi di input, di ogni attivazione intermedia e dei
// parametri (peso+bias di ogni 'linear') lungo la prima pipeline.
std::size_t estimateElementCount(const ir::ModelIR& model, std::size_t batchSize) {
    std::size_t totalElements = shapeElementCount(model.valueById(model.inputValue).shape, batchSize);

    for (const auto& pipeline : model.pipelines) {
        for (const auto& op : pipeline.operations) {
            totalElements += shapeElementCount(model.valueById(op.output).shape, batchSize);

            if (op.kind == ir::OpKind::Linear) {
                const ir::Value& inputValue = model.valueById(op.input);
                auto inFeatures = static_cast<std::size_t>(inputValue.shape.back().literalValue);
                auto outFeatures = static_cast<std::size_t>(op.linearOutFeatures);
                totalElements += inFeatures * outFeatures + outFeatures;  // peso + bias
            }
        }
    }

    return totalElements;
}

}  // namespace

BenchmarkResult runBenchmark(const ir::ModelIR& model, std::size_t batchSize, std::size_t warmupIterations,
                              std::size_t measuredIterations, const std::optional<ir::PrecisionPolicy>& precision) {
    if (model.pipelines.empty()) {
        throw std::invalid_argument("runBenchmark: il modello '" + model.name + "' non ha pipeline da eseguire");
    }
    if (measuredIterations == 0) {
        throw std::invalid_argument("runBenchmark: measuredIterations deve essere positivo");
    }

    Executor executor;
    runtime::Tensor input = executor.makeSyntheticInput(model.valueById(model.inputValue), batchSize);

    for (std::size_t i = 0; i < warmupIterations; ++i) {
        runtime::Tensor output = executor.run(model, input, precision);
        (void)output;
    }

    auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < measuredIterations; ++i) {
        runtime::Tensor output = executor.run(model, input, precision);
        (void)output;
    }
    auto end = std::chrono::steady_clock::now();

    double totalMillis = std::chrono::duration<double, std::milli>(end - start).count();
    double meanMillis = totalMillis / static_cast<double>(measuredIterations);

    BenchmarkResult result;
    result.inputShape = input.shape();
    result.warmupIterations = warmupIterations;
    result.measuredIterations = measuredIterations;
    result.meanMilliseconds = meanMillis;
    result.throughputSamplesPerSecond = meanMillis > 0.0 ? (static_cast<double>(batchSize) * 1000.0) / meanMillis
                                                           : 0.0;
    result.estimatedMemoryBytes = estimateElementCount(model, batchSize) * sizeof(float);
    result.perOperation = measurePerOperation(model, input, warmupIterations, measuredIterations, precision);

    return result;
}

}  // namespace blackforge::backend::cpu
