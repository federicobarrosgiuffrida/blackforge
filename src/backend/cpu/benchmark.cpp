#include "blackforge/backend/cpu/benchmark.hpp"

#include <chrono>
#include <stdexcept>

#include "blackforge/backend/cpu/executor.hpp"

namespace blackforge::backend::cpu {

namespace {

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
                              std::size_t measuredIterations) {
    if (model.pipelines.empty()) {
        throw std::invalid_argument("runBenchmark: il modello '" + model.name + "' non ha pipeline da eseguire");
    }
    if (measuredIterations == 0) {
        throw std::invalid_argument("runBenchmark: measuredIterations deve essere positivo");
    }

    Executor executor;
    runtime::Tensor input = executor.makeSyntheticInput(model.valueById(model.inputValue), batchSize);

    for (std::size_t i = 0; i < warmupIterations; ++i) {
        runtime::Tensor output = executor.run(model, input);
        (void)output;
    }

    auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < measuredIterations; ++i) {
        runtime::Tensor output = executor.run(model, input);
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

    return result;
}

}  // namespace blackforge::backend::cpu
