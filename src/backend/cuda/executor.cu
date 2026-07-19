#include "blackforge/backend/cuda/executor.hpp"

#include <stdexcept>

#include "blackforge/backend/cpu/random_init.hpp"
#include "blackforge/backend/cuda/device_tensor.hpp"
#include "blackforge/backend/cuda/ops.hpp"

namespace blackforge::backend::cuda {

namespace {

constexpr unsigned int kEmbeddingSalt = 0x85EBCA6BU;
constexpr unsigned int kPositionalSalt = 0xC2B2AE35U;
constexpr unsigned int kAttnWqSalt = 0x27D4EB2DU;
constexpr unsigned int kAttnWkSalt = 0x165667B1U;
constexpr unsigned int kAttnWvSalt = 0xD3A2646CU;
constexpr unsigned int kAttnWoutSalt = 0xFD7046C5U;
constexpr unsigned int kFFW1Salt = 0xB55A4F09U;
constexpr unsigned int kFFB1Salt = 0x5BD1E995U;
constexpr unsigned int kFFW2Salt = 0x9E3779B9U;
constexpr unsigned int kFFB2Salt = 0x1B873593U;

}  // namespace

runtime::Tensor Executor::makeSyntheticInput(const ir::ModelIR& model, std::size_t batchSize) const {
    const ir::Value& inputValue = model.valueById(model.inputValue);
    std::vector<std::size_t> shape;
    shape.reserve(inputValue.shape.size());
    for (const auto& dim : inputValue.shape) {
        shape.push_back(dim.isSymbolic ? batchSize : static_cast<std::size_t>(dim.literalValue));
    }

    if (!model.pipelines.empty() && !model.pipelines.front().operations.empty() &&
        model.pipelines.front().operations.front().kind == ir::OpKind::Embedding) {
        auto vocabSize = static_cast<std::size_t>(model.pipelines.front().operations.front().embeddingVocabSize);
        return blackforge::backend::cpu::randomTokenIdTensor(std::move(shape), vocabSize,
                                                               blackforge::backend::cpu::seedFor(seed_, 0, 0x9E3779B1U));
    }
    return blackforge::backend::cpu::randomTensor(std::move(shape),
                                                    blackforge::backend::cpu::seedFor(seed_, 0, 0x9E3779B1U));
}

runtime::Tensor Executor::run(const ir::ModelIR& model, const runtime::Tensor& input) const {
    if (model.pipelines.empty()) {
        throw std::invalid_argument("il modello '" + model.name + "' non ha pipeline da eseguire");
    }

    const ir::Pipeline& pipeline = model.pipelines.front();
    DeviceTensor current = DeviceTensor::fromHost(input);

    for (const auto& op : pipeline.operations) {
        switch (op.kind) {
            case ir::OpKind::Linear: {
                if (current.rank() < 2) {
                    throw std::invalid_argument("linear richiede un input a rango >= 2 [..., features]");
                }
                std::size_t inFeatures = current.shape().back();
                auto outFeatures = static_cast<std::size_t>(op.linearOutFeatures);

                // Stessi pesi (stesso seme) dell'Executor CPU: permette
                // di confrontare direttamente i due backend nei test.
                runtime::Tensor hostWeight = blackforge::backend::cpu::randomTensor(
                    {inFeatures, outFeatures}, blackforge::backend::cpu::seedFor(seed_, op.output, 0x2545F491U));
                runtime::Tensor hostBias = blackforge::backend::cpu::randomTensor(
                    {outFeatures}, blackforge::backend::cpu::seedFor(seed_, op.output, 0x27D4EB2FU));

                DeviceTensor weight = DeviceTensor::fromHost(hostWeight);
                DeviceTensor bias = DeviceTensor::fromHost(hostBias);

                current = linear(current, weight, bias);
                break;
            }
            case ir::OpKind::Silu: current = silu(current); break;
            case ir::OpKind::Relu: current = relu(current); break;
            case ir::OpKind::Gelu: current = gelu(current); break;
            case ir::OpKind::RmsNorm: current = rmsnorm(current); break;
            case ir::OpKind::Softmax: current = softmax(current); break;
            case ir::OpKind::Embedding: {
                auto vocabSize = static_cast<std::size_t>(op.embeddingVocabSize);
                auto dim = static_cast<std::size_t>(op.embeddingDim);
                DeviceTensor table = DeviceTensor::fromHost(blackforge::backend::cpu::randomTensor(
                    {vocabSize, dim}, blackforge::backend::cpu::seedFor(seed_, op.output, kEmbeddingSalt)));
                current = embeddingLookup(current, table);
                break;
            }
            case ir::OpKind::PositionalEmbedding: {
                std::size_t dim = current.shape().back();
                auto maxSeqLen = static_cast<std::size_t>(op.positionalMaxSeqLen);
                DeviceTensor table = DeviceTensor::fromHost(blackforge::backend::cpu::randomTensor(
                    {maxSeqLen, dim}, blackforge::backend::cpu::seedFor(seed_, op.output, kPositionalSalt)));
                current = addPositionalEmbedding(current, table);
                break;
            }
            case ir::OpKind::Attention: {
                std::size_t dim = current.shape().back();
                auto numHeads = static_cast<std::size_t>(op.attentionNumHeads);
                DeviceTensor wq = DeviceTensor::fromHost(blackforge::backend::cpu::randomTensor(
                    {dim, dim}, blackforge::backend::cpu::seedFor(seed_, op.output, kAttnWqSalt)));
                DeviceTensor wk = DeviceTensor::fromHost(blackforge::backend::cpu::randomTensor(
                    {dim, dim}, blackforge::backend::cpu::seedFor(seed_, op.output, kAttnWkSalt)));
                DeviceTensor wv = DeviceTensor::fromHost(blackforge::backend::cpu::randomTensor(
                    {dim, dim}, blackforge::backend::cpu::seedFor(seed_, op.output, kAttnWvSalt)));
                DeviceTensor wout = DeviceTensor::fromHost(blackforge::backend::cpu::randomTensor(
                    {dim, dim}, blackforge::backend::cpu::seedFor(seed_, op.output, kAttnWoutSalt)));
                current = selfAttention(current, wq, wk, wv, wout, numHeads);
                break;
            }
            case ir::OpKind::FeedForward: {
                std::size_t dim = current.shape().back();
                auto hiddenDim = static_cast<std::size_t>(op.feedForwardHiddenDim);
                DeviceTensor w1 = DeviceTensor::fromHost(blackforge::backend::cpu::randomTensor(
                    {dim, hiddenDim}, blackforge::backend::cpu::seedFor(seed_, op.output, kFFW1Salt)));
                DeviceTensor b1 = DeviceTensor::fromHost(blackforge::backend::cpu::randomTensor(
                    {hiddenDim}, blackforge::backend::cpu::seedFor(seed_, op.output, kFFB1Salt)));
                DeviceTensor w2 = DeviceTensor::fromHost(blackforge::backend::cpu::randomTensor(
                    {hiddenDim, dim}, blackforge::backend::cpu::seedFor(seed_, op.output, kFFW2Salt)));
                DeviceTensor b2 = DeviceTensor::fromHost(blackforge::backend::cpu::randomTensor(
                    {dim}, blackforge::backend::cpu::seedFor(seed_, op.output, kFFB2Salt)));
                current = feedForward(current, w1, b1, w2, b2);
                break;
            }
        }
    }

    return current.toHost();
}

}  // namespace blackforge::backend::cuda
