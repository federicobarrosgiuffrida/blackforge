#include "blackforge/backend/cpu/executor.hpp"

#include <stdexcept>

#include "blackforge/backend/cpu/ops.hpp"
#include "blackforge/backend/cpu/quantize.hpp"
#include "blackforge/backend/cpu/random_init.hpp"

namespace blackforge::backend::cpu {

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
        return randomTokenIdTensor(std::move(shape), vocabSize, seedFor(seed_, 0, 0x9E3779B1U));
    }
    return randomTensor(std::move(shape), seedFor(seed_, 0, 0x9E3779B1U));
}

runtime::Tensor Executor::run(const ir::ModelIR& model, const runtime::Tensor& input,
                               const std::optional<ir::PrecisionPolicy>& precision) const {
    if (model.pipelines.empty()) {
        throw std::invalid_argument("il modello '" + model.name + "' non ha pipeline da eseguire");
    }

    const ir::Pipeline& pipeline = model.pipelines.front();
    runtime::Tensor current = input;
    if (precision.has_value()) {
        current = quantize(current, precision->storage);
    }

    for (const auto& op : pipeline.operations) {
        switch (op.kind) {
            case ir::OpKind::Linear: {
                if (current.rank() < 2) {
                    throw std::invalid_argument("linear richiede un input a rango >= 2 [..., features], trovato " +
                                                 current.shapeToString());
                }
                std::size_t inFeatures = current.shape().back();
                auto outFeatures = static_cast<std::size_t>(op.linearOutFeatures);

                runtime::Tensor weight =
                    randomTensor({inFeatures, outFeatures}, seedFor(seed_, op.output, 0x2545F491U));
                runtime::Tensor bias = randomTensor({outFeatures}, seedFor(seed_, op.output, 0x27D4EB2FU));

                if (precision.has_value()) {
                    // Simula il calcolo alla precisione dichiarata:
                    // input e pesi vengono arrotondati al formato
                    // 'compute' prima del prodotto matriciale (il bias
                    // viene sommato dopo, non partecipa al prodotto).
                    current = linear(quantize(current, precision->compute), quantize(weight, precision->compute),
                                      bias);
                } else {
                    current = linear(current, weight, bias);
                }
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
                runtime::Tensor table =
                    randomTensor({vocabSize, dim}, seedFor(seed_, op.output, kEmbeddingSalt));
                current = embeddingLookup(current, table);
                break;
            }
            case ir::OpKind::PositionalEmbedding: {
                std::size_t dim = current.shape().back();
                auto maxSeqLen = static_cast<std::size_t>(op.positionalMaxSeqLen);
                runtime::Tensor table =
                    randomTensor({maxSeqLen, dim}, seedFor(seed_, op.output, kPositionalSalt));
                current = addPositionalEmbedding(current, table);
                break;
            }
            case ir::OpKind::Attention: {
                std::size_t dim = current.shape().back();
                auto numHeads = static_cast<std::size_t>(op.attentionNumHeads);
                runtime::Tensor wq = randomTensor({dim, dim}, seedFor(seed_, op.output, kAttnWqSalt));
                runtime::Tensor wk = randomTensor({dim, dim}, seedFor(seed_, op.output, kAttnWkSalt));
                runtime::Tensor wv = randomTensor({dim, dim}, seedFor(seed_, op.output, kAttnWvSalt));
                runtime::Tensor wout = randomTensor({dim, dim}, seedFor(seed_, op.output, kAttnWoutSalt));
                current = selfAttention(current, wq, wk, wv, wout, numHeads);
                break;
            }
            case ir::OpKind::BidirectionalAttention: {
                // Stessa allocazione di pesi casuali di Attention (stessa
                // forma dei parametri): differisce solo la funzione di
                // forward chiamata (nessuna maschera causale).
                std::size_t dim = current.shape().back();
                auto numHeads = static_cast<std::size_t>(op.attentionNumHeads);
                runtime::Tensor wq = randomTensor({dim, dim}, seedFor(seed_, op.output, kAttnWqSalt));
                runtime::Tensor wk = randomTensor({dim, dim}, seedFor(seed_, op.output, kAttnWkSalt));
                runtime::Tensor wv = randomTensor({dim, dim}, seedFor(seed_, op.output, kAttnWvSalt));
                runtime::Tensor wout = randomTensor({dim, dim}, seedFor(seed_, op.output, kAttnWoutSalt));
                current = bidirectionalSelfAttention(current, wq, wk, wv, wout, numHeads);
                break;
            }
            case ir::OpKind::FeedForward: {
                std::size_t dim = current.shape().back();
                auto hiddenDim = static_cast<std::size_t>(op.feedForwardHiddenDim);
                runtime::Tensor w1 = randomTensor({dim, hiddenDim}, seedFor(seed_, op.output, kFFW1Salt));
                runtime::Tensor b1 = randomTensor({hiddenDim}, seedFor(seed_, op.output, kFFB1Salt));
                runtime::Tensor w2 = randomTensor({hiddenDim, dim}, seedFor(seed_, op.output, kFFW2Salt));
                runtime::Tensor b2 = randomTensor({dim}, seedFor(seed_, op.output, kFFB2Salt));
                current = feedForward(current, w1, b1, w2, b2);
                break;
            }
        }

        if (precision.has_value()) {
            current = quantize(current, precision->storage);
        }
    }

    return current;
}

}  // namespace blackforge::backend::cpu
