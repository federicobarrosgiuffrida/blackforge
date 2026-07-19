#include "blackforge/backend/cuda/model.hpp"

#include <stdexcept>

#include "blackforge/backend/cpu/random_init.hpp"
#include "blackforge/backend/cuda/autodiff.hpp"
#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/backend/cuda/ops.hpp"

namespace blackforge::backend::cuda {

namespace {

constexpr unsigned int kWeightSalt = 0x2545F491U;
constexpr unsigned int kBiasSalt = 0x27D4EB2FU;
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
constexpr int kBlockSize = 256;

int gridSizeFor(std::size_t n) { return static_cast<int>((n + kBlockSize - 1) / kBlockSize); }

__global__ void accumulateKernel(float* target, const float* delta, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        target[i] += delta[i];
    }
}

void accumulate(DeviceTensor& target, const DeviceTensor& delta) {
    std::size_t n = target.elementCount();
    if (n > 0) {
        accumulateKernel<<<gridSizeFor(n), kBlockSize>>>(target.data(), delta.data(), n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
}

// Dimensione delle feature in ingresso a un layer che ne ha bisogno per
// allocare i propri pesi: deve essere concreta, non simbolica (stessa
// esigenza di 'linear', vedi la controparte CPU in model.cpp).
std::size_t concreteLastDim(const ir::Value& value, const std::string& opLabel, std::size_t opOutput) {
    const ir::Dim& dim = value.shape.back();
    if (dim.isSymbolic) {
        throw std::invalid_argument("cuda::Model: la dimensione delle feature in ingresso al layer " + opLabel +
                                     std::to_string(opOutput) + " e' simbolica ('" + dim.symbolicName +
                                     "'); serve una dimensione concreta per allocare i pesi");
    }
    return static_cast<std::size_t>(dim.literalValue);
}

DeviceTensor deviceRandomTensor(std::vector<std::size_t> shape, unsigned int seed) {
    return DeviceTensor::fromHost(blackforge::backend::cpu::randomTensor(std::move(shape), seed));
}

// Appiattisce un tensore device a rango >= 2 a [rows, features]: serve
// per poter chiamare matmulBackward() (primitivo puramente 2D)
// sull'input cachato di un layer 'linear' quando e' a rango > 2 (es.
// [batch, seq, features], come quando 'linear' arriva dopo 'attention'
// o 'feedforward' in una pipeline di modello linguistico). Duplicato
// deliberatamente da src/backend/cuda/autodiff.cu (stessa idea, diversa
// unita' di traduzione, vedi il commento equivalente in model.cpp).
DeviceTensor flatten2D(const DeviceTensor& t) {
    std::size_t features = t.shape().back();
    std::size_t rows = t.elementCount() / features;
    return t.clone().reshaped({rows, features});
}

}  // namespace

Model::Model(const ir::ModelIR& modelIR, unsigned int seed, std::optional<ir::PrecisionPolicy> precision)
    : name_(modelIR.name), useTensorCoreLinear_(precision.has_value() && precision->compute == sema::DType::BF16) {
    if (modelIR.pipelines.empty()) {
        throw std::invalid_argument("cuda::Model: il modello '" + modelIR.name + "' non ha pipeline da addestrare");
    }

    const ir::Pipeline& pipeline = modelIR.pipelines.front();

    for (const auto& op : pipeline.operations) {
        LayerState layer;
        layer.kind = op.kind;

        if (op.kind == ir::OpKind::BidirectionalAttention) {
            // Non ancora implementato su CUDA (vedi backend::cpu::Model
            // per bidirectional_attention/MLM): errore esplicito invece
            // di ignorare silenziosamente il layer (che produrrebbe un
            // risultato quietamente sbagliato, l'input passerebbe
            // attraverso senza alcuna trasformazione).
            throw std::invalid_argument(
                "cuda::Model: 'bidirectional_attention' non e' ancora implementato sul backend CUDA "
                "(solo su CPU). Usa '--device cpu' per un modello linguistico mascherato (MLM).");
        }

        if (op.kind == ir::OpKind::Linear) {
            const ir::Value& inputValue = modelIR.valueById(op.input);
            std::size_t inFeatures = concreteLastDim(inputValue, "linear#", op.output);
            auto outFeatures = static_cast<std::size_t>(op.linearOutFeatures);
            std::string base = "linear" + std::to_string(op.output);

            // Stesso generatore deterministico (stesso seme, stesso
            // "sale" per weight/bias) del backend CPU e dell'Executor
            // CUDA di sola inferenza: permette il confronto diretto
            // CPU/GPU nei test.
            layer.weight = Parameter{base + ".weight",
                                      deviceRandomTensor({inFeatures, outFeatures},
                                                          blackforge::backend::cpu::seedFor(seed, op.output, kWeightSalt)),
                                      DeviceTensor::zeros({inFeatures, outFeatures})};
            layer.bias = Parameter{base + ".bias",
                                    deviceRandomTensor({outFeatures},
                                                        blackforge::backend::cpu::seedFor(seed, op.output, kBiasSalt)),
                                    DeviceTensor::zeros({outFeatures})};
        } else if (op.kind == ir::OpKind::Embedding) {
            auto vocabSize = static_cast<std::size_t>(op.embeddingVocabSize);
            auto dim = static_cast<std::size_t>(op.embeddingDim);
            std::string base = "embedding" + std::to_string(op.output);

            layer.embeddingVocabSize = vocabSize;
            layer.embeddingTable = Parameter{
                base + ".table",
                deviceRandomTensor({vocabSize, dim}, blackforge::backend::cpu::seedFor(seed, op.output, kEmbeddingSalt)),
                DeviceTensor::zeros({vocabSize, dim})};
        } else if (op.kind == ir::OpKind::PositionalEmbedding) {
            const ir::Value& inputValue = modelIR.valueById(op.input);
            std::size_t dim = concreteLastDim(inputValue, "positional_embedding#", op.output);
            auto maxSeqLen = static_cast<std::size_t>(op.positionalMaxSeqLen);
            std::string base = "positional_embedding" + std::to_string(op.output);

            layer.positionalMaxSeqLen = maxSeqLen;
            layer.positionalTable = Parameter{
                base + ".table",
                deviceRandomTensor({maxSeqLen, dim}, blackforge::backend::cpu::seedFor(seed, op.output, kPositionalSalt)),
                DeviceTensor::zeros({maxSeqLen, dim})};
        } else if (op.kind == ir::OpKind::Attention) {
            const ir::Value& inputValue = modelIR.valueById(op.input);
            std::size_t dim = concreteLastDim(inputValue, "attention#", op.output);
            auto numHeads = static_cast<std::size_t>(op.attentionNumHeads);
            if (numHeads == 0 || dim % numHeads != 0) {
                throw std::invalid_argument("cuda::Model: 'attention#" + std::to_string(op.output) +
                                             "' richiede che " + std::to_string(numHeads) +
                                             " teste dividano esattamente dim=" + std::to_string(dim));
            }
            std::string base = "attention" + std::to_string(op.output);

            layer.attentionNumHeads = numHeads;
            layer.attnWq = Parameter{
                base + ".wq", deviceRandomTensor({dim, dim}, blackforge::backend::cpu::seedFor(seed, op.output, kAttnWqSalt)),
                DeviceTensor::zeros({dim, dim})};
            layer.attnWk = Parameter{
                base + ".wk", deviceRandomTensor({dim, dim}, blackforge::backend::cpu::seedFor(seed, op.output, kAttnWkSalt)),
                DeviceTensor::zeros({dim, dim})};
            layer.attnWv = Parameter{
                base + ".wv", deviceRandomTensor({dim, dim}, blackforge::backend::cpu::seedFor(seed, op.output, kAttnWvSalt)),
                DeviceTensor::zeros({dim, dim})};
            layer.attnWout = Parameter{
                base + ".wout",
                deviceRandomTensor({dim, dim}, blackforge::backend::cpu::seedFor(seed, op.output, kAttnWoutSalt)),
                DeviceTensor::zeros({dim, dim})};
        } else if (op.kind == ir::OpKind::FeedForward) {
            const ir::Value& inputValue = modelIR.valueById(op.input);
            std::size_t dim = concreteLastDim(inputValue, "feedforward#", op.output);
            auto hiddenDim = static_cast<std::size_t>(op.feedForwardHiddenDim);
            std::string base = "feedforward" + std::to_string(op.output);

            layer.ffW1 = Parameter{
                base + ".w1",
                deviceRandomTensor({dim, hiddenDim}, blackforge::backend::cpu::seedFor(seed, op.output, kFFW1Salt)),
                DeviceTensor::zeros({dim, hiddenDim})};
            layer.ffB1 = Parameter{
                base + ".b1", deviceRandomTensor({hiddenDim}, blackforge::backend::cpu::seedFor(seed, op.output, kFFB1Salt)),
                DeviceTensor::zeros({hiddenDim})};
            layer.ffW2 = Parameter{
                base + ".w2",
                deviceRandomTensor({hiddenDim, dim}, blackforge::backend::cpu::seedFor(seed, op.output, kFFW2Salt)),
                DeviceTensor::zeros({hiddenDim, dim})};
            layer.ffB2 = Parameter{
                base + ".b2", deviceRandomTensor({dim}, blackforge::backend::cpu::seedFor(seed, op.output, kFFB2Salt)),
                DeviceTensor::zeros({dim})};
        }

        layers_.push_back(std::move(layer));
    }
}

DeviceTensor Model::forward(const DeviceTensor& input) {
    DeviceTensor current = input.clone();

    for (auto& layer : layers_) {
        layer.cachedInput = current.clone();

        switch (layer.kind) {
            case ir::OpKind::Linear:
                current = useTensorCoreLinear_ ? linearBf16(current, layer.weight->value, layer.bias->value)
                                                : linear(current, layer.weight->value, layer.bias->value);
                break;
            case ir::OpKind::Silu: current = silu(current); break;
            case ir::OpKind::Relu: current = relu(current); break;
            case ir::OpKind::Gelu: current = gelu(current); break;
            case ir::OpKind::RmsNorm: current = rmsnorm(current); break;
            case ir::OpKind::Softmax: current = softmax(current); break;
            case ir::OpKind::Embedding: current = embeddingLookup(current, layer.embeddingTable->value); break;
            case ir::OpKind::PositionalEmbedding:
                current = addPositionalEmbedding(current, layer.positionalTable->value);
                break;
            case ir::OpKind::Attention:
                current = useTensorCoreLinear_
                              ? selfAttentionBf16(current, layer.attnWq->value, layer.attnWk->value,
                                                  layer.attnWv->value, layer.attnWout->value, layer.attentionNumHeads)
                              : selfAttention(current, layer.attnWq->value, layer.attnWk->value, layer.attnWv->value,
                                              layer.attnWout->value, layer.attentionNumHeads);
                break;
            case ir::OpKind::FeedForward:
                current = useTensorCoreLinear_ ? feedForwardBf16(current, layer.ffW1->value, layer.ffB1->value,
                                                                  layer.ffW2->value, layer.ffB2->value)
                                                : feedForward(current, layer.ffW1->value, layer.ffB1->value,
                                                              layer.ffW2->value, layer.ffB2->value);
                break;
        }
    }

    return current;
}

void Model::backward(const DeviceTensor& outputGrad) {
    DeviceTensor gradCurrent = outputGrad.clone();

    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        LayerState& layer = *it;

        switch (layer.kind) {
            case ir::OpKind::Linear: {
                AddBiasGrad addGrad = addBiasBackward(gradCurrent);
                MatmulGrad matGrad =
                    useTensorCoreLinear_
                        ? matmulBf16Backward(flatten2D(layer.cachedInput), layer.weight->value,
                                             flatten2D(addGrad.dInput))
                        : matmulBackward(flatten2D(layer.cachedInput), layer.weight->value, flatten2D(addGrad.dInput));

                accumulate(layer.weight->grad, matGrad.dB);
                accumulate(layer.bias->grad, addGrad.dBias);

                gradCurrent = std::move(matGrad.dA).reshaped(layer.cachedInput.shape());
                break;
            }
            case ir::OpKind::Silu: gradCurrent = siluBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::Relu: gradCurrent = reluBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::Gelu: gradCurrent = geluBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::RmsNorm: gradCurrent = rmsnormBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::Softmax: gradCurrent = softmaxBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::Embedding: {
                DeviceTensor dTable = embeddingLookupBackward(layer.cachedInput, gradCurrent, layer.embeddingVocabSize);
                accumulate(layer.embeddingTable->grad, dTable);
                // I token id sono indici, non differenziabili: nessun
                // gradiente da propagare a un eventuale layer precedente.
                gradCurrent = DeviceTensor::zeros(layer.cachedInput.shape());
                break;
            }
            case ir::OpKind::PositionalEmbedding: {
                PositionalEmbeddingGrad g = addPositionalEmbeddingBackward(gradCurrent, layer.positionalMaxSeqLen);
                accumulate(layer.positionalTable->grad, g.dTable);
                gradCurrent = std::move(g.dInput);
                break;
            }
            case ir::OpKind::Attention: {
                SelfAttentionGrad g =
                    useTensorCoreLinear_
                        ? selfAttentionBf16Backward(layer.cachedInput, layer.attnWq->value, layer.attnWk->value,
                                                     layer.attnWv->value, layer.attnWout->value,
                                                     layer.attentionNumHeads, gradCurrent)
                        : selfAttentionBackward(layer.cachedInput, layer.attnWq->value, layer.attnWk->value,
                                                 layer.attnWv->value, layer.attnWout->value, layer.attentionNumHeads,
                                                 gradCurrent);
                accumulate(layer.attnWq->grad, g.dWq);
                accumulate(layer.attnWk->grad, g.dWk);
                accumulate(layer.attnWv->grad, g.dWv);
                accumulate(layer.attnWout->grad, g.dWout);
                gradCurrent = std::move(g.dInput);
                break;
            }
            case ir::OpKind::FeedForward: {
                FeedForwardGrad g =
                    useTensorCoreLinear_
                        ? feedForwardBf16Backward(layer.cachedInput, layer.ffW1->value, layer.ffB1->value,
                                                   layer.ffW2->value, layer.ffB2->value, gradCurrent)
                        : feedForwardBackward(layer.cachedInput, layer.ffW1->value, layer.ffB1->value,
                                               layer.ffW2->value, layer.ffB2->value, gradCurrent);
                accumulate(layer.ffW1->grad, g.dW1);
                accumulate(layer.ffB1->grad, g.dB1);
                accumulate(layer.ffW2->grad, g.dW2);
                accumulate(layer.ffB2->grad, g.dB2);
                gradCurrent = std::move(g.dInput);
                break;
            }
        }
    }
}

DeviceTensor Model::forwardIncremental(const DeviceTensor& newTokenIds) {
    DeviceTensor current = newTokenIds.clone();
    std::size_t newLen = current.rank() >= 2 ? current.dim(1) : 0;

    for (auto& layer : layers_) {
        switch (layer.kind) {
            case ir::OpKind::Linear:
                current = useTensorCoreLinear_ ? linearBf16(current, layer.weight->value, layer.bias->value)
                                                : linear(current, layer.weight->value, layer.bias->value);
                break;
            case ir::OpKind::Silu: current = silu(current); break;
            case ir::OpKind::Relu: current = relu(current); break;
            case ir::OpKind::Gelu: current = gelu(current); break;
            case ir::OpKind::RmsNorm: current = rmsnorm(current); break;
            case ir::OpKind::Softmax: current = softmax(current); break;
            case ir::OpKind::Embedding: current = embeddingLookup(current, layer.embeddingTable->value); break;
            case ir::OpKind::PositionalEmbedding:
                current = addPositionalEmbeddingAt(current, layer.positionalTable->value, generationPosition_);
                break;
            case ir::OpKind::Attention:
                current = useTensorCoreLinear_
                              ? selfAttentionIncrementalBf16(current, layer.attnWq->value, layer.attnWk->value,
                                                              layer.attnWv->value, layer.attnWout->value,
                                                              layer.attentionNumHeads, layer.kvCache)
                              : selfAttentionIncremental(current, layer.attnWq->value, layer.attnWk->value,
                                                          layer.attnWv->value, layer.attnWout->value,
                                                          layer.attentionNumHeads, layer.kvCache);
                break;
            case ir::OpKind::FeedForward:
                current = useTensorCoreLinear_ ? feedForwardBf16(current, layer.ffW1->value, layer.ffB1->value,
                                                                  layer.ffW2->value, layer.ffB2->value)
                                                : feedForward(current, layer.ffW1->value, layer.ffB1->value,
                                                              layer.ffW2->value, layer.ffB2->value);
                break;
        }
    }

    generationPosition_ += newLen;
    return current;
}

void Model::resetGenerationState() {
    generationPosition_ = 0;
    for (auto& layer : layers_) {
        layer.kvCache = KVCache{};
    }
}

std::vector<Parameter*> Model::allParameterSlots(LayerState& layer) {
    std::vector<Parameter*> result;
    for (std::optional<Parameter>* slot :
         {&layer.weight, &layer.bias, &layer.embeddingTable, &layer.positionalTable, &layer.attnWq, &layer.attnWk,
          &layer.attnWv, &layer.attnWout, &layer.ffW1, &layer.ffB1, &layer.ffW2, &layer.ffB2}) {
        if (slot->has_value()) {
            result.push_back(&**slot);
        }
    }
    return result;
}

void Model::zeroGrad() {
    for (auto& layer : layers_) {
        for (Parameter* param : allParameterSlots(layer)) {
            param->grad = DeviceTensor::zeros(param->grad.shape());
        }
    }
}

std::vector<Parameter*> Model::parameters() {
    std::vector<Parameter*> result;
    for (auto& layer : layers_) {
        for (Parameter* param : allParameterSlots(layer)) {
            result.push_back(param);
        }
    }
    return result;
}

}  // namespace blackforge::backend::cuda
