#include "blackforge/backend/cpu/model.hpp"

#include <stdexcept>

#include "blackforge/backend/cpu/autodiff.hpp"
#include "blackforge/backend/cpu/ops.hpp"
#include "blackforge/backend/cpu/quantize.hpp"
#include "blackforge/backend/cpu/random_init.hpp"

namespace blackforge::backend::cpu {

namespace {

constexpr unsigned int kWeightSalt = 0x2545F491U;
constexpr unsigned int kBiasSalt = 0x27D4EB2FU;
constexpr unsigned int kLoraASalt = 0x9E3779B1U;
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

void accumulate(runtime::Tensor& target, const runtime::Tensor& delta) {
    for (std::size_t i = 0; i < target.elementCount(); ++i) {
        target.at(i) += delta.at(i);
    }
}

// Appiattisce un tensore a rango >= 2 a [rows, features] (stesso layout
// flat riga-maggiore, vedi il commento in ops.cpp/addBias): serve per
// poter chiamare matmulBackward() (primitivo puramente 2D) sull'input
// cachato di un layer 'linear' quando e' a rango > 2 (es. [batch, seq,
// features], come succede quando 'linear' arriva dopo 'attention' o
// 'feedforward' in una pipeline di modello linguistico). Il forward
// (linear() in ops.cpp) fa gia' lo stesso appiattimento internamente;
// qui serve rifarlo esplicitamente perche' matmulBackward() lavora sugli
// stessi operandi del forward, non sull'uscita di linear().
runtime::Tensor flatten2D(const runtime::Tensor& t) {
    std::size_t features = t.shape().back();
    std::size_t rows = t.elementCount() / features;
    return runtime::Tensor({rows, features}, t.data());
}

// Dimensione delle feature in ingresso a un layer che ne ha bisogno per
// allocare i propri pesi (positional_embedding/attention/feedforward,
// stessa esigenza di 'linear'): deve essere concreta, non simbolica.
std::size_t concreteLastDim(const ir::Value& value, const std::string& opLabel, std::size_t opOutput) {
    const ir::Dim& dim = value.shape.back();
    if (dim.isSymbolic) {
        throw std::invalid_argument("Model: la dimensione delle feature in ingresso al layer " + opLabel +
                                     std::to_string(opOutput) + " e' simbolica ('" + dim.symbolicName +
                                     "'); serve una dimensione concreta per allocare i pesi");
    }
    return static_cast<std::size_t>(dim.literalValue);
}

}  // namespace

Model::Model(const ir::ModelIR& modelIR, unsigned int seed, std::optional<LoraOptions> lora,
             std::optional<ir::PrecisionPolicy> precision)
    : name_(modelIR.name), loraOptions_(lora), precision_(precision) {
    if (modelIR.pipelines.empty()) {
        throw std::invalid_argument("Model: il modello '" + modelIR.name + "' non ha pipeline da addestrare");
    }
    if (loraOptions_.has_value() && loraOptions_->rank <= 0) {
        throw std::invalid_argument("Model: 'rank' di LoRA deve essere un intero positivo");
    }

    const ir::Pipeline& pipeline = modelIR.pipelines.front();

    for (const auto& op : pipeline.operations) {
        LayerState layer;
        layer.kind = op.kind;
        layer.operationOutputId = op.output;

        if (op.kind == ir::OpKind::Linear) {
            const ir::Value& inputValue = modelIR.valueById(op.input);
            std::size_t inFeatures = concreteLastDim(inputValue, "linear#", op.output);
            auto outFeatures = static_cast<std::size_t>(op.linearOutFeatures);
            std::string base = "linear" + std::to_string(op.output);

            layer.weight = Parameter{base + ".weight",
                                      randomTensor({inFeatures, outFeatures}, seedFor(seed, op.output, kWeightSalt)),
                                      runtime::Tensor::zeros({inFeatures, outFeatures})};
            layer.bias = Parameter{base + ".bias", randomTensor({outFeatures}, seedFor(seed, op.output, kBiasSalt)),
                                    runtime::Tensor::zeros({outFeatures})};

            if (loraOptions_.has_value()) {
                auto rank = static_cast<std::size_t>(loraOptions_->rank);
                // Inizializzazione standard di LoRA: A con valori piccoli
                // (deterministici, non Xavier/Kaiming), B a zero, cosi'
                // il contributo dell'adapter e' esattamente nullo
                // all'inizio dell'addestramento (il modello si comporta
                // come i soli pesi di base finche' A/B non si allenano).
                layer.loraA = Parameter{base + ".loraA", randomTensor({inFeatures, rank}, seedFor(seed, op.output, kLoraASalt)),
                                         runtime::Tensor::zeros({inFeatures, rank})};
                layer.loraB = Parameter{base + ".loraB", runtime::Tensor::zeros({rank, outFeatures}),
                                         runtime::Tensor::zeros({rank, outFeatures})};
            }
        } else if (op.kind == ir::OpKind::Embedding) {
            auto vocabSize = static_cast<std::size_t>(op.embeddingVocabSize);
            auto dim = static_cast<std::size_t>(op.embeddingDim);
            std::string base = "embedding" + std::to_string(op.output);

            layer.embeddingVocabSize = vocabSize;
            layer.embeddingTable = Parameter{base + ".table",
                                              randomTensor({vocabSize, dim}, seedFor(seed, op.output, kEmbeddingSalt)),
                                              runtime::Tensor::zeros({vocabSize, dim})};
        } else if (op.kind == ir::OpKind::PositionalEmbedding) {
            const ir::Value& inputValue = modelIR.valueById(op.input);
            std::size_t dim = concreteLastDim(inputValue, "positional_embedding#", op.output);
            auto maxSeqLen = static_cast<std::size_t>(op.positionalMaxSeqLen);
            std::string base = "positional_embedding" + std::to_string(op.output);

            layer.positionalMaxSeqLen = maxSeqLen;
            layer.positionalTable =
                Parameter{base + ".table", randomTensor({maxSeqLen, dim}, seedFor(seed, op.output, kPositionalSalt)),
                          runtime::Tensor::zeros({maxSeqLen, dim})};
        } else if (op.kind == ir::OpKind::Attention) {
            const ir::Value& inputValue = modelIR.valueById(op.input);
            std::size_t dim = concreteLastDim(inputValue, "attention#", op.output);
            auto numHeads = static_cast<std::size_t>(op.attentionNumHeads);
            if (numHeads == 0 || dim % numHeads != 0) {
                throw std::invalid_argument("Model: 'attention#" + std::to_string(op.output) + "' richiede che " +
                                             std::to_string(numHeads) + " teste dividano esattamente dim=" +
                                             std::to_string(dim));
            }
            std::string base = "attention" + std::to_string(op.output);

            layer.attentionNumHeads = numHeads;
            // Nessun bias per le proiezioni Q/K/V/Out (come in LLaMA, vedi
            // backend::cpu::selfAttention).
            layer.attnWq = Parameter{base + ".wq", randomTensor({dim, dim}, seedFor(seed, op.output, kAttnWqSalt)),
                                      runtime::Tensor::zeros({dim, dim})};
            layer.attnWk = Parameter{base + ".wk", randomTensor({dim, dim}, seedFor(seed, op.output, kAttnWkSalt)),
                                      runtime::Tensor::zeros({dim, dim})};
            layer.attnWv = Parameter{base + ".wv", randomTensor({dim, dim}, seedFor(seed, op.output, kAttnWvSalt)),
                                      runtime::Tensor::zeros({dim, dim})};
            layer.attnWout =
                Parameter{base + ".wout", randomTensor({dim, dim}, seedFor(seed, op.output, kAttnWoutSalt)),
                          runtime::Tensor::zeros({dim, dim})};
        } else if (op.kind == ir::OpKind::FeedForward) {
            const ir::Value& inputValue = modelIR.valueById(op.input);
            std::size_t dim = concreteLastDim(inputValue, "feedforward#", op.output);
            auto hiddenDim = static_cast<std::size_t>(op.feedForwardHiddenDim);
            std::string base = "feedforward" + std::to_string(op.output);

            layer.ffW1 = Parameter{base + ".w1", randomTensor({dim, hiddenDim}, seedFor(seed, op.output, kFFW1Salt)),
                                    runtime::Tensor::zeros({dim, hiddenDim})};
            layer.ffB1 = Parameter{base + ".b1", randomTensor({hiddenDim}, seedFor(seed, op.output, kFFB1Salt)),
                                    runtime::Tensor::zeros({hiddenDim})};
            layer.ffW2 = Parameter{base + ".w2", randomTensor({hiddenDim, dim}, seedFor(seed, op.output, kFFW2Salt)),
                                    runtime::Tensor::zeros({hiddenDim, dim})};
            layer.ffB2 = Parameter{base + ".b2", randomTensor({dim}, seedFor(seed, op.output, kFFB2Salt)),
                                    runtime::Tensor::zeros({dim})};
        }

        layers_.push_back(std::move(layer));
    }
}

runtime::Tensor Model::forward(const runtime::Tensor& input) {
    runtime::Tensor current = input;
    if (precision_.has_value()) {
        current = quantize(current, precision_->storage);
    }

    for (auto& layer : layers_) {
        layer.cachedInput = current;

        switch (layer.kind) {
            case ir::OpKind::Linear: {
                runtime::Tensor computeInput = current;
                runtime::Tensor computeWeight = layer.weight->value;
                if (precision_.has_value()) {
                    computeInput = quantize(computeInput, precision_->compute);
                    computeWeight = quantize(computeWeight, precision_->compute);
                }
                runtime::Tensor base = linear(computeInput, computeWeight, layer.bias->value);

                if (layer.loraA.has_value()) {
                    runtime::Tensor loraInput = current;
                    runtime::Tensor loraA = layer.loraA->value;
                    if (precision_.has_value()) {
                        loraInput = quantize(loraInput, precision_->compute);
                        loraA = quantize(loraA, precision_->compute);
                    }
                    layer.cachedLoraHidden = matmul(loraInput, loraA);
                    runtime::Tensor loraOut = matmul(layer.cachedLoraHidden, layer.loraB->value);
                    auto factor = static_cast<float>(loraOptions_->alpha / static_cast<double>(loraOptions_->rank));
                    current = add(base, scale(loraOut, factor));
                } else {
                    current = base;
                }
                break;
            }
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
                current = selfAttention(current, layer.attnWq->value, layer.attnWk->value, layer.attnWv->value,
                                         layer.attnWout->value, layer.attentionNumHeads);
                break;
            case ir::OpKind::FeedForward:
                current = feedForward(current, layer.ffW1->value, layer.ffB1->value, layer.ffW2->value,
                                       layer.ffB2->value);
                break;
        }

        if (precision_.has_value()) {
            current = quantize(current, precision_->storage);
        }
    }

    return current;
}

void Model::backward(const runtime::Tensor& outputGrad) {
    runtime::Tensor gradCurrent = outputGrad;

    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        LayerState& layer = *it;

        switch (layer.kind) {
            case ir::OpKind::Linear: {
                if (layer.loraA.has_value()) {
                    // output = base(input) + factor * (input @ A) @ B.
                    // 'add' e' un'identita' nel gradiente per entrambi i
                    // rami: gradCurrent raggiunge sia base sia la parte
                    // scalata dell'adapter invariato.
                    auto factor = static_cast<float>(loraOptions_->alpha / static_cast<double>(loraOptions_->rank));
                    runtime::Tensor gradLoraOut = scale(gradCurrent, factor);

                    MatmulGrad loraOutGrad = matmulBackward(layer.cachedLoraHidden, layer.loraB->value, gradLoraOut);
                    accumulate(layer.loraB->grad, loraOutGrad.dB);

                    MatmulGrad hiddenGrad = matmulBackward(layer.cachedInput, layer.loraA->value, loraOutGrad.dA);
                    accumulate(layer.loraA->grad, hiddenGrad.dB);

                    // weight/bias sono congelati: si calcola comunque il
                    // gradiente rispetto all'input (serve per continuare
                    // il backward verso i layer precedenti), ma non si
                    // accumula nulla nei loro .grad.
                    AddBiasGrad addGrad = addBiasBackward(gradCurrent);
                    MatmulGrad baseGrad =
                        matmulBackward(flatten2D(layer.cachedInput), layer.weight->value, flatten2D(addGrad.dInput));
                    runtime::Tensor baseGradDA(layer.cachedInput.shape(), std::move(baseGrad.dA.data()));

                    gradCurrent = add(baseGradDA, hiddenGrad.dA);
                } else {
                    AddBiasGrad addGrad = addBiasBackward(gradCurrent);
                    MatmulGrad matGrad =
                        matmulBackward(flatten2D(layer.cachedInput), layer.weight->value, flatten2D(addGrad.dInput));

                    accumulate(layer.weight->grad, matGrad.dB);
                    accumulate(layer.bias->grad, addGrad.dBias);

                    gradCurrent = runtime::Tensor(layer.cachedInput.shape(), std::move(matGrad.dA.data()));
                }
                break;
            }
            case ir::OpKind::Silu: gradCurrent = siluBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::Relu: gradCurrent = reluBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::Gelu: gradCurrent = geluBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::RmsNorm: gradCurrent = rmsnormBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::Softmax: gradCurrent = softmaxBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::Embedding: {
                runtime::Tensor dTable =
                    embeddingLookupBackward(layer.cachedInput, gradCurrent, layer.embeddingVocabSize);
                accumulate(layer.embeddingTable->grad, dTable);
                // I token id sono indici, non differenziabili: nessun
                // gradiente da propagare a un eventuale layer precedente
                // (Embedding e' tipicamente il primo layer della pipeline).
                gradCurrent = runtime::Tensor::zeros(layer.cachedInput.shape());
                break;
            }
            case ir::OpKind::PositionalEmbedding: {
                PositionalEmbeddingGrad g = addPositionalEmbeddingBackward(gradCurrent, layer.positionalMaxSeqLen);
                accumulate(layer.positionalTable->grad, g.dTable);
                gradCurrent = g.dInput;
                break;
            }
            case ir::OpKind::Attention: {
                SelfAttentionGrad g = selfAttentionBackward(layer.cachedInput, layer.attnWq->value,
                                                             layer.attnWk->value, layer.attnWv->value,
                                                             layer.attnWout->value, layer.attentionNumHeads,
                                                             gradCurrent);
                accumulate(layer.attnWq->grad, g.dWq);
                accumulate(layer.attnWk->grad, g.dWk);
                accumulate(layer.attnWv->grad, g.dWv);
                accumulate(layer.attnWout->grad, g.dWout);
                gradCurrent = g.dInput;
                break;
            }
            case ir::OpKind::FeedForward: {
                FeedForwardGrad g = feedForwardBackward(layer.cachedInput, layer.ffW1->value, layer.ffB1->value,
                                                         layer.ffW2->value, layer.ffB2->value, gradCurrent);
                accumulate(layer.ffW1->grad, g.dW1);
                accumulate(layer.ffB1->grad, g.dB1);
                accumulate(layer.ffW2->grad, g.dW2);
                accumulate(layer.ffB2->grad, g.dB2);
                gradCurrent = g.dInput;
                break;
            }
        }
    }
}

runtime::Tensor Model::forwardIncremental(const runtime::Tensor& newTokenIds) {
    runtime::Tensor current = newTokenIds;
    std::size_t newLen = current.rank() >= 2 ? current.dim(1) : 0;

    for (auto& layer : layers_) {
        switch (layer.kind) {
            case ir::OpKind::Linear: current = linear(current, layer.weight->value, layer.bias->value); break;
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
                current = selfAttentionIncremental(current, layer.attnWq->value, layer.attnWk->value,
                                                    layer.attnWv->value, layer.attnWout->value,
                                                    layer.attentionNumHeads, layer.kvCache);
                break;
            case ir::OpKind::FeedForward:
                current = feedForward(current, layer.ffW1->value, layer.ffB1->value, layer.ffW2->value,
                                       layer.ffB2->value);
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
    auto add = [&](std::optional<Parameter>& slot) {
        if (slot.has_value()) {
            result.push_back(&*slot);
        }
    };
    add(layer.weight);
    add(layer.bias);
    add(layer.loraA);
    add(layer.loraB);
    add(layer.embeddingTable);
    add(layer.positionalTable);
    add(layer.attnWq);
    add(layer.attnWk);
    add(layer.attnWv);
    add(layer.attnWout);
    add(layer.ffW1);
    add(layer.ffB1);
    add(layer.ffW2);
    add(layer.ffB2);
    return result;
}

void Model::zeroGrad() {
    for (auto& layer : layers_) {
        for (Parameter* param : allParameterSlots(layer)) {
            param->grad = runtime::Tensor::zeros(param->grad.shape());
        }
    }
}

std::vector<Parameter*> Model::parameters() {
    std::vector<Parameter*> result;
    for (auto& layer : layers_) {
        if (layer.loraA.has_value()) {
            // LoRA attivo su questo layer: solo l'adapter e' allenabile,
            // weight/bias restano congelati e non compaiono qui.
            result.push_back(&*layer.loraA);
            result.push_back(&*layer.loraB);
        } else {
            if (layer.weight.has_value()) {
                result.push_back(&*layer.weight);
            }
            if (layer.bias.has_value()) {
                result.push_back(&*layer.bias);
            }
        }
        // I nuovi tipi di layer (embedding/positional_embedding/attention/
        // feedforward) non supportano ancora LoRA (vedi LayerState): sono
        // sempre pienamente allenabili.
        for (std::optional<Parameter>* slot :
             {&layer.embeddingTable, &layer.positionalTable, &layer.attnWq, &layer.attnWk, &layer.attnWv,
              &layer.attnWout, &layer.ffW1, &layer.ffB1, &layer.ffW2, &layer.ffB2}) {
            if (slot->has_value()) {
                result.push_back(&**slot);
            }
        }
    }
    return result;
}

std::vector<Parameter*> Model::allParameters() {
    std::vector<Parameter*> result;
    for (auto& layer : layers_) {
        for (Parameter* param : allParameterSlots(layer)) {
            result.push_back(param);
        }
    }
    return result;
}

}  // namespace blackforge::backend::cpu
