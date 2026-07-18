#include "blackforge/backend/cpu/model.hpp"

#include <stdexcept>

#include "blackforge/backend/cpu/autodiff.hpp"
#include "blackforge/backend/cpu/ops.hpp"
#include "blackforge/backend/cpu/random_init.hpp"

namespace blackforge::backend::cpu {

namespace {

constexpr unsigned int kWeightSalt = 0x2545F491U;
constexpr unsigned int kBiasSalt = 0x27D4EB2FU;
constexpr unsigned int kLoraASalt = 0x9E3779B1U;

void accumulate(runtime::Tensor& target, const runtime::Tensor& delta) {
    for (std::size_t i = 0; i < target.elementCount(); ++i) {
        target.at(i) += delta.at(i);
    }
}

}  // namespace

Model::Model(const ir::ModelIR& modelIR, unsigned int seed, std::optional<LoraOptions> lora)
    : name_(modelIR.name), loraOptions_(lora) {
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
            const ir::Dim& inDim = inputValue.shape.back();
            if (inDim.isSymbolic) {
                throw std::invalid_argument("Model: la dimensione delle feature in ingresso al layer linear#" +
                                             std::to_string(op.output) +
                                             " e' simbolica ('" + inDim.symbolicName +
                                             "'); serve una dimensione concreta per allocare i pesi");
            }

            std::size_t inFeatures = static_cast<std::size_t>(inDim.literalValue);
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
        }

        layers_.push_back(std::move(layer));
    }
}

runtime::Tensor Model::forward(const runtime::Tensor& input) {
    runtime::Tensor current = input;

    for (auto& layer : layers_) {
        layer.cachedInput = current;

        switch (layer.kind) {
            case ir::OpKind::Linear: {
                runtime::Tensor base = linear(current, layer.weight->value, layer.bias->value);
                if (layer.loraA.has_value()) {
                    layer.cachedLoraHidden = matmul(current, layer.loraA->value);
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
                    MatmulGrad baseGrad = matmulBackward(layer.cachedInput, layer.weight->value, addGrad.dInput);

                    gradCurrent = add(baseGrad.dA, hiddenGrad.dA);
                } else {
                    AddBiasGrad addGrad = addBiasBackward(gradCurrent);
                    MatmulGrad matGrad = matmulBackward(layer.cachedInput, layer.weight->value, addGrad.dInput);

                    accumulate(layer.weight->grad, matGrad.dB);
                    accumulate(layer.bias->grad, addGrad.dBias);

                    gradCurrent = matGrad.dA;
                }
                break;
            }
            case ir::OpKind::Silu: gradCurrent = siluBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::Relu: gradCurrent = reluBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::Gelu: gradCurrent = geluBackward(layer.cachedInput, gradCurrent); break;
        }
    }
}

void Model::zeroGrad() {
    for (auto& layer : layers_) {
        if (layer.weight.has_value()) {
            layer.weight->grad = runtime::Tensor::zeros(layer.weight->grad.shape());
        }
        if (layer.bias.has_value()) {
            layer.bias->grad = runtime::Tensor::zeros(layer.bias->grad.shape());
        }
        if (layer.loraA.has_value()) {
            layer.loraA->grad = runtime::Tensor::zeros(layer.loraA->grad.shape());
        }
        if (layer.loraB.has_value()) {
            layer.loraB->grad = runtime::Tensor::zeros(layer.loraB->grad.shape());
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
    }
    return result;
}

std::vector<Parameter*> Model::allParameters() {
    std::vector<Parameter*> result;
    for (auto& layer : layers_) {
        if (layer.weight.has_value()) {
            result.push_back(&*layer.weight);
        }
        if (layer.bias.has_value()) {
            result.push_back(&*layer.bias);
        }
        if (layer.loraA.has_value()) {
            result.push_back(&*layer.loraA);
        }
        if (layer.loraB.has_value()) {
            result.push_back(&*layer.loraB);
        }
    }
    return result;
}

}  // namespace blackforge::backend::cpu
