#include "blackforge/backend/cpu/model.hpp"

#include <stdexcept>

#include "blackforge/backend/cpu/autodiff.hpp"
#include "blackforge/backend/cpu/ops.hpp"
#include "blackforge/backend/cpu/random_init.hpp"

namespace blackforge::backend::cpu {

namespace {

constexpr unsigned int kWeightSalt = 0x2545F491U;
constexpr unsigned int kBiasSalt = 0x27D4EB2FU;

void accumulate(runtime::Tensor& target, const runtime::Tensor& delta) {
    for (std::size_t i = 0; i < target.elementCount(); ++i) {
        target.at(i) += delta.at(i);
    }
}

}  // namespace

Model::Model(const ir::ModelIR& modelIR, unsigned int seed) : name_(modelIR.name) {
    if (modelIR.pipelines.empty()) {
        throw std::invalid_argument("Model: il modello '" + modelIR.name + "' non ha pipeline da addestrare");
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
        }

        layers_.push_back(std::move(layer));
    }
}

runtime::Tensor Model::forward(const runtime::Tensor& input) {
    runtime::Tensor current = input;

    for (auto& layer : layers_) {
        layer.cachedInput = current;

        switch (layer.kind) {
            case ir::OpKind::Linear:
                current = linear(current, layer.weight->value, layer.bias->value);
                break;
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
                AddBiasGrad addGrad = addBiasBackward(gradCurrent);
                MatmulGrad matGrad = matmulBackward(layer.cachedInput, layer.weight->value, addGrad.dInput);

                accumulate(layer.weight->grad, matGrad.dB);
                accumulate(layer.bias->grad, addGrad.dBias);

                gradCurrent = matGrad.dA;
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
    }
}

std::vector<Parameter*> Model::parameters() {
    std::vector<Parameter*> result;
    for (auto& layer : layers_) {
        if (layer.weight.has_value()) {
            result.push_back(&*layer.weight);
        }
        if (layer.bias.has_value()) {
            result.push_back(&*layer.bias);
        }
    }
    return result;
}

}  // namespace blackforge::backend::cpu
