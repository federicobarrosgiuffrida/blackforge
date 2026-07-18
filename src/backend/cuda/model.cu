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

}  // namespace

Model::Model(const ir::ModelIR& modelIR, unsigned int seed) : name_(modelIR.name) {
    if (modelIR.pipelines.empty()) {
        throw std::invalid_argument("cuda::Model: il modello '" + modelIR.name + "' non ha pipeline da addestrare");
    }

    const ir::Pipeline& pipeline = modelIR.pipelines.front();

    for (const auto& op : pipeline.operations) {
        LayerState layer;
        layer.kind = op.kind;

        if (op.kind == ir::OpKind::Linear) {
            const ir::Value& inputValue = modelIR.valueById(op.input);
            const ir::Dim& inDim = inputValue.shape.back();
            if (inDim.isSymbolic) {
                throw std::invalid_argument("cuda::Model: la dimensione delle feature in ingresso al layer "
                                             "linear#" +
                                             std::to_string(op.output) + " e' simbolica ('" + inDim.symbolicName +
                                             "'); serve una dimensione concreta per allocare i pesi");
            }

            auto inFeatures = static_cast<std::size_t>(inDim.literalValue);
            auto outFeatures = static_cast<std::size_t>(op.linearOutFeatures);
            std::string base = "linear" + std::to_string(op.output);

            // Stesso generatore deterministico (stesso seme, stesso
            // "sale" per weight/bias) del backend CPU e dell'Executor
            // CUDA di sola inferenza: permette il confronto diretto
            // CPU/GPU nei test.
            runtime::Tensor hostWeight = blackforge::backend::cpu::randomTensor(
                {inFeatures, outFeatures}, blackforge::backend::cpu::seedFor(seed, op.output, kWeightSalt));
            runtime::Tensor hostBias = blackforge::backend::cpu::randomTensor(
                {outFeatures}, blackforge::backend::cpu::seedFor(seed, op.output, kBiasSalt));

            layer.weight = Parameter{base + ".weight", DeviceTensor::fromHost(hostWeight),
                                      DeviceTensor::zeros({inFeatures, outFeatures})};
            layer.bias =
                Parameter{base + ".bias", DeviceTensor::fromHost(hostBias), DeviceTensor::zeros({outFeatures})};
        }

        layers_.push_back(std::move(layer));
    }
}

DeviceTensor Model::forward(const DeviceTensor& input) {
    DeviceTensor current = input.clone();

    for (auto& layer : layers_) {
        layer.cachedInput = current.clone();

        switch (layer.kind) {
            case ir::OpKind::Linear: current = linear(current, layer.weight->value, layer.bias->value); break;
            case ir::OpKind::Silu: current = silu(current); break;
            case ir::OpKind::Relu: current = relu(current); break;
            case ir::OpKind::Gelu: current = gelu(current); break;
            case ir::OpKind::RmsNorm: current = rmsnorm(current); break;
            case ir::OpKind::Softmax: current = softmax(current); break;
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
                MatmulGrad matGrad = matmulBackward(layer.cachedInput, layer.weight->value, addGrad.dInput);

                accumulate(layer.weight->grad, matGrad.dB);
                accumulate(layer.bias->grad, addGrad.dBias);

                gradCurrent = std::move(matGrad.dA);
                break;
            }
            case ir::OpKind::Silu: gradCurrent = siluBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::Relu: gradCurrent = reluBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::Gelu: gradCurrent = geluBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::RmsNorm: gradCurrent = rmsnormBackward(layer.cachedInput, gradCurrent); break;
            case ir::OpKind::Softmax: gradCurrent = softmaxBackward(layer.cachedInput, gradCurrent); break;
        }
    }
}

void Model::zeroGrad() {
    for (auto& layer : layers_) {
        if (layer.weight.has_value()) {
            layer.weight->grad = DeviceTensor::zeros(layer.weight->grad.shape());
        }
        if (layer.bias.has_value()) {
            layer.bias->grad = DeviceTensor::zeros(layer.bias->grad.shape());
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

}  // namespace blackforge::backend::cuda
