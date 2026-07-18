#include "blackforge/backend/cuda/executor.hpp"

#include <stdexcept>

#include "blackforge/backend/cpu/random_init.hpp"
#include "blackforge/backend/cuda/device_tensor.hpp"
#include "blackforge/backend/cuda/ops.hpp"

namespace blackforge::backend::cuda {

runtime::Tensor Executor::makeSyntheticInput(const ir::Value& inputValue, std::size_t batchSize) const {
    std::vector<std::size_t> shape;
    shape.reserve(inputValue.shape.size());
    for (const auto& dim : inputValue.shape) {
        shape.push_back(dim.isSymbolic ? batchSize : static_cast<std::size_t>(dim.literalValue));
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
                if (current.rank() != 2) {
                    throw std::invalid_argument("linear richiede un input a rango 2 [batch, features]");
                }
                std::size_t inFeatures = current.dim(1);
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
        }
    }

    return current.toHost();
}

}  // namespace blackforge::backend::cuda
