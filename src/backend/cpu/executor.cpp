#include "blackforge/backend/cpu/executor.hpp"

#include <stdexcept>

#include "blackforge/backend/cpu/ops.hpp"
#include "blackforge/backend/cpu/random_init.hpp"

namespace blackforge::backend::cpu {

runtime::Tensor Executor::makeSyntheticInput(const ir::Value& inputValue, std::size_t batchSize) const {
    std::vector<std::size_t> shape;
    shape.reserve(inputValue.shape.size());
    for (const auto& dim : inputValue.shape) {
        shape.push_back(dim.isSymbolic ? batchSize : static_cast<std::size_t>(dim.literalValue));
    }
    return randomTensor(std::move(shape), seedFor(seed_, 0, 0x9E3779B1U));
}

runtime::Tensor Executor::run(const ir::ModelIR& model, const runtime::Tensor& input) const {
    if (model.pipelines.empty()) {
        throw std::invalid_argument("il modello '" + model.name + "' non ha pipeline da eseguire");
    }

    const ir::Pipeline& pipeline = model.pipelines.front();
    runtime::Tensor current = input;

    for (const auto& op : pipeline.operations) {
        switch (op.kind) {
            case ir::OpKind::Linear: {
                if (current.rank() != 2) {
                    throw std::invalid_argument("linear richiede un input a rango 2 [batch, features], trovato " +
                                                 current.shapeToString());
                }
                std::size_t inFeatures = current.dim(1);
                auto outFeatures = static_cast<std::size_t>(op.linearOutFeatures);

                runtime::Tensor weight =
                    randomTensor({inFeatures, outFeatures}, seedFor(seed_, op.output, 0x2545F491U));
                runtime::Tensor bias = randomTensor({outFeatures}, seedFor(seed_, op.output, 0x27D4EB2FU));

                current = linear(current, weight, bias);
                break;
            }
            case ir::OpKind::Silu: current = silu(current); break;
            case ir::OpKind::Relu: current = relu(current); break;
            case ir::OpKind::Gelu: current = gelu(current); break;
        }
    }

    return current;
}

}  // namespace blackforge::backend::cpu
