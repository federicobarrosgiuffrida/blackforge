#include "blackforge/backend/cpu/loss.hpp"

#include <stdexcept>

namespace blackforge::backend::cpu {

LossResult meanSquaredError(const runtime::Tensor& prediction, const runtime::Tensor& target) {
    if (prediction.shape() != target.shape()) {
        throw std::invalid_argument("meanSquaredError: forme incompatibili " + prediction.shapeToString() + " e " +
                                     target.shapeToString());
    }

    std::size_t n = prediction.elementCount();
    double sumSquares = 0.0;
    std::vector<float> grad(n);

    for (std::size_t i = 0; i < n; ++i) {
        float diff = prediction.at(i) - target.at(i);
        sumSquares += static_cast<double>(diff) * static_cast<double>(diff);
        // d/dprediction[i] mean((pred-target)^2) = 2*(pred-target)/n
        grad[i] = 2.0F * diff / static_cast<float>(n);
    }

    float value = static_cast<float>(sumSquares / static_cast<double>(n));
    return LossResult{value, runtime::Tensor(prediction.shape(), std::move(grad))};
}

}  // namespace blackforge::backend::cpu
