#include "blackforge/backend/cpu/autodiff.hpp"

#include <cmath>
#include <stdexcept>

namespace blackforge::backend::cpu {

namespace {

constexpr float kGeluCoeff = 0.044715F;
constexpr float kGeluCoeffDeriv = 3.0F * kGeluCoeff;  // 0.134145
constexpr float kSqrt2OverPi = 0.7978845608F;         // sqrt(2/pi)

Tensor elementwiseGrad(const Tensor& input, const Tensor& gradOutput, float (*derivative)(float)) {
    if (input.shape() != gradOutput.shape()) {
        throw std::invalid_argument("backward elementwise: forme incompatibili " + input.shapeToString() + " e " +
                                     gradOutput.shapeToString());
    }
    std::vector<float> result(input.elementCount());
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = gradOutput.at(i) * derivative(input.at(i));
    }
    return Tensor(input.shape(), std::move(result));
}

}  // namespace

MatmulGrad matmulBackward(const Tensor& a, const Tensor& b, const Tensor& gradOutput) {
    if (a.rank() != 2 || b.rank() != 2 || gradOutput.rank() != 2 || a.dim(1) != b.dim(0) ||
        gradOutput.dim(0) != a.dim(0) || gradOutput.dim(1) != b.dim(1)) {
        throw std::invalid_argument("matmulBackward: forme incompatibili A=" + a.shapeToString() +
                                     " B=" + b.shapeToString() + " gradOutput=" + gradOutput.shapeToString());
    }

    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(1);

    std::vector<float> dA(m * k, 0.0F);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            float sum = 0.0F;
            for (std::size_t j = 0; j < n; ++j) {
                sum += gradOutput.at(i * n + j) * b.at(p * n + j);
            }
            dA[i * k + p] = sum;
        }
    }

    std::vector<float> dB(k * n, 0.0F);
    for (std::size_t p = 0; p < k; ++p) {
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0.0F;
            for (std::size_t i = 0; i < m; ++i) {
                sum += a.at(i * k + p) * gradOutput.at(i * n + j);
            }
            dB[p * n + j] = sum;
        }
    }

    return MatmulGrad{Tensor({m, k}, std::move(dA)), Tensor({k, n}, std::move(dB))};
}

AddBiasGrad addBiasBackward(const Tensor& gradOutput) {
    if (gradOutput.rank() != 2) {
        throw std::invalid_argument("addBiasBackward: atteso un gradiente a rango 2 [batch, features], trovato " +
                                     gradOutput.shapeToString());
    }

    std::size_t batch = gradOutput.dim(0);
    std::size_t features = gradOutput.dim(1);

    std::vector<float> dBias(features, 0.0F);
    for (std::size_t row = 0; row < batch; ++row) {
        for (std::size_t col = 0; col < features; ++col) {
            dBias[col] += gradOutput.at(row * features + col);
        }
    }

    return AddBiasGrad{gradOutput, Tensor({features}, std::move(dBias))};
}

Tensor siluBackward(const Tensor& input, const Tensor& gradOutput) {
    return elementwiseGrad(input, gradOutput, [](float x) {
        float sigmoid = 1.0F / (1.0F + std::exp(-x));
        return sigmoid * (1.0F + x * (1.0F - sigmoid));
    });
}

Tensor reluBackward(const Tensor& input, const Tensor& gradOutput) {
    return elementwiseGrad(input, gradOutput, [](float x) { return x > 0.0F ? 1.0F : 0.0F; });
}

Tensor geluBackward(const Tensor& input, const Tensor& gradOutput) {
    return elementwiseGrad(input, gradOutput, [](float x) {
        float inner = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
        float t = std::tanh(inner);
        float innerDerivative = kSqrt2OverPi * (1.0F + kGeluCoeffDeriv * x * x);
        return 0.5F * (1.0F + t) + 0.5F * x * (1.0F - t * t) * innerDerivative;
    });
}

}  // namespace blackforge::backend::cpu
