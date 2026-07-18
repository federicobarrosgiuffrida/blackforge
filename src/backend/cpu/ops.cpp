#include "blackforge/backend/cpu/ops.hpp"

#include <cmath>
#include <stdexcept>

namespace blackforge::backend::cpu {

namespace {

constexpr float kGeluCoeff = 0.044715F;
constexpr float kSqrt2OverPi = 0.7978845608F;  // sqrt(2/pi)

void requireSameShape(const Tensor& a, const Tensor& b, const char* opName) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(std::string(opName) + ": forme incompatibili " + a.shapeToString() + " e " +
                                     b.shapeToString());
    }
}

Tensor elementwise(const Tensor& input, float (*fn)(float)) {
    std::vector<float> result(input.elementCount());
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = fn(input.at(i));
    }
    return Tensor(input.shape(), std::move(result));
}

}  // namespace

Tensor add(const Tensor& a, const Tensor& b) {
    requireSameShape(a, b, "add");
    std::vector<float> result(a.elementCount());
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = a.at(i) + b.at(i);
    }
    return Tensor(a.shape(), std::move(result));
}

Tensor addBias(const Tensor& input, const Tensor& bias) {
    if (input.rank() != 2 || bias.rank() != 1 || input.dim(1) != bias.dim(0)) {
        throw std::invalid_argument("addBias: attesi input [batch, features] e bias [features] con features "
                                     "corrispondenti, trovati " +
                                     input.shapeToString() + " e " + bias.shapeToString());
    }

    std::size_t batch = input.dim(0);
    std::size_t features = input.dim(1);
    std::vector<float> result(input.elementCount());

    for (std::size_t row = 0; row < batch; ++row) {
        for (std::size_t col = 0; col < features; ++col) {
            std::size_t idx = row * features + col;
            result[idx] = input.at(idx) + bias.at(col);
        }
    }

    return Tensor(input.shape(), std::move(result));
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.rank() != 2 || b.rank() != 2 || a.dim(1) != b.dim(0)) {
        throw std::invalid_argument("matmul: forme incompatibili " + a.shapeToString() + " x " + b.shapeToString());
    }

    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(1);

    std::vector<float> result(m * n, 0.0F);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            float aVal = a.at(i * k + p);
            for (std::size_t j = 0; j < n; ++j) {
                result[i * n + j] += aVal * b.at(p * n + j);
            }
        }
    }

    return Tensor({m, n}, std::move(result));
}

Tensor silu(const Tensor& input) {
    return elementwise(input, [](float x) { return x / (1.0F + std::exp(-x)); });
}

Tensor relu(const Tensor& input) {
    return elementwise(input, [](float x) { return x > 0.0F ? x : 0.0F; });
}

Tensor gelu(const Tensor& input) {
    return elementwise(input, [](float x) {
        float inner = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
        return 0.5F * x * (1.0F + std::tanh(inner));
    });
}

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    return addBias(matmul(input, weight), bias);
}

}  // namespace blackforge::backend::cpu
