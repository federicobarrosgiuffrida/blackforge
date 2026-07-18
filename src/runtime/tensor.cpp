#include "blackforge/runtime/tensor.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace blackforge::runtime {

namespace {

std::size_t product(const std::vector<std::size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies<>());
}

}  // namespace

Tensor::Tensor(std::vector<std::size_t> shape, std::vector<float> data)
    : shape_(std::move(shape)), data_(std::move(data)) {
    if (data_.size() != product(shape_)) {
        throw std::invalid_argument("Tensor: la quantita' di dati (" + std::to_string(data_.size()) +
                                     ") non corrisponde alla forma dichiarata (" +
                                     std::to_string(product(shape_)) + " elementi attesi)");
    }
}

Tensor Tensor::zeros(std::vector<std::size_t> shape) { return filled(std::move(shape), 0.0F); }

Tensor Tensor::filled(std::vector<std::size_t> shape, float value) {
    std::size_t count = product(shape);
    return Tensor(std::move(shape), std::vector<float>(count, value));
}

std::size_t Tensor::elementCount() const { return data_.size(); }

float Tensor::min() const {
    if (data_.empty()) {
        throw std::invalid_argument("Tensor::min su un tensore vuoto");
    }
    return *std::min_element(data_.begin(), data_.end());
}

float Tensor::max() const {
    if (data_.empty()) {
        throw std::invalid_argument("Tensor::max su un tensore vuoto");
    }
    return *std::max_element(data_.begin(), data_.end());
}

float Tensor::mean() const {
    if (data_.empty()) {
        throw std::invalid_argument("Tensor::mean su un tensore vuoto");
    }
    double sum = std::accumulate(data_.begin(), data_.end(), 0.0);
    return static_cast<float>(sum / static_cast<double>(data_.size()));
}

std::string Tensor::shapeToString() const {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << shape_[i];
    }
    out << "]";
    return out.str();
}

}  // namespace blackforge::runtime
