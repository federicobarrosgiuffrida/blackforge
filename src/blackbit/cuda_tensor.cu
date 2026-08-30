#include "blackforge/blackbit/cuda_tensor.hpp"

#include <cuda_runtime.h>

#include <numeric>
#include <stdexcept>
#include <utility>

#include "blackforge/backend/cuda/cuda_check.hpp"

namespace blackforge::blackbit::cuda {

namespace {

std::size_t product(const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies<>());
}

}  // namespace

Tensor::Tensor(std::vector<std::size_t> shape, MemoryArena arena)
    : shape_(std::move(shape)), storage_(product(shape_) * sizeof(float), arena) {}

Tensor::Tensor(std::vector<std::size_t> shape, Buffer storage)
    : shape_(std::move(shape)), storage_(std::move(storage)) {}

Tensor Tensor::fromHost(const runtime::Tensor& host, MemoryArena arena) {
    Tensor result(host.shape(), arena);
    if (result.bytes() != 0) {
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(result.data(), host.data().data(), result.bytes(), cudaMemcpyHostToDevice));
    }
    return result;
}

Tensor Tensor::zeros(std::vector<std::size_t> shape, MemoryArena arena) {
    Tensor result(std::move(shape), arena);
    if (result.bytes() != 0) {
        BLACKFORGE_CUDA_CHECK(cudaMemset(result.data(), 0, result.bytes()));
    }
    return result;
}

runtime::Tensor Tensor::toHost() const {
    std::vector<float> values(elementCount());
    if (!values.empty()) {
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(values.data(), data(), bytes(), cudaMemcpyDeviceToHost));
    }
    return runtime::Tensor(shape_, std::move(values));
}

Tensor Tensor::clone(MemoryArena arena) const {
    Tensor result(shape_, arena);
    if (bytes() != 0) {
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(result.data(), data(), bytes(), cudaMemcpyDeviceToDevice));
    }
    return result;
}

Tensor Tensor::reshaped(std::vector<std::size_t> shape) && {
    if (product(shape) != elementCount()) {
        throw std::invalid_argument("CUDA BlackBit Tensor::reshaped: element count differs");
    }
    return Tensor(std::move(shape), std::move(storage_));
}

std::size_t Tensor::elementCount() const { return product(shape_); }

IndexTensor::IndexTensor(std::vector<std::size_t> shape, MemoryArena arena)
    : shape_(std::move(shape)), storage_(product(shape_) * sizeof(int), arena) {}

IndexTensor IndexTensor::fromHost(std::vector<std::size_t> shape, const std::vector<int>& values,
                                  MemoryArena arena) {
    if (product(shape) != values.size()) {
        throw std::invalid_argument("CUDA BlackBit IndexTensor::fromHost: element count differs");
    }
    IndexTensor result(std::move(shape), arena);
    if (!values.empty()) {
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(result.data(), values.data(), result.bytes(), cudaMemcpyHostToDevice));
    }
    return result;
}

std::vector<int> IndexTensor::toHost() const {
    std::vector<int> result(elementCount());
    if (!result.empty()) {
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(result.data(), data(), bytes(), cudaMemcpyDeviceToHost));
    }
    return result;
}

std::size_t IndexTensor::elementCount() const { return product(shape_); }

}  // namespace blackforge::blackbit::cuda
