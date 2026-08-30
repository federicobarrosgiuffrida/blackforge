#pragma once

#include <cstddef>
#include <vector>

#include "blackforge/blackbit/cuda_memory.hpp"
#include "blackforge/runtime/tensor.hpp"

namespace blackforge::blackbit::cuda {

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape, MemoryArena arena = MemoryArena::Temporary);

    static Tensor fromHost(const runtime::Tensor& host, MemoryArena arena = MemoryArena::Activations);
    static Tensor zeros(std::vector<std::size_t> shape, MemoryArena arena = MemoryArena::Temporary);

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    [[nodiscard]] runtime::Tensor toHost() const;
    [[nodiscard]] Tensor clone(MemoryArena arena) const;
    [[nodiscard]] Tensor reshaped(std::vector<std::size_t> shape) &&;

    [[nodiscard]] float* data() { return storage_.as<float>(); }
    [[nodiscard]] const float* data() const { return storage_.as<float>(); }
    [[nodiscard]] const std::vector<std::size_t>& shape() const { return shape_; }
    [[nodiscard]] std::size_t rank() const { return shape_.size(); }
    [[nodiscard]] std::size_t dim(std::size_t axis) const { return shape_.at(axis); }
    [[nodiscard]] std::size_t elementCount() const;
    [[nodiscard]] std::size_t bytes() const { return storage_.bytes(); }
    [[nodiscard]] MemoryArena arena() const { return storage_.arena(); }

private:
    Tensor(std::vector<std::size_t> shape, Buffer storage);

    std::vector<std::size_t> shape_;
    Buffer storage_;
};

class IndexTensor {
public:
    IndexTensor() = default;
    explicit IndexTensor(std::vector<std::size_t> shape, MemoryArena arena = MemoryArena::Temporary);
    static IndexTensor fromHost(std::vector<std::size_t> shape, const std::vector<int>& values,
                                MemoryArena arena = MemoryArena::Activations);

    IndexTensor(const IndexTensor&) = delete;
    IndexTensor& operator=(const IndexTensor&) = delete;
    IndexTensor(IndexTensor&&) noexcept = default;
    IndexTensor& operator=(IndexTensor&&) noexcept = default;

    [[nodiscard]] std::vector<int> toHost() const;
    [[nodiscard]] int* data() { return storage_.as<int>(); }
    [[nodiscard]] const int* data() const { return storage_.as<int>(); }
    [[nodiscard]] const std::vector<std::size_t>& shape() const { return shape_; }
    [[nodiscard]] std::size_t elementCount() const;
    [[nodiscard]] std::size_t bytes() const { return storage_.bytes(); }

private:
    std::vector<std::size_t> shape_;
    Buffer storage_;
};

}  // namespace blackforge::blackbit::cuda
