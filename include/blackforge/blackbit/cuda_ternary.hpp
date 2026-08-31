#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "blackforge/blackbit/cuda_memory.hpp"
#include "blackforge/blackbit/ternary.hpp"

namespace blackforge::blackbit::cuda {

class DecodedTile {
public:
    DecodedTile(std::size_t rows, std::size_t cols);

    [[nodiscard]] float* data() { return storage_.as<float>(); }
    [[nodiscard]] const float* data() const { return storage_.as<float>(); }
    [[nodiscard]] std::size_t rows() const { return rows_; }
    [[nodiscard]] std::size_t cols() const { return cols_; }
    [[nodiscard]] std::size_t bytes() const { return storage_.bytes(); }

private:
    Buffer storage_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
};

// Persistent CUDA representation of a TernaryTensor. The only large device
// buffers are the canonical 5-trits-per-byte words and one FP32 scale per
// group. No decoded full matrix is owned by this class.
class TernaryTensor {
public:
    TernaryTensor() = default;
    explicit TernaryTensor(const blackforge::blackbit::TernaryTensor& host);

    // GPU pack path used by parity tests and small initializers. The int8
    // logical trits are temporary and are discarded as soon as the canonical
    // packed buffer has been produced.
    static TernaryTensor fromTrits(std::vector<std::size_t> shape, const std::vector<std::int8_t>& trits,
                                   const std::vector<float>& scales,
                                   std::size_t groupSize = kDefaultGroupSize);

    TernaryTensor(const TernaryTensor&) = delete;
    TernaryTensor& operator=(const TernaryTensor&) = delete;
    TernaryTensor(TernaryTensor&&) noexcept = default;
    TernaryTensor& operator=(TernaryTensor&&) noexcept = default;

    [[nodiscard]] const std::vector<std::size_t>& shape() const { return shape_; }
    [[nodiscard]] std::size_t rows() const { return rows_; }
    [[nodiscard]] std::size_t rowLength() const { return rowLength_; }
    [[nodiscard]] std::size_t groupSize() const { return groupSize_; }
    [[nodiscard]] std::size_t wordsPerRow() const { return wordsPerRow_; }
    [[nodiscard]] std::size_t groupsPerRow() const { return groupsPerRow_; }
    [[nodiscard]] std::size_t elementCount() const { return rows_ * rowLength_; }
    [[nodiscard]] std::size_t packedByteCount() const { return packed_.bytes(); }
    [[nodiscard]] std::size_t scaleByteCount() const { return scales_.bytes(); }

    [[nodiscard]] const std::uint32_t* packedWords() const { return packed_.as<std::uint32_t>(); }
    [[nodiscard]] std::uint32_t* packedWords() { return packed_.as<std::uint32_t>(); }
    [[nodiscard]] const float* scales() const { return scales_.as<float>(); }
    [[nodiscard]] float* scales() { return scales_.as<float>(); }

    [[nodiscard]] DecodedTile decodeRows(std::size_t firstRow, std::size_t rowCount) const;
    [[nodiscard]] blackforge::blackbit::TernaryTensor download() const;

private:
    TernaryTensor(std::vector<std::size_t> shape, std::size_t groupSize);
    void allocate();

    std::vector<std::size_t> shape_;
    std::size_t rows_ = 0;
    std::size_t rowLength_ = 0;
    std::size_t groupSize_ = kDefaultGroupSize;
    std::size_t wordsPerRow_ = 0;
    std::size_t groupsPerRow_ = 0;
    Buffer packed_;
    Buffer scales_;
};

}  // namespace blackforge::blackbit::cuda
