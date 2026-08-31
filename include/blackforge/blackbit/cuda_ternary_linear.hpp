#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include "blackforge/blackbit/cuda_tensor.hpp"
#include "blackforge/blackbit/cuda_ternary.hpp"
#include "blackforge/blackbit/gradient.hpp"
#include "blackforge/blackbit/ternary_linear.hpp"

namespace blackforge::blackbit::cuda {

class GradientSink {
public:
    virtual ~GradientSink() = default;
    virtual void consumeWeightGradientBlock(const ParameterId& id, std::size_t firstRow, std::size_t rowCount,
                                             const float* deviceBlock) = 0;
    virtual void consumeDenseGradient(const ParameterId&, const float*, std::size_t) {
        throw std::invalid_argument("CUDA BlackBit GradientSink: dense gradients are not supported by this sink");
    }
};

struct GradientLifetimeStats {
    std::size_t liveBytes = 0;
    std::size_t peakLiveBytes = 0;
    std::size_t cumulativeBytes = 0;
    std::size_t blocksProduced = 0;
    std::size_t blocksReleased = 0;

    [[nodiscard]] double reuseRatio() const {
        return peakLiveBytes == 0 ? 0.0 : static_cast<double>(cumulativeBytes) / static_cast<double>(peakLiveBytes);
    }
};

GradientLifetimeStats& gradientLifetimeStats();
void resetGradientLifetimeStats();

struct TernaryLinearMetrics {
    double decodeMs = 0.0;
    double gemmMs = 0.0;
    double totalMs = 0.0;
    std::size_t packedBytesRead = 0;
    std::size_t decodedBytesWritten = 0;
    std::size_t dequantWorkspaceBytes = 0;
    std::size_t cublasWorkspaceBytes = 0;

    [[nodiscard]] double decodeGigabytesPerSecond() const {
        return decodeMs == 0.0 ? 0.0 : static_cast<double>(decodedBytesWritten) / 1.0e6 / decodeMs;
    }
};

class TernaryLinear {
public:
    explicit TernaryLinear(const blackforge::blackbit::TernaryLinear& cpuReference);
    TernaryLinear(std::string name, std::size_t inFeatures, std::size_t outFeatures,
                  blackforge::blackbit::TernaryTensor hostWeight, std::size_t tileRows = 128);

    TernaryLinear(const TernaryLinear&) = delete;
    TernaryLinear& operator=(const TernaryLinear&) = delete;
    TernaryLinear(TernaryLinear&&) noexcept = default;
    TernaryLinear& operator=(TernaryLinear&&) noexcept = default;

    [[nodiscard]] Tensor forward(const Tensor& input) const;
    [[nodiscard]] Tensor backward(const Tensor& input, const Tensor& gradOutput, GradientSink* sink) const;

    // Row-range primitives for the tied embedding/output head. They keep
    // vocabulary work chunk-local; no [tokens, vocab] tensor is created.
    [[nodiscard]] Tensor forwardRows(const Tensor& input, std::size_t firstRow, std::size_t rowCount) const;
    [[nodiscard]] Tensor backwardInputRows(const Tensor& gradOutput, std::size_t firstRow) const;
    [[nodiscard]] Tensor weightGradientRows(const Tensor& input, const Tensor& gradOutput,
                                            std::size_t firstRow) const;

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] std::size_t inFeatures() const { return inFeatures_; }
    [[nodiscard]] std::size_t outFeatures() const { return outFeatures_; }
    [[nodiscard]] std::size_t tileRows() const { return tileRows_; }
    [[nodiscard]] ParameterId parameterId() const { return ParameterId{name_, outFeatures_, inFeatures_}; }
    [[nodiscard]] const cuda::TernaryTensor& weight() const { return weight_; }
    [[nodiscard]] cuda::TernaryTensor& weight() { return weight_; }
    [[nodiscard]] const TernaryLinearMetrics& metrics() const { return metrics_; }

private:
    std::string name_;
    std::size_t inFeatures_ = 0;
    std::size_t outFeatures_ = 0;
    std::size_t tileRows_ = 128;
    cuda::TernaryTensor weight_;
    mutable TernaryLinearMetrics metrics_;
};

}  // namespace blackforge::blackbit::cuda
