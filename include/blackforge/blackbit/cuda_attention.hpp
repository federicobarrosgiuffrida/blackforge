#pragma once

#include "blackforge/blackbit/attention.hpp"
#include "blackforge/blackbit/cuda_ops.hpp"
#include "blackforge/blackbit/cuda_ternary_linear.hpp"

namespace blackforge::blackbit::cuda {

struct AttentionCache {
    Tensor query;
    Tensor key;
    Tensor value;
    Tensor attnOutput;
    Tensor rowMax;
    Tensor rowSum;
};

class GqaAttention {
public:
    GqaAttention(blackforge::blackbit::GqaAttention& cpuReference, const BlackBitConfig& config);

    GqaAttention(const GqaAttention&) = delete;
    GqaAttention& operator=(const GqaAttention&) = delete;
    GqaAttention(GqaAttention&&) noexcept = default;
    GqaAttention& operator=(GqaAttention&&) noexcept = default;

    [[nodiscard]] Tensor forward(const Tensor& input, AttentionCache& cache) const;
    [[nodiscard]] Tensor backward(const Tensor& input, const Tensor& gradOutput,
                                  const AttentionCache& cache, GradientSink* sink) const;

    [[nodiscard]] TernaryLinear& queryProjection() { return q_; }
    [[nodiscard]] TernaryLinear& keyProjection() { return k_; }
    [[nodiscard]] TernaryLinear& valueProjection() { return v_; }
    [[nodiscard]] TernaryLinear& outputProjection() { return o_; }

private:
    BlackBitConfig config_;
    TernaryLinear q_;
    TernaryLinear k_;
    TernaryLinear v_;
    TernaryLinear o_;
};

}  // namespace blackforge::blackbit::cuda
