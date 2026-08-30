#pragma once

#include <string>
#include <vector>

#include "blackforge/blackbit/cuda_ops.hpp"
#include "blackforge/blackbit/cuda_ternary_linear.hpp"
#include "blackforge/blackbit/moe.hpp"

namespace blackforge::blackbit::cuda {

class MoEExpert {
public:
    explicit MoEExpert(blackforge::blackbit::MoEExpert& cpuReference);

    [[nodiscard]] Tensor forward(const Tensor& input) const;
    [[nodiscard]] Tensor backward(const Tensor& input, const Tensor& gradOutput, GradientSink* sink) const;

    [[nodiscard]] TernaryLinear& gate() { return gate_; }
    [[nodiscard]] TernaryLinear& up() { return up_; }
    [[nodiscard]] TernaryLinear& down() { return down_; }

private:
    TernaryLinear gate_;
    TernaryLinear up_;
    TernaryLinear down_;
};

struct MoECache {
    Tensor probabilities;
    IndexTensor expertOfSlot;
    Tensor weightOfSlot;
    Tensor expertOutputOfSlot;
    IndexTensor tokensOfExpert;
    IndexTensor slotsOfExpert;
    IndexTensor assignmentsPerExpert;
    std::vector<int> hostAssignmentsPerExpert;
    std::size_t tokens = 0;
    std::size_t capacity = 0;
};

class MoELayer {
public:
    MoELayer(blackforge::blackbit::MoELayer& cpuReference, const BlackBitConfig& config);

    [[nodiscard]] Tensor forward(const Tensor& input, MoECache& cache, MoERoutingStats& stats) const;
    [[nodiscard]] Tensor backward(const Tensor& input, const Tensor& gradOutput, const MoECache& cache,
                                  const MoERoutingStats& stats, GradientSink* sink) const;

    [[nodiscard]] Tensor& routerWeight() { return routerWeight_; }
    [[nodiscard]] const std::string& routerName() const { return routerName_; }
    [[nodiscard]] ParameterId routerParameterId() const {
        return ParameterId{routerName_, config_.numExperts, config_.hiddenSize};
    }
    [[nodiscard]] std::vector<MoEExpert>& experts() { return experts_; }
    [[nodiscard]] std::size_t capacityFor(std::size_t tokens) const;

private:
    BlackBitConfig config_;
    std::string routerName_;
    Tensor routerWeight_;
    std::vector<MoEExpert> experts_;
};

}  // namespace blackforge::blackbit::cuda
