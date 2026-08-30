#pragma once

#include <string>
#include <vector>

#include "blackforge/blackbit/cuda_attention.hpp"
#include "blackforge/blackbit/cuda_low_rank_optimizer.hpp"
#include "blackforge/blackbit/cuda_moe.hpp"
#include "blackforge/blackbit/model.hpp"

namespace blackforge::blackbit::cuda {

struct BlockCache {
    Tensor normedForAttention;
    RmsNormCache attentionNorm;
    AttentionCache attention;
    Tensor afterAttention;
    Tensor normedForMoE;
    RmsNormCache moeNorm;
    MoECache moe;
    MoERoutingStats routing;
};

class BlackBitBlock {
public:
    BlackBitBlock(blackforge::blackbit::BlackBitBlock& cpuReference, const BlackBitConfig& config);

    [[nodiscard]] Tensor forward(const Tensor& input, BlockCache& cache) const;
    [[nodiscard]] Tensor backward(const Tensor& input, const Tensor& gradOutput,
                                  const BlockCache& cache, GradientSink* sink) const;

    void registerParameters(LowRankProjectedOptimizer& optimizer);

private:
    std::string attentionNormName_;
    std::string moeNormName_;
    Tensor attentionGamma_;
    Tensor moeGamma_;
    GqaAttention attention_;
    MoELayer moe_;
};

struct ForwardCache {
    std::vector<Tensor> blockInputs;
    std::vector<BlockCache> blocks;
    Tensor preNormHidden;
    RmsNormCache finalNorm;
};

class BlackBitModel {
public:
    explicit BlackBitModel(blackforge::blackbit::BlackBitModel& cpuReference);

    [[nodiscard]] Tensor forwardHidden(const std::vector<int>& tokenIds, std::size_t batch,
                                       std::size_t seq, ForwardCache& cache) const;
    BlackBitStepResult trainStep(const std::vector<int>& tokenIds, const std::vector<int>& targets,
                                 std::size_t batch, std::size_t seq, GradientSink* sink);

    void registerParameters(LowRankProjectedOptimizer& optimizer);
    void setRuntimeOptions(const BlackBitRuntimeOptions& options);
    void setVocabChunk(std::size_t chunk);

    [[nodiscard]] const BlackBitConfig& config() const { return config_; }
    [[nodiscard]] TernaryLinear& embedding() { return embedding_; }
    [[nodiscard]] std::vector<BlackBitBlock>& blocks() { return blocks_; }
    [[nodiscard]] Tensor& finalGamma() { return finalGamma_; }
    [[nodiscard]] const BlackBitRuntimeOptions& runtimeOptions() const { return options_; }

private:
    [[nodiscard]] Tensor embedTokens(const IndexTensor& tokenIds) const;

    BlackBitConfig config_;
    TernaryLinear embedding_;
    std::vector<BlackBitBlock> blocks_;
    std::string finalNormName_;
    Tensor finalGamma_;
    BlackBitRuntimeOptions options_;
};

}  // namespace blackforge::blackbit::cuda
