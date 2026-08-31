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

    template <typename Registry>
    void registerParameters(Registry& registry) {
        registry.registerDense(attentionNormName_, attentionGamma_);
        registry.registerDense(moeNormName_, moeGamma_);
        registry.registerTernary(attention_.queryProjection().name(), attention_.queryProjection().weight());
        registry.registerTernary(attention_.keyProjection().name(), attention_.keyProjection().weight());
        registry.registerTernary(attention_.valueProjection().name(), attention_.valueProjection().weight());
        registry.registerTernary(attention_.outputProjection().name(), attention_.outputProjection().weight());
        registry.registerDense(moe_.routerName(), moe_.routerWeight());
        for (MoEExpert& expert : moe_.experts()) {
            registry.registerTernary(expert.gate().name(), expert.gate().weight());
            registry.registerTernary(expert.up().name(), expert.up().weight());
            registry.registerTernary(expert.down().name(), expert.down().weight());
        }
    }

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

    template <typename Registry>
    void registerParameters(Registry& registry) {
        registry.registerTernary(embedding_.name(), embedding_.weight());
        registry.registerDense(finalNormName_, finalGamma_);
        for (BlackBitBlock& block : blocks_) block.registerParameters(registry);
    }
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
