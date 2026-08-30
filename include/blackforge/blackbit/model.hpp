#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "blackforge/blackbit/attention.hpp"
#include "blackforge/blackbit/config.hpp"
#include "blackforge/blackbit/gradient.hpp"
#include "blackforge/blackbit/moe.hpp"
#include "blackforge/blackbit/ternary_linear.hpp"
#include "blackforge/runtime/tensor.hpp"

// Il modello BlackBit completo: embedding condivisa ternaria, N blocchi
// (GQA + MoE, pre-norm con residuo), norm finale e testa di uscita
// legata all'embedding.
//
// Tiny, Small, Medium e 9B-A3B sono LA STESSA classe con una
// BlackBitConfig diversa: un test che passa su Tiny esercita
// esattamente il codice del modello da 9 miliardi di parametri
// (requisito 14).

namespace blackforge::blackbit {

// RMSNorm con fattore di scala allenabile (a differenza della rmsnorm
// senza parametri del resto del motore, vedi backend/cpu/ops.hpp).
// Gamma resta DENSA e in piena precisione: sono hidden valori per
// norm, 172 K su tutto BlackBit-9B, e quantizzarli farebbe risparmiare
// 0,002 % di memoria destabilizzando la normalizzazione (requisito 15).
class RmsNorm {
public:
    RmsNorm(std::string name, std::size_t size);

    ~RmsNorm();
    RmsNorm(const RmsNorm&) = delete;
    RmsNorm& operator=(const RmsNorm&) = delete;
    RmsNorm(RmsNorm&& other) noexcept;
    RmsNorm& operator=(RmsNorm&& other) noexcept;

    [[nodiscard]] runtime::Tensor forward(const runtime::Tensor& input) const;
    [[nodiscard]] runtime::Tensor backward(const runtime::Tensor& input, const runtime::Tensor& gradOutput,
                                            GradientSink* sink) const;

    [[nodiscard]] std::vector<float>& gamma() { return gamma_; }
    [[nodiscard]] const std::vector<float>& gamma() const { return gamma_; }
    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] ParameterId parameterId() const { return ParameterId{name_, 1, gamma_.size()}; }
    [[nodiscard]] std::size_t parameterBytes() const { return gamma_.size() * sizeof(float); }

private:
    std::string name_;
    std::vector<float> gamma_;
    std::size_t accountedBytes_ = 0;
};

struct BlackBitBlockCache {
    // Ingresso del blocco: e' l'unica attivazione conservata al confine
    // fra un layer e il successivo, ed e' quella che il requisito 10
    // trasformera' in punto di ricalcolo.
    runtime::Tensor input;
    runtime::Tensor normedForAttention;
    runtime::Tensor afterAttention;  // ingresso del blocco MoE (dopo il residuo dell'attention)
    runtime::Tensor normedForMoE;
    AttentionCache attention;
    MoECache moe;
    MoERoutingStats routing;
};

// Un blocco: pre-norm + attention GQA + residuo, pre-norm + MoE +
// residuo. La stessa struttura di LLaMA, con il feed-forward denso
// sostituito dal blocco a esperti.
class BlackBitBlock {
public:
    BlackBitBlock(const std::string& namePrefix, const BlackBitConfig& config);

    void initialize(unsigned int seed);

    [[nodiscard]] runtime::Tensor forward(const runtime::Tensor& input, BlackBitBlockCache& cache) const;
    [[nodiscard]] runtime::Tensor backward(const runtime::Tensor& input, const runtime::Tensor& gradOutput,
                                            const BlackBitBlockCache& cache, GradientSink* sink) const;

    [[nodiscard]] GqaAttention& attention() { return attention_; }
    [[nodiscard]] MoELayer& moe() { return moe_; }
    [[nodiscard]] RmsNorm& attentionNorm() { return attentionNorm_; }
    [[nodiscard]] RmsNorm& moeNorm() { return moeNorm_; }

    [[nodiscard]] std::size_t parameterBytes() const;
    void setComputeDType(ComputeDType dtype);

private:
    RmsNorm attentionNorm_;
    RmsNorm moeNorm_;
    GqaAttention attention_;
    MoELayer moe_;
};

// Attivazioni conservate da un forward completo.
struct BlackBitForwardCache {
    std::vector<BlackBitBlockCache> blocks;
    // Uscita dell'ultimo blocco, cioe' l'INGRESSO della norm finale:
    // serve al suo backward e non e' ricostruibile da nessuna cache di
    // blocco senza rifare la ricombinazione degli esperti.
    runtime::Tensor preNormHidden;
};

// Esito di un passo di addestramento completo.
struct BlackBitStepResult {
    float loss = 0.0F;            // cross-entropy media sui token non ignorati
    float auxiliaryLoss = 0.0F;   // media della loss di bilanciamento sui layer
    std::size_t scoredTokens = 0;
    std::vector<MoERoutingStats> routing;  // una voce per layer

    // Diagnostica di stabilita' numerica (requisito 15): se compaiono,
    // l'addestramento va fermato, non proseguito con numeri finti.
    bool sawNaN = false;
    bool sawInf = false;
    std::string firstUnstableTensor;

    [[nodiscard]] double meanRoutingEntropy() const;
    [[nodiscard]] double maxExpertUtilization() const;
    [[nodiscard]] std::size_t droppedAssignments() const;
};

class BlackBitModel {
public:
    explicit BlackBitModel(const BlackBitConfig& config, unsigned int seed = 42);

    // token id [batch, seq] come interi. Restituisce lo stato nascosto
    // finale [batch, seq, hidden] (dopo la norm finale).
    [[nodiscard]] runtime::Tensor forwardHidden(const std::vector<int>& tokenIds, std::size_t batch,
                                                 std::size_t seq, BlackBitForwardCache& cache) const;

    // Logit completi [batch*seq, vocab]. MATERIALIZZA vocab valori per
    // token: 134 MB su BlackBit-9B con seq 512, quindi e' per
    // l'inferenza e per i test sui modelli piccoli. L'addestramento usa
    // trainStep(), che non li materializza mai.
    [[nodiscard]] runtime::Tensor logits(const runtime::Tensor& hidden) const;

    // Un passo completo: forward, cross-entropy a blocchi di
    // vocabolario, backward in streaming con consegna immediata dei
    // gradienti al sink. 'targets' ha batch*seq voci; -1 significa
    // "ignora questa posizione" (stessa convenzione di
    // softmaxCrossEntropyMasked nel resto del motore).
    BlackBitStepResult trainStep(const std::vector<int>& tokenIds, const std::vector<int>& targets,
                                  std::size_t batch, std::size_t seq, GradientSink* sink);

    [[nodiscard]] const BlackBitConfig& config() const { return config_; }
    [[nodiscard]] TernaryLinear& embedding() { return embedding_; }
    [[nodiscard]] std::vector<BlackBitBlock>& blocks() { return blocks_; }
    [[nodiscard]] RmsNorm& finalNorm() { return finalNorm_; }

    // Byte realmente posseduti dai parametri: e' il numero che il
    // benchmark riporta, non una stima.
    [[nodiscard]] std::size_t parameterBytes() const;

    // Registra ogni parametro del modello su un sink che sappia
    // aggiornarli (TernarySgdSink e l'ottimizzatore low-rank).
    template <typename Optimizer>
    void registerParameters(Optimizer& optimizer) {
        optimizer.registerTernary(embedding_.name(), embedding_.weight());
        optimizer.registerDense(finalNorm_.name(), finalNorm_.gamma());
        for (BlackBitBlock& block : blocks_) {
            optimizer.registerDense(block.attentionNorm().name(), block.attentionNorm().gamma());
            optimizer.registerDense(block.moeNorm().name(), block.moeNorm().gamma());
            optimizer.registerTernary(block.attention().queryProjection().name(),
                                       block.attention().queryProjection().weight());
            optimizer.registerTernary(block.attention().keyProjection().name(),
                                       block.attention().keyProjection().weight());
            optimizer.registerTernary(block.attention().valueProjection().name(),
                                       block.attention().valueProjection().weight());
            optimizer.registerTernary(block.attention().outputProjection().name(),
                                       block.attention().outputProjection().weight());
            optimizer.registerDense(block.moe().router().name(), block.moe().router().weight());
            for (MoEExpert& expert : block.moe().experts()) {
                optimizer.registerTernary(expert.gate().name(), expert.gate().weight());
                optimizer.registerTernary(expert.up().name(), expert.up().weight());
                optimizer.registerTernary(expert.down().name(), expert.down().weight());
            }
        }
    }

    void setComputeDType(ComputeDType dtype);

    // Token del vocabolario processati insieme dalla testa di uscita.
    // Non cambia il risultato, solo il picco di memoria: 'vocabChunk'
    // valori per token invece di 'vocabSize'.
    void setVocabChunk(std::size_t chunk);
    [[nodiscard]] std::size_t vocabChunk() const { return vocabChunk_; }

private:
    [[nodiscard]] runtime::Tensor embedTokens(const std::vector<int>& tokenIds, std::size_t batch,
                                               std::size_t seq) const;

    BlackBitConfig config_;
    TernaryLinear embedding_;  // [vocab, hidden]: lookup di ingresso E proiezione di uscita
    std::vector<BlackBitBlock> blocks_;
    RmsNorm finalNorm_;
    std::size_t vocabChunk_ = 1024;
};

}  // namespace blackforge::blackbit
