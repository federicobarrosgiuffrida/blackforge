#include "blackforge/blackbit/model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "blackforge/blackbit/telemetry.hpp"

namespace blackforge::blackbit {

namespace {

constexpr float kRmsNormEps = 1e-6F;

std::size_t rowsOf(const runtime::Tensor& tensor) {
    std::size_t rows = 1;
    for (std::size_t i = 0; i + 1 < tensor.rank(); ++i) {
        rows *= tensor.dim(i);
    }
    return rows;
}

// Cerca il primo valore non finito, per dire QUALE tensore ha
// destabilizzato l'addestramento invece di limitarsi a segnalare che e'
// successo (requisito 15).
bool findInstability(const runtime::Tensor& tensor, bool& sawNaN, bool& sawInf) {
    for (std::size_t i = 0; i < tensor.elementCount(); ++i) {
        const float value = tensor.at(i);
        if (std::isnan(value)) {
            sawNaN = true;
            return true;
        }
        if (std::isinf(value)) {
            sawInf = true;
            return true;
        }
    }
    return false;
}

}  // namespace

// ---------------------------------------------------------------- RMSNorm

RmsNorm::RmsNorm(std::string name, std::size_t size) : name_(std::move(name)), gamma_(size, 1.0F) {
    if (size == 0) {
        throw std::invalid_argument("RmsNorm '" + name_ + "': dimensione nulla");
    }
    accountedBytes_ = parameterBytes();
    MemoryTelemetry::instance().recordAllocation(MemoryArena::Parameter, accountedBytes_);
}

RmsNorm::~RmsNorm() {
    if (accountedBytes_ != 0) {
        MemoryTelemetry::instance().recordRelease(MemoryArena::Parameter, accountedBytes_);
    }
}

RmsNorm::RmsNorm(RmsNorm&& other) noexcept
    : name_(std::move(other.name_)), gamma_(std::move(other.gamma_)), accountedBytes_(other.accountedBytes_) {
    other.accountedBytes_ = 0;
}

RmsNorm& RmsNorm::operator=(RmsNorm&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (accountedBytes_ != 0) {
        MemoryTelemetry::instance().recordRelease(MemoryArena::Parameter, accountedBytes_);
    }
    name_ = std::move(other.name_);
    gamma_ = std::move(other.gamma_);
    accountedBytes_ = other.accountedBytes_;
    other.accountedBytes_ = 0;
    return *this;
}

runtime::Tensor RmsNorm::forward(const runtime::Tensor& input) const {
    const std::size_t size = gamma_.size();
    if (input.shape().back() != size) {
        throw std::invalid_argument("RmsNorm '" + name_ + "': ultima dimensione incoerente con gamma");
    }
    const std::size_t rows = rowsOf(input);

    std::vector<float> output(input.elementCount());
    for (std::size_t r = 0; r < rows; ++r) {
        const float* row = input.data().data() + r * size;
        // Somma dei quadrati in double: e' una riduzione su hidden
        // termini, il tipo di accumulo dove float32 perde precisione in
        // modo visibile (requisito 15).
        double squared = 0.0;
        for (std::size_t i = 0; i < size; ++i) {
            squared += static_cast<double>(row[i]) * static_cast<double>(row[i]);
        }
        const float inverse =
            1.0F / std::sqrt(static_cast<float>(squared / static_cast<double>(size)) + kRmsNormEps);
        for (std::size_t i = 0; i < size; ++i) {
            output[r * size + i] = row[i] * inverse * gamma_[i];
        }
    }
    return runtime::Tensor(input.shape(), std::move(output));
}

runtime::Tensor RmsNorm::backward(const runtime::Tensor& input, const runtime::Tensor& gradOutput,
                                   GradientSink* sink) const {
    const std::size_t size = gamma_.size();
    const std::size_t rows = rowsOf(input);

    std::vector<float> gradInput(input.elementCount(), 0.0F);
    std::vector<float> gradGamma(size, 0.0F);

    for (std::size_t r = 0; r < rows; ++r) {
        const float* row = input.data().data() + r * size;
        const float* dOut = gradOutput.data().data() + r * size;
        float* dIn = gradInput.data() + r * size;

        double squared = 0.0;
        for (std::size_t i = 0; i < size; ++i) {
            squared += static_cast<double>(row[i]) * static_cast<double>(row[i]);
        }
        const float meanSquare = static_cast<float>(squared / static_cast<double>(size));
        const float inverse = 1.0F / std::sqrt(meanSquare + kRmsNormEps);

        // d/dx_i [ x_i * inv * gamma_i ] tenendo conto che inv dipende
        // da tutta la riga: il termine di correzione e'
        // -(inv^3 / size) * x_i * somma_j (dOut_j * gamma_j * x_j).
        double weighted = 0.0;
        for (std::size_t i = 0; i < size; ++i) {
            gradGamma[i] += dOut[i] * row[i] * inverse;
            weighted += static_cast<double>(dOut[i]) * static_cast<double>(gamma_[i]) * static_cast<double>(row[i]);
        }
        const float factor = static_cast<float>(weighted) * inverse * inverse * inverse / static_cast<float>(size);

        for (std::size_t i = 0; i < size; ++i) {
            dIn[i] = dOut[i] * gamma_[i] * inverse - factor * row[i];
        }
    }

    if (sink != nullptr) {
        sink->consumeDenseGradient(parameterId(), gradGamma.data(), gradGamma.size());
    }
    return runtime::Tensor(input.shape(), std::move(gradInput));
}

// ----------------------------------------------------------------- blocco

BlackBitBlock::BlackBitBlock(const std::string& namePrefix, const BlackBitConfig& config)
    : attentionNorm_(namePrefix + ".attn_norm", config.hiddenSize),
      moeNorm_(namePrefix + ".moe_norm", config.hiddenSize),
      attention_(namePrefix + ".attn", config),
      moe_(namePrefix + ".moe", config) {}

void BlackBitBlock::initialize(unsigned int seed) {
    attention_.initialize(seed);
    moe_.initialize(seed + 100);
}

void BlackBitBlock::setComputeDType(ComputeDType dtype) {
    attention_.setComputeDType(dtype);
    moe_.setComputeDType(dtype);
}

std::size_t BlackBitBlock::parameterBytes() const {
    return attentionNorm_.parameterBytes() + moeNorm_.parameterBytes() + attention_.parameterBytes() +
           moe_.parameterBytes();
}

runtime::Tensor BlackBitBlock::forward(const runtime::Tensor& input, BlackBitBlockCache& cache) const {
    cache.input = input;
    cache.normedForAttention = attentionNorm_.forward(input);
    const runtime::Tensor attended = attention_.forward(cache.normedForAttention, cache.attention);

    std::vector<float> afterAttention(input.elementCount());
    for (std::size_t i = 0; i < afterAttention.size(); ++i) {
        afterAttention[i] = input.at(i) + attended.at(i);
    }
    cache.afterAttention = runtime::Tensor(input.shape(), std::move(afterAttention));

    cache.normedForMoE = moeNorm_.forward(cache.afterAttention);
    const runtime::Tensor mixed = moe_.forward(cache.normedForMoE, cache.moe, cache.routing);

    std::vector<float> output(input.elementCount());
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] = cache.afterAttention.at(i) + mixed.at(i);
    }
    return runtime::Tensor(input.shape(), std::move(output));
}

runtime::Tensor BlackBitBlock::backward(const runtime::Tensor& input, const runtime::Tensor& gradOutput,
                                         const BlackBitBlockCache& cache, GradientSink* sink) const {
    // Ramo MoE: il gradiente si biforca fra il residuo e il blocco.
    const runtime::Tensor gradNormedMoE =
        moe_.backward(cache.normedForMoE, gradOutput, cache.moe, cache.routing, sink);
    const runtime::Tensor gradFromMoENorm = moeNorm_.backward(cache.afterAttention, gradNormedMoE, sink);

    std::vector<float> gradAfterAttention(input.elementCount());
    for (std::size_t i = 0; i < gradAfterAttention.size(); ++i) {
        gradAfterAttention[i] = gradOutput.at(i) + gradFromMoENorm.at(i);
    }
    const runtime::Tensor gradAfterAttentionTensor(input.shape(), gradAfterAttention);

    // Ramo attention, stessa struttura.
    const runtime::Tensor gradNormedAttention =
        attention_.backward(cache.normedForAttention, gradAfterAttentionTensor, cache.attention, sink);
    const runtime::Tensor gradFromAttentionNorm = attentionNorm_.backward(input, gradNormedAttention, sink);

    std::vector<float> gradInput(input.elementCount());
    for (std::size_t i = 0; i < gradInput.size(); ++i) {
        gradInput[i] = gradAfterAttention[i] + gradFromAttentionNorm.at(i);
    }
    return runtime::Tensor(input.shape(), std::move(gradInput));
}

// ---------------------------------------------------------------- modello

double BlackBitStepResult::meanRoutingEntropy() const {
    if (routing.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const MoERoutingStats& stats : routing) {
        total += stats.routingEntropy;
    }
    return total / static_cast<double>(routing.size());
}

double BlackBitStepResult::maxExpertUtilization() const {
    double maximum = 0.0;
    for (const MoERoutingStats& stats : routing) {
        maximum = std::max(maximum, stats.maxUtilization());
    }
    return maximum;
}

std::size_t BlackBitStepResult::droppedAssignments() const {
    std::size_t total = 0;
    for (const MoERoutingStats& stats : routing) {
        total += stats.droppedAssignments;
    }
    return total;
}

BlackBitModel::BlackBitModel(const BlackBitConfig& config, unsigned int seed)
    : config_(config),
      embedding_("embedding", config.hiddenSize, config.vocabSize, config.ternaryGroupSize, 128),
      finalNorm_("final_norm", config.hiddenSize) {
    config.validate();
    if (!config.tieEmbeddings) {
        // Un'embedding non legata raddoppierebbe i 201 M parametri della
        // tabella su BlackBit-9B senza alcun beneficio dimostrato. Il
        // campo esiste nella configurazione perche' i conti di memoria
        // lo prevedano, ma il modello non lo implementa ancora: errore
        // esplicito invece di ignorarlo e costruire in silenzio un
        // modello diverso da quello dichiarato.
        throw std::invalid_argument(
            "BlackBitModel: tie_embeddings=false non e' ancora implementato (la tabella di uscita non legata "
            "raddoppierebbe i parametri dell'embedding)");
    }

    blocks_.reserve(config.numLayers);
    for (std::size_t i = 0; i < config.numLayers; ++i) {
        blocks_.emplace_back("layer" + std::to_string(i), config);
    }

    embedding_.initialize(seed);
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        blocks_[i].initialize(static_cast<unsigned int>(seed + 1000 * (i + 1)));
    }

    options_.vocabChunk = std::min<std::size_t>(options_.vocabChunk, config.vocabSize);
}

void BlackBitModel::setComputeDType(ComputeDType dtype) {
    embedding_.setComputeDType(dtype);
    for (BlackBitBlock& block : blocks_) {
        block.setComputeDType(dtype);
    }
}

const char* activationRecomputeName(ActivationRecompute mode) {
    switch (mode) {
        case ActivationRecompute::None: return "none";
        case ActivationRecompute::PerLayer: return "per-layer";
        case ActivationRecompute::EveryNLayers: return "every-n-layers";
        case ActivationRecompute::FullRecompute: return "full-recompute";
    }
    return "?";
}

void BlackBitModel::setRuntimeOptions(const BlackBitRuntimeOptions& options) {
    if (options.recomputeEveryN == 0) {
        throw std::invalid_argument("BlackBitModel: recomputeEveryN deve essere positivo");
    }
    options_ = options;
    setVocabChunk(options.vocabChunk);
}

void BlackBitModel::setVocabChunk(std::size_t chunk) {
    if (chunk == 0) {
        throw std::invalid_argument("BlackBitModel: vocabChunk deve essere positivo");
    }
    options_.vocabChunk = std::min(chunk, config_.vocabSize);
}

std::size_t BlackBitModel::segmentSize() const {
    switch (options_.recompute) {
        case ActivationRecompute::None:
        case ActivationRecompute::PerLayer: return 1;
        case ActivationRecompute::EveryNLayers: return std::max<std::size_t>(options_.recomputeEveryN, 1);
        case ActivationRecompute::FullRecompute: return std::max<std::size_t>(blocks_.size(), 1);
    }
    return 1;
}

std::size_t BlackBitModel::parameterBytes() const {
    std::size_t total = embedding_.parameterBytes() + finalNorm_.parameterBytes();
    for (const BlackBitBlock& block : blocks_) {
        total += block.parameterBytes();
    }
    return total;
}

runtime::Tensor BlackBitModel::embedTokens(const std::vector<int>& tokenIds, std::size_t batch,
                                            std::size_t seq) const {
    if (tokenIds.size() != batch * seq) {
        throw std::invalid_argument("BlackBitModel: numero di token id incoerente con [batch, seq]");
    }

    const std::size_t hidden = config_.hiddenSize;
    std::vector<float> output(batch * seq * hidden);
    std::vector<float> row(hidden);

    for (std::size_t t = 0; t < tokenIds.size(); ++t) {
        const int id = tokenIds[t];
        if (id < 0 || static_cast<std::size_t>(id) >= config_.vocabSize) {
            throw std::invalid_argument("BlackBitModel: token id " + std::to_string(id) + " fuori dal vocabolario");
        }
        // Una riga per volta: il lookup non dequantizza mai piu' di
        // hidden valori, per quanto grande sia il vocabolario.
        embedding_.weight().dequantizeRows(static_cast<std::size_t>(id), 1, row.data());
        std::copy(row.begin(), row.end(), output.begin() + t * hidden);
    }

    return runtime::Tensor({batch, seq, hidden}, std::move(output));
}

runtime::Tensor BlackBitModel::forwardHidden(const std::vector<int>& tokenIds, std::size_t batch, std::size_t seq,
                                              BlackBitForwardCache& cache) const {
    cache.blocks.resize(blocks_.size());
    runtime::Tensor hidden = embedTokens(tokenIds, batch, seq);
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        hidden = blocks_[i].forward(hidden, cache.blocks[i]);
    }
    cache.preNormHidden = hidden;
    return finalNorm_.forward(cache.preNormHidden);
}

runtime::Tensor BlackBitModel::logits(const runtime::Tensor& hidden) const {
    // embedding_ ha peso [vocab, hidden]: la proiezione di uscita e'
    // letteralmente la stessa matrice del lookup (embedding legate),
    // quindi e' la sua forward().
    const std::size_t tokens = rowsOf(hidden);
    const runtime::Tensor flat({tokens, config_.hiddenSize}, hidden.data());
    return embedding_.forward(flat);
}

BlackBitStepResult BlackBitModel::trainStep(const std::vector<int>& tokenIds, const std::vector<int>& targets,
                                             std::size_t batch, std::size_t seq, GradientSink* sink) {
    if (targets.size() != batch * seq) {
        throw std::invalid_argument("BlackBitModel: numero di target incoerente con [batch, seq]");
    }

    BlackBitStepResult result;
    const std::size_t tokens = batch * seq;
    const std::size_t hidden = config_.hiddenSize;
    const std::size_t vocab = config_.vocabSize;
    const std::size_t chunk = options_.vocabChunk;

    // --- forward, con la politica di ricalcolo scelta ---
    //
    // In modalita' None si conservano tutte le cache dei blocchi. Nelle
    // altre si conservano solo gli INGRESSI ai punti di ripristino, e
    // la cache di un blocco vive il tempo di una singola iterazione:
    // il picco di attivazioni passa da L * A_cache a
    // (L/n) * A_in + n * A_cache.
    const std::size_t segment = segmentSize();

    BlackBitForwardCache cache;
    std::vector<runtime::Tensor> checkpoints;
    runtime::Tensor activations = embedTokens(tokenIds, batch, seq);

    if (options_.recompute == ActivationRecompute::None) {
        cache.blocks.resize(blocks_.size());
        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            activations = blocks_[i].forward(activations, cache.blocks[i]);
            result.routing.push_back(cache.blocks[i].routing);
        }
    } else {
        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            if (i % segment == 0) {
                checkpoints.push_back(activations);
            }
            BlackBitBlockCache scratch;
            activations = blocks_[i].forward(activations, scratch);
            result.routing.push_back(scratch.routing);
        }
        if (blocks_.empty()) {
            checkpoints.push_back(activations);
        }
    }
    cache.preNormHidden = activations;
    const runtime::Tensor finalHidden = finalNorm_.forward(cache.preNormHidden);

    for (const MoERoutingStats& routing : result.routing) {
        result.auxiliaryLoss += routing.loadBalancingLoss;
    }
    if (!result.routing.empty()) {
        result.auxiliaryLoss /= static_cast<float>(result.routing.size());
    }

    if (findInstability(finalHidden, result.sawNaN, result.sawInf)) {
        result.firstUnstableTensor = "final_hidden";
        return result;
    }

    // --- testa di uscita a blocchi di vocabolario ---
    //
    // Materializzare i logit costerebbe tokens * vocab float: 134 MB su
    // BlackBit-9B con seq 512, e altrettanti per il loro gradiente. Qui
    // il vocabolario viene attraversato a blocchi di 'chunk' colonne, e
    // nulla di dimensione vocab viene mai tenuto in vita.
    //
    // Servono TRE passaggi, e l'ordine non e' negoziabile:
    //   A) massimo e somma degli esponenziali per token (la softmax ha
    //      bisogno del denominatore globale prima di poter produrre
    //      qualunque probabilita');
    //   B) gradiente rispetto allo stato nascosto, che deve essere
    //      pronto PRIMA del backward dei blocchi;
    //   C) gradiente della tabella di embedding, che si puo' emettere
    //      solo DOPO il backward dei blocchi, perche' la stessa matrice
    //      riceve anche il gradiente del lookup di ingresso e i due
    //      contributi vanno sommati prima di aggiornare il peso.
    // Il costo e' due GEMM di testa in piu' per passo (~7 % del totale);
    // l'alternativa sarebbe aggiornare l'embedding due volte con
    // normalizzazioni diverse, cioe' un ottimizzatore sbagliato.
    const float* h = finalHidden.data().data();

    std::vector<float> rowMax(tokens, -std::numeric_limits<float>::infinity());
    std::vector<float> rowSum(tokens, 0.0F);
    std::vector<float> targetLogit(tokens, 0.0F);
    std::vector<float> chunkLogits(tokens * chunk);
    std::vector<float> chunkWeights(chunk * hidden);
    const ScopedMemory headScope(MemoryArena::Workspace,
                                  (chunkLogits.size() + chunkWeights.size()) * sizeof(float));

    auto computeChunkLogits = [&](std::size_t first, std::size_t count) {
        embedding_.weight().dequantizeRows(first, count, chunkWeights.data());
        for (std::size_t t = 0; t < tokens; ++t) {
            const float* row = h + t * hidden;
            for (std::size_t c = 0; c < count; ++c) {
                const float* w = chunkWeights.data() + c * hidden;
                double dot = 0.0;
                for (std::size_t d = 0; d < hidden; ++d) {
                    dot += static_cast<double>(row[d]) * static_cast<double>(w[d]);
                }
                chunkLogits[t * chunk + c] = static_cast<float>(dot);
            }
        }
    };

    // Passaggio A: statistiche della softmax, in float32 pieno.
    for (std::size_t first = 0; first < vocab; first += chunk) {
        const std::size_t count = std::min(chunk, vocab - first);
        computeChunkLogits(first, count);

        for (std::size_t t = 0; t < tokens; ++t) {
            if (targets[t] < 0) {
                continue;
            }
            const float* row = chunkLogits.data() + t * chunk;
            for (std::size_t c = 0; c < count; ++c) {
                const float value = row[c];
                if (value > rowMax[t]) {
                    // Riscalatura online della somma accumulata finora,
                    // lo stesso schema dell'attention fusa.
                    rowSum[t] *= std::exp(rowMax[t] - value);
                    rowMax[t] = value;
                }
                rowSum[t] += std::exp(value - rowMax[t]);
            }
            const auto target = static_cast<std::size_t>(targets[t]);
            if (target >= first && target < first + count) {
                targetLogit[t] = row[target - first];
            }
        }
    }

    std::size_t scored = 0;
    double lossSum = 0.0;
    for (std::size_t t = 0; t < tokens; ++t) {
        if (targets[t] < 0) {
            continue;
        }
        if (static_cast<std::size_t>(targets[t]) >= vocab) {
            throw std::invalid_argument("BlackBitModel: target " + std::to_string(targets[t]) +
                                         " fuori dal vocabolario");
        }
        // log-sum-exp stabile: max + log(sum) - logit del bersaglio.
        lossSum += static_cast<double>(rowMax[t]) + std::log(static_cast<double>(rowSum[t])) -
                   static_cast<double>(targetLogit[t]);
        ++scored;
    }
    result.scoredTokens = scored;
    result.loss = scored == 0 ? 0.0F : static_cast<float>(lossSum / static_cast<double>(scored));

    if (std::isnan(result.loss) || std::isinf(result.loss)) {
        result.sawNaN = std::isnan(result.loss);
        result.sawInf = std::isinf(result.loss);
        result.firstUnstableTensor = "cross_entropy";
        return result;
    }
    if (scored == 0 || sink == nullptr) {
        return result;
    }

    const float inverseScored = 1.0F / static_cast<float>(scored);

    // Passaggio B: gradiente rispetto allo stato nascosto.
    std::vector<float> gradHidden(tokens * hidden, 0.0F);
    for (std::size_t first = 0; first < vocab; first += chunk) {
        const std::size_t count = std::min(chunk, vocab - first);
        computeChunkLogits(first, count);

        for (std::size_t t = 0; t < tokens; ++t) {
            if (targets[t] < 0) {
                continue;
            }
            const float* row = chunkLogits.data() + t * chunk;
            float* dh = gradHidden.data() + t * hidden;
            const auto target = static_cast<std::size_t>(targets[t]);

            for (std::size_t c = 0; c < count; ++c) {
                float gradLogit = std::exp(row[c] - rowMax[t]) / rowSum[t];
                if (first + c == target) {
                    gradLogit -= 1.0F;
                }
                gradLogit *= inverseScored;
                if (gradLogit == 0.0F) {
                    continue;
                }
                const float* w = chunkWeights.data() + c * hidden;
                for (std::size_t d = 0; d < hidden; ++d) {
                    dh[d] += gradLogit * w[d];
                }
            }
        }
    }

    // Backward attraverso norm finale e blocchi. Ogni blocco riceve il
    // proprio ingresso dalla cache: e' l'unica attivazione conservata al
    // confine fra layer.
    runtime::Tensor gradFinalHidden({batch, seq, hidden}, std::move(gradHidden));
    runtime::Tensor gradBlockInput = finalNorm_.backward(cache.preNormHidden, gradFinalHidden, sink);

    if (options_.recompute == ActivationRecompute::None) {
        for (std::size_t i = blocks_.size(); i-- > 0;) {
            gradBlockInput = blocks_[i].backward(cache.blocks[i].input, gradBlockInput, cache.blocks[i], sink);
        }
    } else if (options_.recompute == ActivationRecompute::FullRecompute) {
        // Un solo ingresso conservato (l'uscita dell'embedding): per il
        // backward del layer i si rifa' il forward dei layer 0..i-1
        // scartandone le cache, e si conserva solo quella del layer i.
        for (std::size_t i = blocks_.size(); i-- > 0;) {
            runtime::Tensor replay = checkpoints.front();
            for (std::size_t j = 0; j < i; ++j) {
                BlackBitBlockCache scratch;
                replay = blocks_[j].forward(replay, scratch);
            }
            BlackBitBlockCache recomputed;
            (void)blocks_[i].forward(replay, recomputed);
            gradBlockInput = blocks_[i].backward(replay, gradBlockInput, recomputed, sink);
        }
    } else {
        // Segmenti: si ricalcola il forward del segmento conservandone
        // le cache, poi lo si differenzia a ritroso, poi le cache
        // muoiono e si passa al segmento precedente.
        for (std::size_t segmentIndex = checkpoints.size(); segmentIndex-- > 0;) {
            const std::size_t first = segmentIndex * segment;
            const std::size_t last = std::min(first + segment, blocks_.size());

            std::vector<BlackBitBlockCache> segmentCaches(last - first);
            runtime::Tensor replay = checkpoints[segmentIndex];
            for (std::size_t i = first; i < last; ++i) {
                replay = blocks_[i].forward(replay, segmentCaches[i - first]);
            }
            for (std::size_t i = last; i-- > first;) {
                gradBlockInput = blocks_[i].backward(segmentCaches[i - first].input, gradBlockInput,
                                                      segmentCaches[i - first], sink);
            }
        }
    }

    // Passaggio C: gradiente della tabella di embedding, somma del
    // contributo della testa di uscita e di quello del lookup.
    std::vector<float> lookupGradient(tokens * hidden);
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t d = 0; d < hidden; ++d) {
            lookupGradient[t * hidden + d] = gradBlockInput.at(t * hidden + d);
        }
    }

    std::vector<float> weightGradBlock(chunk * hidden);
    const ScopedMemory weightScope(MemoryArena::Gradient, weightGradBlock.size() * sizeof(float));
    GradientLifetimeStats& lifetime = gradientLifetimeStats();

    for (std::size_t first = 0; first < vocab; first += chunk) {
        const std::size_t count = std::min(chunk, vocab - first);
        computeChunkLogits(first, count);

        std::fill(weightGradBlock.begin(),
                   weightGradBlock.begin() + static_cast<std::ptrdiff_t>(count * hidden), 0.0F);

        lifetime.liveBytes += count * hidden * sizeof(float);
        lifetime.cumulativeBytes += count * hidden * sizeof(float);
        ++lifetime.blocksProduced;
        lifetime.peakLiveBytes = std::max(lifetime.peakLiveBytes, lifetime.liveBytes);

        for (std::size_t t = 0; t < tokens; ++t) {
            if (targets[t] >= 0) {
                const float* row = chunkLogits.data() + t * chunk;
                const float* hRow = h + t * hidden;
                const auto target = static_cast<std::size_t>(targets[t]);
                for (std::size_t c = 0; c < count; ++c) {
                    float gradLogit = std::exp(row[c] - rowMax[t]) / rowSum[t];
                    if (first + c == target) {
                        gradLogit -= 1.0F;
                    }
                    gradLogit *= inverseScored;
                    if (gradLogit == 0.0F) {
                        continue;
                    }
                    float* gw = weightGradBlock.data() + c * hidden;
                    for (std::size_t d = 0; d < hidden; ++d) {
                        gw[d] += gradLogit * hRow[d];
                    }
                }
            }

            // Contributo del lookup: il token t ha letto la riga
            // tokenIds[t], quindi quella riga riceve il gradiente
            // dell'ingresso del primo blocco.
            const auto id = static_cast<std::size_t>(tokenIds[t]);
            if (id >= first && id < first + count) {
                float* gw = weightGradBlock.data() + (id - first) * hidden;
                for (std::size_t d = 0; d < hidden; ++d) {
                    gw[d] += lookupGradient[t * hidden + d];
                }
            }
        }

        sink->consumeWeightGradientBlock(embedding_.parameterId(), first, count, weightGradBlock.data());

        lifetime.liveBytes -= count * hidden * sizeof(float);
        ++lifetime.blocksReleased;
    }

    return result;
}

}  // namespace blackforge::blackbit
