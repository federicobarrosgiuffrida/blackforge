#include "blackforge/blackbit/moe.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "blackforge/backend/cpu/autodiff.hpp"
#include "blackforge/backend/cpu/ops.hpp"
#include "blackforge/backend/cpu/random_init.hpp"
#include "blackforge/blackbit/telemetry.hpp"

namespace blackforge::blackbit {

namespace {

std::size_t rowsOf(const runtime::Tensor& tensor) {
    std::size_t rows = 1;
    for (std::size_t i = 0; i + 1 < tensor.rank(); ++i) {
        rows *= tensor.dim(i);
    }
    return rows;
}

float siluDerivative(float x) {
    const float sigmoid = 1.0F / (1.0F + std::exp(-x));
    return sigmoid * (1.0F + x * (1.0F - sigmoid));
}

}  // namespace

// ---------------------------------------------------------------- esperto

MoEExpert::MoEExpert(const std::string& namePrefix, std::size_t hiddenSize, std::size_t expertHidden,
                      std::size_t groupSize, std::size_t tileRows)
    : gate_(namePrefix + ".gate", hiddenSize, expertHidden, groupSize, tileRows),
      up_(namePrefix + ".up", hiddenSize, expertHidden, groupSize, tileRows),
      down_(namePrefix + ".down", expertHidden, hiddenSize, groupSize, tileRows) {}

void MoEExpert::initialize(unsigned int seed) {
    gate_.initialize(seed);
    up_.initialize(seed + 1);
    down_.initialize(seed + 2);
}

void MoEExpert::setComputeDType(ComputeDType dtype) {
    gate_.setComputeDType(dtype);
    up_.setComputeDType(dtype);
    down_.setComputeDType(dtype);
}

runtime::Tensor MoEExpert::forward(const runtime::Tensor& input) const {
    const runtime::Tensor gate = gate_.forward(input);
    const runtime::Tensor up = up_.forward(input);

    std::vector<float> hidden(gate.elementCount());
    for (std::size_t i = 0; i < hidden.size(); ++i) {
        const float g = gate.at(i);
        hidden[i] = (g / (1.0F + std::exp(-g))) * up.at(i);  // silu(gate) * up
    }

    return down_.forward(runtime::Tensor(gate.shape(), std::move(hidden)));
}

runtime::Tensor MoEExpert::backward(const runtime::Tensor& input, const runtime::Tensor& gradOutput,
                                     GradientSink* sink) const {
    // Ricalcolo (vedi il commento nell'header): gate/up/silu non sono
    // stati conservati dal forward.
    const runtime::Tensor gate = gate_.forward(input);
    const runtime::Tensor up = up_.forward(input);

    std::vector<float> hidden(gate.elementCount());
    for (std::size_t i = 0; i < hidden.size(); ++i) {
        const float g = gate.at(i);
        hidden[i] = (g / (1.0F + std::exp(-g))) * up.at(i);
    }
    const runtime::Tensor hiddenTensor(gate.shape(), hidden);

    const runtime::Tensor gradHidden = down_.backward(hiddenTensor, gradOutput, sink);

    std::vector<float> gradGate(gate.elementCount());
    std::vector<float> gradUp(gate.elementCount());
    for (std::size_t i = 0; i < gradGate.size(); ++i) {
        const float g = gate.at(i);
        const float siluValue = g / (1.0F + std::exp(-g));
        gradGate[i] = gradHidden.at(i) * up.at(i) * siluDerivative(g);
        gradUp[i] = gradHidden.at(i) * siluValue;
    }

    const runtime::Tensor gradFromGate =
        gate_.backward(input, runtime::Tensor(gate.shape(), std::move(gradGate)), sink);
    const runtime::Tensor gradFromUp = up_.backward(input, runtime::Tensor(gate.shape(), std::move(gradUp)), sink);

    std::vector<float> gradInput(input.elementCount());
    for (std::size_t i = 0; i < gradInput.size(); ++i) {
        gradInput[i] = gradFromGate.at(i) + gradFromUp.at(i);
    }
    return runtime::Tensor(input.shape(), std::move(gradInput));
}

// ----------------------------------------------------------------- router

MoERouter::MoERouter(std::string name, std::size_t hiddenSize, std::size_t numExperts)
    : name_(std::move(name)),
      hiddenSize_(hiddenSize),
      numExperts_(numExperts),
      weight_(numExperts * hiddenSize, 0.0F) {
    if (hiddenSize == 0 || numExperts == 0) {
        throw std::invalid_argument("MoERouter '" + name_ + "': dimensioni nulle");
    }
    accountedBytes_ = parameterBytes();
    MemoryTelemetry::instance().recordAllocation(MemoryArena::Parameter, accountedBytes_);
}

MoERouter::~MoERouter() {
    if (accountedBytes_ != 0) {
        MemoryTelemetry::instance().recordRelease(MemoryArena::Parameter, accountedBytes_);
    }
}

MoERouter::MoERouter(MoERouter&& other) noexcept
    : name_(std::move(other.name_)),
      hiddenSize_(other.hiddenSize_),
      numExperts_(other.numExperts_),
      weight_(std::move(other.weight_)),
      accountedBytes_(other.accountedBytes_) {
    other.accountedBytes_ = 0;
}

MoERouter& MoERouter::operator=(MoERouter&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (accountedBytes_ != 0) {
        MemoryTelemetry::instance().recordRelease(MemoryArena::Parameter, accountedBytes_);
    }
    name_ = std::move(other.name_);
    hiddenSize_ = other.hiddenSize_;
    numExperts_ = other.numExperts_;
    weight_ = std::move(other.weight_);
    accountedBytes_ = other.accountedBytes_;
    other.accountedBytes_ = 0;
    return *this;
}

void MoERouter::initialize(unsigned int seed) {
    // Il router parte PICCOLO (ampiezza 1/sqrt(hidden) ridotta di un
    // ulteriore fattore 10): a inizializzazione i logit devono essere
    // quasi uguali, cosi' la softmax e' quasi uniforme e tutti gli
    // esperti ricevono token nei primi passi. Un router inizializzato
    // "forte" sceglie subito sempre gli stessi due esperti e gli altri
    // non ricevono mai un gradiente — il collasso da cui il MoE non si
    // riprende piu'.
    const float amplitude = 0.1F / std::sqrt(static_cast<float>(hiddenSize_));
    const runtime::Tensor values =
        backend::cpu::randomTensor({numExperts_, hiddenSize_}, backend::cpu::seedFor(seed, 0, 0x120DU));
    for (std::size_t i = 0; i < weight_.size(); ++i) {
        weight_[i] = values.at(i) * (amplitude / 0.1F);
    }
}

runtime::Tensor MoERouter::probabilities(const runtime::Tensor& input) const {
    const std::size_t tokens = rowsOf(input);
    if (input.shape().back() != hiddenSize_) {
        throw std::invalid_argument("MoERouter '" + name_ + "': ingresso con dimensione hidden sbagliata");
    }

    std::vector<float> probs(tokens * numExperts_);
    const float* x = input.data().data();

    for (std::size_t t = 0; t < tokens; ++t) {
        const float* row = x + t * hiddenSize_;
        float* out = probs.data() + t * numExperts_;

        for (std::size_t e = 0; e < numExperts_; ++e) {
            const float* w = weight_.data() + e * hiddenSize_;
            // Accumulo in double: la softmax del router decide QUALE
            // esperto vede il token, ed e' l'unico punto del blocco
            // dove un errore di arrotondamento cambia una scelta
            // discreta invece di spostare un valore (requisito 15).
            double sum = 0.0;
            for (std::size_t k = 0; k < hiddenSize_; ++k) {
                sum += static_cast<double>(row[k]) * static_cast<double>(w[k]);
            }
            out[e] = static_cast<float>(sum);
        }

        float maximum = out[0];
        for (std::size_t e = 1; e < numExperts_; ++e) {
            maximum = std::max(maximum, out[e]);
        }
        double total = 0.0;
        for (std::size_t e = 0; e < numExperts_; ++e) {
            out[e] = std::exp(out[e] - maximum);
            total += out[e];
        }
        for (std::size_t e = 0; e < numExperts_; ++e) {
            out[e] = static_cast<float>(out[e] / total);
        }
    }

    std::vector<std::size_t> shape{tokens, numExperts_};
    return runtime::Tensor(std::move(shape), std::move(probs));
}

void MoERouter::backward(const runtime::Tensor& input, const runtime::Tensor& probabilities,
                          const std::vector<float>& gradProbabilities, runtime::Tensor& gradInput,
                          GradientSink* sink) const {
    const std::size_t tokens = rowsOf(input);
    if (gradProbabilities.size() != tokens * numExperts_) {
        throw std::invalid_argument("MoERouter '" + name_ + "': gradiente delle probabilita' di dimensione errata");
    }

    // Il gradiente della matrice di routing e' numExperts * hidden
    // valori: 24 K su BlackBit-9B. E' abbastanza piccolo da essere
    // costruito intero e consegnato in un colpo solo (consumeDenseGradient),
    // a differenza delle matrici degli esperti.
    std::vector<float> gradWeight(weight_.size(), 0.0F);
    const ScopedMemory scope(MemoryArena::Gradient, gradWeight.size() * sizeof(float));

    const float* x = input.data().data();
    float* dx = gradInput.data().data();

    std::vector<float> gradLogits(numExperts_);
    for (std::size_t t = 0; t < tokens; ++t) {
        const float* p = probabilities.data().data() + t * numExperts_;
        const float* dp = gradProbabilities.data() + t * numExperts_;

        // Backward della softmax: dz_e = p_e * (dp_e - somma_j dp_j p_j).
        double weighted = 0.0;
        for (std::size_t e = 0; e < numExperts_; ++e) {
            weighted += static_cast<double>(dp[e]) * static_cast<double>(p[e]);
        }
        for (std::size_t e = 0; e < numExperts_; ++e) {
            gradLogits[e] = static_cast<float>(static_cast<double>(p[e]) * (static_cast<double>(dp[e]) - weighted));
        }

        const float* row = x + t * hiddenSize_;
        float* dxRow = dx + t * hiddenSize_;
        for (std::size_t e = 0; e < numExperts_; ++e) {
            const float g = gradLogits[e];
            if (g == 0.0F) {
                continue;
            }
            const float* w = weight_.data() + e * hiddenSize_;
            float* gw = gradWeight.data() + e * hiddenSize_;
            for (std::size_t k = 0; k < hiddenSize_; ++k) {
                dxRow[k] += g * w[k];
                gw[k] += g * row[k];
            }
        }
    }

    if (sink != nullptr) {
        sink->consumeDenseGradient(parameterId(), gradWeight.data(), gradWeight.size());
    }
}

// --------------------------------------------------------------- metriche

double MoERoutingStats::utilization(std::size_t expert) const {
    const std::size_t routed = assignments - droppedAssignments;
    if (routed == 0) {
        return 0.0;
    }
    return static_cast<double>(tokensPerExpert.at(expert)) / static_cast<double>(routed);
}

double MoERoutingStats::maxUtilization() const {
    double maximum = 0.0;
    for (std::size_t e = 0; e < tokensPerExpert.size(); ++e) {
        maximum = std::max(maximum, utilization(e));
    }
    return maximum;
}

double MoERoutingStats::minUtilization() const {
    if (tokensPerExpert.empty()) {
        return 0.0;
    }
    double minimum = 1.0;
    for (std::size_t e = 0; e < tokensPerExpert.size(); ++e) {
        minimum = std::min(minimum, utilization(e));
    }
    return minimum;
}

std::string MoERoutingStats::toString() const {
    std::ostringstream out;
    out << "token " << tokens << ", assegnazioni " << assignments << " (scartate " << droppedAssignments << ", "
        << (dropRate() * 100.0) << " %)\n";
    out << "  per esperto:";
    for (std::size_t e = 0; e < tokensPerExpert.size(); ++e) {
        out << " " << tokensPerExpert[e];
    }
    out << "\n  utilizzo min/max " << (minUtilization() * 100.0) << " % / " << (maxUtilization() * 100.0)
        << " %, entropia " << routingEntropy << " nat (max "
        << std::log(static_cast<double>(tokensPerExpert.size())) << "), loss di bilanciamento "
        << loadBalancingLoss << "\n";
    return out.str();
}

// ------------------------------------------------------------------ layer

MoELayer::MoELayer(const std::string& namePrefix, const BlackBitConfig& config)
    : config_(config),
      router_(namePrefix + ".router", config.hiddenSize, config.numExperts) {
    config.validate();
    experts_.reserve(config.numExperts);
    for (std::size_t e = 0; e < config.numExperts; ++e) {
        experts_.emplace_back(namePrefix + ".expert" + std::to_string(e), config.hiddenSize, config.expertHidden,
                               config.ternaryGroupSize, 128);
    }
}

void MoELayer::initialize(unsigned int seed) {
    router_.initialize(seed);
    for (std::size_t e = 0; e < experts_.size(); ++e) {
        experts_[e].initialize(static_cast<unsigned int>(seed + 1 + e * 3));
    }
}

void MoELayer::setComputeDType(ComputeDType dtype) {
    for (MoEExpert& expert : experts_) {
        expert.setComputeDType(dtype);
    }
}

std::size_t MoELayer::parameterBytes() const {
    std::size_t total = router_.parameterBytes();
    for (const MoEExpert& expert : experts_) {
        total += expert.parameterBytes();
    }
    return total;
}

std::size_t MoELayer::capacityFor(std::size_t tokens) const {
    const double fair = static_cast<double>(tokens * config_.expertsPerToken) / static_cast<double>(config_.numExperts);
    const auto capacity = static_cast<std::size_t>(std::ceil(fair * static_cast<double>(config_.expertCapacityFactor)));
    // Almeno un token: con batch minuscoli la quota "equa" puo'
    // arrotondare a zero e nessun esperto riceverebbe nulla.
    return std::max<std::size_t>(capacity, 1);
}

runtime::Tensor MoELayer::forward(const runtime::Tensor& input, MoECache& cache, MoERoutingStats& stats) const {
    const std::size_t tokens = rowsOf(input);
    const std::size_t hidden = config_.hiddenSize;
    const std::size_t topK = config_.expertsPerToken;
    const std::size_t numExperts = config_.numExperts;

    if (input.shape().back() != hidden) {
        throw std::invalid_argument("MoELayer: ingresso con dimensione hidden sbagliata");
    }

    cache.tokens = tokens;
    cache.probabilities = router_.probabilities(input);
    cache.expertOfSlot.assign(tokens * topK, -1);
    cache.weightOfSlot.assign(tokens * topK, 0.0F);
    cache.expertOutputOfSlot.assign(tokens * topK * hidden, 0.0F);
    cache.assignmentsPerExpert.assign(numExperts, 0);

    stats = MoERoutingStats{};
    stats.tokensPerExpert.assign(numExperts, 0);
    stats.tokens = tokens;
    stats.assignments = tokens * topK;

    const std::size_t capacity = capacityFor(tokens);

    // --- selezione top-k, deterministica ---
    // Il dispatch e' una lista di indici per esperto: nessun tensore di
    // forma [token, esperti, hidden] viene mai creato.
    std::vector<std::vector<std::uint32_t>> tokensOfExpert(numExperts);
    std::vector<std::vector<std::size_t>> slotOfExpert(numExperts);
    std::vector<std::size_t> order(numExperts);
    std::vector<double> probabilityMass(numExperts, 0.0);

    for (std::size_t t = 0; t < tokens; ++t) {
        const float* p = cache.probabilities.data().data() + t * numExperts;

        double entropy = 0.0;
        for (std::size_t e = 0; e < numExperts; ++e) {
            probabilityMass[e] += p[e];
            if (p[e] > 0.0F) {
                entropy -= static_cast<double>(p[e]) * std::log(static_cast<double>(p[e]));
            }
        }
        stats.routingEntropy += entropy;

        std::iota(order.begin(), order.end(), 0);
        // Ordinamento STABILE con confronto stretto: a parita' di
        // probabilita' vince l'indice piu' basso, quindi due esecuzioni
        // identiche instradano identicamente (modalita' deterministica
        // richiesta dal requisito 4).
        std::stable_sort(order.begin(), order.end(),
                          [&](std::size_t a, std::size_t b) { return p[a] > p[b]; });

        float selectedMass = 0.0F;
        for (std::size_t slot = 0; slot < topK; ++slot) {
            selectedMass += p[order[slot]];
        }
        if (selectedMass <= 0.0F) {
            selectedMass = 1.0F;
        }

        for (std::size_t slot = 0; slot < topK; ++slot) {
            const std::size_t expert = order[slot];
            const std::size_t slotIndex = t * topK + slot;

            if (tokensOfExpert[expert].size() >= capacity) {
                // Capacita' esaurita: l'assegnazione viene scartata. Il
                // token non e' perso (gli altri slot restano validi) ma
                // il contributo di questo esperto sparisce, e la cosa
                // deve comparire nelle metriche invece di essere
                // silenziosa.
                ++stats.droppedAssignments;
                continue;
            }

            cache.expertOfSlot[slotIndex] = static_cast<int>(expert);
            // Rinormalizzazione sui soli esperti scelti (convenzione di
            // Mixtral): i pesi sommano a 1 anche quando il router e'
            // incerto, cosi' la scala dell'uscita non dipende dalla sua
            // sicurezza.
            cache.weightOfSlot[slotIndex] = p[expert] / selectedMass;

            tokensOfExpert[expert].push_back(static_cast<std::uint32_t>(t));
            slotOfExpert[expert].push_back(slotIndex);
            ++stats.tokensPerExpert[expert];
            ++cache.assignmentsPerExpert[expert];
        }
    }

    if (tokens > 0) {
        stats.routingEntropy /= static_cast<double>(tokens);
    }

    // Loss di bilanciamento (Switch Transformer): numExperts * somma_e
    // f_e * P_e. Vale 1 con carico perfetto, numExperts nel collasso.
    const std::size_t routed = stats.assignments - stats.droppedAssignments;
    double auxiliary = 0.0;
    if (routed > 0 && tokens > 0) {
        for (std::size_t e = 0; e < numExperts; ++e) {
            const double fraction = static_cast<double>(stats.tokensPerExpert[e]) / static_cast<double>(routed);
            const double meanProbability = probabilityMass[e] / static_cast<double>(tokens);
            auxiliary += fraction * meanProbability;
        }
        auxiliary *= static_cast<double>(numExperts);
    }
    stats.loadBalancingLoss = static_cast<float>(auxiliary);

    // --- esecuzione per esperto, su matrici compatte ---
    std::vector<float> output(tokens * hidden, 0.0F);
    const float* x = input.data().data();

    for (std::size_t e = 0; e < numExperts; ++e) {
        const std::vector<std::uint32_t>& assigned = tokensOfExpert[e];
        if (assigned.empty()) {
            continue;
        }

        std::vector<float> gathered(assigned.size() * hidden);
        const ScopedMemory scope(MemoryArena::Activation, gathered.size() * sizeof(float));
        for (std::size_t i = 0; i < assigned.size(); ++i) {
            std::copy_n(x + static_cast<std::size_t>(assigned[i]) * hidden, hidden, gathered.begin() + i * hidden);
        }

        const runtime::Tensor expertOutput =
            experts_[e].forward(runtime::Tensor({assigned.size(), hidden}, std::move(gathered)));

        for (std::size_t i = 0; i < assigned.size(); ++i) {
            const std::size_t slotIndex = slotOfExpert[e][i];
            const std::size_t token = assigned[i];
            const float weight = cache.weightOfSlot[slotIndex];

            float* combined = output.data() + token * hidden;
            float* saved = cache.expertOutputOfSlot.data() + slotIndex * hidden;
            for (std::size_t k = 0; k < hidden; ++k) {
                const float value = expertOutput.at(i * hidden + k);
                saved[k] = value;
                combined[k] += weight * value;
            }
        }
    }

    return runtime::Tensor(input.shape(), std::move(output));
}

runtime::Tensor MoELayer::backward(const runtime::Tensor& input, const runtime::Tensor& gradOutput,
                                    const MoECache& cache, const MoERoutingStats& stats,
                                    GradientSink* sink) const {
    const std::size_t tokens = cache.tokens;
    const std::size_t hidden = config_.hiddenSize;
    const std::size_t topK = config_.expertsPerToken;
    const std::size_t numExperts = config_.numExperts;

    if (rowsOf(gradOutput) != tokens || gradOutput.shape().back() != hidden) {
        throw std::invalid_argument("MoELayer: gradiente dell'uscita con forma incoerente con la cache");
    }

    runtime::Tensor gradInput = runtime::Tensor::zeros(input.shape());
    std::vector<float> gradProbabilities(tokens * numExperts, 0.0F);

    const float* dy = gradOutput.data().data();

    // --- gradiente dei pesi di combinazione ---
    // y_token = somma_slot w_slot * uscita_slot, quindi
    // dw_slot = <dy_token, uscita_slot>. Con la rinormalizzazione
    // w_i = p_i / S (S = somma delle p selezionate) si ha
    // dp_i = (dw_i - somma_j dw_j w_j) / S.
    std::vector<float> gradWeightOfSlot(tokens * topK, 0.0F);
    for (std::size_t t = 0; t < tokens; ++t) {
        const float* dyRow = dy + t * hidden;
        for (std::size_t slot = 0; slot < topK; ++slot) {
            const std::size_t slotIndex = t * topK + slot;
            if (cache.expertOfSlot[slotIndex] < 0) {
                continue;
            }
            const float* saved = cache.expertOutputOfSlot.data() + slotIndex * hidden;
            double dot = 0.0;
            for (std::size_t k = 0; k < hidden; ++k) {
                dot += static_cast<double>(dyRow[k]) * static_cast<double>(saved[k]);
            }
            gradWeightOfSlot[slotIndex] = static_cast<float>(dot);
        }
    }

    for (std::size_t t = 0; t < tokens; ++t) {
        const float* p = cache.probabilities.data().data() + t * numExperts;

        float selectedMass = 0.0F;
        double correction = 0.0;
        for (std::size_t slot = 0; slot < topK; ++slot) {
            const std::size_t slotIndex = t * topK + slot;
            const int expert = cache.expertOfSlot[slotIndex];
            if (expert < 0) {
                continue;
            }
            selectedMass += p[static_cast<std::size_t>(expert)];
            correction += static_cast<double>(gradWeightOfSlot[slotIndex]) *
                          static_cast<double>(cache.weightOfSlot[slotIndex]);
        }
        if (selectedMass <= 0.0F) {
            continue;
        }

        for (std::size_t slot = 0; slot < topK; ++slot) {
            const std::size_t slotIndex = t * topK + slot;
            const int expert = cache.expertOfSlot[slotIndex];
            if (expert < 0) {
                continue;
            }
            gradProbabilities[t * numExperts + static_cast<std::size_t>(expert)] +=
                static_cast<float>((static_cast<double>(gradWeightOfSlot[slotIndex]) - correction) /
                                    static_cast<double>(selectedMass));
        }
    }

    // --- gradiente della loss ausiliaria rispetto alle probabilita' ---
    // L_aux = numExperts * somma_e f_e * P_e, con P_e = media_t p[t, e].
    // f_e conta assegnazioni (discreto, non differenziabile): il
    // gradiente passa solo per P_e, che e' esattamente il modo in cui la
    // loss di Switch Transformer spinge il router verso il
    // bilanciamento.
    const std::size_t routed = stats.assignments - stats.droppedAssignments;
    if (config_.routerAuxLossWeight > 0.0F && routed > 0 && tokens > 0) {
        for (std::size_t e = 0; e < numExperts; ++e) {
            const double fraction = static_cast<double>(stats.tokensPerExpert[e]) / static_cast<double>(routed);
            const double gradient = static_cast<double>(config_.routerAuxLossWeight) *
                                     static_cast<double>(numExperts) * fraction / static_cast<double>(tokens);
            for (std::size_t t = 0; t < tokens; ++t) {
                gradProbabilities[t * numExperts + e] += static_cast<float>(gradient);
            }
        }
    }

    // --- backward degli esperti, di nuovo su matrici compatte ---
    for (std::size_t e = 0; e < numExperts; ++e) {
        std::vector<std::uint32_t> assigned;
        std::vector<std::size_t> slots;
        assigned.reserve(cache.assignmentsPerExpert[e]);
        slots.reserve(cache.assignmentsPerExpert[e]);

        for (std::size_t t = 0; t < tokens; ++t) {
            for (std::size_t slot = 0; slot < topK; ++slot) {
                const std::size_t slotIndex = t * topK + slot;
                if (cache.expertOfSlot[slotIndex] == static_cast<int>(e)) {
                    assigned.push_back(static_cast<std::uint32_t>(t));
                    slots.push_back(slotIndex);
                }
            }
        }
        if (assigned.empty()) {
            continue;
        }

        std::vector<float> gathered(assigned.size() * hidden);
        std::vector<float> gathedGrad(assigned.size() * hidden);
        const ScopedMemory scope(MemoryArena::Activation, gathered.size() * sizeof(float));

        const float* x = input.data().data();
        for (std::size_t i = 0; i < assigned.size(); ++i) {
            const std::size_t token = assigned[i];
            const float weight = cache.weightOfSlot[slots[i]];
            std::copy_n(x + token * hidden, hidden, gathered.begin() + i * hidden);
            for (std::size_t k = 0; k < hidden; ++k) {
                gathedGrad[i * hidden + k] = weight * dy[token * hidden + k];
            }
        }

        const runtime::Tensor gatheredInput({assigned.size(), hidden}, std::move(gathered));
        const runtime::Tensor expertGradInput = experts_[e].backward(
            gatheredInput, runtime::Tensor({assigned.size(), hidden}, std::move(gathedGrad)), sink);

        for (std::size_t i = 0; i < assigned.size(); ++i) {
            const std::size_t token = assigned[i];
            for (std::size_t k = 0; k < hidden; ++k) {
                gradInput.at(token * hidden + k) += expertGradInput.at(i * hidden + k);
            }
        }
    }

    router_.backward(input, cache.probabilities, gradProbabilities, gradInput, sink);

    return gradInput;
}

}  // namespace blackforge::blackbit
