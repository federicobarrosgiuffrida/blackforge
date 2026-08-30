#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "blackforge/blackbit/config.hpp"
#include "blackforge/blackbit/gradient.hpp"
#include "blackforge/blackbit/ternary_linear.hpp"
#include "blackforge/runtime/tensor.hpp"

// Blocco feed-forward Mixture-of-Experts di BlackBit (requisito 4).
//
//     input -> router -> top-k -> esperti selezionati -> combinazione pesata
//
// COSA NON VIENE MAI ALLOCATO
//
// L'implementazione ingenua di un MoE costruisce una maschera densa
// [token, esperti, hidden] e moltiplica tutto per zero tranne le
// posizioni scelte. Per BlackBit-9B con seq 512 sarebbero
// 512 * 8 * 3072 * 4 B = 50 MB per layer, 1,4 GB per il modello, per
// rappresentare un'informazione che sta in due interi per token. Qui il
// dispatch e' una LISTA DI INDICI per esperto: la memoria e'
// proporzionale ai token realmente instradati, e ogni esperto vede una
// matrice [token_suoi, hidden] compatta — che e' anche l'unica forma su
// cui un GEMM sia efficiente.
//
// PRECISIONE
//
// Il router resta denso e in float32 (BF16 sul percorso GPU): sono
// hidden * numExperts pesi, 24 K per layer contro 292 M di esperti, e
// sono il punto piu' sensibile del blocco — un errore di
// quantizzazione qui cambia QUALE esperto viene scelto, non di quanto
// contribuisce. Gli esperti, che sono il 99,99 % dei parametri, sono
// interamente ternari.

namespace blackforge::blackbit {

// Un esperto: SwiGLU con le tre proiezioni in ternario impacchettato.
//
//     y = down( silu(gate(x)) * up(x) )
class MoEExpert {
public:
    MoEExpert(const std::string& namePrefix, std::size_t hiddenSize, std::size_t expertHidden,
               std::size_t groupSize, std::size_t tileRows);

    void initialize(unsigned int seed);

    [[nodiscard]] runtime::Tensor forward(const runtime::Tensor& input) const;

    // Il backward RICALCOLA gate/up/silu dall'ingresso invece di
    // leggerli da una cache: sono 2 * token * expertHidden float per
    // esperto (8 MB per layer su BlackBit-9B con seq 512, 230 MB sul
    // modello) contro il costo di due GEMM in piu'. Su un motore che
    // esiste per stare in 8 GB, il baratto e' ovvio — ed e' la stessa
    // politica che il requisito 10 generalizza a tutto il modello.
    [[nodiscard]] runtime::Tensor backward(const runtime::Tensor& input, const runtime::Tensor& gradOutput,
                                            GradientSink* sink) const;

    [[nodiscard]] const TernaryLinear& gate() const { return gate_; }
    [[nodiscard]] const TernaryLinear& up() const { return up_; }
    [[nodiscard]] const TernaryLinear& down() const { return down_; }
    [[nodiscard]] TernaryLinear& gate() { return gate_; }
    [[nodiscard]] TernaryLinear& up() { return up_; }
    [[nodiscard]] TernaryLinear& down() { return down_; }

    [[nodiscard]] std::size_t parameterBytes() const {
        return gate_.parameterBytes() + up_.parameterBytes() + down_.parameterBytes();
    }

    void setComputeDType(ComputeDType dtype);

private:
    TernaryLinear gate_;
    TernaryLinear up_;
    TernaryLinear down_;
};

// Metriche di routing di un singolo forward (requisito 4). Non sono
// diagnostica opzionale: un MoE che instrada l'85 % dei token allo
// stesso esperto ha, di fatto, un ottavo dei parametri che dichiara, e
// senza queste metriche il fenomeno e' invisibile finche' non si guarda
// la loss finale.
struct MoERoutingStats {
    std::vector<std::size_t> tokensPerExpert;
    std::size_t tokens = 0;
    std::size_t assignments = 0;      // token * expertsPerToken previsti
    std::size_t droppedAssignments = 0;  // scartati per capacita' esaurita

    // Entropia media della distribuzione del router per token, in nat.
    // Il massimo e' log(numExperts) (router indeciso), il minimo 0
    // (router certo). Serve a distinguere "collassato su un esperto"
    // (entropia bassa E utilizzo squilibrato) da "ancora indeciso"
    // (entropia alta a inizio addestramento).
    double routingEntropy = 0.0;

    // Loss ausiliaria di bilanciamento (Switch Transformer, Fedus et
    // al. 2021): numExperts * somma_e f_e * P_e, con f_e frazione di
    // assegnazioni all'esperto e e P_e probabilita' media che il router
    // gli assegna. Vale 1 a carico perfettamente bilanciato e cresce
    // fino a numExperts nel collasso totale.
    float loadBalancingLoss = 0.0F;

    [[nodiscard]] double utilization(std::size_t expert) const;
    [[nodiscard]] double maxUtilization() const;
    [[nodiscard]] double minUtilization() const;
    [[nodiscard]] double dropRate() const {
        return assignments == 0 ? 0.0 : static_cast<double>(droppedAssignments) / static_cast<double>(assignments);
    }

    [[nodiscard]] std::string toString() const;
};

// Router: una proiezione densa hidden -> numExperts, softmax, selezione
// dei top-k con rinormalizzazione dei pesi (convenzione di Mixtral: i
// pesi dei soli esperti scelti sommano a 1, cosi' la scala dell'uscita
// non dipende da quanto il router fosse sicuro).
//
// DETERMINISTICO per costruzione: nessun rumore di esplorazione, e le
// parita' fra logit uguali si rompono sull'indice piu' basso. Due
// esecuzioni con gli stessi pesi instradano gli stessi token.
class MoERouter {
public:
    MoERouter(std::string name, std::size_t hiddenSize, std::size_t numExperts);

    // Stessa contabilita' di TernaryLinear: il peso e' registrato nella
    // telemetria alla costruzione e scaricato alla distruzione.
    ~MoERouter();
    MoERouter(const MoERouter&) = delete;
    MoERouter& operator=(const MoERouter&) = delete;
    MoERouter(MoERouter&& other) noexcept;
    MoERouter& operator=(MoERouter&& other) noexcept;

    void initialize(unsigned int seed);

    // probs [tokens, numExperts]: softmax dei logit, in float32 anche
    // quando il resto calcola in precisione ridotta (requisito 15).
    [[nodiscard]] runtime::Tensor probabilities(const runtime::Tensor& input) const;

    // Aggiunge a 'gradInput' il gradiente rispetto all'ingresso e
    // consegna al sink il gradiente della matrice di routing, dato il
    // gradiente rispetto alle probabilita'.
    void backward(const runtime::Tensor& input, const runtime::Tensor& probabilities,
                   const std::vector<float>& gradProbabilities, runtime::Tensor& gradInput,
                   GradientSink* sink) const;

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] std::vector<float>& weight() { return weight_; }
    [[nodiscard]] const std::vector<float>& weight() const { return weight_; }
    [[nodiscard]] ParameterId parameterId() const { return ParameterId{name_, numExperts_, hiddenSize_}; }
    [[nodiscard]] std::size_t parameterBytes() const { return weight_.size() * sizeof(float); }

private:
    std::string name_;
    std::size_t hiddenSize_;
    std::size_t numExperts_;
    std::vector<float> weight_;  // [numExperts, hiddenSize], layout come TernaryLinear
    std::size_t accountedBytes_ = 0;
};

// Stato del forward necessario al backward. Contiene SOLO quantita'
// piccole: probabilita' [token, esperti], indici e pesi per slot
// [token * topK], e le uscite degli esperti per slot
// [token * topK, hidden] (che servono al gradiente dei pesi di
// combinazione). Nessun tensore di forma [token, esperti, hidden].
struct MoECache {
    runtime::Tensor probabilities;        // [tokens, numExperts]
    std::vector<int> expertOfSlot;        // tokens * topK, -1 se scartato
    std::vector<float> weightOfSlot;      // tokens * topK
    std::vector<float> expertOutputOfSlot;  // tokens * topK * hidden
    std::vector<std::size_t> assignmentsPerExpert;
    std::size_t tokens = 0;
};

class MoELayer {
public:
    MoELayer(const std::string& namePrefix, const BlackBitConfig& config);

    void initialize(unsigned int seed);

    // input [..., hidden] -> output [..., hidden]. Popola 'cache' e
    // 'stats'.
    [[nodiscard]] runtime::Tensor forward(const runtime::Tensor& input, MoECache& cache,
                                           MoERoutingStats& stats) const;

    // Richiede la cache dell'ultima forward() sullo STESSO input.
    [[nodiscard]] runtime::Tensor backward(const runtime::Tensor& input, const runtime::Tensor& gradOutput,
                                            const MoECache& cache, const MoERoutingStats& stats,
                                            GradientSink* sink) const;

    [[nodiscard]] MoERouter& router() { return router_; }
    [[nodiscard]] const MoERouter& router() const { return router_; }
    [[nodiscard]] std::vector<MoEExpert>& experts() { return experts_; }
    [[nodiscard]] const std::vector<MoEExpert>& experts() const { return experts_; }

    [[nodiscard]] std::size_t parameterBytes() const;
    void setComputeDType(ComputeDType dtype);

    // Capacita' per esperto dato il numero di token del batch: oltre
    // questa soglia le assegnazioni vengono scartate (e contate).
    [[nodiscard]] std::size_t capacityFor(std::size_t tokens) const;

private:
    BlackBitConfig config_;
    MoERouter router_;
    std::vector<MoEExpert> experts_;
};

}  // namespace blackforge::blackbit
