#pragma once

#include <optional>
#include <string>
#include <vector>

#include "blackforge/backend/cpu/ops.hpp"
#include "blackforge/ir/module.hpp"
#include "blackforge/runtime/tensor.hpp"

namespace blackforge::backend::cpu {

// Un parametro allenabile: il suo valore corrente e il gradiente
// accumulato dall'ultima chiamata a Model::zeroGrad().
struct Parameter {
    std::string name;
    runtime::Tensor value;
    runtime::Tensor grad;
};

// Configurazione di un adapter LoRA (Low-Rank Adaptation, Hu et al.
// 2021), applicato a ogni layer 'linear' del modello. weight/bias
// originali restano congelati (nessun gradiente accumulato); si
// allenano solo le matrici a basso rango A [inFeatures, rank] e
// B [rank, outFeatures], con contributo scalato di alpha/rank e sommato
// all'uscita del layer originale: output = linear(x) + (alpha/rank) *
// (x @ A) @ B.
struct LoraOptions {
    long long rank;
    double alpha;
};

// Un modello CPU allenabile, costruito dalla pipeline di un ir::ModelIR.
//
// A differenza di Executor (milestone del backend CPU di riferimento,
// che rigenera pesi casuali ad ogni chiamata solo per ispezionare
// rapidamente un'esecuzione), Model possiede i propri parametri e li
// mantiene tra una forward() e la successiva: e' il tipo pensato per
// essere effettivamente allenato con un optimizer.
//
// NOTA: l'inizializzazione dei pesi resta deterministica ma non
// statisticamente valida (vedi random_init.hpp); una strategia di init
// seria (Xavier/Kaiming) e' lavoro futuro. Il caricamento di pesi
// pre-allenati e' supportato tramite checkpoint.hpp.
class Model {
public:
    // Lancia std::invalid_argument se il modello non ha pipeline, se ha
    // piu' di una dimensione di feature simbolica in ingresso a un
    // layer 'linear' (serve una dimensione concreta per allocare i
    // pesi). Se 'lora' e' presente, ogni layer 'linear' riceve anche un
    // adapter a basso rango (vedi LoraOptions); weight/bias restano
    // inizializzati come al solito, ma vanno tipicamente sovrascritti
    // con pesi pre-allenati (caricando un checkpoint) prima
    // dell'addestramento: allenare un adapter su pesi casuali non ha
    // senso.
    //
    // Se 'precision' e' presente, forward() applica la quantizzazione
    // simulata (vedi quantize.hpp) alle attivazioni (formato 'storage')
    // e agli operandi di ogni matmul (formato 'compute'). ATTENZIONE:
    // questo e' pensato SOLO per l'inferenza (es. blackforge forecast).
    // backward() calcola comunque il gradiente della funzione originale
    // non quantizzata rispetto alle attivazioni cachate (che, se
    // precision e' attivo, sono gia' quantizzate): non e' uno
    // straight-through estimator corretto per l'addestramento. Per
    // questo train_runner.cpp non passa mai 'precision' a Model: il
    // training resta sempre a piena precisione float32.
    explicit Model(const ir::ModelIR& modelIR, unsigned int seed = 42,
                   std::optional<LoraOptions> lora = std::nullopt,
                   std::optional<ir::PrecisionPolicy> precision = std::nullopt);

    // Esegue la pipeline e salva le attivazioni intermedie necessarie a
    // backward(). Va chiamata prima di ogni backward().
    runtime::Tensor forward(const runtime::Tensor& input);

    // Calcola i gradienti dei parametri a partire dal gradiente
    // dell'uscita (dL/doutput), usando le attivazioni salvate
    // dall'ultima forward(). Li ACCUMULA (non li azzera): chiamare
    // zeroGrad() prima di ogni step di addestramento.
    void backward(const runtime::Tensor& outputGrad);

    void zeroGrad();

    // Genera un output autoregressivo incrementale: 'newTokenIds'
    // contiene SOLO i token nuovi rispetto all'ultima chiamata (l'intero
    // prompt alla prima chiamata dopo resetGenerationState(), un solo
    // token ad ogni chiamata successiva). Mantiene una cache K/V per
    // ogni layer 'attention' della pipeline (vedi ops.hpp::KVCache),
    // crescente ad ogni chiamata: evita di ricalcolare l'attention
    // sull'intera sequenza generata finora ad ogni nuovo token. Non
    // salva alcuno stato per backward() (la generazione e' solo
    // inferenza): chiamarla non interferisce con forward()/backward(),
    // ma nemmeno vale il contrario — non mescolare le due modalita'
    // sulla stessa istanza senza chiamare resetGenerationState() in
    // mezzo. Layer diversi da Embedding/PositionalEmbedding/Attention/
    // FeedForward/Linear/Silu/Relu/Gelu/RmsNorm/Softmax non sono
    // previsti (nessuno lo e' ancora nel linguaggio).
    runtime::Tensor forwardIncremental(const runtime::Tensor& newTokenIds);

    // Azzera la cache K/V di ogni layer 'attention' e la posizione
    // assoluta corrente: va chiamata prima di iniziare una nuova
    // sessione di generazione (o tra una sessione e la successiva).
    void resetGenerationState();

    // Parametri ALLENABILI: pesi/bias dei layer 'linear' normalmente,
    // oppure solo gli adapter LoRA (A/B) per i layer con LoRA attivo
    // (weight/bias restano congelati e non compaiono qui). E' quello
    // che va passato a un Optimizer.
    [[nodiscard]] std::vector<Parameter*> parameters();

    // TUTTI i parametri esistenti, inclusi weight/bias congelati quando
    // LoRA e' attivo: e' quello che checkpoint.hpp usa per
    // salvare/caricare, cosi' un checkpoint salvato durante un
    // addestramento LoRA contiene sia i pesi di base sia gli adapter
    // (un modello autosufficiente), e un checkpoint di pesi di base puo'
    // essere caricato in un Model con LoRA attivo (gli adapter, non
    // presenti nel file, restano ai valori di inizializzazione).
    [[nodiscard]] std::vector<Parameter*> allParameters();

    [[nodiscard]] const std::string& name() const { return name_; }

private:
    struct LayerState {
        ir::OpKind kind;
        std::size_t operationOutputId;
        std::optional<Parameter> weight;  // valido solo se kind == Linear
        std::optional<Parameter> bias;    // valido solo se kind == Linear
        runtime::Tensor cachedInput;      // popolato da forward()

        // Validi solo se kind == Linear e Model e' stato costruito con
        // 'lora' attivo.
        std::optional<Parameter> loraA;
        std::optional<Parameter> loraB;
        runtime::Tensor cachedLoraHidden;  // cachedInput @ loraA, popolato da forward()

        // Validi solo se kind == Embedding.
        std::optional<Parameter> embeddingTable;
        std::size_t embeddingVocabSize = 0;

        // Valido solo se kind == PositionalEmbedding.
        std::optional<Parameter> positionalTable;
        std::size_t positionalMaxSeqLen = 0;

        // Validi solo se kind == Attention (nessun bias, come in LLaMA:
        // vedi backend::cpu::selfAttention).
        std::optional<Parameter> attnWq;
        std::optional<Parameter> attnWk;
        std::optional<Parameter> attnWv;
        std::optional<Parameter> attnWout;
        std::size_t attentionNumHeads = 0;
        KVCache kvCache;  // usata solo da forwardIncremental()

        // Validi solo se kind == FeedForward.
        std::optional<Parameter> ffW1;
        std::optional<Parameter> ffB1;
        std::optional<Parameter> ffW2;
        std::optional<Parameter> ffB2;
    };

    // Tutti gli slot Parameter potenzialmente presenti in un layer,
    // indipendentemente dal loro tipo: usato da zeroGrad()/allParameters(),
    // che trattano ogni parametro allo stesso modo (parameters() resta
    // separata perche' applica la logica speciale di LoRA solo a 'linear').
    static std::vector<Parameter*> allParameterSlots(LayerState& layer);

    std::string name_;
    std::vector<LayerState> layers_;
    std::optional<LoraOptions> loraOptions_;
    std::optional<ir::PrecisionPolicy> precision_;
    std::size_t generationPosition_ = 0;  // usata solo da forwardIncremental()
};

}  // namespace blackforge::backend::cpu
