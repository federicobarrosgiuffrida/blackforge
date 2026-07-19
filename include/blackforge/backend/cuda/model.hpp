#pragma once

#include <optional>
#include <string>
#include <vector>

#include "blackforge/backend/cuda/device_tensor.hpp"
#include "blackforge/ir/module.hpp"

namespace blackforge::backend::cuda {

// Un parametro allenabile residente su device: il suo valore corrente e
// il gradiente accumulato dall'ultima chiamata a Model::zeroGrad().
struct Parameter {
    std::string name;
    DeviceTensor value;
    DeviceTensor grad;
};

// Un modello CUDA allenabile, costruito dalla pipeline di un ir::ModelIR
// — la controparte su device di backend::cpu::Model, con la stessa
// interfaccia (forward/backward/zeroGrad/parameters).
//
// LIMITAZIONI ESPLICITE di questa prima versione (percorso minimo
// stabile, non un'omissione nascosta):
//   - Nessun supporto LoRA: costruire un Model con un adapter a basso
//     rango non e' possibile ancora su CUDA. backend::cuda::runTraining
//     lancia un errore esplicito se il blocco 'train' dichiara 'lora'.
//   - Nessuna quantizzazione di precisione (vedi
//     backend::cpu::quantize): un blocco 'precision' dichiarato viene
//     ignorato durante l'addestramento su GPU, come gia' accade per
//     'blackforge run --device cuda' (nota stampata dalla CLI).
//   - I pesi sono inizializzati con lo stesso generatore deterministico
//     del backend CPU (blackforge::backend::cpu::randomTensor), non una
//     strategia statisticamente valida (Xavier/Kaiming): permette il
//     confronto diretto CPU/GPU nei test, non produce un modello pronto
//     per un addestramento serio da zero.
class Model {
public:
    // Lancia std::invalid_argument se il modello non ha pipeline o se
    // una dimensione delle feature in ingresso a un layer 'linear' e'
    // ancora simbolica (serve una dimensione concreta per allocare i
    // pesi).
    explicit Model(const ir::ModelIR& modelIR, unsigned int seed = 42);

    // Esegue la pipeline e salva le attivazioni intermedie (su device)
    // necessarie a backward(). Va chiamata prima di ogni backward().
    DeviceTensor forward(const DeviceTensor& input);

    // Calcola i gradienti dei parametri a partire dal gradiente
    // dell'uscita (dL/doutput), usando le attivazioni salvate
    // dall'ultima forward(). Li ACCUMULA (non li azzera): chiamare
    // zeroGrad() prima di ogni step di addestramento.
    void backward(const DeviceTensor& outputGrad);

    void zeroGrad();

    // Parametri allenabili: pesi/bias di ogni layer 'linear'. E' quello
    // che va passato a un Optimizer.
    [[nodiscard]] std::vector<Parameter*> parameters();

    [[nodiscard]] const std::string& name() const { return name_; }

private:
    struct LayerState {
        ir::OpKind kind;
        std::optional<Parameter> weight;  // valido solo se kind == Linear
        std::optional<Parameter> bias;    // valido solo se kind == Linear
        DeviceTensor cachedInput;         // popolato da forward()

        // Valido solo se kind == Embedding.
        std::optional<Parameter> embeddingTable;
        std::size_t embeddingVocabSize = 0;

        // Valido solo se kind == PositionalEmbedding.
        std::optional<Parameter> positionalTable;
        std::size_t positionalMaxSeqLen = 0;

        // Validi solo se kind == Attention.
        std::optional<Parameter> attnWq;
        std::optional<Parameter> attnWk;
        std::optional<Parameter> attnWv;
        std::optional<Parameter> attnWout;
        std::size_t attentionNumHeads = 0;

        // Validi solo se kind == FeedForward.
        std::optional<Parameter> ffW1;
        std::optional<Parameter> ffB1;
        std::optional<Parameter> ffW2;
        std::optional<Parameter> ffB2;
    };

    static std::vector<Parameter*> allParameterSlots(LayerState& layer);

    std::string name_;
    std::vector<LayerState> layers_;
};

}  // namespace blackforge::backend::cuda
