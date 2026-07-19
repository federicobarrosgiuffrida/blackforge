#pragma once

#include <optional>
#include <string>
#include <vector>

#include "blackforge/backend/cuda/device_tensor.hpp"
#include "blackforge/backend/cuda/ops.hpp"
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
//   - Quantizzazione simulata (vedi backend::cpu::quantize) non ancora
//     supportata: un blocco 'precision' con 'storage'/'accumulate'
//     diversi da fp32 e' ignorato. 'compute bf16' e' pero' un caso
//     speciale: NON e' quantizzazione simulata (che qui non esiste),
//     ma Tensor Core BF16 REALE (cuBLASLt, vedi ops.hpp/matmulBf16) per
//     il prodotto matriciale di ogni layer 'linear' — forward E
//     backward, quindi utilizzabile anche durante l'addestramento (a
//     differenza della quantizzazione simulata del backend CPU, che
//     resta solo-inferenza perche' l'arrotondamento in se' non e'
//     differenziabile: qui invece si calcola davvero in BF16, non si
//     arrotonda un risultato FP32 gia' calcolato, quindi il backward
//     differenzia esattamente l'operazione che il forward ha davvero
//     eseguito).
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
    // pesi). Se 'precision' e' presente e 'precision->compute ==
    // sema::DType::BF16', ogni layer 'linear' usa matmulBf16 (Tensor
    // Core reale) al posto di matmul (SGEMM float32) sia in forward sia
    // in backward.
    explicit Model(const ir::ModelIR& modelIR, unsigned int seed = 42,
                   std::optional<ir::PrecisionPolicy> precision = std::nullopt);

    // Esegue la pipeline e salva le attivazioni intermedie (su device)
    // necessarie a backward(). Va chiamata prima di ogni backward().
    //
    // 'inputRangeTrusted': se true E il primo layer della pipeline e'
    // 'embedding', salta la validazione del range dei token id
    // sull'host per QUEL layer (nessun round-trip device->host->device —
    // vedi embeddingLookupPreValidated in ops.hpp). PRECONDIZIONE,
    // responsabilita' del chiamante: 'input' deve gia' contenere solo
    // token id in [0, vocabSize) — usato dall'hot loop di addestramento
    // (train_runner.cu/multi_gpu_train_runner.cu), che valida gli stessi
    // valori sull'host PRIMA di caricarli su device, rendendo la
    // validazione interna ridondante. Ignorato (nessun effetto, nessun
    // rischio) se il primo layer non e' 'embedding'.
    DeviceTensor forward(const DeviceTensor& input, bool inputRangeTrusted = false);

    // Calcola i gradienti dei parametri a partire dal gradiente
    // dell'uscita (dL/doutput), usando le attivazioni salvate
    // dall'ultima forward(). Li ACCUMULA (non li azzera): chiamare
    // zeroGrad() prima di ogni step di addestramento.
    //
    // 'inputRangeTrusted': stessa semantica/precondizione di forward() —
    // deve avere lo STESSO valore passato all'ultima forward() (la
    // validazione riguarda gli stessi token id cachati da quella
    // chiamata).
    void backward(const DeviceTensor& outputGrad, bool inputRangeTrusted = false);

    void zeroGrad();

    // Genera un output autoregressivo incrementale su device (vedi
    // backend::cpu::Model::forwardIncremental per la semantica
    // completa, identica qui): 'newTokenIds' contiene SOLO i token
    // nuovi rispetto all'ultima chiamata. Mantiene una cache K/V
    // (backend::cuda::KVCache) per ogni layer 'attention' della
    // pipeline. Non salva alcuno stato per backward(): non mescolare
    // forward()/backward() con forwardIncremental() sulla stessa
    // istanza senza chiamare resetGenerationState() in mezzo.
    DeviceTensor forwardIncremental(const DeviceTensor& newTokenIds);

    // Azzera la cache K/V di ogni layer 'attention' e la posizione
    // assoluta corrente: va chiamata prima di iniziare una nuova
    // sessione di generazione.
    void resetGenerationState();

    // Parametri allenabili: pesi/bias di ogni layer 'linear'. E' quello
    // che va passato a un Optimizer.
    [[nodiscard]] std::vector<Parameter*> parameters();

    [[nodiscard]] const std::string& name() const { return name_; }

    // Se il primo layer della pipeline e' 'embedding', restituisce la
    // dimensione del suo vocabolario (per permettere al chiamante di
    // validare i token id sull'host prima di caricarli su device, vedi
    // forward()/backward() sopra); std::nullopt altrimenti.
    [[nodiscard]] std::optional<std::size_t> firstLayerEmbeddingVocabSize() const;

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
        KVCache kvCache;  // usata solo da forwardIncremental()

        // Validi solo se kind == FeedForward.
        std::optional<Parameter> ffW1;
        std::optional<Parameter> ffB1;
        std::optional<Parameter> ffW2;
        std::optional<Parameter> ffB2;
    };

    static std::vector<Parameter*> allParameterSlots(LayerState& layer);

    std::string name_;
    std::vector<LayerState> layers_;
    bool useTensorCoreLinear_ = false;
    std::size_t generationPosition_ = 0;  // usata solo da forwardIncremental()
};

}  // namespace blackforge::backend::cuda
