#pragma once

#include <cstddef>
#include <string>

#include "blackforge/sema/dtype.hpp"

// Configurazione architetturale di un modello BlackBit e contabilita'
// di parametri/memoria che ne deriva.
//
// I modelli BlackBit (Tiny, Small, Medium, 9B-A3B) differiscono SOLO
// per i valori di questa struttura: non esiste una seconda
// implementazione "piccola" da tenere allineata. Un test che passa su
// BlackBit-Tiny esercita esattamente lo stesso codice del modello da
// 9 miliardi di parametri.

namespace blackforge::blackbit {

struct BlackBitConfig {
    std::string name = "blackbit-tiny";

    std::size_t vocabSize = 8192;
    std::size_t hiddenSize = 384;
    std::size_t numLayers = 6;

    // Attention a query raggruppate (GQA): numHeads query condividono
    // numKvHeads teste di chiave/valore. numHeads deve essere un
    // multiplo di numKvHeads.
    std::size_t numHeads = 6;
    std::size_t numKvHeads = 2;
    std::size_t headDim = 64;

    // Mixture-of-Experts: numExperts esperti per blocco feed-forward,
    // di cui expertsPerToken selezionati per ogni token.
    std::size_t numExperts = 4;
    std::size_t expertsPerToken = 2;
    std::size_t expertHidden = 1024;

    std::size_t maxSeqLen = 512;

    // Formato di memorizzazione delle matrici grandi (proiezioni di
    // attention, esperti, embedding). TERNARY_1P58 e' il caso reale;
    // FP32 esiste come modalita' di riferimento per il debug (vedi
    // BlackBitRuntimeOptions::referenceDenseWeights) e NON e'
    // utilizzabile su BlackBit-9B (18 GB di pesi).
    sema::DType weightDtype = sema::DType::TERNARY_1P58;

    // I parametri numericamente sensibili restano densi: sono ~0,01 %
    // del modello, comprimerli non farebbe risparmiare memoria
    // misurabile e destabilizzerebbe l'addestramento (requisito 15).
    sema::DType normDtype = sema::DType::BF16;
    sema::DType routerDtype = sema::DType::BF16;

    // Embedding di ingresso e proiezione di uscita condividono la
    // stessa matrice.
    bool tieEmbeddings = true;

    // Pesi che condividono una scala di quantizzazione (vedi
    // ternary.hpp): multiplo di kTritsPerWord.
    std::size_t ternaryGroupSize = 160;

    // Peso della loss ausiliaria di bilanciamento del carico fra
    // esperti, sommata alla loss principale.
    float routerAuxLossWeight = 0.01F;

    // Capacita' di ogni esperto come multiplo della quota "equa"
    // (tokens * expertsPerToken / numExperts). I token oltre la
    // capacita' vengono scartati per quell'esperto e contati nelle
    // metriche di routing.
    float expertCapacityFactor = 1.25F;

    // Lancia std::invalid_argument descrivendo il primo vincolo
    // violato (divisibilita' teste/gruppi, top-k <= esperti,
    // groupSize allineato alle parole, dimensioni non nulle, ...).
    void validate() const;

    [[nodiscard]] std::size_t queryDim() const { return numHeads * headDim; }
    [[nodiscard]] std::size_t kvDim() const { return numKvHeads * headDim; }
    [[nodiscard]] std::size_t headsPerKvGroup() const { return numHeads / numKvHeads; }
};

// Configurazioni di riferimento. Le prime tre servono a validare
// l'implementazione a costi di calcolo ragionevoli prima di
// istanziare quella finale (requisito 14).
BlackBitConfig blackBitTiny();      // ~34 M parametri
BlackBitConfig blackBitSmall();     // ~118 M parametri
BlackBitConfig blackBitMedium();    // ~440 M parametri
BlackBitConfig blackBit9bA3b();     // ~9,05 G parametri, ~2,9 G attivi/token

// Cerca una configurazione di riferimento per nome ("tiny", "small",
// "medium", "9b" / "9b-a3b"); lancia std::invalid_argument se il nome
// non corrisponde a nessuna.
BlackBitConfig blackBitPreset(const std::string& name);

// Numero di parametri per categoria, sull'INTERO modello (tutti i
// layer). Separare ternari e densi non e' cosmetico: e' esattamente la
// separazione che determina l'occupazione di memoria, dato che i due
// gruppi hanno costi per parametro diversi di un fattore 10.
struct ParameterCount {
    std::size_t attention = 0;   // q/k/v/o di tutti i layer (ternari)
    std::size_t experts = 0;     // gate/up/down di tutti gli esperti (ternari)
    std::size_t embedding = 0;   // tabella condivisa ingresso/uscita (ternaria)
    std::size_t router = 0;      // matrici di routing (dense)
    std::size_t norms = 0;       // gamma di ogni RMSNorm (dense)

    [[nodiscard]] std::size_t ternary() const { return attention + experts + embedding; }
    [[nodiscard]] std::size_t dense() const { return router + norms; }
    [[nodiscard]] std::size_t total() const { return ternary() + dense(); }
};

ParameterCount countParameters(const BlackBitConfig& config);

// Parametri effettivamente coinvolti nel calcolo di UN token: tutta
// l'attention, solo gli expertsPerToken esperti selezionati, il router,
// le norm e la proiezione di uscita legata all'embedding (l'embedding
// di INGRESSO e' un lookup: contribuisce con una riga sola, non con
// l'intera tabella, e non viene contata qui).
std::size_t countActiveParameters(const BlackBitConfig& config);

// Forma del passo di addestramento su cui fare i conti di memoria.
struct TrainingShape {
    std::size_t microBatch = 1;
    std::size_t seqLen = 512;
};

// Parametri del percorso a bassa memoria che influenzano la memoria
// stimata.
struct LowMemoryOptions {
    // Rango del sottospazio in cui vive lo stato dell'optimizer
    // (requisito 7).
    std::size_t optimizerRank = 32;

    // Byte per valore dello stato dell'optimizer. 4 = float32, che e'
    // cio' che l'implementazione di riferimento usa davvero oggi
    // (runtime::Tensor e' float32 in tutto questo motore); 2 = BF16,
    // il passo successivo previsto dal requisito 7. La stima riporta il
    // valore che si passa, non quello che si vorrebbe.
    std::size_t optimizerStateBytes = 4;

    // Righe di peso dequantizzate contemporaneamente da TernaryLinear.
    std::size_t dequantTileRows = 128;

    // Righe di gradiente di peso calcolate e consumate in un colpo dal
    // backward in streaming. dW[n, k] = sum_m dY[m, n] * X[m, k]: la
    // riga n del gradiente dipende solo dalla colonna n di dY, quindi
    // un blocco di righe si puo' calcolare, proiettare e distruggere
    // senza mai vedere le altre. Senza questo blocco il gradiente della
    // tabella di embedding di BlackBit-9B sarebbe da solo
    // 65536 * 3072 * 4 B = 805 MB.
    std::size_t gradientTileRows = 512;

    // true: si conserva una sola attivazione per layer e il resto viene
    // ricalcolato nel backward.
    bool activationCheckpointing = true;
};

// STIMA di progetto della memoria di un passo di addestramento, voce
// per voce. NON e' una misura: serve a decidere se una configurazione
// sta nel budget PRIMA di allocare (requisito 11) e a confrontare la
// previsione con quello che la telemetria misurera' davvero. Ogni voce
// e' calcolata dalle formule documentate in docs/blackbit.md §2.
struct MemoryEstimate {
    std::size_t packedWeightBytes = 0;   // pesi ternari impacchettati
    std::size_t scaleBytes = 0;          // scale per gruppo
    std::size_t denseParameterBytes = 0; // router + norm
    std::size_t optimizerBytes = 0;      // stato low-rank (m, v)
    std::size_t activationBytes = 0;     // attivazioni conservate
    std::size_t gradientPeakBytes = 0;   // gradiente del blocco di righe piu' grande
    std::size_t workspaceBytes = 0;      // tile dequantizzati + logit a blocchi

    [[nodiscard]] std::size_t total() const {
        return packedWeightBytes + scaleBytes + denseParameterBytes + optimizerBytes + activationBytes +
               gradientPeakBytes + workspaceBytes;
    }

    // Byte che sarebbero necessari con l'approccio ordinario, per
    // confronto: master copy BF16 + gradiente denso a dimensione
    // modello + stato AdamW completo. Non viene mai allocato: e' il
    // numero che giustifica l'intero progetto.
    std::size_t conventionalBytes = 0;
};

MemoryEstimate estimateTrainingMemory(const BlackBitConfig& config, const TrainingShape& shape,
                                       const LowMemoryOptions& options);

// Carica/salva una configurazione in JSON (oggetto piatto di
// numeri/booleani/stringhe, vedi configs/*.json). Lancia
// std::runtime_error con riga e motivo se il file non e' leggibile o
// contiene una chiave sconosciuta: un refuso in un nome di campo deve
// essere un errore, non un valore di default applicato in silenzio.
BlackBitConfig loadConfigFromJson(const std::string& path);
BlackBitConfig parseConfigJson(const std::string& text, const std::string& origin = "<memoria>");
std::string toJson(const BlackBitConfig& config);

}  // namespace blackforge::blackbit
