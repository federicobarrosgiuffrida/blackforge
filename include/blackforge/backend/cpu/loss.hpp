#pragma once

#include "blackforge/runtime/tensor.hpp"

namespace blackforge::backend::cpu {

struct LossResult {
    float value;
    runtime::Tensor grad;  // dL/dprediction, stessa forma di 'prediction'
};

// Errore quadratico medio: mean((prediction - target)^2) su tutti gli
// elementi. Adatta alla regressione (compreso il forecasting).
LossResult meanSquaredError(const runtime::Tensor& prediction, const runtime::Tensor& target);

// Cross-entropy softmax per la classificazione multiclasse.
// 'logits' sono le uscite grezze del modello (PRIMA di una softmax:
// questa funzione applica softmax internamente, in modo numericamente
// stabile sottraendo il massimo per riga prima di esponenziare) con
// forma [..., classi] (rango >= 2: es. [batch, classi] per la
// classificazione, [batch, seq, classi] per la next-token-prediction di
// un modello linguistico). 'target' ha la stessa forma e rappresenta una
// distribuzione di probabilita' per ogni "riga" (tutte le dimensioni
// tranne l'ultima; tipicamente one-hot, ma un'etichetta soft e'
// accettata allo stesso modo). Il valore restituito e' la cross-entropy
// media su tutte le righe: -mean_r( sum_c target[r,c] *
// log(softmax(logits)[r,c]) ). Il gradiente rispetto a 'logits' e' la
// forma chiusa standard per softmax+cross-entropy combinati,
// (softmax(logits) - target) / righe, quindi non serve un backward di
// una softmax separata. Lancia std::invalid_argument se le forme non
// coincidono o se non sono a rango >= 2.
LossResult softmaxCrossEntropy(const runtime::Tensor& logits, const runtime::Tensor& target);

// Variante sparsa di softmaxCrossEntropy: 'targetIndices' ha rango
// logits.rank() - 1 (la stessa forma di 'logits' senza l'ultima
// dimensione) e contiene, per ogni riga, l'INDICE della classe
// corretta (un intero non negativo rappresentato come float,
// arrotondato con std::lround — la stessa convenzione degli id di
// token altrove) invece di un vettore one-hot denso. Matematicamente
// identica a softmaxCrossEntropy() con un target one-hot equivalente
// (stessa loss, stesso gradiente), ma senza mai materializzare un
// target di dimensione [..., classi]: essenziale quando 'classi' e' un
// vocabolario grande (next-token-prediction di un modello linguistico),
// dove un target denso sprecherebbe memoria proporzionale al
// vocabolario. Lancia std::invalid_argument se le forme non
// corrispondono o se un indice e' fuori da [0, classi).
LossResult softmaxCrossEntropySparse(const runtime::Tensor& logits, const runtime::Tensor& targetIndices);

// Variante MASCHERATA di softmaxCrossEntropySparse, pensata per
// l'addestramento di un modello linguistico mascherato (MLM: alcuni
// token dell'ingresso sono sostituiti con un token <mask>, e solo QUEI
// token contribuiscono alla loss — non l'intera sequenza, a differenza
// della next-token-prediction causale). 'targetIndices' ha la stessa
// semantica di softmaxCrossEntropySparse (indice di classe per riga),
// ma il valore sentinella **-1** in una riga significa "ignora questa
// riga": nessun contributo alla loss ne' al gradiente (gradiente
// esattamente zero per tutte le classi di quella riga). La loss finale
// e' la media SOLO sulle righe non ignorate. Se NESSUNA riga e'
// mascherata (puo' capitare per caso con un mascheramento casuale su
// batch piccoli), la loss e' 0 e il gradiente e' tutto zero — non e'
// un errore, solo un batch senza nulla da imparare. Lancia
// std::invalid_argument se le forme non corrispondono o se un indice
// (diverso da -1) e' fuori da [0, classi).
LossResult softmaxCrossEntropyMasked(const runtime::Tensor& logits, const runtime::Tensor& targetIndices);

}  // namespace blackforge::backend::cpu
