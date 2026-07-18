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
// forma [batch, classi]. 'target' ha la stessa forma e rappresenta una
// distribuzione di probabilita' per esempio (tipicamente one-hot, ma
// un'etichetta soft e' accettata allo stesso modo). Il valore restituito
// e' la cross-entropy media sul batch: -mean_b( sum_c target[b,c] *
// log(softmax(logits)[b,c]) ). Il gradiente rispetto a 'logits' e' la
// forma chiusa standard per softmax+cross-entropy combinati,
// (softmax(logits) - target) / batch, quindi non serve un backward di
// una softmax separata. Lancia std::invalid_argument se le forme non
// coincidono o se non sono a rango 2.
LossResult softmaxCrossEntropy(const runtime::Tensor& logits, const runtime::Tensor& target);

}  // namespace blackforge::backend::cpu
