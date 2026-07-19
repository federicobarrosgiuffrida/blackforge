#pragma once

#include "blackforge/backend/cuda/device_tensor.hpp"

namespace blackforge::backend::cuda {

struct LossResult {
    float value;
    DeviceTensor grad;  // dL/dprediction, stessa forma di 'prediction', residente su device
};

// Errore quadratico medio: mean((prediction - target)^2) su tutti gli
// elementi, calcolato interamente su device (grad via kernel
// elementwise, value via una riduzione device-side a singolo blocco).
// L'unico dato che lascia il device e' lo scalare finale 'value'
// (necessario comunque per stamparlo o loggarlo): non e' un fallback
// della computazione sulla CPU, e' l'unica cosa che una stampa a
// schermo puo' mai richiedere.
//
LossResult meanSquaredError(const DeviceTensor& prediction, const DeviceTensor& target);

// Cross-entropy softmax per la classificazione multiclasse, calcolata
// interamente su device (un blocco per "riga", riduzione in shared
// memory). Vedi backend::cpu::softmaxCrossEntropy per la semantica
// completa (identica: generalizzata a rango >= 2, [..., classi], la loss
// e' la media su tutte le righe). Lancia std::invalid_argument se le
// forme non coincidono o se non sono a rango >= 2.
LossResult softmaxCrossEntropy(const DeviceTensor& logits, const DeviceTensor& target);

// Variante sparsa (vedi backend::cpu::softmaxCrossEntropySparse per la
// semantica completa, identica): 'targetIndices' contiene, per ogni
// riga, l'indice della classe corretta invece di un vettore one-hot
// denso — essenziale per vocabolari grandi (next-token-prediction),
// dove un target denso sul device sprecherebbe memoria proporzionale al
// vocabolario. Lancia std::invalid_argument se le forme non
// corrispondono o se un indice e' fuori da [0, classi).
LossResult softmaxCrossEntropySparse(const DeviceTensor& logits, const DeviceTensor& targetIndices);

}  // namespace blackforge::backend::cuda
