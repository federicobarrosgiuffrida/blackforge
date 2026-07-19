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

// Varianti "Accumulate" per l'hot loop di addestramento (vedi
// train_runner.cu/multi_gpu_train_runner.cu): il gradiente e' calcolato
// e restituito subito (serve al backward immediato), ma la loss
// scalare NON viene letta sull'host ad ogni chiamata — al suo posto, la
// somma delle loss di riga di QUESTA chiamata (non ancora divisa per il
// numero di righe) viene sommata (atomicAdd, device-side) dentro
// 'lossAccumulator' (un DeviceTensor di un solo elemento, di proprieta'
// del chiamante, azzerato a inizio epoca). Una singola cudaMemcpy a
// fine epoca legge il totale accumulato, invece di bloccare la
// pipeline GPU una volta per ogni singolo step — la stessa
// trasformazione della cache dei piani cuBLASLt (vedi ops_tensorcore.cu),
// applicata qui alla sincronizzazione host invece che alla creazione di
// stato cuBLASLt. Il chiamante e' responsabile di dividere il totale
// accumulato per (righe per chiamata * numero di chiamate nell'epoca)
// per ottenere la stessa media-delle-medie che l'API eager produceva.
DeviceTensor meanSquaredErrorAccumulate(const DeviceTensor& prediction, const DeviceTensor& target,
                                         DeviceTensor& lossAccumulator);
DeviceTensor softmaxCrossEntropyAccumulate(const DeviceTensor& logits, const DeviceTensor& target,
                                            DeviceTensor& lossAccumulator);

// Come softmaxCrossEntropyAccumulate, ma per la variante sparsa — con
// un'ulteriore differenza rispetto a softmaxCrossEntropySparse(): NON
// valida il range di 'targetIndices' sull'host (nessun round-trip
// device->host->device). PRECONDIZIONE, responsabilita' del chiamante:
// ogni indice deve gia' essere in [0, classi) — comportamento indefinito
// altrimenti (un kernel CUDA non puo' lanciare eccezioni). Pensata per
// essere chiamata quando il chiamante ha GIA' gli stessi valori sull'host
// PRIMA di caricarli su device (es. train_runner.cu valida
// 'batch.target' appena letto dal dataset, prima di DeviceTensor::fromHost:
// stessi valori, zero round-trip aggiuntivo verso il device).
DeviceTensor softmaxCrossEntropySparseAccumulate(const DeviceTensor& logits, const DeviceTensor& targetIndices,
                                                  DeviceTensor& lossAccumulator);

}  // namespace blackforge::backend::cuda
