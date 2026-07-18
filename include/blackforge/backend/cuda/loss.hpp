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
// Solo questa loss e' implementata sul backend CUDA per ora:
// cross-entropy (softmax) su GPU e' lavoro futuro, vedi
// backend::cuda::runTraining per l'errore esplicito se richiesta.
LossResult meanSquaredError(const DeviceTensor& prediction, const DeviceTensor& target);

}  // namespace blackforge::backend::cuda
