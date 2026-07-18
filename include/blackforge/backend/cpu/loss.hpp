#pragma once

#include "blackforge/runtime/tensor.hpp"

namespace blackforge::backend::cpu {

struct LossResult {
    float value;
    runtime::Tensor grad;  // dL/dprediction, stessa forma di 'prediction'
};

// Errore quadratico medio: mean((prediction - target)^2) su tutti gli
// elementi. E' l'unica loss implementata per ora: altre (es.
// cross-entropy per la classificazione) arriveranno quando il
// linguaggio avra' una sintassi per dichiarare il tipo di compito.
LossResult meanSquaredError(const runtime::Tensor& prediction, const runtime::Tensor& target);

}  // namespace blackforge::backend::cpu
