#pragma once

#include "blackforge/backend/cuda/device_tensor.hpp"

namespace blackforge::backend::cuda {

// Somma elementwise di due tensori device con la stessa forma.
DeviceTensor add(const DeviceTensor& a, const DeviceTensor& b);

// Somma un bias device di forma [features] a un tensore [batch, features],
// trasmettendolo (broadcast) su ogni riga del batch.
DeviceTensor addBias(const DeviceTensor& input, const DeviceTensor& bias);

// Prodotto matriciale 2D via cuBLAS SGEMM: [M, K] x [K, N] -> [M, N].
DeviceTensor matmul(const DeviceTensor& a, const DeviceTensor& b);

DeviceTensor silu(const DeviceTensor& input);
DeviceTensor relu(const DeviceTensor& input);
DeviceTensor gelu(const DeviceTensor& input);

// RMSNorm senza gamma allenabile (vedi backend::cpu::rmsnorm per i
// dettagli): un blocco CUDA per riga del batch, con una riduzione in
// shared memory per calcolare la somma dei quadrati. eps fisso a 1e-6,
// identico al backend CPU (kRmsNormEps in ops_elementwise.cu), per
// permettere il confronto diretto CPU/GPU nei test.
DeviceTensor rmsnorm(const DeviceTensor& input);

// Softmax riga per riga su un tensore [batch, classi] (vedi
// backend::cpu::softmax per i dettagli, stessa semantica). eps non
// applicabile qui: la stabilita' numerica viene dalla sottrazione del
// massimo per riga, non da un epsilon additivo.
DeviceTensor softmax(const DeviceTensor& input);

// Layer lineare: input [batch, inFeatures], weight [inFeatures, outFeatures],
// bias [outFeatures] -> output [batch, outFeatures].
DeviceTensor linear(const DeviceTensor& input, const DeviceTensor& weight, const DeviceTensor& bias);

}  // namespace blackforge::backend::cuda
