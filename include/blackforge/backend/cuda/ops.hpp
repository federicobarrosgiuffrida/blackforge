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

// Layer lineare: input [batch, inFeatures], weight [inFeatures, outFeatures],
// bias [outFeatures] -> output [batch, outFeatures].
DeviceTensor linear(const DeviceTensor& input, const DeviceTensor& weight, const DeviceTensor& bias);

}  // namespace blackforge::backend::cuda
