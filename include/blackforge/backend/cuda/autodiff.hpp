#pragma once

#include "blackforge/backend/cuda/device_tensor.hpp"

namespace blackforge::backend::cuda {

// Funzioni di backward (reverse-mode) per le operazioni del backend
// CUDA. Formule analitiche identiche a quelle del backend CPU
// (src/backend/cpu/autodiff.cpp), riscritte come kernel CUDA. Per
// correttezza-prima-che-prestazioni (vedi note di progetto), matmulBackward
// e addBiasBackward(dBias) usano kernel "ingenui" (un thread per elemento
// di output, loop di riduzione dentro il thread) invece di cuBLAS con
// trasposizioni: e' la stessa struttura a triplo loop del backend CPU,
// solo parallelizzata, il che rende piu' facile fidarsi della
// corrispondenza esatta con la versione CPU gia' verificata.

struct MatmulGrad {
    DeviceTensor dA;
    DeviceTensor dB;
};
// matmul: C = A @ B, A:[M,K], B:[K,N], C:[M,N].
MatmulGrad matmulBackward(const DeviceTensor& a, const DeviceTensor& b, const DeviceTensor& gradOutput);

struct AddBiasGrad {
    DeviceTensor dInput;
    DeviceTensor dBias;
};
// addBias: output = input + bias (broadcast su ogni riga del batch).
AddBiasGrad addBiasBackward(const DeviceTensor& gradOutput);

DeviceTensor siluBackward(const DeviceTensor& input, const DeviceTensor& gradOutput);
DeviceTensor reluBackward(const DeviceTensor& input, const DeviceTensor& gradOutput);
DeviceTensor geluBackward(const DeviceTensor& input, const DeviceTensor& gradOutput);

// rmsnorm: y = x / sqrt(mean(x^2) + eps), riga per riga.
DeviceTensor rmsnormBackward(const DeviceTensor& input, const DeviceTensor& gradOutput);

}  // namespace blackforge::backend::cuda
