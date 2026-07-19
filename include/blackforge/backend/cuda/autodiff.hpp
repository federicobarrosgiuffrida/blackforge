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

// Backward di matmulBf16 (vedi ops.hpp): stessa formula analitica di
// matmulBackward (dA = gradOutput @ B^T, dB = A^T @ gradOutput), ma
// entrambi i prodotti eseguiti via Tensor Core BF16 invece che SGEMM
// float32 — coerente con l'uso di matmulBf16 nel forward corrispondente.
MatmulGrad matmulBf16Backward(const DeviceTensor& a, const DeviceTensor& b, const DeviceTensor& gradOutput);

struct MatmulTransposeBGrad {
    DeviceTensor dA;
    DeviceTensor dB;
};
// matmulTransposeB: C = A @ B^T, A:[M,K], B:[N,K], C:[M,N].
MatmulTransposeBGrad matmulTransposeBBackward(const DeviceTensor& a, const DeviceTensor& b,
                                               const DeviceTensor& gradOutput);

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

// softmax: ricalcola y = softmax(input) internamente (la formula del
// gradiente ha bisogno dell'uscita, non dell'ingresso grezzo), stessa
// scelta della controparte CPU (vedi backend::cpu::softmaxBackward).
DeviceTensor softmaxBackward(const DeviceTensor& input, const DeviceTensor& gradOutput);

// embeddingLookup: nessun gradiente rispetto ai token id, solo dTable
// (vedi backend::cpu::embeddingLookupBackward, stessa semantica). Lo
// scatter-add usa atomicAdd (piu' occorrenze dello stesso token in
// thread diversi possono scrivere concorrentemente sulla stessa riga).
DeviceTensor embeddingLookupBackward(const DeviceTensor& tokenIds, const DeviceTensor& gradOutput,
                                      std::size_t vocabSize);

struct PositionalEmbeddingGrad {
    DeviceTensor dInput;
    DeviceTensor dTable;
};
// addPositionalEmbedding (vedi backend::cpu::addPositionalEmbeddingBackward).
PositionalEmbeddingGrad addPositionalEmbeddingBackward(const DeviceTensor& gradOutput, std::size_t maxSeqLen);

struct FeedForwardGrad {
    DeviceTensor dInput;
    DeviceTensor dW1;
    DeviceTensor dB1;
    DeviceTensor dW2;
    DeviceTensor dB2;
};
// feedForward (vedi backend::cpu::feedForwardBackward, stessa
// semantica: ricalcola internamente rmsnorm/linear1/silu).
FeedForwardGrad feedForwardBackward(const DeviceTensor& input, const DeviceTensor& w1, const DeviceTensor& b1,
                                     const DeviceTensor& w2, const DeviceTensor& b2, const DeviceTensor& gradOutput);

struct SelfAttentionGrad {
    DeviceTensor dInput;
    DeviceTensor dWq;
    DeviceTensor dWk;
    DeviceTensor dWv;
    DeviceTensor dWout;
};
// selfAttention (vedi backend::cpu::selfAttentionBackward, stessa
// semantica: ricalcola internamente l'intero forward).
SelfAttentionGrad selfAttentionBackward(const DeviceTensor& input, const DeviceTensor& wq, const DeviceTensor& wk,
                                         const DeviceTensor& wv, const DeviceTensor& wout, std::size_t numHeads,
                                         const DeviceTensor& gradOutput);

}  // namespace blackforge::backend::cuda
