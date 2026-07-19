#pragma once

#include "blackforge/backend/cuda/device_tensor.hpp"

namespace blackforge::backend::cuda {

// Somma elementwise di due tensori device con la stessa forma.
DeviceTensor add(const DeviceTensor& a, const DeviceTensor& b);

// Somma un bias device di forma [features] a un tensore [..., features],
// trasmettendolo (broadcast) su ogni "riga" (tutte le dimensioni tranne
// l'ultima). Generalizzato a rango >= 2 (vedi backend::cpu::addBias per
// i dettagli, stessa semantica).
DeviceTensor addBias(const DeviceTensor& input, const DeviceTensor& bias);

// Prodotto matriciale 2D via cuBLAS SGEMM: [M, K] x [K, N] -> [M, N].
// Primitivo puramente 2D (non generalizzato): e' 'linear' che si
// occupa di appiattire un input a rango > 2 prima di chiamarlo.
DeviceTensor matmul(const DeviceTensor& a, const DeviceTensor& b);

// a: [M, K], b: [N, K] -> a @ b^T: [M, N]. Kernel scritto a mano (non
// cuBLAS con trasposizioni): serve per gli score di attention (Q @ K^T),
// stessa scelta di correttezza-prima-che-prestazioni di matmulBackward.
DeviceTensor matmulTransposeB(const DeviceTensor& a, const DeviceTensor& b);

DeviceTensor silu(const DeviceTensor& input);
DeviceTensor relu(const DeviceTensor& input);
DeviceTensor gelu(const DeviceTensor& input);

// RMSNorm senza gamma allenabile (vedi backend::cpu::rmsnorm per i
// dettagli): un blocco CUDA per riga, con una riduzione in shared
// memory per calcolare la somma dei quadrati. Generalizzato a rango
// >= 2. eps fisso a 1e-6, identico al backend CPU (kRmsNormEps in
// ops_elementwise.cu), per permettere il confronto diretto CPU/GPU nei
// test.
DeviceTensor rmsnorm(const DeviceTensor& input);

// Softmax lungo l'ultima dimensione, riga per riga (vedi
// backend::cpu::softmax per i dettagli, stessa semantica).
// Generalizzato a rango >= 2. eps non applicabile qui: la stabilita'
// numerica viene dalla sottrazione del massimo per riga.
DeviceTensor softmax(const DeviceTensor& input);

// Lookup di embedding: tokenIds [batch, seq], table [vocabSize, dim]
// -> output [batch, seq, dim] (vedi backend::cpu::embeddingLookup per
// i dettagli, stessa semantica). Lancia std::invalid_argument se un
// token id e' fuori da [0, vocabSize).
DeviceTensor embeddingLookup(const DeviceTensor& tokenIds, const DeviceTensor& table);

// Aggiunge un embedding posizionale allenabile: input [batch, seq, dim],
// table [maxSeqLen, dim] -> output[b,s,d] = input[b,s,d] + table[s,d]
// (vedi backend::cpu::addPositionalEmbedding, stessa semantica).
DeviceTensor addPositionalEmbedding(const DeviceTensor& input, const DeviceTensor& table);

// Blocco feed-forward pre-norm con residual (vedi
// backend::cpu::feedForward per i dettagli, stessa semantica):
// y = x + Linear2(SiLU(Linear1(RMSNorm(x)))).
DeviceTensor feedForward(const DeviceTensor& input, const DeviceTensor& w1, const DeviceTensor& b1,
                          const DeviceTensor& w2, const DeviceTensor& b2);

// Self-attention causale multi-head, pre-norm con residual (vedi
// backend::cpu::selfAttention per i dettagli, stessa semantica):
// y = x + Wout(MultiHeadAttention(Q,K,V da RMSNorm(x))). dim deve
// essere divisibile per numHeads.
DeviceTensor selfAttention(const DeviceTensor& input, const DeviceTensor& wq, const DeviceTensor& wk,
                            const DeviceTensor& wv, const DeviceTensor& wout, std::size_t numHeads);

// Layer lineare: input [..., inFeatures], weight [inFeatures, outFeatures],
// bias [outFeatures] -> output [..., outFeatures]. Generalizzato a
// rango >= 2 (vedi DeviceTensor::reshaped): per rango > 2 appiattisce
// temporaneamente a [rows, inFeatures] prima del prodotto matriciale.
DeviceTensor linear(const DeviceTensor& input, const DeviceTensor& weight, const DeviceTensor& bias);

}  // namespace blackforge::backend::cuda
