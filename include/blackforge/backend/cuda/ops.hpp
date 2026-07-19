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

// Prodotto matriciale 2D via cuBLASLt su Tensor Core: [M, K] x [K, N]
// -> [M, N]. A differenza di matmul() (SGEMM, calcolo interamente in
// float32 su CUDA core), qui gli operandi vengono convertiti in BF16
// prima del prodotto (Tensor Core reale, non simulato: throughput
// significativamente maggiore su hardware Blackwell/Hopper/Ampere),
// con accumulo e uscita in float32 (lo schema "mixed precision"
// standard usato per l'addestramento di modelli linguistici — BF16 ha
// lo stesso range di esponente di FP32, quindi a differenza di FP16
// non serve loss scaling per evitare overflow/underflow). L'input
// resta float32 (stesso contratto di ogni altra funzione di questo
// modulo): la conversione a BF16 e' un dettaglio interno.
DeviceTensor matmulBf16(const DeviceTensor& a, const DeviceTensor& b);

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

// Come addPositionalEmbedding, ma la riga della tabella usata per la
// posizione s e' 'offset + s' invece di semplicemente 's' (vedi
// backend::cpu::addPositionalEmbeddingAt, stessa semantica: serve alla
// generazione incrementale, dove ogni chiamata processa solo i token
// NUOVI di una sequenza che sta crescendo).
DeviceTensor addPositionalEmbeddingAt(const DeviceTensor& input, const DeviceTensor& table, std::size_t offset);

// Blocco feed-forward pre-norm con residual (vedi
// backend::cpu::feedForward per i dettagli, stessa semantica):
// y = x + Linear2(SiLU(Linear1(RMSNorm(x)))).
DeviceTensor feedForward(const DeviceTensor& input, const DeviceTensor& w1, const DeviceTensor& b1,
                          const DeviceTensor& w2, const DeviceTensor& b2);

// Come feedForward(), ma i due prodotti matriciali interni (Linear1,
// Linear2) usano linearBf16() (Tensor Core) invece di linear() (SGEMM
// float32) — stessa semantica, stesso schema "mixed precision" di
// linearBf16/matmulBf16 (vedi sopra). Usata da cuda::Model quando
// 'precision { compute bf16 }' e' dichiarato (vedi
// cuda::Model::useTensorCoreLinear_), esattamente come linearBf16 lo e'
// gia' per il layer di pipeline 'linear'.
DeviceTensor feedForwardBf16(const DeviceTensor& input, const DeviceTensor& w1, const DeviceTensor& b1,
                              const DeviceTensor& w2, const DeviceTensor& b2);

// Self-attention causale multi-head, pre-norm con residual (vedi
// backend::cpu::selfAttention per i dettagli, stessa semantica):
// y = x + Wout(MultiHeadAttention(Q,K,V da RMSNorm(x))). dim deve
// essere divisibile per numHeads.
DeviceTensor selfAttention(const DeviceTensor& input, const DeviceTensor& wq, const DeviceTensor& wk,
                            const DeviceTensor& wv, const DeviceTensor& wout, std::size_t numHeads);

// Come selfAttention(), ma le proiezioni Q/K/V/Out usano matmulBf16()
// (Tensor Core) invece di matmul() (SGEMM float32) — stessa semantica,
// stesso schema "mixed precision". Il calcolo dell'attention vera e
// propria (Q@K^T scalato, maschera, softmax, probs@V — vedi
// fused_attention.hpp) resta in float32: solo le proiezioni lineari
// (la parte dominante del costo computazionale) passano per il Tensor
// Core, come richiesto esplicitamente per questa milestone.
DeviceTensor selfAttentionBf16(const DeviceTensor& input, const DeviceTensor& wq, const DeviceTensor& wk,
                                const DeviceTensor& wv, const DeviceTensor& wout, std::size_t numHeads);

// Chiavi/valori accumulati di un layer 'attention' attraverso una
// sessione di generazione autoregressiva incrementale sul device (vedi
// backend::cpu::KVCache, stessa semantica): 'k'/'v' hanno forma
// [batch, length, dim], crescente di 'newLen' ad ogni chiamata a
// selfAttentionIncremental. Un'istanza per-layer, non condivisa tra
// layer diversi.
struct KVCache {
    DeviceTensor k;
    DeviceTensor v;
    std::size_t length = 0;
};

// Variante incrementale di selfAttention su device, pensata per la
// generazione autoregressiva token per token (vedi
// backend::cpu::selfAttentionIncremental per la semantica completa,
// identica qui: produce esattamente lo stesso risultato che
// selfAttention() darebbe per ogni posizione nuova se le venisse
// passata l'intera sequenza fino a quel punto — un'ottimizzazione, non
// un'approssimazione). 'cache' deve essere una KVCache{} appena
// costruita (length == 0) all'inizio di una nuova sequenza di
// generazione. Lancia std::invalid_argument se dim non e' divisibile
// per numHeads o se l'input non e' a rango 3.
DeviceTensor selfAttentionIncremental(const DeviceTensor& newInput, const DeviceTensor& wq, const DeviceTensor& wk,
                                       const DeviceTensor& wv, const DeviceTensor& wout, std::size_t numHeads,
                                       KVCache& cache);

// Come selfAttentionIncremental(), ma le proiezioni Q/K/V/Out usano
// matmulBf16() invece di matmul() (stessa relazione di
// selfAttentionBf16() rispetto a selfAttention()): usata da
// Model::forwardIncremental quando 'precision { compute bf16 }' e'
// dichiarato, per coerenza con il resto della pipeline addestrata in
// BF16.
DeviceTensor selfAttentionIncrementalBf16(const DeviceTensor& newInput, const DeviceTensor& wq,
                                           const DeviceTensor& wk, const DeviceTensor& wv, const DeviceTensor& wout,
                                           std::size_t numHeads, KVCache& cache);

// Layer lineare: input [..., inFeatures], weight [inFeatures, outFeatures],
// bias [outFeatures] -> output [..., outFeatures]. Generalizzato a
// rango >= 2 (vedi DeviceTensor::reshaped): per rango > 2 appiattisce
// temporaneamente a [rows, inFeatures] prima del prodotto matriciale.
DeviceTensor linear(const DeviceTensor& input, const DeviceTensor& weight, const DeviceTensor& bias);

// Come linear(), ma il prodotto matriciale usa matmulBf16() (Tensor
// Core reale) invece di matmul() (SGEMM float32): stessa generalizzazione
// a rango >= 2, stesso bias in float32 pieno (solo il prodotto matriciale
// passa per il Tensor Core, non l'addizione del bias).
DeviceTensor linearBf16(const DeviceTensor& input, const DeviceTensor& weight, const DeviceTensor& bias);

}  // namespace blackforge::backend::cuda
