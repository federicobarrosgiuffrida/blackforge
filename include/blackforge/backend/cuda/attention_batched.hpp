#pragma once

#include <cstddef>

#include "blackforge/backend/cuda/device_tensor.hpp"

namespace blackforge::backend::cuda {

// Nucleo batchizzato dell'attention multi-testa, condiviso da
// selfAttention/selfAttentionIncremental (ops_transformer.cu) e da
// selfAttentionBackward (autodiff.cu).
//
// PERCHE': l'implementazione precedente calcolava gli score/l'output
// dell'attention con un doppio loop host-side 'for batch: for testa:',
// lanciando ~9 kernel separati (extractHead x3, matmulTransposeB,
// scale, mask, softmax, matmul, scatterHead) per OGNI combinazione
// (batch, testa). Con batch_size=8 e 8 teste, sono 64 iterazioni per
// singolo layer 'attention' — misurato che un batch a batch_size=8
// costa circa 8 volte un batch a batch_size=1 (0.46s vs 0.06s), quasi
// esattamente il fattore di scala del loop: il collo di bottiglia
// dominante era il NUMERO di kernel lanciati, non il calcolo in se'.
//
// Le funzioni qui sotto calcolano lo score/l'output dell'attention per
// TUTTE le combinazioni (batch, testa) in un singolo kernel (thread
// per elemento di output finale, indicizzazione strided sul layout
// [batch, seq, dim] con le teste contigue in 'dim'): il numero di
// lanci di kernel non dipende piu' da batch_size ne' da numHeads.
//
// GENERALI rispetto a newLen/totalLen (non solo seq==seq): permettono
// di condividere lo stesso codice tra l'attention "piena" (newLen ==
// totalLen == seq, oldLen == 0 — usata da selfAttention) e
// l'attention incrementale con cache K/V (newLen == token nuovi,
// totalLen == lunghezza della cache, oldLen == totalLen - newLen —
// usata da selfAttentionIncremental): la maschera causale ordinaria
// e' semplicemente il caso oldLen == 0 della maschera incrementale.

// Q@K^T scalato per tutte le teste in un colpo solo: q [batch, newLen,
// dim], k [batch, totalLen, dim] (heads contigue in 'dim', headDim =
// dim / numHeads) -> scores [batch, numHeads, newLen, totalLen].
DeviceTensor batchedQK(const DeviceTensor& q, const DeviceTensor& k, std::size_t numHeads, float scaleFactor);

// Maschera causale (eventualmente incrementale, vedi sopra) applicata
// a scores [batch, numHeads, newLen, totalLen]: la query alla riga i
// (posizione assoluta oldLen + i) puo' attendere solo alle chiavi
// j <= oldLen + i. oldLen == 0 riproduce la maschera causale
// ordinaria (query e chiavi coincidono, nessuna cache pregressa).
DeviceTensor batchedMask(const DeviceTensor& scores, std::size_t oldLen);

// probs @ V per tutte le teste, scritto DIRETTAMENTE nel layout
// unito [batch, newLen, dim] (equivalente a un batched matmul seguito
// da un "merge delle teste", ma senza il passaggio intermedio): probs
// [batch, numHeads, newLen, totalLen], v [batch, totalLen, dim] ->
// merged [batch, newLen, dim].
DeviceTensor batchedPV(const DeviceTensor& probs, const DeviceTensor& v, std::size_t numHeads);

// Gradienti di batchedQK rispetto a q/k, dati i gradienti degli score
// (dScores, stessa forma di scores). Stesse forme newLen/totalLen di
// batchedQK. Usata solo dal training (selfAttentionBackward): la
// generazione incrementale e' solo inferenza, nessun backward.
struct BatchedQKGrad {
    DeviceTensor dQ;
    DeviceTensor dK;
};
BatchedQKGrad batchedQKBackward(const DeviceTensor& dScores, const DeviceTensor& q, const DeviceTensor& k,
                                 std::size_t numHeads, float scaleFactor);

// Gradienti di batchedPV rispetto a probs/v, dato il gradiente
// dell'uscita unita (dMerged, stessa forma di merged).
struct BatchedPVGrad {
    DeviceTensor dProbs;
    DeviceTensor dV;
};
BatchedPVGrad batchedPVBackward(const DeviceTensor& dMerged, const DeviceTensor& probs, const DeviceTensor& v,
                                 std::size_t numHeads);

}  // namespace blackforge::backend::cuda
