#pragma once

#include "blackforge/runtime/tensor.hpp"

namespace blackforge::backend::cpu {

using runtime::Tensor;

// Funzioni di backward (reverse-mode) per le operazioni del backend
// CPU. Ognuna calcola i gradienti degli ingressi di un'operazione a
// partire dal gradiente della sua uscita (dL/doutput), secondo la
// regola della catena. Sono formule analitiche scritte a mano per il
// piccolo insieme di operazioni attualmente supportato, non un
// autodiff generico basato su un grafo di espressioni.

// matmul: C = A @ B, A:[M,K], B:[K,N], C:[M,N]. Restituisce dA, dB.
struct MatmulGrad {
    Tensor dA;
    Tensor dB;
};
MatmulGrad matmulBackward(const Tensor& a, const Tensor& b, const Tensor& gradOutput);

// matmulTransposeB: C = A @ B^T, A:[M,K], B:[N,K], C:[M,N]. Restituisce
// dA, dB. Serve per il backward degli score di attention (Q @ K^T).
struct MatmulTransposeBGrad {
    Tensor dA;
    Tensor dB;
};
MatmulTransposeBGrad matmulTransposeBBackward(const Tensor& a, const Tensor& b, const Tensor& gradOutput);

// addBias: output = input + bias (bias trasmesso su ogni "riga", vedi
// ops.hpp/addBias per cosa si intende a rango > 2). Restituisce dInput
// (== gradOutput, l'addizione e' un'identita' nel gradiente) e dBias
// (somma di gradOutput lungo tutte le righe).
struct AddBiasGrad {
    Tensor dInput;
    Tensor dBias;
};
AddBiasGrad addBiasBackward(const Tensor& gradOutput);

Tensor siluBackward(const Tensor& input, const Tensor& gradOutput);
Tensor reluBackward(const Tensor& input, const Tensor& gradOutput);
Tensor geluBackward(const Tensor& input, const Tensor& gradOutput);

// rmsnorm: y = x / rms(x), rms(x) = sqrt(mean(x^2) + eps), applicato
// riga per riga (rango >= 2, vedi ops.hpp/rmsnorm). 'input' e'
// l'ingresso cachato dal forward (serve a ricalcolare rms(x) e la
// somma pesata usata dalla derivata). Lancia std::invalid_argument se
// le forme non coincidono o il rango e' minore di 2.
Tensor rmsnormBackward(const Tensor& input, const Tensor& gradOutput);

// softmax: ricalcola y = softmax(input) internamente (la formula del
// gradiente, dx_j = y_j * (gOut_j - sum_i(gOut_i * y_i)), ha bisogno
// dell'uscita y, non dell'ingresso grezzo) invece di richiedere al
// chiamante di cachare l'uscita anziche' l'ingresso: un ricalcolo in
// piu' trascurabile, per restare coerenti con la firma (input,
// gradOutput) di tutte le altre funzioni di questo file.
Tensor softmaxBackward(const Tensor& input, const Tensor& gradOutput);

// embeddingLookup: output[b,s,:] = table[tokenIds[b,s],:]. Nessun
// gradiente rispetto ai token id (indici, non differenziabili): solo
// dTable, via scatter-add (piu' occorrenze dello stesso token
// accumulano il gradiente sulla stessa riga). Lancia
// std::invalid_argument se le forme non corrispondono o un token id e'
// fuori da [0, vocabSize).
Tensor embeddingLookupBackward(const Tensor& tokenIds, const Tensor& gradOutput, std::size_t vocabSize);

// addPositionalEmbedding: output[b,s,d] = input[b,s,d] + table[s,d].
// dInput e' un'identita' nel gradiente; dTable somma il gradiente su
// tutto il batch, posizione per posizione.
struct PositionalEmbeddingGrad {
    Tensor dInput;
    Tensor dTable;
};
PositionalEmbeddingGrad addPositionalEmbeddingBackward(const Tensor& gradOutput, std::size_t maxSeqLen);

// feedForward: y = x + Linear2(SiLU(Linear1(RMSNorm(x)))) (vedi
// ops.hpp/feedForward). Ricalcola internamente rmsnorm/linear1/silu
// (come softmaxBackward ricalcola softmax): 'input' e' l'unico stato
// che il chiamante deve aver cachato dal forward.
struct FeedForwardGrad {
    Tensor dInput;
    Tensor dW1;
    Tensor dB1;
    Tensor dW2;
    Tensor dB2;
};
FeedForwardGrad feedForwardBackward(const Tensor& input, const Tensor& w1, const Tensor& b1, const Tensor& w2,
                                     const Tensor& b2, const Tensor& gradOutput);

// selfAttention: y = x + Wout(MultiHeadCausalAttention(...)) (vedi
// ops.hpp/selfAttention). Ricalcola internamente l'intero forward
// (normed, Q/K/V, score mascherati, probabilita', concatenazione):
// stessa scelta di feedForwardBackward, 'input' e' l'unico stato da
// cachare.
struct SelfAttentionGrad {
    Tensor dInput;
    Tensor dWq;
    Tensor dWk;
    Tensor dWv;
    Tensor dWout;
};
SelfAttentionGrad selfAttentionBackward(const Tensor& input, const Tensor& wq, const Tensor& wk, const Tensor& wv,
                                         const Tensor& wout, std::size_t numHeads, const Tensor& gradOutput);

}  // namespace blackforge::backend::cpu
