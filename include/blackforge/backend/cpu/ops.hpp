#pragma once

#include "blackforge/runtime/tensor.hpp"

namespace blackforge::backend::cpu {

using runtime::Tensor;

// Somma elementwise di due tensori con la stessa forma.
Tensor add(const Tensor& a, const Tensor& b);

// Moltiplica ogni elemento per uno scalare.
Tensor scale(const Tensor& input, float factor);

// Somma un bias di forma [features] a un tensore [..., features],
// trasmettendolo (broadcast) su ogni "riga" (tutte le dimensioni tranne
// l'ultima: per un input a rango 2 [batch, features] equivale al
// broadcast per riga classico; per un input a rango 3 [batch, seq,
// features], uscita di 'embedding', si applica ad ogni token di ogni
// sequenza). Lancia std::invalid_argument se il rango e' minore di 2.
Tensor addBias(const Tensor& input, const Tensor& bias);

// Prodotto matriciale 2D: [M, K] x [K, N] -> [M, N]. Primitivo
// puramente 2D (non generalizzato a ranghi maggiori): e' 'linear' che
// si occupa di appiattire un input a rango > 2 prima di chiamarlo.
Tensor matmul(const Tensor& a, const Tensor& b);

// a: [M, K], b: [N, K] -> a @ b^T: [M, N]. Serve per gli score di
// attention (Q @ K^T) senza dover trasporre esplicitamente K.
Tensor matmulTransposeB(const Tensor& a, const Tensor& b);

Tensor silu(const Tensor& input);
Tensor relu(const Tensor& input);
Tensor gelu(const Tensor& input);

// RMSNorm (Zhang & Sennrich, 2019): normalizza ogni "riga" (tutte le
// dimensioni tranne l'ultima) di un tensore per la sua root-mean-square,
// y = x / sqrt(mean(x^2) + eps). A differenza della formulazione più
// comune (usata ad es. in LLaMA), questa versione NON ha un fattore di
// scala gamma allenabile: è normalizzazione pura, senza parametri. eps
// è fisso a 1e-6 (kRmsNormEps in ops.cpp), non configurabile dal
// linguaggio. Generalizzata a rango >= 2 (vedi addBias). Lancia
// std::invalid_argument se il rango è minore di 2.
Tensor rmsnorm(const Tensor& input);

// Softmax lungo l'ultima dimensione di un tensore, riga per riga (vedi
// addBias per cosa si intende per "riga" a rango > 2): y_j =
// exp(x_j - max) / sum_k exp(x_k - max) (sottrazione del massimo per
// stabilità numerica). A differenza di softmaxCrossEntropy in loss.hpp
// (che applica softmax internamente in una formula combinata con la
// cross-entropy, più efficiente e stabile per l'addestramento), questa
// è softmax come operazione di pipeline a sé stante: serve a ottenere
// probabilità esplicite in uscita da un modello, e viene riusata anche
// dentro l'attention (sugli score [seq, seq]). Lancia
// std::invalid_argument se il rango è minore di 2.
Tensor softmax(const Tensor& input);

// Layer lineare: input [..., inFeatures], weight [inFeatures, outFeatures],
// bias [outFeatures] -> output [..., outFeatures] (stessa forma
// dell'input tranne l'ultima dimensione). Per rango > 2 (es. [batch,
// seq, inFeatures]) appiattisce temporaneamente a [batch*seq,
// inFeatures] prima del prodotto matriciale (stesso layout flat
// riga-maggiore, nessuna vera riorganizzazione dei dati).
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);

// Lookup di embedding: tokenIds [batch, seq] (valori non negativi che
// rappresentano indici interi, memorizzati come float — BlackForge non
// ha ancora un dtype intero dedicato, è una semplificazione pragmatica
// documentata), table [vocabSize, dim] -> output [batch, seq, dim].
// Lancia std::invalid_argument se un token id è fuori da [0, vocabSize).
Tensor embeddingLookup(const Tensor& tokenIds, const Tensor& table);

// Aggiunge un embedding posizionale allenabile: input [batch, seq, dim],
// table [maxSeqLen, dim] -> output[b,s,d] = input[b,s,d] + table[s,d]
// (trasmesso su ogni esempio del batch). Lancia std::invalid_argument
// se seq supera maxSeqLen o se dim non coincide.
Tensor addPositionalEmbedding(const Tensor& input, const Tensor& table);

// Blocco feed-forward pre-norm con residual (convenzione usata da
// LLaMA/GPT-NeoX e simili): y = x + Linear2(SiLU(Linear1(RMSNorm(x)))).
// input: [batch, seq, dim] (o [batch, dim]). w1: [dim, hiddenDim],
// b1: [hiddenDim]. w2: [hiddenDim, dim], b2: [dim]. Il residual e il
// pre-norm sono incorporati nell'operazione stessa: il linguaggio
// BlackForge non ha ancora una sintassi per esprimere connessioni
// residuali esplicite nella pipeline (che resta una sequenza lineare
// |>), quindi 'feedforward' le include internamente — e' l'unico modo,
// con la sintassi attuale, di costruire un blocco transformer profondo
// che si allena davvero.
Tensor feedForward(const Tensor& input, const Tensor& w1, const Tensor& b1, const Tensor& w2, const Tensor& b2);

// Self-attention causale multi-head, pre-norm con residual: y = x +
// Wout(MultiHeadAttention(Q,K,V da RMSNorm(x))), con maschera causale
// (ogni posizione puo' attendere solo a se stessa e alle precedenti —
// necessaria per un modello linguistico autoregressivo). input:
// [batch, seq, dim]. wq/wk/wv/wout: [dim, dim], SENZA bias (come in
// LLaMA: meno parametri, empiricamente equivalente per questo tipo di
// proiezioni). dim deve essere divisibile per numHeads. Come
// 'feedforward', residual e pre-norm sono incorporati nell'operazione.
// Lancia std::invalid_argument se dim non e' divisibile per numHeads o
// se l'input non e' a rango 3.
Tensor selfAttention(const Tensor& input, const Tensor& wq, const Tensor& wk, const Tensor& wv, const Tensor& wout,
                      std::size_t numHeads);

}  // namespace blackforge::backend::cpu
