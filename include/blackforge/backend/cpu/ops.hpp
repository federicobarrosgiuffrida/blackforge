#pragma once

#include "blackforge/runtime/tensor.hpp"

namespace blackforge::backend::cpu {

using runtime::Tensor;

// Somma elementwise di due tensori con la stessa forma.
Tensor add(const Tensor& a, const Tensor& b);

// Moltiplica ogni elemento per uno scalare.
Tensor scale(const Tensor& input, float factor);

// Somma un bias di forma [features] a un tensore [batch, features],
// trasmettendolo (broadcast) su ogni riga del batch.
Tensor addBias(const Tensor& input, const Tensor& bias);

// Prodotto matriciale 2D: [M, K] x [K, N] -> [M, N].
Tensor matmul(const Tensor& a, const Tensor& b);

Tensor silu(const Tensor& input);
Tensor relu(const Tensor& input);
Tensor gelu(const Tensor& input);

// RMSNorm (Zhang & Sennrich, 2019): normalizza ogni riga di un tensore
// [batch, features] per la sua root-mean-square, y = x / sqrt(mean(x^2)
// + eps). A differenza della formulazione piu' comune (usata ad es. in
// LLaMA), questa versione NON ha un fattore di scala gamma allenabile:
// e' normalizzazione pura, senza parametri. eps e' fisso a 1e-6
// (kRmsNormEps in ops.cpp), non configurabile dal linguaggio. Lancia
// std::invalid_argument se il tensore non e' a rango 2.
Tensor rmsnorm(const Tensor& input);

// Softmax riga per riga su un tensore [batch, classi]: y_j =
// exp(x_j - max) / sum_k exp(x_k - max) (sottrazione del massimo per
// stabilita' numerica). A differenza di softmaxCrossEntropy in
// loss.hpp (che applica softmax internamente in una formula combinata
// con la cross-entropy, piu' efficiente e stabile per l'addestramento),
// questa e' softmax come operazione di pipeline a se stante: serve a
// ottenere probabilita' esplicite in uscita da un modello (es. per
// l'inferenza), non e' pensata per essere seguita da cross-entropy
// nello stesso grafo (in quel caso conviene la loss combinata). Lancia
// std::invalid_argument se il tensore non e' a rango 2.
Tensor softmax(const Tensor& input);

// Layer lineare: input [batch, inFeatures], weight [inFeatures, outFeatures],
// bias [outFeatures] -> output [batch, outFeatures].
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);

}  // namespace blackforge::backend::cpu
