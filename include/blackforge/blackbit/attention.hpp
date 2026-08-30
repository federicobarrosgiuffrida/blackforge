#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "blackforge/blackbit/config.hpp"
#include "blackforge/blackbit/gradient.hpp"
#include "blackforge/blackbit/ternary_linear.hpp"
#include "blackforge/runtime/tensor.hpp"

// Attention causale a query raggruppate (GQA) di BlackBit
// (requisito 5), con posizioni rotanti (RoPE).
//
// GQA SENZA DUPLICARE K/V
//
// 24 teste di query condividono 6 teste di chiave/valore. L'implementazione
// diffusa "ripete" K e V fino a 24 teste con una repeat_interleave e poi
// usa il kernel di attention ordinario: su BlackBit-9B con seq 4096
// sarebbero 4 * (24-6) * 4096 * 128 * 4 B = 150 MB per layer di copie
// pure, e altrettanti nel backward. Qui il raggruppamento resta
// IMPLICITO: la testa di query h legge la testa K/V h / (24/6) tramite
// un offset di indice. Nessun byte duplicato, in nessun momento.
//
// MAI LA MATRICE DI SCORE
//
// Come fusedAttention del backend CUDA di questo motore, il forward
// scorre le chiavi accumulando massimo e somma correnti (softmax
// online, Milakov & Gimelshein 2018) e non scrive mai la matrice
// [batch, teste, seq, seq]: per seq 4096 sarebbero 6,4 GB per layer. Si
// conservano solo le due statistiche per riga di query, O(seq), che il
// backward riusa per ricostruire le probabilita' una riga per volta.
//
// POSIZIONI ROTANTI
//
// RoPE (Su et al., 2021) invece della tabella posizionale allenabile
// usata dal resto del motore: zero parametri, si estende oltre la
// lunghezza vista in addestramento, ed e' quello che ogni modello
// recente usa. Viene applicata a q e k dopo le proiezioni.

namespace blackforge::blackbit {

// Statistiche softmax per riga di query, salvate dal forward e riusate
// dal backward: O(batch * teste * seq), non O(seq^2).
struct AttentionCache {
    runtime::Tensor query;      // [batch, seq, numHeads * headDim], dopo RoPE
    runtime::Tensor key;        // [batch, seq, numKvHeads * headDim], dopo RoPE
    runtime::Tensor value;      // [batch, seq, numKvHeads * headDim]
    runtime::Tensor attnOutput; // [batch, seq, numHeads * headDim], prima di o_proj
    std::vector<float> rowMax;  // [batch, numHeads, seq]
    std::vector<float> rowSum;  // [batch, numHeads, seq]
};

class GqaAttention {
public:
    GqaAttention(const std::string& namePrefix, const BlackBitConfig& config);

    void initialize(unsigned int seed);

    // input [batch, seq, hidden] -> output [batch, seq, hidden].
    [[nodiscard]] runtime::Tensor forward(const runtime::Tensor& input, AttentionCache& cache) const;

    [[nodiscard]] runtime::Tensor backward(const runtime::Tensor& input, const runtime::Tensor& gradOutput,
                                            const AttentionCache& cache, GradientSink* sink) const;

    [[nodiscard]] TernaryLinear& queryProjection() { return q_; }
    [[nodiscard]] TernaryLinear& keyProjection() { return k_; }
    [[nodiscard]] TernaryLinear& valueProjection() { return v_; }
    [[nodiscard]] TernaryLinear& outputProjection() { return o_; }

    [[nodiscard]] std::size_t parameterBytes() const {
        return q_.parameterBytes() + k_.parameterBytes() + v_.parameterBytes() + o_.parameterBytes();
    }

    void setComputeDType(ComputeDType dtype);

private:
    BlackBitConfig config_;
    TernaryLinear q_;
    TernaryLinear k_;
    TernaryLinear v_;
    TernaryLinear o_;
};

}  // namespace blackforge::blackbit
