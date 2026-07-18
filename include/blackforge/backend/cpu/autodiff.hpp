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

// addBias: output = input + bias (bias trasmesso su ogni riga del
// batch). Restituisce dInput (== gradOutput, l'addizione e' un'identita'
// nel gradiente) e dBias (somma di gradOutput lungo il batch).
struct AddBiasGrad {
    Tensor dInput;
    Tensor dBias;
};
AddBiasGrad addBiasBackward(const Tensor& gradOutput);

Tensor siluBackward(const Tensor& input, const Tensor& gradOutput);
Tensor reluBackward(const Tensor& input, const Tensor& gradOutput);
Tensor geluBackward(const Tensor& input, const Tensor& gradOutput);

// rmsnorm: y = x / rms(x), rms(x) = sqrt(mean(x^2) + eps), applicato
// riga per riga. 'input' e' l'ingresso cachato dal forward (serve a
// ricalcolare rms(x) e la somma pesata usata dalla derivata). Lancia
// std::invalid_argument se le forme non coincidono o non sono a rango 2.
Tensor rmsnormBackward(const Tensor& input, const Tensor& gradOutput);

// softmax: ricalcola y = softmax(input) internamente (la formula del
// gradiente, dx_j = y_j * (gOut_j - sum_i(gOut_i * y_i)), ha bisogno
// dell'uscita y, non dell'ingresso grezzo) invece di richiedere al
// chiamante di cachare l'uscita anziche' l'ingresso: un ricalcolo in
// piu' trascurabile, per restare coerenti con la firma (input,
// gradOutput) di tutte le altre funzioni di questo file.
Tensor softmaxBackward(const Tensor& input, const Tensor& gradOutput);

}  // namespace blackforge::backend::cpu
