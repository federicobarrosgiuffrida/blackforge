#pragma once

#include <cmath>
#include <cstddef>

#include "blackforge/blackbit/device_shared.hpp"

// Posizioni rotanti (RoPE, Su et al. 2021).
//
// Ogni coppia di componenti (2i, 2i+1) di una testa viene ruotata di un
// angolo proporzionale alla posizione assoluta del token:
//
//     theta_i = base^(-2i / headDim)
//     angolo  = posizione * theta_i
//
// Il prodotto scalare fra due query/chiavi ruotate dipende allora solo
// dalla loro distanza relativa, che e' la proprieta' che si vuole.
//
// Zero parametri (a differenza della tabella posizionale allenabile che
// il resto del motore usa per i modelli del linguaggio BlackForge), e
// definita per qualunque posizione: la lunghezza di contesto puo'
// crescere senza reinizializzare nulla.
//
// Funzione pura, condivisa host/device: la stessa formula per il
// backend CPU e per il kernel CUDA.

namespace blackforge::blackbit {

inline constexpr float kRopeBase = 10000.0F;

// Angolo della coppia 'pair' (0 <= pair < headDim/2) alla posizione
// 'position'.
BLACKFORGE_HOST_DEVICE inline float ropeAngle(std::size_t position, std::size_t pair, std::size_t headDim) {
    const float exponent = -2.0F * static_cast<float>(pair) / static_cast<float>(headDim);
    return static_cast<float>(position) * std::pow(kRopeBase, exponent);
}

// Ruota in place le componenti di UNA testa (headDim valori contigui)
// per la posizione data. headDim deve essere pari.
BLACKFORGE_HOST_DEVICE inline void applyRope(float* head, std::size_t headDim, std::size_t position) {
    const std::size_t pairs = headDim / 2;
    for (std::size_t p = 0; p < pairs; ++p) {
        const float angle = ropeAngle(position, p, headDim);
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        // Coppie (p, p + pairs), la convenzione "meta' e meta'" usata da
        // LLaMA: le due componenti ruotate sono lontane headDim/2 nel
        // vettore, non adiacenti. Cambiare convenzione qui cambierebbe
        // i pesi salvati, quindi resta fissata.
        const float a = head[p];
        const float b = head[p + pairs];
        head[p] = a * cosine - b * sine;
        head[p + pairs] = a * sine + b * cosine;
    }
}

// Rotazione inversa, usata dal backward: RoPE e' una rotazione, quindi
// la sua trasposta (che e' cio' che serve per propagare il gradiente) e'
// la rotazione dell'angolo opposto.
BLACKFORGE_HOST_DEVICE inline void applyRopeTranspose(float* head, std::size_t headDim, std::size_t position) {
    const std::size_t pairs = headDim / 2;
    for (std::size_t p = 0; p < pairs; ++p) {
        const float angle = ropeAngle(position, p, headDim);
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float a = head[p];
        const float b = head[p + pairs];
        head[p] = a * cosine + b * sine;
        head[p + pairs] = -a * sine + b * cosine;
    }
}

}  // namespace blackforge::blackbit
