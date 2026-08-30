#pragma once

#include <cmath>
#include <cstdint>

#include "blackforge/blackbit/device_shared.hpp"

// Arrotondamento stocastico e generatore pseudocasuale A CONTATORE
// (requisito 8).
//
// PERCHE' SERVE
//
// Un peso ternario ha una griglia di passo 'scala' (tipicamente ~0,03
// per una matrice inizializzata). Un aggiornamento tipico di
// addestramento vale lr * gradiente ~ 1e-4, cioe' ~300 volte piu'
// piccolo del passo della griglia. Arrotondato al valore piu' vicino,
// SPARISCE: il modello smetterebbe di imparare senza che nulla lo
// segnali. Con l'arrotondamento stocastico lo stesso aggiornamento
// diventa un cambio di trit con probabilita' 1/300 — e in valore atteso
// l'aggiornamento e' applicato ESATTAMENTE:
//
//     E[round_stocastico(x)] = x
//
// Su 292 M pesi per layer, "probabilita' 1/300" significa circa un
// milione di trit che cambiano davvero ad ogni passo.
//
// PERCHE' UN GENERATORE A CONTATORE
//
// Un generatore con stato (mt19937, xorshift con seme che avanza) e'
// inutilizzabile in un kernel CUDA senza dare a ogni thread il proprio
// stato e senza rendere il risultato dipendente dall'ordine di
// esecuzione. Qui il numero casuale e' una FUNZIONE PURA di (seme,
// indice): ogni thread calcola il proprio senza stato condiviso, e
// due esecuzioni con lo stesso seme producono bit identici
// indipendentemente da come i thread sono schedulati. E' anche cio'
// che rende i test riproducibili.
//
// L'hash e' splitmix64 (Steele, Lea & Flood 2014): un passo di mixing
// a 64 bit con ottime proprieta' statistiche e costo di poche
// istruzioni, senza tabelle.

namespace blackforge::blackbit {

// Mescola seme e indice in 64 bit pseudocasuali di buona qualita'.
BLACKFORGE_HOST_DEVICE inline std::uint64_t splitMix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

BLACKFORGE_HOST_DEVICE inline std::uint64_t counterRandom(std::uint64_t seed, std::uint64_t index) {
    // Due passi di mixing: il primo decorrela il seme, il secondo
    // l'indice. Con un solo passo indici consecutivi restano
    // correlati nei bit alti, che e' esattamente cio' che
    // counterUniform() usa.
    return splitMix64(splitMix64(seed) ^ (index + 0x9E3779B97F4A7C15ULL));
}

// Uniforme in [0, 1): 24 bit di mantissa, il massimo che un float
// rappresenta senza perdita.
BLACKFORGE_HOST_DEVICE inline float counterUniform(std::uint64_t seed, std::uint64_t index) {
    const std::uint32_t bits = static_cast<std::uint32_t>(counterRandom(seed, index) >> 40);  // 24 bit
    return static_cast<float>(bits) * (1.0F / 16777216.0F);
}

// Arrotonda 'value' a un intero in modo stocastico: verso il basso con
// probabilita' pari alla distanza dal soffitto, verso l'alto
// altrimenti. E' esatto in valore atteso (E[risultato] == value) per
// ogni value finito.
BLACKFORGE_HOST_DEVICE inline float stochasticRound(float value, std::uint64_t seed, std::uint64_t index) {
    const float lower = std::floor(value);
    const float fraction = value - lower;
    // '<' e non '<=': con fraction == 0 (value gia' intero) nessun
    // sorteggio puo' spostarlo, come deve essere.
    return counterUniform(seed, index) < fraction ? lower + 1.0F : lower;
}

// Arrotonda 'value' (gia' espresso in unita' di scala, cioe' peso /
// scala) al trit piu' vicino in modo stocastico, saturando a
// [-1, +1].
//
// ATTENZIONE alla saturazione: e' l'unico punto in cui
// l'arrotondamento NON e' esatto in valore atteso. E' inevitabile —
// la griglia ternaria non ha valori oltre +-1 — ed e' anche
// desiderabile: impedisce a un gradiente anomalo di spingere un peso
// fuori dal formato. Chi ha bisogno della proprieta' di non
// distorsione (i test statistici) deve restare dentro [-1, +1].
BLACKFORGE_HOST_DEVICE inline int stochasticRoundToTrit(float value, std::uint64_t seed, std::uint64_t index) {
    const float rounded = stochasticRound(value, seed, index);
    if (rounded <= -1.0F) {
        return -1;
    }
    if (rounded >= 1.0F) {
        return 1;
    }
    return static_cast<int>(rounded);
}

}  // namespace blackforge::blackbit
