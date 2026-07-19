#pragma once

#include <cstddef>

#include "blackforge/backend/cuda/device_tensor.hpp"

namespace blackforge::backend::cuda {

// Attention fusa a "online softmax" (stile FlashAttention), sostituisce
// il nucleo batchedQK/batchedMask/softmax/batchedPV (che materializzava
// l'intera matrice di score [batch,numHeads,newLen,totalLen] in memoria
// device, con 4 lanci di kernel e altrettanti round-trip in memoria
// globale). Un solo kernel per il forward: un blocco CUDA per ogni
// combinazione (batch, testa, riga di query), che scorre le posizioni
// chiave fino al limite causale mantenendo massimo/somma correnti
// (online softmax, Milakov & Gimelshein 2018 — lo stesso trucco
// numerico alla base di FlashAttention) e accumula l'uscita pesata SENZA
// mai scrivere la matrice di score in memoria globale: vive solo in
// shared memory/registri per la durata del blocco. Le posizioni oltre
// il limite causale non vengono nemmeno visitate (a differenza di
// calcolare l'intera matrice e poi mascherarla), dimezzando circa il
// lavoro effettivo oltre a eliminare il traffico di memoria.
//
// Rispetto a un vero FlashAttention (CUTLASS/tensor core, tiling
// multi-riga con K/V caricati una volta e riusati da piu' righe di
// query nello stesso blocco): qui ogni blocco ricarica K/V dalla
// memoria globale ad ogni posizione chiave (un blocco per riga, non per
// tile di righe) — meno efficiente in banda di memoria del vero
// FlashAttention, ma comunque elimina il collo di bottiglia dominante
// misurato (la matrice di score materializzata) restando un'unica
// implementazione ragionevole da scrivere e verificare correttamente a
// mano, coerente con la scelta di correttezza-prima-che-prestazioni di
// questo backend.

// Risultato del forward: 'output' e' l'uscita dell'attention (prima
// della proiezione Wout, stessa forma di q: [batch, newLen, dim]); 'm'
// (massimo corrente) e 'l' (somma corrente) sono le statistiche softmax
// per riga di query [batch, numHeads, newLen], salvate per il backward
// (che le riusa per evitare di dover ricalcolare l'intera normalizzazione
// softmax, seguendo lo stesso principio "salva solo O(seq), mai
// O(seq^2)" del forward).
struct FusedAttentionForwardResult {
    DeviceTensor output;
    DeviceTensor m;
    DeviceTensor l;
};

// q: [batch, newLen, dim], k/v: [batch, totalLen, dim] (teste contigue
// in dim, headDim = dim / numHeads). 'oldLen' generalizza la maschera
// causale ordinaria (oldLen == 0, newLen == totalLen, usata da
// selfAttention) al caso incrementale con cache K/V (oldLen ==
// totalLen - newLen, usata da selfAttentionIncremental): la query alla
// riga i (posizione assoluta oldLen + i) attende solo alle chiavi
// j <= oldLen + i.
FusedAttentionForwardResult fusedAttentionForward(const DeviceTensor& q, const DeviceTensor& k,
                                                    const DeviceTensor& v, std::size_t numHeads, float scaleFactor,
                                                    std::size_t oldLen);

// Gradienti di fusedAttentionForward rispetto a q/k/v, dato il gradiente
// dell'uscita (dOutput, stessa forma di 'output'). Ricalcola gli score
// (stesso ciclo del forward, stesso costo asintotico) invece di
// salvarli: usa l'identita' D[i] = dot(output[i,:], dOutput[i,:]) per
// ottenere il termine di correzione della softmax-backward senza dover
// mai materializzare le probabilita' complete (vedi il commento nel
// file .cu per la derivazione completa). dK/dV sono accumulati con
// atomicAdd: piu' righe di query (blocchi diversi) contribuiscono alla
// stessa posizione chiave quando la attendono.
struct FusedAttentionGrad {
    DeviceTensor dQ;
    DeviceTensor dK;
    DeviceTensor dV;
};
FusedAttentionGrad fusedAttentionBackward(const DeviceTensor& dOutput, const DeviceTensor& q, const DeviceTensor& k,
                                           const DeviceTensor& v, const DeviceTensor& output, const DeviceTensor& m,
                                           const DeviceTensor& l, std::size_t numHeads, float scaleFactor,
                                           std::size_t oldLen);

}  // namespace blackforge::backend::cuda
