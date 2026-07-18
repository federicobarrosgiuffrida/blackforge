#pragma once

#include "blackforge/runtime/tensor.hpp"
#include "blackforge/sema/dtype.hpp"

namespace blackforge::backend::cpu {

// Arrotonda ogni elemento di 'input' alla precisione rappresentabile
// dal formato 'dtype', simulando la perdita di precisione che si
// avrebbe memorizzando/calcolando davvero in quel formato.
//
// I dati restano fisicamente float32 (ne' il backend CPU ne' quello
// CUDA hanno oggi un tipo di storage a byte ridotto: vedi le
// limitazioni gia' documentate altrove): questa funzione azzera i bit
// di mantissa che il formato target non potrebbe rappresentare e
// satura i valori troppo grandi, cosi' i NUMERI riflettono davvero
// l'arrotondamento dichiarato in un blocco 'precision', anche se
// l'occupazione di memoria non cambia.
//
// NOTA sull'accuratezza: bf16 usa un algoritmo bit-esatto standard
// (round-to-nearest-even sui 16 bit alti di un float32, lo stesso
// approccio di TensorFlow/PyTorch). fp16/tf32/fp8.e4m3/fp8.e5m2 usano
// un arrotondamento generico della mantissa via frexp/ldexp, corretto
// per il range normale ma senza gestione bit-esatta dei subnormali; i
// valori fuori range vengono saturati al massimo rappresentabile
// invece di produrre infinito, per non propagare NaN/Inf in modo
// sorprendente in un motore che deve restare numericamente stabile.
// fp32 e' un'identita' (nessun arrotondamento).
runtime::Tensor quantize(const runtime::Tensor& input, sema::DType dtype);

// Arrotonda un singolo valore, usata anche da quantize() e dai test.
float quantizeScalar(float value, sema::DType dtype);

}  // namespace blackforge::backend::cpu
