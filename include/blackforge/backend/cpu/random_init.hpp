#pragma once

#include <cstddef>
#include <vector>

#include "blackforge/runtime/tensor.hpp"

namespace blackforge::backend::cpu {

// Genera un tensore con valori pseudocasuali uniformi in [-0.1, 0.1],
// deterministico rispetto al seme dato. Non e' una strategia di
// inizializzazione statisticamente valida (Xavier/Kaiming, ecc.): serve
// a rendere l'esecuzione e i test riproducibili, non a produrre un
// modello pronto per l'addestramento da zero con garanzie di
// convergenza.
runtime::Tensor randomTensor(std::vector<std::size_t> shape, unsigned int seed);

// Genera un tensore di token id pseudocasuali uniformi in [0, vocabSize),
// deterministico rispetto al seme dato: usato per l'input sintetico di
// 'blackforge run'/'blackforge benchmark' quando la pipeline inizia con
// 'embedding' (un input a valori continui in [-0.1, 0.1] come
// randomTensor() non sarebbe un id di vocabolario valido).
runtime::Tensor randomTokenIdTensor(std::vector<std::size_t> shape, std::size_t vocabSize, unsigned int seed);

// Combina un seme base con l'id di un valore/operazione IR e un "salt"
// per ottenere un seme deterministico ma diverso per ogni utilizzo
// (es. pesi e bias dello stesso layer usano salt differenti).
unsigned int seedFor(unsigned int base, std::size_t valueId, unsigned int salt);

}  // namespace blackforge::backend::cpu
