#pragma once

#include <cstddef>

#include "blackforge/ir/module.hpp"
#include "blackforge/runtime/tensor.hpp"

namespace blackforge::backend::cpu {

// Esegue la prima pipeline di una ModelIR sul backend CPU di riferimento.
//
// NOTA: i pesi (weight/bias dei layer 'linear') sono generati in modo
// deterministico da un generatore pseudocasuale seminato con l'id
// dell'operazione, non caricati da un checkpoint ne' inizializzati con
// una strategia statisticamente valida (Xavier/Kaiming). Servono a
// rendere l'esecuzione riproducibile per i test e per 'blackforge run',
// non a produrre un modello allenabile: il caricamento di pesi reali
// arrivera' con la milestone del training/checkpoint.
class Executor {
public:
    explicit Executor(unsigned int seed = 42) : seed_(seed) {}

    // Risolve ogni dimensione simbolica del Value in ingresso (es.
    // 'batch') a batchSize e genera un tensore di input deterministico
    // con quella forma.
    [[nodiscard]] runtime::Tensor makeSyntheticInput(const ir::Value& inputValue, std::size_t batchSize) const;

    // Esegue la prima pipeline del modello sul tensore di input fornito.
    // Lancia std::invalid_argument se il modello non ha pipeline o se
    // la forma dell'input non e' compatibile con la prima operazione.
    [[nodiscard]] runtime::Tensor run(const ir::ModelIR& model, const runtime::Tensor& input) const;

private:
    unsigned int seed_;
};

}  // namespace blackforge::backend::cpu
