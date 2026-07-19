#pragma once

#include <cstddef>
#include <optional>

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
    // con quella forma. Se la prima operazione della pipeline e'
    // 'embedding', genera token id interi uniformi in [0, vocabolario)
    // invece di valori continui in [-0.1, 0.1] (che non sarebbero id di
    // vocabolario validi): serve a rendere 'blackforge run'/'blackforge
    // benchmark' utilizzabili anche su modelli che iniziano con
    // 'embedding', senza dover fornire per forza un dataset reale.
    [[nodiscard]] runtime::Tensor makeSyntheticInput(const ir::ModelIR& model, std::size_t batchSize) const;

    // Esegue la prima pipeline del modello sul tensore di input fornito.
    // Se 'precision' e' presente, applica la quantizzazione simulata
    // (vedi quantize.hpp) a input/pesi (formato 'compute', prima di
    // ogni matmul) e alle attivazioni intermedie (formato 'storage',
    // dopo ogni operazione): e' cosi' che un blocco 'precision' del
    // programma influisce davvero sui numeri, non solo sulla
    // validazione. Lancia std::invalid_argument se il modello non ha
    // pipeline o se la forma dell'input non e' compatibile con la
    // prima operazione.
    [[nodiscard]] runtime::Tensor run(const ir::ModelIR& model, const runtime::Tensor& input,
                                       const std::optional<ir::PrecisionPolicy>& precision = std::nullopt) const;

private:
    unsigned int seed_;
};

}  // namespace blackforge::backend::cpu
