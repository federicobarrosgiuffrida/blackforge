#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "blackforge/ast/ast.hpp"
#include "blackforge/ir/module.hpp"
#include "blackforge/runtime/tensor.hpp"

namespace blackforge::backend::cpu {

struct ForecastRunResult {
    std::string modelName;
    std::vector<runtime::Tensor> steps;  // un tensore per ciascuno degli 'horizon' passi generati
};

// Esegue la sessione di forecasting autoregressivo descritta dal primo
// blocco 'forecast' trovato in 'program'.
//
// Carica il modello referenziato con i pesi da 'fromCheckpointPath'
// (obbligatorio: un modello con pesi casuali non produce previsioni
// sensate), genera un input sintetico iniziale con la forma dichiarata
// dal modello (risolvendo le dimensioni simboliche a 'batchSize'), poi
// applica il modello ripetutamente per 'horizon' passi usando ogni
// output come input del passo successivo. Questo richiede che l'ultima
// dimensione della forma di input e di quella di output del modello
// coincidano (altrimenti l'output non potrebbe diventare l'input del
// passo successivo): lancia std::runtime_error se non e' cosi', o se
// qualunque altro riferimento del blocco 'forecast' non e' risolvibile.
ForecastRunResult runForecast(const ast::Program& program, const ir::Module& module,
                               const std::string& fromCheckpointPath, std::size_t batchSize,
                               std::ostream* progressOutput = nullptr);

}  // namespace blackforge::backend::cpu
