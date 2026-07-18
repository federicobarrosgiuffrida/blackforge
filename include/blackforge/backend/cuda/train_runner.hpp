#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "blackforge/ast/ast.hpp"
#include "blackforge/ir/module.hpp"

namespace blackforge::backend::cuda {

struct TrainRunResult {
    std::string modelName;
    std::vector<double> epochLosses;  // loss media per epoca, in ordine di esecuzione
};

// Esegue l'addestramento descritto dal primo blocco 'train' trovato in
// 'program' interamente sul backend CUDA (forward, loss, backward e
// optimizer.step tutti su device; solo il dataset e i checkpoint
// vengono letti/scritti sull'host, come su CPU, e caricati batch per
// batch sul device). 'fromCheckpointPath'/'saveCheckpointPath' usano lo
// stesso formato binario del backend CPU (vedi
// blackforge/backend/cuda/checkpoint.hpp): un checkpoint salvato da un
// training CPU e' caricabile qui e viceversa.
//
// LIMITAZIONI ESPLICITE di questa prima versione (percorso minimo
// stabile): lancia std::runtime_error, SENZA fare fallback silenzioso
// sulla CPU, se il blocco 'train' usa una funzionalita' non ancora
// supportata su GPU:
//   - 'loss' diversa da 'mse' (cross-entropy su CUDA e' lavoro futuro);
//   - un blocco 'lora' (nessun supporto adapter a basso rango su CUDA
//     ancora).
//
// Presuppone che program/module abbiano gia' superato l'analisi
// semantica. Se 'progressOutput' non e' nullptr, stampa lì
// l'avanzamento; altrimenti l'addestramento avviene silenziosamente
// (usato dai test).
TrainRunResult runTraining(const ast::Program& program, const ir::Module& module,
                            const std::string& fromCheckpointPath, const std::string& saveCheckpointPath,
                            std::ostream* progressOutput = nullptr);

}  // namespace blackforge::backend::cuda
