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
// optimizer.step tutti su device; solo il dataset viene caricato da
// disco sull'host, come su CPU, e caricato batch per batch sul device).
//
// LIMITAZIONI ESPLICITE di questa prima versione (percorso minimo
// stabile): lancia std::runtime_error, SENZA fare fallback silenzioso
// sulla CPU, se il blocco 'train' usa una funzionalita' non ancora
// supportata su GPU:
//   - 'loss' diversa da 'mse' (cross-entropy su CUDA e' lavoro futuro);
//   - un blocco 'lora' (nessun supporto adapter a basso rango su CUDA
//     ancora);
//   - 'fromCheckpointPath'/'saveCheckpointPath' non vuoti (il formato
//     di checkpoint non e' ancora collegato a cuda::Model).
//
// Presuppone che program/module abbiano gia' superato l'analisi
// semantica. Se 'progressOutput' non e' nullptr, stampa lì
// l'avanzamento; altrimenti l'addestramento avviene silenziosamente
// (usato dai test).
TrainRunResult runTraining(const ast::Program& program, const ir::Module& module,
                            const std::string& fromCheckpointPath, const std::string& saveCheckpointPath,
                            std::ostream* progressOutput = nullptr);

}  // namespace blackforge::backend::cuda
