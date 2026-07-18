#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "blackforge/ast/ast.hpp"
#include "blackforge/ir/module.hpp"

namespace blackforge::backend::cpu {

struct TrainRunResult {
    std::string modelName;
    std::vector<double> epochLosses;  // loss media per epoca, in ordine di esecuzione
};

// Esegue l'addestramento descritto dal primo blocco 'train' trovato in
// 'program', usando la IR gia' costruita in 'module' (build 'blackforge
// check' non lo chiama: e' usata da 'blackforge train' e dai test).
//
// Presuppone che program/module abbiano gia' superato l'analisi
// semantica (non la ripete): un blocco 'train' semanticamente invalido
// non e' un caso gestito qui. Lancia std::runtime_error se qualcosa
// nel 'train' non e' risolvibile a runtime (dataset/modello non
// trovato nella IR, file dataset non apribile, dataset piu' piccolo
// del batch_size, ...).
//
// Se 'progressOutput' non e' nullptr, stampa lì l'avanzamento
// (esempi/batch, loss per epoca); altrimenti l'addestramento avviene
// silenziosamente (usato dai test).
TrainRunResult runTraining(const ast::Program& program, const ir::Module& module,
                            const std::string& fromCheckpointPath, const std::string& saveCheckpointPath,
                            std::ostream* progressOutput = nullptr);

}  // namespace blackforge::backend::cpu
