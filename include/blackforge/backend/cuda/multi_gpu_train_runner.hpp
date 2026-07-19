#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "blackforge/ast/ast.hpp"
#include "blackforge/backend/cuda/train_runner.hpp"
#include "blackforge/ir/module.hpp"

namespace blackforge::backend::cuda {

// Addestramento data-parallelo su piu' GPU dello stesso processo: una
// replica completa del modello per ogni indice in 'deviceIndices'
// (stesso seme di inizializzazione => pesi iniziali identici), ognuna
// che processa una porzione (shard) disgiunta di ogni batch sulla
// propria GPU. Dopo ogni forward+backward, i gradienti di ogni
// parametro vengono mediati tra tutte le repliche (ALL-REDUCE) prima
// dello step dell'optimizer, cosi' che ogni replica applichi lo STESSO
// aggiornamento e resti sincronizzata con le altre — matematicamente
// equivalente ad addestrare con l'intero batch su una singola GPU
// (a meno delle differenze di somma in virgola mobile dovute
// all'ordine diverso delle operazioni), non un'approssimazione.
//
// STRATEGIA DI ALL-REDUCE: NCCL non supporta nativamente Windows (dove
// questo progetto viene sviluppato e testato), quindi l'all-reduce e'
// implementato con uno STAGING via HOST: il gradiente di ogni
// parametro viene copiato dalla GPU alla RAM (una per replica), sommato
// e mediato sulla CPU, poi ricopiato su ciascuna GPU. E' piu' lento di
// un all-reduce diretto GPU-to-GPU (P2P/NVLink) o di NCCL, ma e'
// PORTABILE su qualunque combinazione di GPU — non richiede che le GPU
// supportino il peer access (NVLink o stesso switch PCIe), rilevante
// per GPU affittate su cloud dove il collegamento fisico tra le
// schede non e' garantito. Una scelta esplicita di correttezza e
// portabilita' prima delle prestazioni, coerente con il resto del
// progetto.
//
// LIMITAZIONI ESPLICITE (non un'omissione nascosta):
//   - richiede 'deviceIndices.size() >= 2' (per una sola GPU, usa
//     runTraining()); lancia std::invalid_argument altrimenti;
//   - 'batch_size' deve essere divisibile per il numero di GPU (ogni
//     replica riceve uno shard della stessa dimensione — nessuno
//     sharding sbilanciato); lancia std::runtime_error altrimenti;
//   - stesse limitazioni di runTraining() (niente 'lora' su CUDA);
//   - indici RIPETUTI in 'deviceIndices' sono permessi deliberatamente
//     (es. {0, 0}): non danno alcun vantaggio di velocita' (piu'
//     repliche sullo stesso hardware fisico, nessun parallelismo
//     reale), ma sono un caso limite ben definito e utile per
//     verificare l'INTERA logica di sharding/all-reduce/sincronizzazione
//     su una macchina con una sola GPU fisica — esattamente il caso di
//     questa macchina di sviluppo (RTX 5060);
//   - NON ANCORA VERIFICATO SU HARDWARE MULTI-GPU REALE (piu' GPU
//     fisicamente distinte): questa macchina di sviluppo ha una sola
//     GPU. Verificato invece, sulla singola GPU disponibile, che il
//     percorso multi-GPU degenere ({0, 0}) produca una loss finale
//     numericamente equivalente all'addestramento a singola GPU
//     sull'intero batch — la logica di sharding/all-reduce e'
//     quindi verificata, la parallelizzazione reale su hardware
//     distinto resta da confermare quando sara' disponibile.
//
// Presuppone che program/module abbiano gia' superato l'analisi
// semantica. Se 'progressOutput' non e' nullptr, stampa lì
// l'avanzamento; altrimenti l'addestramento avviene silenziosamente.
TrainRunResult runMultiGpuTraining(const ast::Program& program, const ir::Module& module,
                                    const std::vector<int>& deviceIndices, const std::string& fromCheckpointPath,
                                    const std::string& saveCheckpointPath, std::ostream* progressOutput = nullptr);

}  // namespace blackforge::backend::cuda
