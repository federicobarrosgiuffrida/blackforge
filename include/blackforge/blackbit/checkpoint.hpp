#pragma once

#include <cstdint>
#include <string>

#include "blackforge/blackbit/config.hpp"
#include "blackforge/blackbit/low_rank_optimizer.hpp"
#include "blackforge/blackbit/model.hpp"

// Formato di checkpoint di BlackBit (requisito 16).
//
// I PESI RESTANO IMPACCHETTATI SU DISCO
//
// Espandere i pesi ternari in FP16 per salvarli renderebbe un
// checkpoint di BlackBit-9B 18 GB invece di 1,9 GB, e soprattutto
// significherebbe che la rappresentazione canonica del modello NON e'
// quella impacchettata — cioe' esattamente cio' che il requisito 3
// vieta, spostato dalla VRAM al disco. Qui il TernaryTensor viene
// serializzato come tale, trit e scale.
//
// FORMATO (little-endian nativo, come il BFCKPT1 esistente)
//
//   magic          8 byte, "BFBIT\0\0\0"
//   formatVersion  uint32   (kBlackBitCheckpointVersion)
//   configJson     uint32 lunghezza + byte (la configurazione completa,
//                           cosi' un checkpoint e' autodescrittivo e non
//                           dipende da un file esterno rimasto indietro)
//   step           uint64   passo di addestramento raggiunto
//   tokensSeen     uint64   token processati
//   learningRate   float32  stato dello scheduler
//   rngSeed        uint64   seme dell'arrotondamento stocastico
//   optimizerStep  uint64   passo interno dell'ottimizzatore
//   hasOptimizer   uint8    1 se seguono gli stati low-rank
//   parameterCount uint32
//   per parametro:
//     nameLength   uint32 + nome
//     kind         uint8  (0 = ternario, 1 = denso)
//     ternario:    blocco TernaryTensor::serialize()
//     denso:       uint64 conteggio + float32
//   se hasOptimizer:
//     stateCount   uint32
//     per stato:   nome, rank uint64, cols uint64, tre buffer float32
//
// COMPATIBILITA'
//
// La versione e' controllata al caricamento: un file di versione
// sconosciuta viene RIFIUTATO con un messaggio che dice quale versione
// ha e quale ci si aspettava, invece di essere interpretato male. La
// configurazione salvata viene confrontata con quella del modello in
// memoria: un checkpoint di un modello a 28 layer non puo' finire per
// sbaglio in uno a 12.

namespace blackforge::blackbit {

inline constexpr std::uint32_t kBlackBitCheckpointVersion = 1;

struct BlackBitTrainingState {
    std::uint64_t step = 0;
    std::uint64_t tokensSeen = 0;
    float learningRate = 0.0F;
    std::uint64_t rngSeed = 0;
    std::uint64_t optimizerStep = 0;
};

// Salva modello e stato di addestramento. Se 'optimizer' non e' nullo,
// salva anche lo stato low-rank, cosi' la ripresa non riparte con
// momenti azzerati (che su Adam vale diversi passi di
// riscaldamento persi).
void saveCheckpoint(const std::string& path, BlackBitModel& model, const BlackBitTrainingState& state,
                     LowRankProjectedOptimizer* optimizer = nullptr);

// Carica in un modello gia' costruito. Lancia std::runtime_error se il
// magic, la versione, la configurazione o le forme non corrispondono.
BlackBitTrainingState loadCheckpoint(const std::string& path, BlackBitModel& model,
                                      LowRankProjectedOptimizer* optimizer = nullptr);

// Legge SOLO l'intestazione: serve a costruire il modello con la
// configurazione giusta prima di caricarlo.
BlackBitConfig readCheckpointConfig(const std::string& path);

}  // namespace blackforge::blackbit
