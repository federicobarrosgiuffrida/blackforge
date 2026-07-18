#pragma once

#include <string>

#include "blackforge/backend/cpu/model.hpp"

namespace blackforge::backend::cpu {

// Salva/carica i parametri di un Model su disco in un formato binario
// minimale e proprietario di BlackForge (non compatibile con formati
// esterni come safetensors): sufficiente per riprendere un
// addestramento o eseguire un modello gia' allenato all'interno di
// BlackForge stesso.
//
// Formato (little-endian, cosi' come scritto nativamente su hardware
// x86/ARM comuni):
//   magic:            8 byte, "BFCKPT1\0"
//   parameterCount:   uint32
//   per ogni parametro:
//     nameLength:     uint32
//     name:           nameLength byte (UTF-8, senza terminatore)
//     shapeRank:      uint32
//     shape:          shapeRank * uint64
//     data:           (prodotto della shape) * float32

// Lancia std::runtime_error se il file non puo' essere scritto.
void saveCheckpoint(Model& model, const std::string& path);

// Lancia std::runtime_error se il file non puo' essere letto, ha un
// magic non valido, oppure se un parametro nel file non corrisponde
// (per nome o per forma) a un parametro del modello.
void loadCheckpoint(Model& model, const std::string& path);

}  // namespace blackforge::backend::cpu
