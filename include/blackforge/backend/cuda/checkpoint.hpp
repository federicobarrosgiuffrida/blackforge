#pragma once

#include <string>

#include "blackforge/backend/cuda/model.hpp"

namespace blackforge::backend::cuda {

// Salva/carica i parametri di un cuda::Model su disco nello STESSO
// formato binario di backend::cpu::checkpoint (magic "BFCKPT1", vedi
// blackforge/backend/cpu/checkpoint.hpp per il layout esatto): un
// checkpoint salvato da un training CPU e' caricabile da un Model CUDA
// e viceversa, dato che il formato memorizza sempre float32 su disco
// indipendentemente da dove sono stati calcolati i valori.
//
// I dati vengono trasferiti device<->host solo per il tempo dell'I/O su
// disco (che deve comunque avvenire sull'host): non e' un fallback
// della computazione sulla CPU, l'addestramento resta interamente su
// device.

// Lancia std::runtime_error se il file non puo' essere scritto.
void saveCheckpoint(Model& model, const std::string& path);

// Lancia std::runtime_error se il file non puo' essere letto, ha un
// magic non valido, oppure se un parametro nel file non corrisponde
// (per nome o per forma) a un parametro del modello.
void loadCheckpoint(Model& model, const std::string& path);

}  // namespace blackforge::backend::cuda
