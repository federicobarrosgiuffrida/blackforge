#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "blackforge/blackbit/config.hpp"

// Dove vive un parametro (requisito 12).
//
// Per BlackBit-9B i pesi impacchettati (1,9 GB) stanno comodamente in
// VRAM e questa astrazione non serve. Serve per BlackBit-16B-A3B e
// BlackBit-24B-A3B, dove non ci staranno: gli esperti MoE sono il posto
// naturale dove intervenire, perche' per ogni token ne servono due su
// otto e gli altri sei potrebbero non essere in VRAM affatto.
//
// COSA C'E' QUI E COSA NO
//
// C'e' l'API di proprieta' e uno STRUMENTO DI PIANIFICAZIONE che, data
// una politica, dice quanti byte finirebbero in ciascuno stato usando
// le dimensioni reali del formato impacchettato. E' utile subito: serve
// a rispondere a "se tengo in VRAM solo 2 esperti per layer, quanto
// risparmio?" prima di scrivere una riga di codice di paginazione.
//
// NON c'e' l'implementazione del trasferimento: nessun percorso di
// esecuzione oggi legge un parametro che non sia residente. Chiedere di
// eseguire con parametri non residenti e' un errore esplicito, non un
// caricamento silenzioso che maschererebbe l'assenza della
// funzionalita'.

namespace blackforge::blackbit {

enum class ResidencyState {
    // Nella memoria del dispositivo di calcolo. L'unico stato in cui un
    // parametro puo' essere usato oggi.
    GpuResident,
    // In memoria host non paginabile: trasferibile su device con DMA
    // asincrona, sovrapponibile al calcolo del layer precedente.
    CpuPinned,
    // Fuori dalla memoria di lavoro (host paginabile o disco): richiede
    // un caricamento esplicito prima dell'uso.
    Paged,
};

const char* residencyStateName(ResidencyState state);

struct ResidencyPolicy {
    // Esperti per layer da tenere residenti. 0 = tutti (il default:
    // e' cio' che BlackBit-9B fa e deve continuare a fare).
    std::size_t hotExpertsPerLayer = 0;

    // Stato in cui finiscono gli esperti non "caldi".
    ResidencyState coldExpertState = ResidencyState::CpuPinned;

    // Attention, router, norm ed embedding restano sempre residenti:
    // sono il 9 % dei parametri ma servono a OGNI token, quindi
    // spostarli costerebbe un trasferimento per token risparmiando
    // poco.
    bool keepSharedResident = true;
};

// Byte che finirebbero in ciascuno stato con una data politica,
// calcolati con le dimensioni reali del formato impacchettato (scale
// incluse), non con una stima a spanne.
struct ResidencyPlan {
    std::array<std::size_t, 3> bytesByState{};

    [[nodiscard]] std::size_t bytes(ResidencyState state) const {
        return bytesByState[static_cast<std::size_t>(state)];
    }
    [[nodiscard]] std::size_t total() const {
        return bytesByState[0] + bytesByState[1] + bytesByState[2];
    }
    [[nodiscard]] std::size_t residentBytes() const { return bytes(ResidencyState::GpuResident); }

    // Trasferimenti host->device per token, nel caso peggiore: ogni
    // esperto selezionato che non e' caldo va portato dentro. E' il
    // numero che dice se una politica e' praticabile o se saturera' il
    // bus PCIe.
    std::size_t worstCaseTransferBytesPerToken = 0;

    [[nodiscard]] std::string report() const;
};

ResidencyPlan planResidency(const BlackBitConfig& config, const ResidencyPolicy& policy);

}  // namespace blackforge::blackbit
