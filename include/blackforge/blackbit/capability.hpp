#pragma once

#include <optional>
#include <string>

#include "blackforge/blackbit/ternary_linear.hpp"

// Capacita' di calcolo del dispositivo e scelta del formato di calcolo
// (requisito 13).
//
// PERCHE' ESISTE
//
// La memorizzazione ternaria NON implica aritmetica ternaria. La
// catena a cui BlackBit punta e'
//
//     ternario impacchettato -> tile decodificato -> FP4/BF16 -> Tensor Core
//                                                             -> accumulo BF16/FP32
//
// e il formato del terzo passo dipende dall'hardware: su Blackwell
// (sm_120, la RTX 5060 bersaglio) i Tensor Core hanno percorsi a bassa
// precisione che su Ampere non esistono. Questo modulo isola quella
// scelta in UN posto, cosi' che aggiungere FP4 domani non richieda di
// toccare TernaryLinear, MoEExpert o GqaAttention: cambiera' solo cosa
// preferredComputeDType() restituisce.
//
// COSA RIPORTA OGGI
//
// La capacita' reale del dispositivo quando la build ha CUDA
// (backend::cuda::enumerateDevices), std::nullopt altrimenti. La scelta
// del formato di calcolo restituisce BF16 su tutto cio' che ha Tensor
// Core e FP32 altrove: FP4 comparira' qui quando ci sara' un kernel che
// lo implementa, non prima. Una funzione che dichiarasse FP4 senza un
// percorso dietro renderebbe silenziosamente falso ogni rapporto di
// prestazioni.

namespace blackforge::blackbit {

enum class TensorCoreGeneration {
    None,      // nessun Tensor Core (CPU, o GPU pre-Volta)
    Volta,     // sm_70: FP16
    Turing,    // sm_75: FP16, INT8, INT4
    Ampere,    // sm_80/86: BF16, TF32
    Hopper,    // sm_90: FP8
    Blackwell  // sm_100/120: FP4, FP6, FP8
};

const char* tensorCoreGenerationName(TensorCoreGeneration generation);

struct DeviceCapability {
    int major = 0;
    int minor = 0;
    std::string name;
    std::size_t totalMemoryBytes = 0;

    [[nodiscard]] TensorCoreGeneration tensorCores() const;
    [[nodiscard]] bool supportsBf16TensorCores() const;

    // Percorsi a bassa precisione dei Tensor Core Blackwell. true NON
    // significa che BlackForge li usi: significa che l'hardware li ha.
    [[nodiscard]] bool supportsFp8() const;
    [[nodiscard]] bool supportsFp4() const;

    [[nodiscard]] std::string describe() const;
};

// Capacita' del dispositivo CUDA indicato (0 = il primo).
// std::nullopt se la build non ha CUDA o se non ci sono GPU visibili:
// l'assenza di una GPU e' un esito normale del rilevamento, non un
// errore (stessa convenzione di backend::cuda::enumerateDevices).
std::optional<DeviceCapability> detectDeviceCapability(int index = 0);

// Formato in cui decodificare i tile ternari su questo dispositivo.
//
// Oggi: BF16 dove ci sono Tensor Core che lo supportano, FP32 altrove.
// Quando esistera' un GEMM FP4, questa funzione restituira' FP4 su
// Blackwell e nessun altro file dovra' cambiare.
ComputeDType preferredComputeDType(const std::optional<DeviceCapability>& capability);

}  // namespace blackforge::blackbit
