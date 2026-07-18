#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace blackforge::backend::cuda {

struct CudaDeviceInfo {
    int index;
    std::string name;
    int computeCapabilityMajor;
    int computeCapabilityMinor;
    std::size_t totalMemoryBytes;
};

// Elenca le GPU NVIDIA visibili tramite il driver CUDA. Restituisce un
// elenco vuoto (senza lanciare eccezioni) se non ci sono GPU o il
// driver non e' inizializzabile: l'assenza di una GPU e' un esito
// normale del rilevamento hardware, non un errore.
std::vector<CudaDeviceInfo> enumerateDevices();

}  // namespace blackforge::backend::cuda
