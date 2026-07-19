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

// Imposta il dispositivo CUDA attivo (cudaSetDevice) per il thread
// corrente: le allocazioni ed i kernel successivi (DeviceTensor,
// Executor) vengono eseguiti su questo indice. Lancia
// std::runtime_error se l'indice non e' valido. Permette a
// 'blackforge run/benchmark --device cuda:N' di scegliere la GPU
// quando ne sono presenti piu' di una. E' anche il meccanismo con cui
// blackforge::backend::cuda::runMultiGpuTraining alterna il device
// attivo tra una replica del modello e l'altra dello stesso processo
// (vedi multi_gpu_train_runner.hpp) per il training data-parallelo su
// piu' GPU.
void setActiveDevice(int index);

}  // namespace blackforge::backend::cuda
