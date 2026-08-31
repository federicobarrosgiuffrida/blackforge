#pragma once
// STUB — NON fa parte della build.
//
// Serve a un solo scopo: far passare i file .cu di BlackBit attraverso
// g++ -fsyntax-only, per verificarne il C++ in ambienti dove nvcc non
// esiste (container di sviluppo, CI senza GPU). Non emula la semantica
// CUDA: emula soltanto abbastanza dichiarazioni da rendere il codice
// analizzabile. Vedi tools/check_cuda_syntax.sh.
//
// Cosa CATTURA: errori di sintassi, tipi sbagliati, argomenti dei
// kernel non corrispondenti alla firma, membri inesistenti, name lookup.
// Cosa NON cattura: errori specifici di nvcc (limiti di registri,
// __launch_bounds__ non soddisfatti, uso di funzioni host nel device),
// e ovviamente qualunque errore di esecuzione.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

using cudaError_t = int;
enum : cudaError_t { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 };

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
};

enum cudaDeviceAttr { cudaDevAttrMemoryPoolsSupported = 115 };

struct CUevent_st;
struct CUstream_st;
using cudaEvent_t = CUevent_st*;
using cudaStream_t = CUstream_st*;

struct uint3 {
    unsigned int x = 0, y = 0, z = 0;
};

struct dim3 {
    unsigned int x = 1, y = 1, z = 1;
    dim3() = default;
    // NOLINTNEXTLINE(google-explicit-constructor)
    dim3(unsigned int px, unsigned int py = 1, unsigned int pz = 1) : x(px), y(py), z(pz) {}
};

// Variabili predefinite del device. Nel modello di compilazione vero
// sono per-thread; qui bastano oggetti globali perche' non eseguiamo.
extern uint3 threadIdx;
extern uint3 blockIdx;
extern dim3 blockDim;
extern dim3 gridDim;
inline constexpr int warpSize = 32;

struct cudaDeviceProp {
    char name[256] = {};
    std::size_t totalGlobalMem = 0;
    std::size_t sharedMemPerBlock = 0;
    int major = 0;
    int minor = 0;
    int multiProcessorCount = 0;
    int maxThreadsPerBlock = 0;
    int maxThreadsPerMultiProcessor = 0;
    int warpSize = 32;
    int clockRate = 0;
    int memoryClockRate = 0;
    int memoryBusWidth = 0;
    int l2CacheSize = 0;
    int regsPerBlock = 0;
    int integrated = 0;
    int unifiedAddressing = 0;
};

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int) { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "stub"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int*) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int*) { return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaDeviceGetAttribute(int*, cudaDeviceAttr, int) { return cudaSuccess; }
inline cudaError_t cudaMemGetInfo(std::size_t*, std::size_t*) { return cudaSuccess; }
inline cudaError_t cudaMalloc(void**, std::size_t) { return cudaSuccess; }
inline cudaError_t cudaFree(void*) { return cudaSuccess; }
inline cudaError_t cudaMallocAsync(void**, std::size_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaFreeAsync(void*, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaMemcpy(void*, const void*, std::size_t, cudaMemcpyKind) { return cudaSuccess; }
inline cudaError_t cudaMemset(void*, int, std::size_t) { return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t*) { return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t) { return cudaSuccess; }

// Qualificatori: spariscono, cosi' i kernel diventano funzioni normali
// e le chiamate (private del loro <<<...>>> dallo script) vengono
// controllate sugli argomenti come qualunque altra chiamata.
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__
#define __restrict__
#define __launch_bounds__(...)

inline void __syncthreads() {}

template <typename T>
T atomicAdd(T* address, T value) {
    const T previous = *address;
    *address += value;
    return previous;
}

template <typename T>
T atomicMax(T* address, T value) {
    const T previous = *address;
    if (value > *address) *address = value;
    return previous;
}

template <typename T>
T atomicCAS(T* address, T compare, T value) {
    const T previous = *address;
    if (previous == compare) *address = value;
    return previous;
}

template <typename T>
T __shfl_down_sync(unsigned int, T value, unsigned int, int = warpSize) {
    return value;
}

template <typename T>
T __shfl_sync(unsigned int, T value, int, int = warpSize) {
    return value;
}

inline float __expf(float value) { return std::exp(value); }
inline float __logf(float value) { return std::log(value); }
inline float __fdividef(float a, float b) { return a / b; }
inline float rsqrtf(float value) { return 1.0F / std::sqrt(value); }

// CUDA mette queste allo scope globale, senza qualificatore std::. Il
// codice dei kernel le usa non qualificate, quindi devono esistere qui
// o l'analisi fallisce su codice che nvcc accetta.
inline bool isfinite(float value) { return std::isfinite(value); }
inline bool isfinite(double value) { return std::isfinite(value); }
inline bool isnan(float value) { return std::isnan(value); }
inline bool isnan(double value) { return std::isnan(value); }
inline bool isinf(float value) { return std::isinf(value); }
inline bool isinf(double value) { return std::isinf(value); }

template <typename T>
constexpr T min(T a, T b) {
    return b < a ? b : a;
}

template <typename T>
constexpr T max(T a, T b) {
    return a < b ? b : a;
}

// Riceve la configurazione di lancio che check_cuda_syntax.sh estrae da
// `<<<...>>>`: serve solo a mantenerne "usate" le espressioni, cosi' le
// variabili che compaiono soltanto nella griglia non diventano falsi
// -Wunused-variable.
template <typename... Args>
inline int blackforgeLaunchConfiguration(const Args&...) {
    return 0;
}
