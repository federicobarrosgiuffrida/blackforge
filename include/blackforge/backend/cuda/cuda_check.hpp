#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace blackforge::backend::cuda {

// Converte un errore del runtime CUDA in un'eccezione con messaggio
// chiaro (file/riga della chiamata, codice e descrizione CUDA).
inline void checkCudaError(cudaError_t result, const char* expression, const char* file, int line) {
    if (result != cudaSuccess) {
        std::ostringstream out;
        out << file << ":" << line << ": chiamata CUDA fallita '" << expression
            << "': " << cudaGetErrorString(result) << " (codice " << static_cast<int>(result) << ")";
        throw std::runtime_error(out.str());
    }
}

inline const char* cublasErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "errore cuBLAS sconosciuto";
    }
}

inline void checkCublasStatus(cublasStatus_t result, const char* expression, const char* file, int line) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream out;
        out << file << ":" << line << ": chiamata cuBLAS fallita '" << expression
            << "': " << cublasErrorString(result);
        throw std::runtime_error(out.str());
    }
}

}  // namespace blackforge::backend::cuda

#define BLACKFORGE_CUDA_CHECK(expr) ::blackforge::backend::cuda::checkCudaError((expr), #expr, __FILE__, __LINE__)

#define BLACKFORGE_CUBLAS_CHECK(expr) \
    ::blackforge::backend::cuda::checkCublasStatus((expr), #expr, __FILE__, __LINE__)
