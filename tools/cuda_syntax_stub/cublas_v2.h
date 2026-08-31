#pragma once
// STUB — vedi cuda_runtime.h.
#include <cstddef>

using cublasStatus_t = int;
enum : cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED,
    CUBLAS_STATUS_ALLOC_FAILED,
    CUBLAS_STATUS_INVALID_VALUE,
    CUBLAS_STATUS_ARCH_MISMATCH,
    CUBLAS_STATUS_MAPPING_ERROR,
    CUBLAS_STATUS_EXECUTION_FAILED,
    CUBLAS_STATUS_INTERNAL_ERROR,
    CUBLAS_STATUS_NOT_SUPPORTED,
    CUBLAS_STATUS_LICENSE_ERROR,
};

enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 };
enum cublasComputeType_t {
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
};
enum cudaDataType_t { CUDA_R_32F = 0, CUDA_R_16F = 2, CUDA_R_16BF = 14, CUDA_R_8F_E4M3 = 28 };
using cudaDataType = cudaDataType_t;

struct cublasContext;
using cublasHandle_t = cublasContext*;

inline cublasStatus_t cublasCreate(cublasHandle_t*) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasDestroy(cublasHandle_t) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSetStream(cublasHandle_t, void*) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                  const float*, const float*, int, const float*, int, const float*, float*,
                                  int) {
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int,
                                                int, const float*, const float*, int, long long, const float*,
                                                int, long long, const float*, float*, int, long long, int) {
    return CUBLAS_STATUS_SUCCESS;
}
