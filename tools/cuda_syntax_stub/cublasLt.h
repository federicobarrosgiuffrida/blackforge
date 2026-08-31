#pragma once
// STUB — vedi cuda_runtime.h.
#include <cstddef>

#include "cublas_v2.h"
#include "cuda_runtime.h"

struct cublasLtContext;
struct cublasLtMatmulDescOpaque;
struct cublasLtMatrixLayoutOpaque;
struct cublasLtMatmulPreferenceOpaque;

using cublasLtHandle_t = cublasLtContext*;
using cublasLtMatmulDesc_t = cublasLtMatmulDescOpaque*;
using cublasLtMatrixLayout_t = cublasLtMatrixLayoutOpaque*;
using cublasLtMatmulPreference_t = cublasLtMatmulPreferenceOpaque*;

enum cublasLtMatmulDescAttributes_t { CUBLASLT_MATMUL_DESC_TRANSA = 3, CUBLASLT_MATMUL_DESC_TRANSB = 4 };
enum cublasLtMatrixLayoutAttribute_t { CUBLASLT_MATRIX_LAYOUT_ORDER = 1 };
enum cublasLtOrder_t { CUBLASLT_ORDER_COL = 0, CUBLASLT_ORDER_ROW = 1 };
enum cublasLtMatmulPreferenceAttributes_t { CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1 };

struct cublasLtMatmulAlgo_t {
    std::uint64_t data[8];
};

struct cublasLtMatmulHeuristicResult_t {
    cublasLtMatmulAlgo_t algo;
    std::size_t workspaceSize = 0;
    cublasStatus_t state = CUBLAS_STATUS_SUCCESS;
    float wavesCount = 0.0F;
    int reserved[4];
};

inline cublasStatus_t cublasLtCreate(cublasLtHandle_t*) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasLtDestroy(cublasLtHandle_t) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t*, cublasComputeType_t, cudaDataType_t) {
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t,
                                                     const void*, std::size_t) {
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t*, cudaDataType_t, std::uint64_t,
                                                 std::uint64_t, std::int64_t) {
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t,
                                                       const void*, std::size_t) {
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t*) {
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t) {
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t,
                                                           cublasLtMatmulPreferenceAttributes_t, const void*,
                                                           std::size_t) {
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t, cublasLtMatmulDesc_t,
                                                     cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                                     cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                                     cublasLtMatmulPreference_t, int,
                                                     cublasLtMatmulHeuristicResult_t*, int*) {
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasLtMatmul(cublasLtHandle_t, cublasLtMatmulDesc_t, const void*, const void*,
                                     cublasLtMatrixLayout_t, const void*, cublasLtMatrixLayout_t, const void*,
                                     const void*, cublasLtMatrixLayout_t, void*, cublasLtMatrixLayout_t,
                                     const cublasLtMatmulAlgo_t*, void*, std::size_t, cudaStream_t) {
    return CUBLAS_STATUS_SUCCESS;
}
