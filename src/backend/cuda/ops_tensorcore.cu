#include <cuda_bf16.h>
#include <cublasLt.h>

#include <stdexcept>
#include <unordered_map>

#include "blackforge/backend/cuda/autodiff.hpp"
#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/backend/cuda/ops.hpp"

namespace blackforge::backend::cuda {

namespace {

constexpr int kBlockSize = 256;
int gridSizeFor(std::size_t n) { return static_cast<int>((n + kBlockSize - 1) / kBlockSize); }

__global__ void floatToBf16Kernel(__nv_bfloat16* out, const float* in, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2bfloat16(in[i]);
    }
}

// Buffer BF16 grezzo su device (RAII): DeviceTensor memorizza sempre
// float32 per contratto (vedi device_tensor.hpp), quindi lo scratch
// BF16 usato solo internamente da questo file per alimentare Tensor
// Core non puo' essere un DeviceTensor — e' un dettaglio implementativo
// che non deve mai attraversare l'API pubblica (che resta sempre
// float32 in ingresso/uscita, vedi matmulBf16 sotto).
class DeviceBf16Buffer {
public:
    explicit DeviceBf16Buffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            BLACKFORGE_CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(__nv_bfloat16)));
        }
    }
    ~DeviceBf16Buffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }
    DeviceBf16Buffer(const DeviceBf16Buffer&) = delete;
    DeviceBf16Buffer& operator=(const DeviceBf16Buffer&) = delete;

    DeviceBf16Buffer(DeviceBf16Buffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBf16Buffer& operator=(DeviceBf16Buffer&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    [[nodiscard]] __nv_bfloat16* data() { return ptr_; }
    [[nodiscard]] const __nv_bfloat16* data() const { return ptr_; }

private:
    __nv_bfloat16* ptr_ = nullptr;
    std::size_t count_;
};

DeviceBf16Buffer toBf16(const DeviceTensor& t) {
    DeviceBf16Buffer buffer(t.elementCount());
    if (t.elementCount() > 0) {
        floatToBf16Kernel<<<gridSizeFor(t.elementCount()), kBlockSize>>>(buffer.data(), t.data(), t.elementCount());
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return buffer;
}

// Un handle cuBLASLt per DISPOSITIVO CUDA, stessa scelta e stesso
// motivo di sharedHandle() in ops_gemm.cu per il cuBLAS classico
// (essenziale per il training multi-GPU: un handle creato mentre e'
// attivo un device non e' valido su un device diverso).
cublasLtHandle_t sharedLtHandle() {
    static std::unordered_map<int, cublasLtHandle_t> handles;
    int device = 0;
    BLACKFORGE_CUDA_CHECK(cudaGetDevice(&device));
    auto it = handles.find(device);
    if (it != handles.end()) {
        return it->second;
    }
    cublasLtHandle_t h;
    BLACKFORGE_CUBLAS_CHECK(cublasLtCreate(&h));
    handles.emplace(device, h);
    return h;
}

// Esegue C[resultRows, resultCols] = op(A) @ op(B) in row-major, con A
// e B gia' in BF16 (Tensor Core) e accumulo/uscita in FP32 (lo schema
// di "mixed precision" standard: attivazioni/pesi in BF16 per il
// calcolo, ma niente overflow/perdita di range nell'accumulo, a
// differenza di un accumulo BF16 puro). 'aRows'/'aCols' e
// 'bRows'/'bCols' sono le dimensioni di A/B COSI' COME MEMORIZZATE
// (prima di un'eventuale trasposizione logica via transA/transB): la
// stessa convenzione di cuBLAS/cuBLASLt classici. A differenza del
// trucco riga-maggiore/colonna-maggiore usato da matmul() (ops_gemm.cu,
// cuBLAS classico), cuBLASLt supporta nativamente CUBLASLT_ORDER_ROW,
// quindi qui non serve scambiare operandi/dimensioni.
//
// Crea e distrugge i descrittori (e alloca/libera il workspace) ad
// ogni chiamata invece di metterli in cache per riusarli su forme
// ripetute: correttezza-prima-che-prestazioni, come altrove nel
// progetto — un'ottimizzazione futura, non una scorciatoia rischiosa
// per la correttezza.
void bf16Gemm(cublasOperation_t transA, cublasOperation_t transB, const __nv_bfloat16* a, std::size_t aRows,
              std::size_t aCols, const __nv_bfloat16* b, std::size_t bRows, std::size_t bCols, float* c,
              std::size_t resultRows, std::size_t resultCols) {
    cublasLtHandle_t handle = sharedLtHandle();

    cublasLtMatmulDesc_t opDesc = nullptr;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;

    cublasLtMatrixLayout_t aLayout = nullptr;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&aLayout, CUDA_R_16BF, static_cast<std::uint64_t>(aRows),
                                                        static_cast<std::uint64_t>(aCols),
                                                        static_cast<std::int64_t>(aCols)));
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(aLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    cublasLtMatrixLayout_t bLayout = nullptr;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&bLayout, CUDA_R_16BF, static_cast<std::uint64_t>(bRows),
                                                        static_cast<std::uint64_t>(bCols),
                                                        static_cast<std::int64_t>(bCols)));
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(bLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    cublasLtMatrixLayout_t cLayout = nullptr;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cLayout, CUDA_R_32F, static_cast<std::uint64_t>(resultRows),
                                                        static_cast<std::uint64_t>(resultCols),
                                                        static_cast<std::int64_t>(resultCols)));
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(cLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    cublasLtMatmulPreference_t preference = nullptr;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    std::size_t workspaceBytes = 4 * 1024 * 1024;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, opDesc, aLayout, bLayout, cLayout, cLayout,
                                                            preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(aLayout);
        cublasLtMatrixLayoutDestroy(bLayout);
        cublasLtMatrixLayoutDestroy(cLayout);
        cublasLtMatmulDescDestroy(opDesc);
        throw std::runtime_error("bf16Gemm: nessun algoritmo Tensor Core disponibile per questa forma/architettura");
    }

    void* workspace = nullptr;
    if (heuristicResult.workspaceSize > 0) {
        BLACKFORGE_CUDA_CHECK(cudaMalloc(&workspace, heuristicResult.workspaceSize));
    }

    float alpha = 1.0F;
    float beta = 0.0F;
    cublasStatus_t status =
        cublasLtMatmul(handle, opDesc, &alpha, a, aLayout, b, bLayout, &beta, c, cLayout, c, cLayout,
                        &heuristicResult.algo, workspace, heuristicResult.workspaceSize, /*stream=*/nullptr);

    if (workspace != nullptr) {
        cudaFree(workspace);
    }
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(aLayout);
    cublasLtMatrixLayoutDestroy(bLayout);
    cublasLtMatrixLayoutDestroy(cLayout);
    cublasLtMatmulDescDestroy(opDesc);

    BLACKFORGE_CUBLAS_CHECK(status);
}

}  // namespace

DeviceTensor matmulBf16(const DeviceTensor& a, const DeviceTensor& b) {
    if (a.rank() != 2 || b.rank() != 2 || a.dim(1) != b.dim(0)) {
        throw std::invalid_argument("matmulBf16: forme incompatibili sul device");
    }
    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(1);

    DeviceBf16Buffer aBf16 = toBf16(a);
    DeviceBf16Buffer bBf16 = toBf16(b);
    DeviceTensor result({m, n});

    if (m > 0 && k > 0 && n > 0) {
        bf16Gemm(CUBLAS_OP_N, CUBLAS_OP_N, aBf16.data(), m, k, bBf16.data(), k, n, result.data(), m, n);
    } else {
        BLACKFORGE_CUDA_CHECK(cudaMemset(result.data(), 0, result.elementCount() * sizeof(float)));
    }

    return result;
}

DeviceTensor linearBf16(const DeviceTensor& input, const DeviceTensor& weight, const DeviceTensor& bias) {
    if (input.rank() == 2) {
        return addBias(matmulBf16(input, weight), bias);
    }
    if (input.rank() < 2) {
        throw std::invalid_argument("linearBf16: richiede un tensore a rango >= 2 sul device");
    }

    std::size_t inFeatures = input.shape().back();
    std::size_t rows = input.elementCount() / inFeatures;
    DeviceTensor flatOutput = addBias(matmulBf16(input.clone().reshaped({rows, inFeatures}), weight), bias);

    std::vector<std::size_t> outputShape = input.shape();
    outputShape.back() = weight.dim(1);
    return std::move(flatOutput).reshaped(std::move(outputShape));
}

MatmulGrad matmulBf16Backward(const DeviceTensor& a, const DeviceTensor& b, const DeviceTensor& gradOutput) {
    if (a.rank() != 2 || b.rank() != 2 || gradOutput.rank() != 2 || a.dim(1) != b.dim(0) ||
        gradOutput.dim(0) != a.dim(0) || gradOutput.dim(1) != b.dim(1)) {
        throw std::invalid_argument("matmulBf16Backward: forme incompatibili sul device");
    }
    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(1);

    DeviceBf16Buffer aBf16 = toBf16(a);
    DeviceBf16Buffer bBf16 = toBf16(b);
    DeviceBf16Buffer gradBf16 = toBf16(gradOutput);

    DeviceTensor dA({m, k});
    DeviceTensor dB({k, n});

    if (m > 0 && k > 0 && n > 0) {
        // dA[m,k] = gradOutput[m,n] @ B^T: op(gradOutput) senza
        // trasposizione, op(B) trasposta (B e' memorizzata [k,n]).
        bf16Gemm(CUBLAS_OP_N, CUBLAS_OP_T, gradBf16.data(), m, n, bBf16.data(), k, n, dA.data(), m, k);
        // dB[k,n] = A^T @ gradOutput: op(A) trasposta (A e' memorizzata
        // [m,k]), op(gradOutput) senza trasposizione.
        bf16Gemm(CUBLAS_OP_T, CUBLAS_OP_N, aBf16.data(), m, k, gradBf16.data(), m, n, dB.data(), k, n);
    } else {
        BLACKFORGE_CUDA_CHECK(cudaMemset(dA.data(), 0, dA.elementCount() * sizeof(float)));
        BLACKFORGE_CUDA_CHECK(cudaMemset(dB.data(), 0, dB.elementCount() * sizeof(float)));
    }

    return MatmulGrad{std::move(dA), std::move(dB)};
}

}  // namespace blackforge::backend::cuda
