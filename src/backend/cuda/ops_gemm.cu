#include <stdexcept>

#include <cublas_v2.h>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/backend/cuda/ops.hpp"

namespace blackforge::backend::cuda {

namespace {

constexpr int kBlockSize = 256;
int gridSizeFor(std::size_t n) { return static_cast<int>((n + kBlockSize - 1) / kBlockSize); }

// a: [m,k], b: [n,k] -> a @ b^T: [m,n]. Un thread per elemento di
// output con riduzione seriale interna: kernel scritto a mano, non
// cuBLAS con trasposizioni (stessa scelta di correttezza-prima-che-
// prestazioni gia' fatta per i kernel di backward del matmul standard).
__global__ void matmulTransposeBKernel(float* out, const float* a, const float* b, std::size_t m, std::size_t k,
                                        std::size_t n) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= m * n) {
        return;
    }
    std::size_t i = idx / n;
    std::size_t j = idx % n;

    float sum = 0.0F;
    for (std::size_t p = 0; p < k; ++p) {
        sum += a[i * k + p] * b[j * k + p];
    }
    out[idx] = sum;
}

// Un handle cuBLAS per processo, creato alla prima chiamata e mai
// distrutto esplicitamente (il runtime CUDA lo libera all'uscita del
// processo): crearne e distruggerne uno ad ogni singola matmul() e'
// corretto ma non ottimale, un handle e' pensato per essere riusato.
// Assunzione esplicita: nessun accesso concorrente da piu' thread
// (l'intero progetto, CLI compresa, e' a singolo thread quando esegue
// codice CUDA — nessun lock qui, ne servirebbe uno se cambiasse).
cublasHandle_t sharedHandle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h;
        BLACKFORGE_CUBLAS_CHECK(cublasCreate(&h));
        return h;
    }();
    return handle;
}

}  // namespace

DeviceTensor matmul(const DeviceTensor& a, const DeviceTensor& b) {
    if (a.rank() != 2 || b.rank() != 2 || a.dim(1) != b.dim(0)) {
        throw std::invalid_argument("matmul: forme incompatibili sul device");
    }

    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(1);

    DeviceTensor result({m, n});

    cublasHandle_t handle = sharedHandle();

    // I nostri tensori sono memorizzati row-major (come ogni buffer C
    // "normale"), ma cuBLAS lavora in column-major. Un buffer row-major
    // [r, c] e' pero' identico, byte per byte, a un buffer column-major
    // [c, r]: e' la sua trasposta "gratis". Sfruttando questo si calcola
    // C = A @ B (row-major) chiedendo a cuBLAS di calcolare
    // C^T = B^T @ A^T (column-major) sugli stessi buffer, invertendo
    // l'ordine degli operandi e scambiando m <-> n. Risultato: lo stesso
    // buffer 'result', letto come row-major [m, n], contiene proprio
    // A @ B. E' la tecnica standard per usare cuBLAS con dati row-major.
    float alpha = 1.0F;
    float beta = 0.0F;
    BLACKFORGE_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(m),
                                         static_cast<int>(k), &alpha, b.data(), static_cast<int>(n), a.data(),
                                         static_cast<int>(k), &beta, result.data(), static_cast<int>(n)));

    return result;
}

DeviceTensor matmulTransposeB(const DeviceTensor& a, const DeviceTensor& b) {
    if (a.rank() != 2 || b.rank() != 2 || a.dim(1) != b.dim(1)) {
        throw std::invalid_argument("matmulTransposeB: forme incompatibili sul device");
    }
    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(0);

    DeviceTensor result({m, n});
    std::size_t total = m * n;
    if (total > 0) {
        matmulTransposeBKernel<<<gridSizeFor(total), kBlockSize>>>(result.data(), a.data(), b.data(), m, k, n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor linear(const DeviceTensor& input, const DeviceTensor& weight, const DeviceTensor& bias) {
    if (input.rank() == 2) {
        return addBias(matmul(input, weight), bias);
    }
    if (input.rank() < 2) {
        throw std::invalid_argument("linear: richiede un tensore a rango >= 2 sul device");
    }

    // Rango > 2 (es. [batch, seq, features]): appiattisce a
    // [batch*seq, features] (reshape senza copia, vedi
    // DeviceTensor::reshaped) prima del prodotto matriciale, poi
    // ripristina la forma originale sostituendo solo l'ultima
    // dimensione con le feature in uscita.
    std::size_t inFeatures = input.shape().back();
    std::size_t rows = input.elementCount() / inFeatures;
    DeviceTensor flatOutput = addBias(matmul(input.clone().reshaped({rows, inFeatures}), weight), bias);

    std::vector<std::size_t> outputShape = input.shape();
    outputShape.back() = weight.dim(1);
    return std::move(flatOutput).reshaped(std::move(outputShape));
}

}  // namespace blackforge::backend::cuda
