#include <stdexcept>

#include <cublas_v2.h>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/backend/cuda/ops.hpp"

namespace blackforge::backend::cuda {

namespace {

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

DeviceTensor linear(const DeviceTensor& input, const DeviceTensor& weight, const DeviceTensor& bias) {
    return addBias(matmul(input, weight), bias);
}

}  // namespace blackforge::backend::cuda
