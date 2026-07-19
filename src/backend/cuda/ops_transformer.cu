#include <cmath>
#include <stdexcept>
#include <vector>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/backend/cuda/ops.hpp"

namespace blackforge::backend::cuda {

namespace {

constexpr int kBlockSize = 256;
int gridSizeFor(std::size_t n) { return static_cast<int>((n + kBlockSize - 1) / kBlockSize); }

// Estrae la "testa" h dell'esempio b da un tensore device [batch, seq,
// dim] (concettualmente [batch, seq, numHeads, headDim], con le teste
// contigue nell'ultima dimensione) in un buffer [seq, headDim] gia'
// allocato. Un thread per elemento di output.
__global__ void extractHeadKernel(float* out, const float* full, std::size_t b, std::size_t h, std::size_t seq,
                                   std::size_t dim, std::size_t headDim) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = seq * headDim;
    if (idx >= total) {
        return;
    }
    std::size_t s = idx / headDim;
    std::size_t d = idx % headDim;
    out[idx] = full[(b * seq + s) * dim + h * headDim + d];
}

// Scrive una "testa" [seq, headDim] dentro un buffer [batch, seq, dim]:
// le teste partizionano esattamente 'dim', quindi e' una scrittura
// diretta (nessuna sovrapposizione tra teste diverse, non serve
// atomicAdd).
__global__ void scatterHeadKernel(float* full, const float* headSlice, std::size_t b, std::size_t h,
                                   std::size_t seq, std::size_t dim, std::size_t headDim) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = seq * headDim;
    if (idx >= total) {
        return;
    }
    std::size_t s = idx / headDim;
    std::size_t d = idx % headDim;
    full[(b * seq + s) * dim + h * headDim + d] = headSlice[idx];
}

// Imposta a -infinito le posizioni (i,j) con j > i di una matrice
// [seq, seq] (maschera causale): un thread per elemento, condizionale.
__global__ void applyCausalMaskKernel(float* scores, std::size_t seq) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = seq * seq;
    if (idx >= total) {
        return;
    }
    std::size_t i = idx / seq;
    std::size_t j = idx % seq;
    if (j > i) {
        // -INFINITY (macro di <math.h>, non std::numeric_limits) e'
        // utilizzabile dentro un kernel __global__ senza bisogno di
        // --expt-relaxed-constexpr: stessa convenzione gia' usata in
        // softmaxKernel (ops_elementwise.cu).
        scores[idx] = -INFINITY;
    }
}

DeviceTensor extractHead(const DeviceTensor& full, std::size_t b, std::size_t h, std::size_t seq, std::size_t dim,
                          std::size_t headDim) {
    DeviceTensor result({seq, headDim});
    std::size_t total = seq * headDim;
    if (total > 0) {
        extractHeadKernel<<<gridSizeFor(total), kBlockSize>>>(result.data(), full.data(), b, h, seq, dim, headDim);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

void scatterHead(DeviceTensor& full, const DeviceTensor& headSlice, std::size_t b, std::size_t h, std::size_t seq,
                  std::size_t dim, std::size_t headDim) {
    std::size_t total = seq * headDim;
    if (total > 0) {
        scatterHeadKernel<<<gridSizeFor(total), kBlockSize>>>(full.data(), headSlice.data(), b, h, seq, dim,
                                                               headDim);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
}

DeviceTensor applyCausalMask(const DeviceTensor& scores) {
    DeviceTensor result = scores.clone();
    std::size_t seq = result.dim(0);
    std::size_t total = seq * seq;
    if (total > 0) {
        applyCausalMaskKernel<<<gridSizeFor(total), kBlockSize>>>(result.data(), seq);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

// scale() non e' esposta in ops.hpp (e' interna): un fattore scalare
// moltiplicato elemento per elemento, riusa lo stesso kernel pattern
// delle altre operazioni elementwise di questo backend.
__global__ void scaleKernel(float* out, const float* input, float factor, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = input[i] * factor;
    }
}

DeviceTensor scale(const DeviceTensor& input, float factor) {
    DeviceTensor result(input.shape());
    std::size_t n = input.elementCount();
    if (n > 0) {
        scaleKernel<<<gridSizeFor(n), kBlockSize>>>(result.data(), input.data(), factor, n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

}  // namespace

DeviceTensor feedForward(const DeviceTensor& input, const DeviceTensor& w1, const DeviceTensor& b1,
                          const DeviceTensor& w2, const DeviceTensor& b2) {
    DeviceTensor normed = rmsnorm(input);
    DeviceTensor hidden = silu(linear(normed, w1, b1));
    DeviceTensor out = linear(hidden, w2, b2);
    return add(input, out);
}

DeviceTensor selfAttention(const DeviceTensor& input, const DeviceTensor& wq, const DeviceTensor& wk,
                            const DeviceTensor& wv, const DeviceTensor& wout, std::size_t numHeads) {
    if (input.rank() != 3) {
        throw std::invalid_argument("selfAttention: richiede un input a rango 3 [batch, seq, dim] sul device");
    }
    std::size_t batch = input.dim(0);
    std::size_t seq = input.dim(1);
    std::size_t dim = input.dim(2);
    if (numHeads == 0 || dim % numHeads != 0) {
        throw std::invalid_argument("selfAttention: numHeads deve dividere esattamente dim");
    }
    std::size_t headDim = dim / numHeads;
    float scaleFactor = 1.0F / std::sqrt(static_cast<float>(headDim));

    // Le proiezioni Q/K/V/Out non hanno bias (come in LLaMA e come la
    // controparte CPU): matmul() diretto invece di linear(), su
    // 'normed' appiattito a 2D (reshape senza copia via
    // DeviceTensor::reshaped, tranne il clone() necessario perche'
    // 'normed' serve tre volte mentre reshaped() consuma il tensore).
    DeviceTensor normed = rmsnorm(input);
    DeviceTensor q = matmul(normed.clone().reshaped({batch * seq, dim}), wq).reshaped({batch, seq, dim});
    DeviceTensor k = matmul(normed.clone().reshaped({batch * seq, dim}), wk).reshaped({batch, seq, dim});
    DeviceTensor v = matmul(std::move(normed).reshaped({batch * seq, dim}), wv).reshaped({batch, seq, dim});

    DeviceTensor concatenated = DeviceTensor::zeros({batch, seq, dim});
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < numHeads; ++h) {
            DeviceTensor qHead = extractHead(q, b, h, seq, dim, headDim);
            DeviceTensor kHead = extractHead(k, b, h, seq, dim, headDim);
            DeviceTensor vHead = extractHead(v, b, h, seq, dim, headDim);

            DeviceTensor scores = scale(matmulTransposeB(qHead, kHead), scaleFactor);
            DeviceTensor maskedScores = applyCausalMask(scores);
            DeviceTensor probs = softmax(maskedScores);
            DeviceTensor headOut = matmul(probs, vHead);

            scatterHead(concatenated, headOut, b, h, seq, dim, headDim);
        }
    }

    DeviceTensor projected =
        matmul(std::move(concatenated).reshaped({batch * seq, dim}), wout).reshaped({batch, seq, dim});
    return add(input, projected);
}

}  // namespace blackforge::backend::cuda
