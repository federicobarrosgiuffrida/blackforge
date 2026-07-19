#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "blackforge/backend/cuda/attention_batched.hpp"
#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/backend/cuda/ops.hpp"

namespace blackforge::backend::cuda {

namespace {

constexpr int kBlockSize = 256;
int gridSizeFor(std::size_t n) { return static_cast<int>((n + kBlockSize - 1) / kBlockSize); }

// Concatena due tensori device [batch, seqA, dim] e [batch, seqB, dim]
// lungo la dimensione di sequenza, producendo [batch, seqA+seqB, dim]:
// usato da selfAttentionIncremental per accumulare K/V nella cache
// (vedi backend::cpu::concatSeq, stessa semantica). Un thread per
// elemento di output.
__global__ void concatSeqKernel(float* out, const float* a, const float* b, std::size_t seqA, std::size_t seqB,
                                 std::size_t dim, std::size_t totalElements) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= totalElements) {
        return;
    }
    std::size_t seqTotal = seqA + seqB;
    std::size_t perBatch = seqTotal * dim;
    std::size_t batchIdx = idx / perBatch;
    std::size_t withinBatch = idx % perBatch;
    std::size_t s = withinBatch / dim;
    std::size_t d = withinBatch % dim;
    if (s < seqA) {
        out[idx] = a[(batchIdx * seqA + s) * dim + d];
    } else {
        out[idx] = b[(batchIdx * seqB + (s - seqA)) * dim + d];
    }
}

DeviceTensor concatSeq(const DeviceTensor& a, const DeviceTensor& b) {
    std::size_t batch = a.dim(0);
    std::size_t seqA = a.dim(1);
    std::size_t seqB = b.dim(1);
    std::size_t dim = a.dim(2);
    std::size_t seqTotal = seqA + seqB;

    DeviceTensor result({batch, seqTotal, dim});
    std::size_t total = batch * seqTotal * dim;
    if (total > 0) {
        concatSeqKernel<<<gridSizeFor(total), kBlockSize>>>(result.data(), a.data(), b.data(), seqA, seqB, dim,
                                                              total);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

// Le proiezioni lineari (Q/K/V/Out di selfAttention, W1/W2 di
// feedForward) passano per matmul()/linear() (SGEMM float32) o per
// matmulBf16()/linearBf16() (Tensor Core) a seconda di 'useBf16': un
// solo punto di scelta condiviso da tutte le proiezioni di questo
// file, invece di duplicare interamente selfAttention/feedForward per
// la variante BF16 (stesso principio gia' usato per causale/
// bidirezionale in ops.hpp).
DeviceTensor projectionMatmul(const DeviceTensor& a, const DeviceTensor& b, bool useBf16) {
    return useBf16 ? matmulBf16(a, b) : matmul(a, b);
}

DeviceTensor projectionLinear(const DeviceTensor& input, const DeviceTensor& weight, const DeviceTensor& bias,
                               bool useBf16) {
    return useBf16 ? linearBf16(input, weight, bias) : linear(input, weight, bias);
}

DeviceTensor feedForwardImpl(const DeviceTensor& input, const DeviceTensor& w1, const DeviceTensor& b1,
                              const DeviceTensor& w2, const DeviceTensor& b2, bool useBf16) {
    DeviceTensor normed = rmsnorm(input);
    DeviceTensor hidden = silu(projectionLinear(normed, w1, b1, useBf16));
    DeviceTensor out = projectionLinear(hidden, w2, b2, useBf16);
    return add(input, out);
}

DeviceTensor selfAttentionImpl(const DeviceTensor& input, const DeviceTensor& wq, const DeviceTensor& wk,
                                const DeviceTensor& wv, const DeviceTensor& wout, std::size_t numHeads,
                                bool useBf16, const char* callerName) {
    if (input.rank() != 3) {
        throw std::invalid_argument(std::string(callerName) +
                                     ": richiede un input a rango 3 [batch, seq, dim] sul device");
    }
    std::size_t batch = input.dim(0);
    std::size_t seq = input.dim(1);
    std::size_t dim = input.dim(2);
    if (numHeads == 0 || dim % numHeads != 0) {
        throw std::invalid_argument(std::string(callerName) + ": numHeads deve dividere esattamente dim");
    }
    std::size_t headDim = dim / numHeads;
    float scaleFactor = 1.0F / std::sqrt(static_cast<float>(headDim));

    // Le proiezioni Q/K/V/Out non hanno bias (come in LLaMA e come la
    // controparte CPU): matmul()/matmulBf16() diretto invece di
    // linear(), su 'normed' appiattito a 2D (reshape senza copia via
    // DeviceTensor::reshaped, tranne il clone() necessario perche'
    // 'normed' serve tre volte mentre reshaped() consuma il tensore).
    DeviceTensor normed = rmsnorm(input);
    DeviceTensor q = projectionMatmul(normed.clone().reshaped({batch * seq, dim}), wq, useBf16)
                          .reshaped({batch, seq, dim});
    DeviceTensor k = projectionMatmul(normed.clone().reshaped({batch * seq, dim}), wk, useBf16)
                          .reshaped({batch, seq, dim});
    DeviceTensor v = projectionMatmul(std::move(normed).reshaped({batch * seq, dim}), wv, useBf16)
                          .reshaped({batch, seq, dim});

    // Nucleo batchizzato (vedi attention_batched.hpp): Q@K^T, maschera
    // causale e probs@V per TUTTE le combinazioni (batch, testa) in
    // pochi lanci di kernel, invece di un lancio per ciascuna delle
    // batch*numHeads combinazioni. Resta sempre in float32 anche
    // quando useBf16 e' attivo: solo le proiezioni lineari (la parte
    // dominante del costo computazionale) passano per il Tensor Core.
    // softmax() e' gia' generica rispetto alle dimensioni iniziali
    // (opera per "riga", l'ultima dimensione), quindi si applica
    // direttamente a scores [batch,numHeads,seq,seq] senza bisogno di
    // una versione batchizzata dedicata.
    DeviceTensor scores = batchedQK(q, k, numHeads, scaleFactor);
    DeviceTensor maskedScores = batchedMask(scores, /*oldLen=*/0);
    DeviceTensor probs = softmax(maskedScores);
    DeviceTensor concatenated = batchedPV(probs, v, numHeads);

    DeviceTensor projected =
        projectionMatmul(std::move(concatenated).reshaped({batch * seq, dim}), wout, useBf16).reshaped({batch, seq, dim});
    return add(input, projected);
}

DeviceTensor selfAttentionIncrementalImpl(const DeviceTensor& newInput, const DeviceTensor& wq,
                                           const DeviceTensor& wk, const DeviceTensor& wv, const DeviceTensor& wout,
                                           std::size_t numHeads, KVCache& cache, bool useBf16,
                                           const char* callerName) {
    if (newInput.rank() != 3) {
        throw std::invalid_argument(std::string(callerName) +
                                     ": richiede un input a rango 3 [batch, seq, dim] sul device");
    }
    std::size_t batch = newInput.dim(0);
    std::size_t newLen = newInput.dim(1);
    std::size_t dim = newInput.dim(2);
    if (numHeads == 0 || dim % numHeads != 0) {
        throw std::invalid_argument(std::string(callerName) + ": numHeads deve dividere esattamente dim");
    }

    // Pre-norm: e' per-posizione (nessuna dipendenza tra posizioni),
    // quindi applicarla solo ai token nuovi e' esatto, non
    // un'approssimazione (stesso ragionamento della controparte CPU).
    DeviceTensor normed = rmsnorm(newInput);
    DeviceTensor qNew = projectionMatmul(normed.clone().reshaped({batch * newLen, dim}), wq, useBf16)
                             .reshaped({batch, newLen, dim});
    DeviceTensor kNew = projectionMatmul(normed.clone().reshaped({batch * newLen, dim}), wk, useBf16)
                             .reshaped({batch, newLen, dim});
    DeviceTensor vNew = projectionMatmul(std::move(normed).reshaped({batch * newLen, dim}), wv, useBf16)
                             .reshaped({batch, newLen, dim});

    std::size_t oldLen = cache.length;
    if (oldLen == 0) {
        cache.k = std::move(kNew);
        cache.v = std::move(vNew);
    } else {
        cache.k = concatSeq(cache.k, kNew);
        cache.v = concatSeq(cache.v, vNew);
    }
    cache.length = oldLen + newLen;

    // Stesso nucleo batchizzato di selfAttentionImpl(): qui newLen (i
    // token nuovi) e totalLen (l'intera cache K/V aggiornata) possono
    // differire, e oldLen != 0 generalizza la maschera causale a
    // "posizioni gia' in cache sempre visibili" (vedi
    // attention_batched.hpp).
    float scaleFactor = 1.0F / std::sqrt(static_cast<float>(dim / numHeads));
    DeviceTensor scores = batchedQK(qNew, cache.k, numHeads, scaleFactor);
    DeviceTensor maskedScores = batchedMask(scores, oldLen);
    DeviceTensor probs = softmax(maskedScores);
    DeviceTensor concatenated = batchedPV(probs, cache.v, numHeads);

    DeviceTensor projected = projectionMatmul(std::move(concatenated).reshaped({batch * newLen, dim}), wout, useBf16)
                                  .reshaped({batch, newLen, dim});
    return add(newInput, projected);
}

}  // namespace

DeviceTensor feedForward(const DeviceTensor& input, const DeviceTensor& w1, const DeviceTensor& b1,
                          const DeviceTensor& w2, const DeviceTensor& b2) {
    return feedForwardImpl(input, w1, b1, w2, b2, /*useBf16=*/false);
}

DeviceTensor feedForwardBf16(const DeviceTensor& input, const DeviceTensor& w1, const DeviceTensor& b1,
                              const DeviceTensor& w2, const DeviceTensor& b2) {
    return feedForwardImpl(input, w1, b1, w2, b2, /*useBf16=*/true);
}

DeviceTensor selfAttention(const DeviceTensor& input, const DeviceTensor& wq, const DeviceTensor& wk,
                            const DeviceTensor& wv, const DeviceTensor& wout, std::size_t numHeads) {
    return selfAttentionImpl(input, wq, wk, wv, wout, numHeads, /*useBf16=*/false, "selfAttention");
}

DeviceTensor selfAttentionBf16(const DeviceTensor& input, const DeviceTensor& wq, const DeviceTensor& wk,
                                const DeviceTensor& wv, const DeviceTensor& wout, std::size_t numHeads) {
    return selfAttentionImpl(input, wq, wk, wv, wout, numHeads, /*useBf16=*/true, "selfAttentionBf16");
}

DeviceTensor selfAttentionIncremental(const DeviceTensor& newInput, const DeviceTensor& wq, const DeviceTensor& wk,
                                       const DeviceTensor& wv, const DeviceTensor& wout, std::size_t numHeads,
                                       KVCache& cache) {
    return selfAttentionIncrementalImpl(newInput, wq, wk, wv, wout, numHeads, cache, /*useBf16=*/false,
                                         "selfAttentionIncremental");
}

DeviceTensor selfAttentionIncrementalBf16(const DeviceTensor& newInput, const DeviceTensor& wq,
                                           const DeviceTensor& wk, const DeviceTensor& wv, const DeviceTensor& wout,
                                           std::size_t numHeads, KVCache& cache) {
    return selfAttentionIncrementalImpl(newInput, wq, wk, wv, wout, numHeads, cache, /*useBf16=*/true,
                                         "selfAttentionIncrementalBf16");
}

}  // namespace blackforge::backend::cuda
