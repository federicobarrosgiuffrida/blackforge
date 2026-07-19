#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/backend/cuda/fused_attention.hpp"
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

// Come projectionMatmul(), ma usa matmulBf16CachedWeight() al posto di
// matmulBf16() quando useBf16 e' attivo: solo per le varianti
// *ForwardCached, il cui contratto di invalidazione e' gestito
// interamente da cuda::Model (vedi il commento su matmulBf16CachedWeight
// in ops.hpp).
DeviceTensor projectionMatmulCached(const DeviceTensor& a, const DeviceTensor& b, bool useBf16) {
    return useBf16 ? matmulBf16CachedWeight(a, b) : matmul(a, b);
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

    // Nucleo fuso a online softmax (vedi fused_attention.hpp): mai una
    // matrice di score materializzata in memoria globale, un solo
    // lancio di kernel invece di quattro (Q@K^T, maschera, softmax,
    // probs@V separati). Resta sempre in float32 anche quando useBf16
    // e' attivo: solo le proiezioni lineari (la parte dominante del
    // costo computazionale) passano per il Tensor Core.
    FusedAttentionForwardResult attn = fusedAttentionForward(q, k, v, numHeads, scaleFactor, /*oldLen=*/0);

    DeviceTensor projected = projectionMatmul(std::move(attn.output).reshaped({batch * seq, dim}), wout, useBf16)
                                  .reshaped({batch, seq, dim});
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

    // Stesso nucleo fuso di selfAttentionImpl(): qui newLen (i token
    // nuovi) e totalLen (l'intera cache K/V aggiornata) possono
    // differire, e oldLen != 0 generalizza la maschera causale a
    // "posizioni gia' in cache sempre visibili" (vedi
    // fused_attention.hpp).
    float scaleFactor = 1.0F / std::sqrt(static_cast<float>(dim / numHeads));
    FusedAttentionForwardResult attn = fusedAttentionForward(qNew, cache.k, cache.v, numHeads, scaleFactor, oldLen);

    DeviceTensor projected = projectionMatmul(std::move(attn.output).reshaped({batch * newLen, dim}), wout, useBf16)
                                  .reshaped({batch, newLen, dim});
    return add(newInput, projected);
}

DeviceTensor feedForwardForwardCachedImpl(const DeviceTensor& input, const DeviceTensor& w1, const DeviceTensor& b1,
                                           const DeviceTensor& w2, const DeviceTensor& b2, bool useBf16,
                                           FeedForwardCache& cache) {
    // A differenza di feedForwardImpl(), qui il GEMM grezzo (prima di
    // bias/silu) viene tenuto separato invece di passare per
    // projectionLinearCached() (che fonde subito il bias): serve cachare
    // gemm1 SENZA bias/silu applicati, cosi' addBiasSiluBackward puo'
    // ricalcolare bias+silu al volo nel backward invece di leggerli da
    // un tensore intermedio in piu' (vedi FeedForwardCache in ops.hpp).
    cache.normed = rmsnorm(input);

    std::size_t dim = cache.normed.shape().back();
    std::size_t rows = cache.normed.elementCount() / dim;
    std::size_t hiddenDim = w1.dim(1);

    DeviceTensor gemm1Flat = projectionMatmulCached(cache.normed.clone().reshaped({rows, dim}), w1, useBf16);
    std::vector<std::size_t> hiddenShape = cache.normed.shape();
    hiddenShape.back() = hiddenDim;
    cache.gemm1 = std::move(gemm1Flat).reshaped(hiddenShape);
    // Fonde addBias(gemm1,b1)+silu(...) in un solo kernel (vedi ops.hpp).
    cache.hidden = addBiasSilu(cache.gemm1, b1);

    std::size_t hiddenRows = cache.hidden.elementCount() / hiddenDim;
    std::size_t outDim = w2.dim(1);
    DeviceTensor gemm2Flat =
        projectionMatmulCached(cache.hidden.clone().reshaped({hiddenRows, hiddenDim}), w2, useBf16);
    std::vector<std::size_t> outShape = input.shape();
    outShape.back() = outDim;
    DeviceTensor gemm2 = std::move(gemm2Flat).reshaped(outShape);

    // Fonde addBias(gemm2,b2)+add(input,...) (residual) in un solo
    // kernel (vedi ops.hpp).
    return addBiasResidual(gemm2, b2, input);
}

DeviceTensor selfAttentionForwardCachedImpl(const DeviceTensor& input, const DeviceTensor& wq, const DeviceTensor& wk,
                                             const DeviceTensor& wv, const DeviceTensor& wout, std::size_t numHeads,
                                             bool useBf16, SelfAttentionCache& cache, const char* callerName) {
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

    // rmsnorm(input) e' gia' un prvalue indipendente (non alias di
    // input): reshaped() diretto sul risultato, zero clone (a
    // differenza di selfAttentionImpl sopra, che deve clonare 'normed'
    // perche' li' resta anche a rango 3 per essere riusato tre volte —
    // qui invece serve solo la forma appiattita, riusata per
    // riferimento costante da tutte e tre le proiezioni).
    cache.normedFlat = rmsnorm(input).reshaped({batch * seq, dim});
    cache.q = projectionMatmulCached(cache.normedFlat, wq, useBf16).reshaped({batch, seq, dim});
    cache.k = projectionMatmulCached(cache.normedFlat, wk, useBf16).reshaped({batch, seq, dim});
    cache.v = projectionMatmulCached(cache.normedFlat, wv, useBf16).reshaped({batch, seq, dim});

    FusedAttentionForwardResult attn = fusedAttentionForward(cache.q, cache.k, cache.v, numHeads, scaleFactor,
                                                               /*oldLen=*/0);
    cache.attnOutput = std::move(attn.output);
    cache.attnM = std::move(attn.m);
    cache.attnL = std::move(attn.l);

    // cache.attnOutput deve restare valido (rango 3) per il backward:
    // un clone qui, a differenza di selfAttentionImpl che puo' consumare
    // 'attn.output' direttamente dato che non lo riusa dopo.
    DeviceTensor projected =
        projectionMatmulCached(cache.attnOutput.clone().reshaped({batch * seq, dim}), wout, useBf16)
            .reshaped({batch, seq, dim});
    return add(input, projected);
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

DeviceTensor feedForwardForwardCached(const DeviceTensor& input, const DeviceTensor& w1, const DeviceTensor& b1,
                                       const DeviceTensor& w2, const DeviceTensor& b2, bool useBf16,
                                       FeedForwardCache& cache) {
    return feedForwardForwardCachedImpl(input, w1, b1, w2, b2, useBf16, cache);
}

DeviceTensor selfAttentionForwardCached(const DeviceTensor& input, const DeviceTensor& wq, const DeviceTensor& wk,
                                         const DeviceTensor& wv, const DeviceTensor& wout, std::size_t numHeads,
                                         bool useBf16, SelfAttentionCache& cache) {
    return selfAttentionForwardCachedImpl(input, wq, wk, wv, wout, numHeads, useBf16, cache,
                                           "selfAttentionForwardCached");
}

}  // namespace blackforge::backend::cuda
