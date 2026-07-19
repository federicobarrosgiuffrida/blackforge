#include "blackforge/backend/cuda/attention_batched.hpp"

#include "blackforge/backend/cuda/cuda_check.hpp"

namespace blackforge::backend::cuda {

namespace {

constexpr int kBlockSize = 256;
int gridSizeFor(std::size_t n) { return static_cast<int>((n + kBlockSize - 1) / kBlockSize); }

// scores[b,h,i,j] = scaleFactor * sum_d q[b,i,h*headDim+d] * k[b,j,h*headDim+d].
// Un thread per elemento di 'scores' (batch*numHeads*newLen*totalLen):
// nessuna estrazione di testa intermedia, indicizzazione strided
// diretta su q/k (heads contigue in 'dim').
__global__ void batchedQKKernel(float* scores, const float* q, const float* k, std::size_t numHeads,
                                 std::size_t newLen, std::size_t totalLen, std::size_t dim, std::size_t headDim,
                                 float scaleFactor, std::size_t total) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    std::size_t perBh = newLen * totalLen;
    std::size_t bh = idx / perBh;
    std::size_t rem = idx % perBh;
    std::size_t i = rem / totalLen;
    std::size_t j = rem % totalLen;
    std::size_t b = bh / numHeads;
    std::size_t h = bh % numHeads;

    std::size_t qBase = (b * newLen + i) * dim + h * headDim;
    std::size_t kBase = (b * totalLen + j) * dim + h * headDim;
    float sum = 0.0F;
    for (std::size_t d = 0; d < headDim; ++d) {
        sum += q[qBase + d] * k[kBase + d];
    }
    scores[idx] = sum * scaleFactor;
}

// Maschera causale (eventualmente incrementale): scores[.,.,i,j] con
// j > oldLen + i viene impostato a -infinito. Un thread per elemento.
__global__ void batchedMaskKernel(float* scores, std::size_t newLen, std::size_t totalLen, std::size_t oldLen,
                                   std::size_t total) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    std::size_t perBh = newLen * totalLen;
    std::size_t rem = idx % perBh;
    std::size_t i = rem / totalLen;
    std::size_t j = rem % totalLen;
    if (j > oldLen + i) {
        scores[idx] = -INFINITY;
    }
}

// merged[b,s,h*headDim+d] = sum_k probs[b,h,s,k] * v[b,k,h*headDim+d]:
// scrive direttamente nel layout "unito" [batch,newLen,dim], senza un
// passaggio di scatter separato. Un thread per elemento di 'merged'.
__global__ void batchedPVKernel(float* merged, const float* probs, const float* v, std::size_t numHeads,
                                 std::size_t newLen, std::size_t totalLen, std::size_t dim, std::size_t headDim,
                                 std::size_t total) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    std::size_t perBatch = newLen * dim;
    std::size_t b = idx / perBatch;
    std::size_t rem = idx % perBatch;
    std::size_t s = rem / dim;
    std::size_t fullD = rem % dim;
    std::size_t h = fullD / headDim;

    std::size_t probsBase = (b * numHeads + h) * newLen * totalLen + s * totalLen;
    float sum = 0.0F;
    for (std::size_t k = 0; k < totalLen; ++k) {
        sum += probs[probsBase + k] * v[(b * totalLen + k) * dim + fullD];
    }
    merged[idx] = sum;
}

// dQ[b,i,fullD] = scaleFactor * sum_j dScores[b,h,i,j] * k[b,j,fullD].
// Un thread per elemento di 'dQ' (stessa forma di q).
__global__ void batchedQKBackwardDQKernel(float* dQ, const float* dScores, const float* k, std::size_t numHeads,
                                           std::size_t newLen, std::size_t totalLen, std::size_t dim,
                                           std::size_t headDim, float scaleFactor, std::size_t total) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    std::size_t perBatch = newLen * dim;
    std::size_t b = idx / perBatch;
    std::size_t rem = idx % perBatch;
    std::size_t i = rem / dim;
    std::size_t fullD = rem % dim;
    std::size_t h = fullD / headDim;

    std::size_t scoresBase = (b * numHeads + h) * newLen * totalLen + i * totalLen;
    float sum = 0.0F;
    for (std::size_t j = 0; j < totalLen; ++j) {
        sum += dScores[scoresBase + j] * k[(b * totalLen + j) * dim + fullD];
    }
    dQ[idx] = sum * scaleFactor;
}

// dK[b,j,fullD] = scaleFactor * sum_i dScores[b,h,i,j] * q[b,i,fullD].
// Un thread per elemento di 'dK' (stessa forma di k).
__global__ void batchedQKBackwardDKKernel(float* dK, const float* dScores, const float* q, std::size_t numHeads,
                                           std::size_t newLen, std::size_t totalLen, std::size_t dim,
                                           std::size_t headDim, float scaleFactor, std::size_t total) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    std::size_t perBatch = totalLen * dim;
    std::size_t b = idx / perBatch;
    std::size_t rem = idx % perBatch;
    std::size_t j = rem / dim;
    std::size_t fullD = rem % dim;
    std::size_t h = fullD / headDim;

    float sum = 0.0F;
    for (std::size_t i = 0; i < newLen; ++i) {
        std::size_t scoresIdx = (b * numHeads + h) * newLen * totalLen + i * totalLen + j;
        sum += dScores[scoresIdx] * q[(b * newLen + i) * dim + fullD];
    }
    dK[idx] = sum * scaleFactor;
}

// dV[b,k,fullD] = sum_s dMerged[b,s,fullD] * probs[b,h,s,k]. Un thread
// per elemento di 'dV' (stessa forma di v).
__global__ void batchedPVBackwardDVKernel(float* dV, const float* dMerged, const float* probs, std::size_t numHeads,
                                           std::size_t newLen, std::size_t totalLen, std::size_t dim,
                                           std::size_t headDim, std::size_t total) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    std::size_t perBatch = totalLen * dim;
    std::size_t b = idx / perBatch;
    std::size_t rem = idx % perBatch;
    std::size_t k = rem / dim;
    std::size_t fullD = rem % dim;
    std::size_t h = fullD / headDim;

    float sum = 0.0F;
    for (std::size_t s = 0; s < newLen; ++s) {
        sum += dMerged[(b * newLen + s) * dim + fullD] * probs[(b * numHeads + h) * newLen * totalLen + s * totalLen + k];
    }
    dV[idx] = sum;
}

// dProbs[b,h,s,k] = sum_d dMerged[b,s,h*headDim+d] * v[b,k,h*headDim+d].
// Un thread per elemento di 'dProbs' (stessa forma di probs).
__global__ void batchedPVBackwardDProbsKernel(float* dProbs, const float* dMerged, const float* v,
                                               std::size_t numHeads, std::size_t newLen, std::size_t totalLen,
                                               std::size_t dim, std::size_t headDim, std::size_t total) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    std::size_t perBh = newLen * totalLen;
    std::size_t bh = idx / perBh;
    std::size_t rem = idx % perBh;
    std::size_t s = rem / totalLen;
    std::size_t k = rem % totalLen;
    std::size_t b = bh / numHeads;
    std::size_t h = bh % numHeads;

    std::size_t dMergedBase = (b * newLen + s) * dim + h * headDim;
    std::size_t vBase = (b * totalLen + k) * dim + h * headDim;
    float sum = 0.0F;
    for (std::size_t d = 0; d < headDim; ++d) {
        sum += dMerged[dMergedBase + d] * v[vBase + d];
    }
    dProbs[idx] = sum;
}

}  // namespace

DeviceTensor batchedQK(const DeviceTensor& q, const DeviceTensor& k, std::size_t numHeads, float scaleFactor) {
    std::size_t batch = q.dim(0);
    std::size_t newLen = q.dim(1);
    std::size_t dim = q.dim(2);
    std::size_t totalLen = k.dim(1);
    std::size_t headDim = dim / numHeads;

    DeviceTensor scores({batch, numHeads, newLen, totalLen});
    std::size_t total = batch * numHeads * newLen * totalLen;
    if (total > 0) {
        batchedQKKernel<<<gridSizeFor(total), kBlockSize>>>(scores.data(), q.data(), k.data(), numHeads, newLen,
                                                             totalLen, dim, headDim, scaleFactor, total);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return scores;
}

DeviceTensor batchedMask(const DeviceTensor& scores, std::size_t oldLen) {
    DeviceTensor result = scores.clone();
    std::size_t newLen = result.dim(2);
    std::size_t totalLen = result.dim(3);
    std::size_t total = result.elementCount();
    if (total > 0) {
        batchedMaskKernel<<<gridSizeFor(total), kBlockSize>>>(result.data(), newLen, totalLen, oldLen, total);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor batchedPV(const DeviceTensor& probs, const DeviceTensor& v, std::size_t numHeads) {
    std::size_t batch = probs.dim(0);
    std::size_t newLen = probs.dim(2);
    std::size_t totalLen = probs.dim(3);
    std::size_t dim = v.dim(2);
    std::size_t headDim = dim / numHeads;

    DeviceTensor merged({batch, newLen, dim});
    std::size_t total = batch * newLen * dim;
    if (total > 0) {
        batchedPVKernel<<<gridSizeFor(total), kBlockSize>>>(merged.data(), probs.data(), v.data(), numHeads, newLen,
                                                             totalLen, dim, headDim, total);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return merged;
}

BatchedQKGrad batchedQKBackward(const DeviceTensor& dScores, const DeviceTensor& q, const DeviceTensor& k,
                                 std::size_t numHeads, float scaleFactor) {
    std::size_t newLen = q.dim(1);
    std::size_t totalLen = k.dim(1);
    std::size_t dim = q.dim(2);
    std::size_t headDim = dim / numHeads;

    DeviceTensor dQ(q.shape());
    std::size_t totalQ = q.elementCount();
    if (totalQ > 0) {
        batchedQKBackwardDQKernel<<<gridSizeFor(totalQ), kBlockSize>>>(dQ.data(), dScores.data(), k.data(), numHeads,
                                                                        newLen, totalLen, dim, headDim, scaleFactor,
                                                                        totalQ);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }

    DeviceTensor dK(k.shape());
    std::size_t totalK = k.elementCount();
    if (totalK > 0) {
        batchedQKBackwardDKKernel<<<gridSizeFor(totalK), kBlockSize>>>(dK.data(), dScores.data(), q.data(), numHeads,
                                                                        newLen, totalLen, dim, headDim, scaleFactor,
                                                                        totalK);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }

    return BatchedQKGrad{std::move(dQ), std::move(dK)};
}

BatchedPVGrad batchedPVBackward(const DeviceTensor& dMerged, const DeviceTensor& probs, const DeviceTensor& v,
                                 std::size_t numHeads) {
    std::size_t newLen = probs.dim(2);
    std::size_t totalLen = probs.dim(3);
    std::size_t dim = v.dim(2);
    std::size_t headDim = dim / numHeads;

    DeviceTensor dV(v.shape());
    std::size_t totalV = v.elementCount();
    if (totalV > 0) {
        batchedPVBackwardDVKernel<<<gridSizeFor(totalV), kBlockSize>>>(dV.data(), dMerged.data(), probs.data(),
                                                                        numHeads, newLen, totalLen, dim, headDim,
                                                                        totalV);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }

    DeviceTensor dProbs(probs.shape());
    std::size_t totalProbs = probs.elementCount();
    if (totalProbs > 0) {
        batchedPVBackwardDProbsKernel<<<gridSizeFor(totalProbs), kBlockSize>>>(
            dProbs.data(), dMerged.data(), v.data(), numHeads, newLen, totalLen, dim, headDim, totalProbs);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }

    return BatchedPVGrad{std::move(dProbs), std::move(dV)};
}

}  // namespace blackforge::backend::cuda
