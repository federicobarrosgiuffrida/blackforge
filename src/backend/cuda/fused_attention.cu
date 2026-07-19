#include <stdexcept>
#include <string>

#include "blackforge/backend/cuda/fused_attention.hpp"

#include "blackforge/backend/cuda/cuda_check.hpp"

namespace blackforge::backend::cuda {

namespace {

constexpr int kWarpSize = 32;
// Righe di query per blocco: un warp per riga (blockDim = kWarpSize x
// kBr), cosi' K/V vengono caricati in shared memory UNA VOLTA per
// gruppo di kBr righe invece che una volta per singola riga — la vera
// tecnica di FlashAttention (vedi il commento sopra
// tiledFusedAttentionForwardKernel per il perche').
constexpr int kBr = 8;
// Posizioni chiave per tile caricato in shared memory.
constexpr int kBc = 32;
// headDim / kWarpSize arrotondato per eccesso: quanti elementi di
// headDim possiede ciascuna corsia del warp (headDim puo' superare
// kWarpSize, es. 64). Limite statico che sostiene headDim fino a
// kMaxElemsPerThread * kWarpSize = 256 — oltre, il costruttore lancia
// std::invalid_argument invece di produrre un risultato silenziosamente
// sbagliato (indici fuori dall'array locale).
constexpr int kMaxElemsPerThread = 8;

// Riduzione a somma dentro un singolo warp via shuffle: nessun
// __syncthreads, nessuna shared memory — i 5 passi (offset 16,8,4,2,1)
// scambiano direttamente i registri tra le corsie del warp.
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFFU, val, offset);
    }
    return val;
}

// Forward fuso a "online softmax" CON TILING multi-riga (stile
// FlashAttention): un blocco copre kBr righe di query consecutive per
// una combinazione (batch,testa) — non piu' una riga sola. Ogni riga e'
// di proprieta' di un warp (blockDim = (kWarpSize, kBr)); le corsie del
// warp si dividono gli elementi di headDim (elemsPerThread ciascuna,
// strided di kWarpSize).
//
// La differenza cruciale rispetto alla versione precedente (un blocco
// per riga): K/V vengono caricati in shared memory in tile di kBc
// posizioni, UNA VOLTA per tile, cooperativamente da TUTTI i thread del
// blocco (non solo dalla riga che li usa) — poi OGNI riga del blocco
// riusa lo stesso tile dalla shared memory per calcolare il proprio
// contributo, prima di passare al tile successivo. Senza tiling, ogni
// riga rileggeva K/V dalla memoria globale in modo completamente
// indipendente: con newLen righe che condividono lo stesso K/V, quello
// era traffico di memoria globale ripetuto ~newLen volte piu' del
// necessario. Con il tiling, il traffico scende di un fattore ~kBr, e
// la sincronizzazione (__syncthreads, necessaria solo per il
// caricamento/uso condiviso del tile) avviene una volta per tile
// (~totalLen/kBc volte) invece che una volta per singola posizione
// chiave.
__global__ void tiledFusedAttentionForwardKernel(float* out, float* mOut, float* lOut, const float* q,
                                                   const float* k, const float* v, std::size_t numHeads,
                                                   std::size_t newLen, std::size_t totalLen, std::size_t dim,
                                                   std::size_t headDim, float scaleFactor, std::size_t oldLen,
                                                   std::size_t numQueryTiles) {
    extern __shared__ float smem[];
    float* kTile = smem;                                    // [kBc][headDim]
    float* vTile = smem + static_cast<std::size_t>(kBc) * headDim;  // [kBc][headDim]

    std::size_t blockLinear = blockIdx.x;
    std::size_t queryTileIdx = blockLinear % numQueryTiles;
    std::size_t bh = blockLinear / numQueryTiles;
    std::size_t h = bh % numHeads;
    std::size_t b = bh / numHeads;

    auto row = static_cast<std::size_t>(threadIdx.y);   // riga LOCALE nel tile, 0..kBr-1
    auto lane = static_cast<std::size_t>(threadIdx.x);   // corsia nel warp, 0..31
    std::size_t i = queryTileIdx * static_cast<std::size_t>(kBr) + row;  // riga GLOBALE di query
    bool rowActive = i < newLen;

    auto elemsPerThread = static_cast<int>((headDim + kWarpSize - 1) / kWarpSize);

    float qReg[kMaxElemsPerThread];
    for (int e = 0; e < elemsPerThread; ++e) {
        std::size_t d = lane + static_cast<std::size_t>(e) * kWarpSize;
        qReg[e] = (rowActive && d < headDim) ? q[(b * newLen + i) * dim + h * headDim + d] : 0.0F;
    }

    float m = -INFINITY;
    float l = 0.0F;
    float acc[kMaxElemsPerThread];
    for (int e = 0; e < elemsPerThread; ++e) {
        acc[e] = 0.0F;
    }

    std::size_t maxKeyForRow = oldLen + i;  // valido solo se rowActive, ma innocuo altrimenti (non usato)
    std::size_t linearTid = row * static_cast<std::size_t>(kWarpSize) + lane;
    std::size_t blockThreads = static_cast<std::size_t>(kBr) * kWarpSize;
    std::size_t tileElems = static_cast<std::size_t>(kBc) * headDim;

    for (std::size_t tileStart = 0; tileStart < totalLen; tileStart += static_cast<std::size_t>(kBc)) {
        // Caricamento cooperativo del tile K/V: TUTTI i thread del
        // blocco partecipano (non solo quelli della riga proprietaria),
        // cosi' il tile si carica in una frazione delle iterazioni.
        for (std::size_t idx = linearTid; idx < tileElems; idx += blockThreads) {
            std::size_t localKpos = idx / headDim;
            std::size_t d = idx % headDim;
            std::size_t globalKpos = tileStart + localKpos;
            bool valid = globalKpos < totalLen && d < headDim;
            kTile[idx] = valid ? k[(b * totalLen + globalKpos) * dim + h * headDim + d] : 0.0F;
            vTile[idx] = valid ? v[(b * totalLen + globalKpos) * dim + h * headDim + d] : 0.0F;
        }
        __syncthreads();

        if (rowActive) {
            for (std::size_t localKpos = 0; localKpos < static_cast<std::size_t>(kBc); ++localKpos) {
                std::size_t globalKpos = tileStart + localKpos;
                if (globalKpos >= totalLen || globalKpos > maxKeyForRow) {
                    break;  // il tile e' ordinato per posizione: le successive sono anch'esse oltre il limite
                }

                float partial = 0.0F;
                for (int e = 0; e < elemsPerThread; ++e) {
                    std::size_t d = lane + static_cast<std::size_t>(e) * kWarpSize;
                    float kVal = (d < headDim) ? kTile[localKpos * headDim + d] : 0.0F;
                    partial += qReg[e] * kVal;
                }
                float score = warpReduceSum(partial);
                score = __shfl_sync(0xFFFFFFFFU, score, 0) * scaleFactor;

                float newM = fmaxf(m, score);
                float correction = expf(m - newM);  // exp(-inf - finito) = 0 alla prima iterazione, corretto
                float p = expf(score - newM);
                for (int e = 0; e < elemsPerThread; ++e) {
                    std::size_t d = lane + static_cast<std::size_t>(e) * kWarpSize;
                    float vVal = (d < headDim) ? vTile[localKpos * headDim + d] : 0.0F;
                    acc[e] = acc[e] * correction + p * vVal;
                }
                l = l * correction + p;
                m = newM;
            }
        }
        __syncthreads();  // nessun thread puo' iniziare a sovrascrivere il tile finche' tutte le righe l'hanno usato
    }

    if (rowActive) {
        for (int e = 0; e < elemsPerThread; ++e) {
            std::size_t d = lane + static_cast<std::size_t>(e) * kWarpSize;
            if (d < headDim) {
                out[(b * newLen + i) * dim + h * headDim + d] = acc[e] / l;
            }
        }
        if (lane == 0) {
            mOut[(b * numHeads + h) * newLen + i] = m;
            lOut[(b * numHeads + h) * newLen + i] = l;
        }
    }
}

// D[b,h,i] = dot(output[i,:], dOutput[i,:]) (sulla sola fetta di 'dim'
// della testa h): il termine di correzione della softmax-backward senza
// dover mai materializzare le probabilita' complete — sfrutta
// l'identita' dot(output,dOutput) = sum_j p[j]*dot(v[j,:],dOutput[i,:])
// = sum_j p[j]*dP[j]. Un warp per riga di query (stessa struttura del
// forward, ma senza bisogno di tiling: un solo dot product per riga,
// non un ciclo su tutte le posizioni chiave).
__global__ void computeDKernel(float* dOut, const float* output, const float* dOutput, std::size_t numHeads,
                                std::size_t newLen, std::size_t dim, std::size_t headDim) {
    std::size_t blockId = blockIdx.x;
    std::size_t i = blockId % newLen;
    std::size_t bh = blockId / newLen;
    std::size_t h = bh % numHeads;
    std::size_t b = bh / numHeads;

    auto lane = static_cast<std::size_t>(threadIdx.x);
    auto elemsPerThread = static_cast<int>((headDim + kWarpSize - 1) / kWarpSize);

    float partial = 0.0F;
    for (int e = 0; e < elemsPerThread; ++e) {
        std::size_t d = lane + static_cast<std::size_t>(e) * kWarpSize;
        if (d < headDim) {
            float oVal = output[(b * newLen + i) * dim + h * headDim + d];
            float dVal = dOutput[(b * newLen + i) * dim + h * headDim + d];
            partial += oVal * dVal;
        }
    }
    float total = warpReduceSum(partial);
    if (lane == 0) {
        dOut[(b * numHeads + h) * newLen + i] = total;
    }
}

// Backward fuso CON TILING multi-riga, stessa idea del forward: un
// blocco copre kBr righe di query, K/V caricati in shared memory una
// volta per tile e riusati da tutte le righe del blocco. Ricalcola
// score[j] (invece di averlo salvato) e deriva p[j] dalle statistiche
// (m,l) salvate dal forward. dQ[i,:] e' scritto una volta sola da
// questo blocco (nessuna race). dK/dV ricevono contributi da OGNI riga
// di query che attende alla posizione chiave j — piu' blocchi diversi
// (query tile differenti) scrivono sulla stessa locazione, serve
// atomicAdd.
__global__ void tiledFusedAttentionBackwardKernel(float* dQ, float* dK, float* dV, const float* dOutput,
                                                    const float* q, const float* k, const float* v,
                                                    const float* mSaved, const float* lSaved, const float* dTerm,
                                                    std::size_t numHeads, std::size_t newLen, std::size_t totalLen,
                                                    std::size_t dim, std::size_t headDim, float scaleFactor,
                                                    std::size_t oldLen, std::size_t numQueryTiles) {
    extern __shared__ float smem[];
    float* kTile = smem;
    float* vTile = smem + static_cast<std::size_t>(kBc) * headDim;

    std::size_t blockLinear = blockIdx.x;
    std::size_t queryTileIdx = blockLinear % numQueryTiles;
    std::size_t bh = blockLinear / numQueryTiles;
    std::size_t h = bh % numHeads;
    std::size_t b = bh / numHeads;

    auto row = static_cast<std::size_t>(threadIdx.y);
    auto lane = static_cast<std::size_t>(threadIdx.x);
    std::size_t i = queryTileIdx * static_cast<std::size_t>(kBr) + row;
    bool rowActive = i < newLen;

    auto elemsPerThread = static_cast<int>((headDim + kWarpSize - 1) / kWarpSize);

    float qReg[kMaxElemsPerThread];
    float dOutReg[kMaxElemsPerThread];
    for (int e = 0; e < elemsPerThread; ++e) {
        std::size_t d = lane + static_cast<std::size_t>(e) * kWarpSize;
        bool valid = rowActive && d < headDim;
        qReg[e] = valid ? q[(b * newLen + i) * dim + h * headDim + d] : 0.0F;
        dOutReg[e] = valid ? dOutput[(b * newLen + i) * dim + h * headDim + d] : 0.0F;
    }

    float mRow = rowActive ? mSaved[(b * numHeads + h) * newLen + i] : 0.0F;
    float lRow = rowActive ? lSaved[(b * numHeads + h) * newLen + i] : 1.0F;
    float dRow = rowActive ? dTerm[(b * numHeads + h) * newLen + i] : 0.0F;

    float dQAcc[kMaxElemsPerThread];
    for (int e = 0; e < elemsPerThread; ++e) {
        dQAcc[e] = 0.0F;
    }

    std::size_t maxKeyForRow = oldLen + i;
    std::size_t linearTid = row * static_cast<std::size_t>(kWarpSize) + lane;
    std::size_t blockThreads = static_cast<std::size_t>(kBr) * kWarpSize;
    std::size_t tileElems = static_cast<std::size_t>(kBc) * headDim;

    for (std::size_t tileStart = 0; tileStart < totalLen; tileStart += static_cast<std::size_t>(kBc)) {
        for (std::size_t idx = linearTid; idx < tileElems; idx += blockThreads) {
            std::size_t localKpos = idx / headDim;
            std::size_t d = idx % headDim;
            std::size_t globalKpos = tileStart + localKpos;
            bool valid = globalKpos < totalLen && d < headDim;
            kTile[idx] = valid ? k[(b * totalLen + globalKpos) * dim + h * headDim + d] : 0.0F;
            vTile[idx] = valid ? v[(b * totalLen + globalKpos) * dim + h * headDim + d] : 0.0F;
        }
        __syncthreads();

        if (rowActive) {
            for (std::size_t localKpos = 0; localKpos < static_cast<std::size_t>(kBc); ++localKpos) {
                std::size_t globalKpos = tileStart + localKpos;
                if (globalKpos >= totalLen || globalKpos > maxKeyForRow) {
                    break;
                }

                float scorePartial = 0.0F;
                float dPPartial = 0.0F;
                for (int e = 0; e < elemsPerThread; ++e) {
                    std::size_t d = lane + static_cast<std::size_t>(e) * kWarpSize;
                    float kVal = (d < headDim) ? kTile[localKpos * headDim + d] : 0.0F;
                    float vVal = (d < headDim) ? vTile[localKpos * headDim + d] : 0.0F;
                    scorePartial += qReg[e] * kVal;
                    dPPartial += dOutReg[e] * vVal;
                }
                float score = __shfl_sync(0xFFFFFFFFU, warpReduceSum(scorePartial), 0) * scaleFactor;
                float dP = __shfl_sync(0xFFFFFFFFU, warpReduceSum(dPPartial), 0);

                float p = expf(score - mRow) / lRow;
                float dScore = p * (dP - dRow);

                for (int e = 0; e < elemsPerThread; ++e) {
                    std::size_t d = lane + static_cast<std::size_t>(e) * kWarpSize;
                    if (d >= headDim) {
                        continue;
                    }
                    float kVal = kTile[localKpos * headDim + d];
                    dQAcc[e] += dScore * scaleFactor * kVal;
                    atomicAdd(&dK[(b * totalLen + globalKpos) * dim + h * headDim + d], dScore * scaleFactor * qReg[e]);
                    atomicAdd(&dV[(b * totalLen + globalKpos) * dim + h * headDim + d], p * dOutReg[e]);
                }
            }
        }
        __syncthreads();
    }

    if (rowActive) {
        for (int e = 0; e < elemsPerThread; ++e) {
            std::size_t d = lane + static_cast<std::size_t>(e) * kWarpSize;
            if (d < headDim) {
                dQ[(b * newLen + i) * dim + h * headDim + d] = dQAcc[e];
            }
        }
    }
}

void checkHeadDimSupported(std::size_t headDim, const char* callerName) {
    if (headDim > static_cast<std::size_t>(kMaxElemsPerThread) * kWarpSize) {
        throw std::invalid_argument(std::string(callerName) + ": headDim (" + std::to_string(headDim) +
                                     ") supera il limite supportato dal kernel fuso a tiling (" +
                                     std::to_string(kMaxElemsPerThread * kWarpSize) + ")");
    }
}

}  // namespace

FusedAttentionForwardResult fusedAttentionForward(const DeviceTensor& q, const DeviceTensor& k,
                                                    const DeviceTensor& v, std::size_t numHeads, float scaleFactor,
                                                    std::size_t oldLen) {
    std::size_t batch = q.dim(0);
    std::size_t newLen = q.dim(1);
    std::size_t dim = q.dim(2);
    std::size_t totalLen = k.dim(1);
    std::size_t headDim = dim / numHeads;
    checkHeadDimSupported(headDim, "fusedAttentionForward");

    DeviceTensor output({batch, newLen, dim});
    DeviceTensor mOut({batch, numHeads, newLen});
    DeviceTensor lOut({batch, numHeads, newLen});

    std::size_t numQueryTiles = (newLen + kBr - 1) / kBr;
    std::size_t numBlocks = batch * numHeads * numQueryTiles;
    if (numBlocks > 0) {
        dim3 blockDim(static_cast<unsigned int>(kWarpSize), static_cast<unsigned int>(kBr));
        auto sharedBytes = static_cast<std::size_t>(2 * kBc) * headDim * sizeof(float);
        tiledFusedAttentionForwardKernel<<<static_cast<unsigned int>(numBlocks), blockDim, sharedBytes>>>(
            output.data(), mOut.data(), lOut.data(), q.data(), k.data(), v.data(), numHeads, newLen, totalLen, dim,
            headDim, scaleFactor, oldLen, numQueryTiles);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }

    return FusedAttentionForwardResult{std::move(output), std::move(mOut), std::move(lOut)};
}

FusedAttentionGrad fusedAttentionBackward(const DeviceTensor& dOutput, const DeviceTensor& q, const DeviceTensor& k,
                                           const DeviceTensor& v, const DeviceTensor& output, const DeviceTensor& m,
                                           const DeviceTensor& l, std::size_t numHeads, float scaleFactor,
                                           std::size_t oldLen) {
    std::size_t batch = q.dim(0);
    std::size_t newLen = q.dim(1);
    std::size_t dim = q.dim(2);
    std::size_t totalLen = k.dim(1);
    std::size_t headDim = dim / numHeads;
    checkHeadDimSupported(headDim, "fusedAttentionBackward");

    std::size_t numBlocksPerRow = batch * numHeads * newLen;

    DeviceTensor dTerm({batch, numHeads, newLen});
    if (numBlocksPerRow > 0) {
        computeDKernel<<<static_cast<int>(numBlocksPerRow), kWarpSize>>>(dTerm.data(), output.data(),
                                                                          dOutput.data(), numHeads, newLen, dim,
                                                                          headDim);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }

    DeviceTensor dQ({batch, newLen, dim});
    DeviceTensor dK = DeviceTensor::zeros({batch, totalLen, dim});
    DeviceTensor dV = DeviceTensor::zeros({batch, totalLen, dim});

    std::size_t numQueryTiles = (newLen + kBr - 1) / kBr;
    std::size_t numBlocks = batch * numHeads * numQueryTiles;
    if (numBlocks > 0) {
        dim3 blockDim(static_cast<unsigned int>(kWarpSize), static_cast<unsigned int>(kBr));
        auto sharedBytes = static_cast<std::size_t>(2 * kBc) * headDim * sizeof(float);
        tiledFusedAttentionBackwardKernel<<<static_cast<unsigned int>(numBlocks), blockDim, sharedBytes>>>(
            dQ.data(), dK.data(), dV.data(), dOutput.data(), q.data(), k.data(), v.data(), m.data(), l.data(),
            dTerm.data(), numHeads, newLen, totalLen, dim, headDim, scaleFactor, oldLen, numQueryTiles);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }

    return FusedAttentionGrad{std::move(dQ), std::move(dK), std::move(dV)};
}

}  // namespace blackforge::backend::cuda
