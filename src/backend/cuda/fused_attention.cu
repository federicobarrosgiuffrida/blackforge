#include "blackforge/backend/cuda/fused_attention.hpp"

#include "blackforge/backend/cuda/cuda_check.hpp"

namespace blackforge::backend::cuda {

namespace {

constexpr int kWarpSize = 32;

// Dimensione del blocco CUDA per una data headDim: sempre un multiplo
// di 32 (allineato ai warp), il piu' piccolo che copre headDim — cosi'
// blockReduceSum() sotto puo' contare su blockDim.x multiplo di
// kWarpSize incondizionatamente. I thread "di riempimento" (indice >=
// headDim) restano inattivi nelle letture/scritture in memoria globale
// ma partecipano comunque alla riduzione (contribuendo 0).
std::size_t blockSizeForHeadDim(std::size_t headDim) {
    std::size_t warps = (headDim + kWarpSize - 1) / kWarpSize;
    return warps * static_cast<std::size_t>(kWarpSize);
}

// Riduzione a somma dentro un singolo warp via shuffle: nessun
// __syncthreads, nessuna shared memory — i 5 passi (offset 16,8,4,2,1)
// scambiano direttamente i registri tra le corsie del warp. Il
// risultato e' corretto solo nella corsia 0 al termine.
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFFU, val, offset);
    }
    return val;
}

// Riduzione a somma sull'intero blocco, a due livelli: prima una
// riduzione rapida intra-warp via shuffle (nessuna barriera), poi UNA
// sola __syncthreads per la riduzione finale tra i warp (agisce su
// 'numWarps' valori in shared memory, non su blockDim.x) — invece
// della classica riduzione ad albero in shared memory che userebbe
// log2(blockDim.x) barriere per OGNI chiamata. Con centinaia di
// posizioni chiave visitate per blocco (una chiamata a
// blockReduceSum per posizione), il numero di barriere synced e'
// dominante: passare da log2(blockDim.x) a 1 e' il guadagno principale
// di questa versione rispetto alla riduzione ad albero iniziale.
// 'shared' deve avere almeno blockDim.x/32 elementi allocati dal
// chiamante (dynamic shared memory). Il risultato finale e' visibile a
// TUTTI i thread del blocco (broadcast via shared[0]), non solo alla
// corsia 0.
__device__ __forceinline__ float blockReduceSum(float val, float* shared) {
    int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    int warpId = static_cast<int>(threadIdx.x) / kWarpSize;
    auto numWarps = static_cast<int>((blockDim.x + kWarpSize - 1) / kWarpSize);

    val = warpReduceSum(val);
    if (lane == 0) {
        shared[warpId] = val;
    }
    __syncthreads();

    float total = 0.0F;
    if (threadIdx.x == 0) {
        for (int w = 0; w < numWarps; ++w) {
            total += shared[w];
        }
        shared[0] = total;
    }
    __syncthreads();
    return shared[0];
}

// Forward fuso a "online softmax": un blocco per (batch, testa, riga di
// query), un thread per dimensione della testa (headDim, con
// riempimento al multiplo di 32 successivo). Per ogni posizione chiave
// j <= oldLen + i (limite causale): ricalcola score[j] = scaleFactor *
// dot(q,k[j]) con blockReduceSum, poi applica l'aggiornamento a
// "softmax online" (massimo corrente m, somma corrente l, accumulatore
// pesato acc) — mai una matrice di score materializzata in memoria
// globale. mOut/lOut sono le statistiche per riga di query, salvate per
// il backward (vedi fusedAttentionBackwardKernel).
__global__ void fusedAttentionForwardKernel(float* out, float* mOut, float* lOut, const float* q, const float* k,
                                             const float* v, std::size_t numHeads, std::size_t newLen,
                                             std::size_t totalLen, std::size_t dim, std::size_t headDim,
                                             float scaleFactor, std::size_t oldLen) {
    extern __shared__ float warpPartials[];

    std::size_t blockId = blockIdx.x;
    std::size_t i = blockId % newLen;
    std::size_t bh = blockId / newLen;
    std::size_t h = bh % numHeads;
    std::size_t b = bh / numHeads;

    std::size_t d = threadIdx.x;
    bool active = d < headDim;

    float qVal = active ? q[(b * newLen + i) * dim + h * headDim + d] : 0.0F;

    float m = -INFINITY;
    float l = 0.0F;
    float acc = 0.0F;

    std::size_t maxKey = oldLen + i;  // inclusivo

    for (std::size_t kpos = 0; kpos <= maxKey; ++kpos) {
        float kVal = active ? k[(b * totalLen + kpos) * dim + h * headDim + d] : 0.0F;
        float score = blockReduceSum(qVal * kVal, warpPartials) * scaleFactor;

        float newM = fmaxf(m, score);
        float correction = expf(m - newM);  // exp(-inf - finito) = 0 alla prima iterazione, corretto
        float p = expf(score - newM);
        float vVal = active ? v[(b * totalLen + kpos) * dim + h * headDim + d] : 0.0F;
        acc = acc * correction + p * vVal;
        l = l * correction + p;
        m = newM;
    }

    if (active) {
        out[(b * newLen + i) * dim + h * headDim + d] = acc / l;
    }
    if (d == 0) {
        mOut[(b * numHeads + h) * newLen + i] = m;
        lOut[(b * numHeads + h) * newLen + i] = l;
    }
}

// D[b,h,i] = dot(output[i,:], dOutput[i,:]) (sulla sola fetta di
// 'dim' della testa h): il termine di correzione della softmax-backward
// senza dover mai materializzare le probabilita' complete — sfrutta
// l'identita' dot(output,dOutput) = sum_j p[j]*dot(v[j,:],dOutput[i,:])
// = sum_j p[j]*dP[j] (la stessa quantita' che la formula standard di
// softmax-backward userebbe come "somma pesata dei gradienti").
__global__ void computeDKernel(float* dOut, const float* output, const float* dOutput, std::size_t numHeads,
                                std::size_t newLen, std::size_t dim, std::size_t headDim) {
    extern __shared__ float warpPartials[];

    std::size_t blockId = blockIdx.x;
    std::size_t i = blockId % newLen;
    std::size_t bh = blockId / newLen;
    std::size_t h = bh % numHeads;
    std::size_t b = bh / numHeads;

    std::size_t d = threadIdx.x;
    bool active = d < headDim;

    float oVal = active ? output[(b * newLen + i) * dim + h * headDim + d] : 0.0F;
    float dVal = active ? dOutput[(b * newLen + i) * dim + h * headDim + d] : 0.0F;
    float total = blockReduceSum(oVal * dVal, warpPartials);
    if (d == 0) {
        dOut[(b * numHeads + h) * newLen + i] = total;
    }
}

// Backward fuso: stessa struttura a blocco-per-(batch,testa,riga) del
// forward, ricalcola score[j] (stesso ciclo, stesso costo asintotico
// del forward) invece di averlo salvato, deriva p[j] dalle statistiche
// (m,l) salvate dal forward, e accumula dQ/dK/dV con la formula
// standard di backward per attention (vedi fused_attention.hpp per la
// derivazione completa). dQ[i,:] e' scritto una volta sola da questo
// blocco (nessuna race: nessun altro blocco scrive li'). dK/dV invece
// ricevono contributi da OGNI riga di query i che attende alla
// posizione chiave j — piu' blocchi diversi scrivono sulla stessa
// locazione, serve atomicAdd.
__global__ void fusedAttentionBackwardKernel(float* dQ, float* dK, float* dV, const float* dOutput, const float* q,
                                              const float* k, const float* v, const float* mSaved,
                                              const float* lSaved, const float* dTerm, std::size_t numHeads,
                                              std::size_t newLen, std::size_t totalLen, std::size_t dim,
                                              std::size_t headDim, float scaleFactor, std::size_t oldLen) {
    extern __shared__ float warpPartials[];

    std::size_t blockId = blockIdx.x;
    std::size_t i = blockId % newLen;
    std::size_t bh = blockId / newLen;
    std::size_t h = bh % numHeads;
    std::size_t b = bh / numHeads;

    std::size_t d = threadIdx.x;
    bool active = d < headDim;

    float qVal = active ? q[(b * newLen + i) * dim + h * headDim + d] : 0.0F;
    float dOutVal = active ? dOutput[(b * newLen + i) * dim + h * headDim + d] : 0.0F;

    float mRow = mSaved[(b * numHeads + h) * newLen + i];
    float lRow = lSaved[(b * numHeads + h) * newLen + i];
    float dRow = dTerm[(b * numHeads + h) * newLen + i];

    float dQAcc = 0.0F;
    std::size_t maxKey = oldLen + i;

    for (std::size_t kpos = 0; kpos <= maxKey; ++kpos) {
        float kVal = active ? k[(b * totalLen + kpos) * dim + h * headDim + d] : 0.0F;
        float score = blockReduceSum(qVal * kVal, warpPartials) * scaleFactor;
        float p = expf(score - mRow) / lRow;

        float vVal = active ? v[(b * totalLen + kpos) * dim + h * headDim + d] : 0.0F;
        float dP = blockReduceSum(dOutVal * vVal, warpPartials);

        float dScore = p * (dP - dRow);

        dQAcc += dScore * scaleFactor * kVal;

        if (active) {
            atomicAdd(&dK[(b * totalLen + kpos) * dim + h * headDim + d], dScore * scaleFactor * qVal);
            atomicAdd(&dV[(b * totalLen + kpos) * dim + h * headDim + d], p * dOutVal);
        }
    }

    if (active) {
        dQ[(b * newLen + i) * dim + h * headDim + d] = dQAcc;
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

    DeviceTensor output({batch, newLen, dim});
    DeviceTensor mOut({batch, numHeads, newLen});
    DeviceTensor lOut({batch, numHeads, newLen});

    std::size_t numBlocks = batch * numHeads * newLen;
    if (numBlocks > 0) {
        auto blockSize = static_cast<int>(blockSizeForHeadDim(headDim));
        auto numWarps = static_cast<std::size_t>((blockSize + kWarpSize - 1) / kWarpSize);
        auto sharedBytes = numWarps * sizeof(float);
        fusedAttentionForwardKernel<<<static_cast<int>(numBlocks), blockSize, sharedBytes>>>(
            output.data(), mOut.data(), lOut.data(), q.data(), k.data(), v.data(), numHeads, newLen, totalLen, dim,
            headDim, scaleFactor, oldLen);
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

    std::size_t numBlocks = batch * numHeads * newLen;
    auto blockSize = static_cast<int>(blockSizeForHeadDim(headDim));
    auto numWarps = static_cast<std::size_t>((blockSize + kWarpSize - 1) / kWarpSize);
    auto sharedBytes = numWarps * sizeof(float);

    DeviceTensor dTerm({batch, numHeads, newLen});
    if (numBlocks > 0) {
        computeDKernel<<<static_cast<int>(numBlocks), blockSize, sharedBytes>>>(
            dTerm.data(), output.data(), dOutput.data(), numHeads, newLen, dim, headDim);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }

    DeviceTensor dQ({batch, newLen, dim});
    DeviceTensor dK = DeviceTensor::zeros({batch, totalLen, dim});
    DeviceTensor dV = DeviceTensor::zeros({batch, totalLen, dim});

    if (numBlocks > 0) {
        fusedAttentionBackwardKernel<<<static_cast<int>(numBlocks), blockSize, sharedBytes>>>(
            dQ.data(), dK.data(), dV.data(), dOutput.data(), q.data(), k.data(), v.data(), m.data(), l.data(),
            dTerm.data(), numHeads, newLen, totalLen, dim, headDim, scaleFactor, oldLen);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }

    return FusedAttentionGrad{std::move(dQ), std::move(dK), std::move(dV)};
}

}  // namespace blackforge::backend::cuda
