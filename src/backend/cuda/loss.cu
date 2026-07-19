#include "blackforge/backend/cuda/loss.hpp"

#include <cmath>
#include <stdexcept>

#include "blackforge/backend/cuda/cuda_check.hpp"

namespace blackforge::backend::cuda {

namespace {

constexpr int kBlockSize = 256;

int gridSizeFor(std::size_t n) { return static_cast<int>((n + kBlockSize - 1) / kBlockSize); }

__global__ void mseGradKernel(float* grad, float* squaredDiff, const float* prediction, const float* target,
                               std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff = prediction[i] - target[i];
        squaredDiff[i] = diff * diff;
        grad[i] = 2.0F * diff / static_cast<float>(n);
    }
}

// Riduzione a singolo blocco: ogni thread somma con un grid-stride loop
// la propria porzione di 'data' (di lunghezza n, che puo' superare
// blockDim.x), poi una riduzione ad albero in shared memory produce la
// somma totale in out[0]. Un solo blocco e' corretto e sufficiente per
// le dimensioni di batch/tensori di questo linguaggio (non e' pensato
// per dataset da GB); correttezza prima delle prestazioni.
__global__ void sumReduceKernel(float* out, const float* data, std::size_t n) {
    extern __shared__ float shared[];

    float localSum = 0.0F;
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
        localSum += data[i];
    }
    shared[threadIdx.x] = localSum;
    __syncthreads();

    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[0] = shared[0];
    }
}

// Un blocco per "riga" (rango >= 2 appiattito a [righe, classi], stessa
// idea di rmsnorm/softmax): tre fasi con riduzione in shared memory
// (massimo per riga per la stabilita' numerica, somma degli
// esponenziali, poi somma della loss di riga), grid-stride loop dentro
// ogni fase per gestire numClasses > blockDim.x.
__global__ void softmaxCrossEntropyKernel(float* grad, float* rowLoss, const float* logits, const float* target,
                                           std::size_t rows, std::size_t numClasses) {
    extern __shared__ float shared[];

    std::size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float* rowLogits = logits + row * numClasses;
    const float* rowTarget = target + row * numClasses;
    float* rowGrad = grad + row * numClasses;

    float localMax = -INFINITY;
    for (std::size_t c = threadIdx.x; c < numClasses; c += blockDim.x) {
        localMax = fmaxf(localMax, rowLogits[c]);
    }
    shared[threadIdx.x] = localMax;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    __shared__ float rowMax;
    if (threadIdx.x == 0) {
        rowMax = shared[0];
    }
    __syncthreads();

    float localSumExp = 0.0F;
    for (std::size_t c = threadIdx.x; c < numClasses; c += blockDim.x) {
        localSumExp += expf(rowLogits[c] - rowMax);
    }
    shared[threadIdx.x] = localSumExp;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    __shared__ float sumExp;
    if (threadIdx.x == 0) {
        sumExp = shared[0];
    }
    __syncthreads();

    float localLoss = 0.0F;
    for (std::size_t c = threadIdx.x; c < numClasses; c += blockDim.x) {
        float prob = expf(rowLogits[c] - rowMax) / sumExp;
        float t = rowTarget[c];
        if (t != 0.0F) {
            // 1e-12 evita log(0), stessa soglia della controparte CPU.
            localLoss += -t * logf(fmaxf(prob, 1e-12F));
        }
        rowGrad[c] = (prob - t) / static_cast<float>(rows);
    }
    shared[threadIdx.x] = localLoss;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        rowLoss[row] = shared[0];
    }
}

// Come softmaxCrossEntropyKernel, ma legge un solo indice di classe per
// riga (gia' validato sull'host, vedi softmaxCrossEntropySparse) invece
// di un target denso [righe, classi]: evita di materializzare mai un
// target di dimensione proporzionale al vocabolario sul device.
__global__ void softmaxCrossEntropySparseKernel(float* grad, float* rowLoss, const float* logits,
                                                  const float* targetIndices, std::size_t rows,
                                                  std::size_t numClasses) {
    extern __shared__ float shared[];

    std::size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float* rowLogits = logits + row * numClasses;
    float* rowGrad = grad + row * numClasses;
    auto targetClass = static_cast<std::size_t>(lroundf(targetIndices[row]));

    float localMax = -INFINITY;
    for (std::size_t c = threadIdx.x; c < numClasses; c += blockDim.x) {
        localMax = fmaxf(localMax, rowLogits[c]);
    }
    shared[threadIdx.x] = localMax;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    __shared__ float rowMax;
    if (threadIdx.x == 0) {
        rowMax = shared[0];
    }
    __syncthreads();

    float localSumExp = 0.0F;
    for (std::size_t c = threadIdx.x; c < numClasses; c += blockDim.x) {
        localSumExp += expf(rowLogits[c] - rowMax);
    }
    shared[threadIdx.x] = localSumExp;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    __shared__ float sumExp;
    if (threadIdx.x == 0) {
        sumExp = shared[0];
    }
    __syncthreads();

    for (std::size_t c = threadIdx.x; c < numClasses; c += blockDim.x) {
        float prob = expf(rowLogits[c] - rowMax) / sumExp;
        float t = (c == targetClass) ? 1.0F : 0.0F;
        rowGrad[c] = (prob - t) / static_cast<float>(rows);
    }
    if (threadIdx.x == 0) {
        float targetProb = expf(rowLogits[targetClass] - rowMax) / sumExp;
        rowLoss[row] = -logf(fmaxf(targetProb, 1e-12F));
    }
}

}  // namespace

LossResult meanSquaredError(const DeviceTensor& prediction, const DeviceTensor& target) {
    if (prediction.shape() != target.shape()) {
        throw std::invalid_argument("meanSquaredError: forme incompatibili sul device");
    }

    std::size_t n = prediction.elementCount();
    DeviceTensor grad(prediction.shape());
    DeviceTensor squaredDiff({n});

    float value = 0.0F;
    if (n > 0) {
        mseGradKernel<<<gridSizeFor(n), kBlockSize>>>(grad.data(), squaredDiff.data(), prediction.data(),
                                                       target.data(), n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());

        DeviceTensor sumResult({1});
        sumReduceKernel<<<1, kBlockSize, kBlockSize * sizeof(float)>>>(sumResult.data(), squaredDiff.data(), n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());

        float sumSquares = 0.0F;
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(&sumSquares, sumResult.data(), sizeof(float), cudaMemcpyDeviceToHost));
        value = sumSquares / static_cast<float>(n);
    }

    return LossResult{value, std::move(grad)};
}

LossResult softmaxCrossEntropy(const DeviceTensor& logits, const DeviceTensor& target) {
    if (logits.shape() != target.shape()) {
        throw std::invalid_argument("softmaxCrossEntropy: forme incompatibili sul device");
    }
    if (logits.rank() < 2) {
        throw std::invalid_argument("softmaxCrossEntropy: richiede un tensore a rango >= 2 sul device");
    }

    std::size_t numClasses = logits.shape().back();
    std::size_t rows = logits.elementCount() / numClasses;

    DeviceTensor grad(logits.shape());
    float value = 0.0F;
    if (rows > 0 && numClasses > 0) {
        DeviceTensor rowLoss({rows});
        std::size_t sharedBytes = kBlockSize * sizeof(float);
        softmaxCrossEntropyKernel<<<static_cast<int>(rows), kBlockSize, sharedBytes>>>(
            grad.data(), rowLoss.data(), logits.data(), target.data(), rows, numClasses);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());

        DeviceTensor sumResult({1});
        sumReduceKernel<<<1, kBlockSize, kBlockSize * sizeof(float)>>>(sumResult.data(), rowLoss.data(), rows);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());

        float totalLoss = 0.0F;
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(&totalLoss, sumResult.data(), sizeof(float), cudaMemcpyDeviceToHost));
        value = totalLoss / static_cast<float>(rows);
    }

    return LossResult{value, std::move(grad)};
}

LossResult softmaxCrossEntropySparse(const DeviceTensor& logits, const DeviceTensor& targetIndices) {
    if (logits.rank() < 2) {
        throw std::invalid_argument("softmaxCrossEntropySparse: richiede logits a rango >= 2 sul device");
    }
    std::size_t numClasses = logits.shape().back();
    std::size_t rows = logits.elementCount() / numClasses;

    if (targetIndices.elementCount() != rows) {
        throw std::invalid_argument("softmaxCrossEntropySparse: targetIndices ha un numero di elementi "
                                     "incompatibile con le righe di logits sul device");
    }

    // Un kernel CUDA non puo' lanciare eccezioni: la validazione del
    // range degli indici avviene sull'host, prima del lancio (stessa
    // scelta di embeddingLookup/embeddingLookupBackward).
    if (rows > 0) {
        runtime::Tensor hostIndices = targetIndices.toHost();
        for (std::size_t i = 0; i < hostIndices.elementCount(); ++i) {
            auto idx = static_cast<long long>(std::lround(hostIndices.at(i)));
            if (idx < 0 || static_cast<std::size_t>(idx) >= numClasses) {
                throw std::invalid_argument("softmaxCrossEntropySparse: indice di classe " + std::to_string(idx) +
                                             " fuori da [0, " + std::to_string(numClasses) + ")");
            }
        }
    }

    DeviceTensor grad(logits.shape());
    float value = 0.0F;
    if (rows > 0 && numClasses > 0) {
        DeviceTensor rowLoss({rows});
        std::size_t sharedBytes = kBlockSize * sizeof(float);
        softmaxCrossEntropySparseKernel<<<static_cast<int>(rows), kBlockSize, sharedBytes>>>(
            grad.data(), rowLoss.data(), logits.data(), targetIndices.data(), rows, numClasses);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());

        DeviceTensor sumResult({1});
        sumReduceKernel<<<1, kBlockSize, kBlockSize * sizeof(float)>>>(sumResult.data(), rowLoss.data(), rows);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());

        float totalLoss = 0.0F;
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(&totalLoss, sumResult.data(), sizeof(float), cudaMemcpyDeviceToHost));
        value = totalLoss / static_cast<float>(rows);
    }

    return LossResult{value, std::move(grad)};
}

}  // namespace blackforge::backend::cuda
