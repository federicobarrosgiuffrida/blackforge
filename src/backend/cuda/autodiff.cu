#include "blackforge/backend/cuda/autodiff.hpp"

#include <cmath>
#include <stdexcept>

#include "blackforge/backend/cuda/attention_batched.hpp"
#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/backend/cuda/ops.hpp"

namespace blackforge::backend::cuda {

namespace {

constexpr int kBlockSize = 256;
constexpr float kGeluCoeff = 0.044715F;
constexpr float kGeluCoeffDeriv = 3.0F * kGeluCoeff;
constexpr float kSqrt2OverPi = 0.7978845608F;
constexpr float kRmsNormEps = 1e-6F;  // deve coincidere con ops_elementwise.cu e src/backend/cpu/ops.cpp

int gridSizeFor(std::size_t n) { return static_cast<int>((n + kBlockSize - 1) / kBlockSize); }

// dA[i,p] = sum_j gradOutput[i,j] * B[p,j]. Un thread per elemento di
// dA (m*k thread totali), loop seriale su n dentro ogni thread: stessa
// struttura del triplo loop CPU, solo con il loop esterno parallelizzato.
__global__ void matmulBackwardDAKernel(float* dA, const float* gradOutput, const float* b, std::size_t m,
                                        std::size_t k, std::size_t n) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= m * k) {
        return;
    }
    std::size_t i = idx / k;
    std::size_t p = idx % k;

    float sum = 0.0F;
    for (std::size_t j = 0; j < n; ++j) {
        sum += gradOutput[i * n + j] * b[p * n + j];
    }
    dA[idx] = sum;
}

// dB[p,j] = sum_i A[i,p] * gradOutput[i,j].
__global__ void matmulBackwardDBKernel(float* dB, const float* a, const float* gradOutput, std::size_t m,
                                        std::size_t k, std::size_t n) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= k * n) {
        return;
    }
    std::size_t p = idx / n;
    std::size_t j = idx % n;

    float sum = 0.0F;
    for (std::size_t i = 0; i < m; ++i) {
        sum += a[i * k + p] * gradOutput[i * n + j];
    }
    dB[idx] = sum;
}

// matmulTransposeB: C = A @ B^T, A:[m,k], B:[n,k], C:[m,n].
// dA[i,p] = sum_j gradOutput[i,j] * B[j,p] (= matmul(gradOutput, B)).
__global__ void matmulTransposeBBackwardDAKernel(float* dA, const float* gradOutput, const float* b, std::size_t m,
                                                  std::size_t k, std::size_t n) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= m * k) {
        return;
    }
    std::size_t i = idx / k;
    std::size_t p = idx % k;

    float sum = 0.0F;
    for (std::size_t j = 0; j < n; ++j) {
        sum += gradOutput[i * n + j] * b[j * k + p];
    }
    dA[idx] = sum;
}

// dB[j,p] = sum_i gradOutput[i,j] * A[i,p].
__global__ void matmulTransposeBBackwardDBKernel(float* dB, const float* a, const float* gradOutput, std::size_t m,
                                                  std::size_t k, std::size_t n) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n * k) {
        return;
    }
    std::size_t j = idx / k;
    std::size_t p = idx % k;

    float sum = 0.0F;
    for (std::size_t i = 0; i < m; ++i) {
        sum += gradOutput[i * n + j] * a[i * k + p];
    }
    dB[idx] = sum;
}

__global__ void addBiasBackwardDBiasKernel(float* dBias, const float* gradOutput, std::size_t batch,
                                            std::size_t features) {
    std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= features) {
        return;
    }
    float sum = 0.0F;
    for (std::size_t row = 0; row < batch; ++row) {
        sum += gradOutput[row * features + col];
    }
    dBias[col] = sum;
}

__global__ void siluBackwardKernel(float* out, const float* input, const float* gradOutput, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        float sigmoid = 1.0F / (1.0F + expf(-x));
        out[i] = gradOutput[i] * sigmoid * (1.0F + x * (1.0F - sigmoid));
    }
}

__global__ void reluBackwardKernel(float* out, const float* input, const float* gradOutput, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = input[i] > 0.0F ? gradOutput[i] : 0.0F;
    }
}

__global__ void geluBackwardKernel(float* out, const float* input, const float* gradOutput, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        float inner = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
        float t = tanhf(inner);
        float innerDerivative = kSqrt2OverPi * (1.0F + kGeluCoeffDeriv * x * x);
        float derivative = 0.5F * (1.0F + t) + 0.5F * x * (1.0F - t * t) * innerDerivative;
        out[i] = gradOutput[i] * derivative;
    }
}

// Un blocco per riga: riduzione in shared memory per sumSquares (serve
// a ricalcolare r) e weightedSum (S = sum_j gradOutput_j * x_j, la
// derivata ha bisogno di entrambe). dL/dx_i = gOut_i/r - x_i*S/(d*r^3).
__global__ void rmsnormBackwardKernel(float* out, const float* input, const float* gradOutput, std::size_t batch,
                                       std::size_t features) {
    extern __shared__ float shared[];
    float* sumSquaresShared = shared;
    float* weightedSumShared = shared + blockDim.x;

    std::size_t row = blockIdx.x;
    if (row >= batch) {
        return;
    }
    const float* rowIn = input + row * features;
    const float* rowGrad = gradOutput + row * features;
    float* rowOut = out + row * features;

    float localSumSquares = 0.0F;
    float localWeightedSum = 0.0F;
    for (std::size_t col = threadIdx.x; col < features; col += blockDim.x) {
        float x = rowIn[col];
        localSumSquares += x * x;
        localWeightedSum += rowGrad[col] * x;
    }
    sumSquaresShared[threadIdx.x] = localSumSquares;
    weightedSumShared[threadIdx.x] = localWeightedSum;
    __syncthreads();

    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sumSquaresShared[threadIdx.x] += sumSquaresShared[threadIdx.x + stride];
            weightedSumShared[threadIdx.x] += weightedSumShared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    __shared__ float r;
    __shared__ float weightedSum;
    if (threadIdx.x == 0) {
        float meanSquare = sumSquaresShared[0] / static_cast<float>(features);
        r = sqrtf(meanSquare + kRmsNormEps);
        weightedSum = weightedSumShared[0];
    }
    __syncthreads();

    for (std::size_t col = threadIdx.x; col < features; col += blockDim.x) {
        float x = rowIn[col];
        float gOut = rowGrad[col];
        rowOut[col] = gOut / r - x * weightedSum / (static_cast<float>(features) * r * r * r);
    }
}

// Un blocco per riga: dx_j = y_j * (gOut_j - S), S = sum_i(gOut_i*y_i).
// 'y' e' l'uscita di softmax(input), gia' calcolata da un kernel
// separato (softmaxKernel in ops_elementwise.cu) prima di questo.
__global__ void softmaxBackwardKernel(float* out, const float* y, const float* gradOutput, std::size_t batch,
                                       std::size_t features) {
    extern __shared__ float shared[];

    std::size_t row = blockIdx.x;
    if (row >= batch) {
        return;
    }
    const float* rowY = y + row * features;
    const float* rowGrad = gradOutput + row * features;
    float* rowOut = out + row * features;

    float localWeightedSum = 0.0F;
    for (std::size_t col = threadIdx.x; col < features; col += blockDim.x) {
        localWeightedSum += rowGrad[col] * rowY[col];
    }
    shared[threadIdx.x] = localWeightedSum;
    __syncthreads();

    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    __shared__ float weightedSum;
    if (threadIdx.x == 0) {
        weightedSum = shared[0];
    }
    __syncthreads();

    for (std::size_t col = threadIdx.x; col < features; col += blockDim.x) {
        rowOut[col] = rowY[col] * (rowGrad[col] - weightedSum);
    }
}

// Appiattisce un tensore device a rango >= 2 a [rows, features] (stesso
// layout flat riga-maggiore): serve per poter chiamare matmulBackward()
// (primitivo puramente 2D) su input a rango > 2 come [batch, seq,
// features]. Sempre una clone(): il tensore originale a rango > 2
// resta valido e riutilizzabile dal chiamante dopo questa chiamata
// (stessa idea di flatten2D in src/backend/cpu/autodiff.cpp).
DeviceTensor flatten2D(const DeviceTensor& t) {
    std::size_t features = t.shape().back();
    std::size_t rows = t.elementCount() / features;
    return t.clone().reshaped({rows, features});
}

// Scatter-add: dTable[tokenId,:] += gradOutput[token,:]. Piu' token
// nella stessa chiamata possono avere lo stesso id (thread diversi che
// scrivono sulla stessa riga della tabella concorrentemente): serve
// atomicAdd, a differenza di scatterHeadKernel (ops_transformer.cu) le
// cui teste non si sovrappongono mai.
__global__ void embeddingLookupBackwardKernel(float* dTable, const float* tokenIds, const float* gradOutput,
                                               std::size_t totalTokens, std::size_t dim) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = totalTokens * dim;
    if (idx >= total) {
        return;
    }
    std::size_t token = idx / dim;
    std::size_t d = idx % dim;

    long long tokenId = static_cast<long long>(lroundf(tokenIds[token]));
    // Il range e' gia' stato validato sull'host in
    // embeddingLookupBackward() prima di lanciare questo kernel.
    atomicAdd(&dTable[static_cast<std::size_t>(tokenId) * dim + d], gradOutput[idx]);
}

// dTable[s,d] = sum_b gradOutput[b,s,d]: un thread per (s,d), somma
// seriale sul batch (il batch e' tipicamente piccolo in questo
// linguaggio, non serve una riduzione in shared memory).
__global__ void positionalEmbeddingBackwardKernel(float* dTable, const float* gradOutput, std::size_t batch,
                                                    std::size_t seq, std::size_t dim) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = seq * dim;
    if (idx >= total) {
        return;
    }
    float sum = 0.0F;
    for (std::size_t b = 0; b < batch; ++b) {
        sum += gradOutput[b * total + idx];
    }
    dTable[idx] = sum;
}

}  // namespace

MatmulGrad matmulBackward(const DeviceTensor& a, const DeviceTensor& b, const DeviceTensor& gradOutput) {
    if (a.rank() != 2 || b.rank() != 2 || gradOutput.rank() != 2 || a.dim(1) != b.dim(0) ||
        gradOutput.dim(0) != a.dim(0) || gradOutput.dim(1) != b.dim(1)) {
        throw std::invalid_argument("matmulBackward: forme incompatibili sul device");
    }

    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(1);

    DeviceTensor dA({m, k});
    DeviceTensor dB({k, n});

    std::size_t dACount = m * k;
    if (dACount > 0) {
        matmulBackwardDAKernel<<<gridSizeFor(dACount), kBlockSize>>>(dA.data(), gradOutput.data(), b.data(), m, k,
                                                                      n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }

    std::size_t dBCount = k * n;
    if (dBCount > 0) {
        matmulBackwardDBKernel<<<gridSizeFor(dBCount), kBlockSize>>>(dB.data(), a.data(), gradOutput.data(), m, k,
                                                                      n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }

    return MatmulGrad{std::move(dA), std::move(dB)};
}

MatmulTransposeBGrad matmulTransposeBBackward(const DeviceTensor& a, const DeviceTensor& b,
                                               const DeviceTensor& gradOutput) {
    if (a.rank() != 2 || b.rank() != 2 || gradOutput.rank() != 2 || a.dim(1) != b.dim(1) ||
        gradOutput.dim(0) != a.dim(0) || gradOutput.dim(1) != b.dim(0)) {
        throw std::invalid_argument("matmulTransposeBBackward: forme incompatibili sul device");
    }

    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(0);

    DeviceTensor dA({m, k});
    DeviceTensor dB({n, k});

    std::size_t dACount = m * k;
    if (dACount > 0) {
        matmulTransposeBBackwardDAKernel<<<gridSizeFor(dACount), kBlockSize>>>(dA.data(), gradOutput.data(),
                                                                                b.data(), m, k, n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }

    std::size_t dBCount = n * k;
    if (dBCount > 0) {
        matmulTransposeBBackwardDBKernel<<<gridSizeFor(dBCount), kBlockSize>>>(dB.data(), a.data(),
                                                                                gradOutput.data(), m, k, n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }

    return MatmulTransposeBGrad{std::move(dA), std::move(dB)};
}

AddBiasGrad addBiasBackward(const DeviceTensor& gradOutput) {
    // Generalizzato a rango >= 2 (vedi il commento equivalente in
    // src/backend/cpu/autodiff.cpp).
    if (gradOutput.rank() < 2) {
        throw std::invalid_argument("addBiasBackward: atteso un gradiente a rango >= 2 sul device");
    }

    std::size_t features = gradOutput.shape().back();
    std::size_t rows = gradOutput.elementCount() / features;

    // dInput e' un'identita' nel gradiente: si copia semplicemente il
    // buffer (device-to-device), non serve un kernel elementwise.
    DeviceTensor dInput(gradOutput.shape());
    if (gradOutput.elementCount() > 0) {
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(dInput.data(), gradOutput.data(), gradOutput.elementCount() * sizeof(float),
                                         cudaMemcpyDeviceToDevice));
    }

    DeviceTensor dBias({features});
    if (features > 0) {
        addBiasBackwardDBiasKernel<<<gridSizeFor(features), kBlockSize>>>(dBias.data(), gradOutput.data(), rows,
                                                                           features);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }

    return AddBiasGrad{std::move(dInput), std::move(dBias)};
}

DeviceTensor siluBackward(const DeviceTensor& input, const DeviceTensor& gradOutput) {
    if (input.shape() != gradOutput.shape()) {
        throw std::invalid_argument("siluBackward: forme incompatibili sul device");
    }
    DeviceTensor result(input.shape());
    std::size_t n = input.elementCount();
    if (n > 0) {
        siluBackwardKernel<<<gridSizeFor(n), kBlockSize>>>(result.data(), input.data(), gradOutput.data(), n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor reluBackward(const DeviceTensor& input, const DeviceTensor& gradOutput) {
    if (input.shape() != gradOutput.shape()) {
        throw std::invalid_argument("reluBackward: forme incompatibili sul device");
    }
    DeviceTensor result(input.shape());
    std::size_t n = input.elementCount();
    if (n > 0) {
        reluBackwardKernel<<<gridSizeFor(n), kBlockSize>>>(result.data(), input.data(), gradOutput.data(), n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor geluBackward(const DeviceTensor& input, const DeviceTensor& gradOutput) {
    if (input.shape() != gradOutput.shape()) {
        throw std::invalid_argument("geluBackward: forme incompatibili sul device");
    }
    DeviceTensor result(input.shape());
    std::size_t n = input.elementCount();
    if (n > 0) {
        geluBackwardKernel<<<gridSizeFor(n), kBlockSize>>>(result.data(), input.data(), gradOutput.data(), n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor rmsnormBackward(const DeviceTensor& input, const DeviceTensor& gradOutput) {
    if (input.shape() != gradOutput.shape()) {
        throw std::invalid_argument("rmsnormBackward: forme incompatibili sul device");
    }
    if (input.rank() < 2) {
        throw std::invalid_argument("rmsnormBackward: richiede un tensore a rango >= 2 sul device");
    }

    std::size_t features = input.shape().back();
    std::size_t rows = input.elementCount() / features;
    DeviceTensor result(input.shape());
    if (rows > 0 && features > 0) {
        std::size_t sharedBytes = 2 * kBlockSize * sizeof(float);
        rmsnormBackwardKernel<<<static_cast<int>(rows), kBlockSize, sharedBytes>>>(
            result.data(), input.data(), gradOutput.data(), rows, features);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor softmaxBackward(const DeviceTensor& input, const DeviceTensor& gradOutput) {
    if (input.shape() != gradOutput.shape()) {
        throw std::invalid_argument("softmaxBackward: forme incompatibili sul device");
    }
    if (input.rank() < 2) {
        throw std::invalid_argument("softmaxBackward: richiede un tensore a rango >= 2 sul device");
    }

    DeviceTensor y = softmax(input);
    std::size_t features = input.shape().back();
    std::size_t rows = input.elementCount() / features;
    DeviceTensor result(input.shape());
    if (rows > 0 && features > 0) {
        std::size_t sharedBytes = kBlockSize * sizeof(float);
        softmaxBackwardKernel<<<static_cast<int>(rows), kBlockSize, sharedBytes>>>(
            result.data(), y.data(), gradOutput.data(), rows, features);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor embeddingLookupBackward(const DeviceTensor& tokenIds, const DeviceTensor& gradOutput,
                                      std::size_t vocabSize) {
    if (tokenIds.rank() != 2 || gradOutput.rank() != 3 || tokenIds.dim(0) != gradOutput.dim(0) ||
        tokenIds.dim(1) != gradOutput.dim(1)) {
        throw std::invalid_argument("embeddingLookupBackward: forme incompatibili sul device");
    }

    std::size_t dim = gradOutput.dim(2);

    // Stessa validazione del range fatta sull'host in embeddingLookup()
    // (forward): un kernel CUDA non puo' lanciare eccezioni.
    runtime::Tensor tokenIdsHost = tokenIds.toHost();
    for (std::size_t i = 0; i < tokenIdsHost.elementCount(); ++i) {
        auto tokenId = static_cast<long long>(std::lround(tokenIdsHost.at(i)));
        if (tokenId < 0 || static_cast<std::size_t>(tokenId) >= vocabSize) {
            throw std::invalid_argument("embeddingLookupBackward: token id " + std::to_string(tokenId) +
                                         " fuori dal vocabolario [0, " + std::to_string(vocabSize) + ")");
        }
    }

    DeviceTensor dTable = DeviceTensor::zeros({vocabSize, dim});
    std::size_t totalTokens = tokenIds.elementCount();
    std::size_t total = totalTokens * dim;
    if (total > 0) {
        embeddingLookupBackwardKernel<<<gridSizeFor(total), kBlockSize>>>(dTable.data(), tokenIds.data(),
                                                                           gradOutput.data(), totalTokens, dim);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return dTable;
}

PositionalEmbeddingGrad addPositionalEmbeddingBackward(const DeviceTensor& gradOutput, std::size_t maxSeqLen) {
    if (gradOutput.rank() != 3) {
        throw std::invalid_argument("addPositionalEmbeddingBackward: atteso un gradiente a rango 3 sul device");
    }

    std::size_t batch = gradOutput.dim(0);
    std::size_t seq = gradOutput.dim(1);
    std::size_t dim = gradOutput.dim(2);
    if (seq > maxSeqLen) {
        throw std::invalid_argument("addPositionalEmbeddingBackward: la sequenza supera maxSeqLen");
    }

    // dInput e' un'identita' nel gradiente: copia device-to-device.
    DeviceTensor dInput(gradOutput.shape());
    if (gradOutput.elementCount() > 0) {
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(dInput.data(), gradOutput.data(), gradOutput.elementCount() * sizeof(float),
                                         cudaMemcpyDeviceToDevice));
    }

    // Le righe di dTable oltre 'seq' (se maxSeqLen > seq) non sono
    // state usate in questo forward: restano a zero, come da
    // DeviceTensor::zeros.
    DeviceTensor dTable = DeviceTensor::zeros({maxSeqLen, dim});
    std::size_t total = seq * dim;
    if (total > 0) {
        positionalEmbeddingBackwardKernel<<<gridSizeFor(total), kBlockSize>>>(dTable.data(), gradOutput.data(),
                                                                               batch, seq, dim);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }

    return PositionalEmbeddingGrad{std::move(dInput), std::move(dTable)};
}

FeedForwardGrad feedForwardBackward(const DeviceTensor& input, const DeviceTensor& w1, const DeviceTensor& b1,
                                     const DeviceTensor& w2, const DeviceTensor& b2, const DeviceTensor& gradOutput) {
    // b2 non serve al calcolo (vedi il commento equivalente in
    // src/backend/cpu/autodiff.cpp): resta nella firma solo per
    // simmetria con feedForward().
    (void)b2;

    // Ricalcola gli stessi passaggi del forward (vedi
    // ops_transformer.cu/feedForward).
    DeviceTensor normed = rmsnorm(input);
    DeviceTensor preActivation = linear(normed, w1, b1);
    DeviceTensor hidden = silu(preActivation);

    AddBiasGrad addGrad2 = addBiasBackward(gradOutput);
    MatmulGrad matGrad2 = matmulBackward(flatten2D(hidden), w2, flatten2D(addGrad2.dInput));
    DeviceTensor dHidden = std::move(matGrad2.dA).reshaped(hidden.shape());

    DeviceTensor dPreActivation = siluBackward(preActivation, dHidden);

    AddBiasGrad addGrad1 = addBiasBackward(dPreActivation);
    MatmulGrad matGrad1 = matmulBackward(flatten2D(normed), w1, flatten2D(addGrad1.dInput));
    DeviceTensor dNormed = std::move(matGrad1.dA).reshaped(normed.shape());

    // y = input + linear2(...): il residual manda gradOutput sia
    // direttamente a dInput sia, tramite il ramo rmsnorm/linear1/
    // silu/linear2, di nuovo indietro fino a dInput (i due contributi
    // si sommano).
    DeviceTensor dInputFromBranch = rmsnormBackward(input, dNormed);
    DeviceTensor dInput = add(gradOutput, dInputFromBranch);

    return FeedForwardGrad{std::move(dInput), std::move(matGrad1.dB), std::move(addGrad1.dBias),
                            std::move(matGrad2.dB), std::move(addGrad2.dBias)};
}

SelfAttentionGrad selfAttentionBackward(const DeviceTensor& input, const DeviceTensor& wq, const DeviceTensor& wk,
                                         const DeviceTensor& wv, const DeviceTensor& wout, std::size_t numHeads,
                                         const DeviceTensor& gradOutput) {
    std::size_t batch = input.dim(0);
    std::size_t seq = input.dim(1);
    std::size_t dim = input.dim(2);
    std::size_t headDim = dim / numHeads;
    float scaleFactor = 1.0F / std::sqrt(static_cast<float>(headDim));

    // Ricalcola lo stesso forward BATCHIZZATO di ops_transformer.cu/
    // selfAttention (vedi attention_batched.hpp per il perche': un
    // singolo kernel per Q@K^T/maschera/probs@V su TUTTE le
    // combinazioni (batch,testa), non un loop host-side con un lancio
    // di kernel per combinazione). normedFlat e' riusato (const&, ne'
    // matmul() ne' matmulBackward() lo consumano) sia qui sia nel
    // backward delle proiezioni Q/K/V piu' sotto.
    DeviceTensor normed = rmsnorm(input);
    DeviceTensor normedFlat = flatten2D(normed);
    DeviceTensor q = matmul(normedFlat, wq).reshaped({batch, seq, dim});
    DeviceTensor k = matmul(normedFlat, wk).reshaped({batch, seq, dim});
    DeviceTensor v = matmul(normedFlat, wv).reshaped({batch, seq, dim});

    DeviceTensor scores = batchedQK(q, k, numHeads, scaleFactor);
    DeviceTensor maskedScores = batchedMask(scores, /*oldLen=*/0);
    DeviceTensor probs = softmax(maskedScores);
    DeviceTensor concatenated = batchedPV(probs, v, numHeads);

    // --- backward attraverso la proiezione Wout ---
    MatmulGrad woutGrad = matmulBackward(flatten2D(concatenated), wout, flatten2D(gradOutput));
    DeviceTensor dConcatenated = std::move(woutGrad.dA).reshaped({batch, seq, dim});

    // --- backward attraverso probs@V (batchizzato) ---
    BatchedPVGrad pvGrad = batchedPVBackward(dConcatenated, probs, v, numHeads);
    // pvGrad.dProbs, pvGrad.dV

    // --- backward attraverso softmax ---
    DeviceTensor dMaskedScores = softmaxBackward(maskedScores, pvGrad.dProbs);
    // La maschera causale non richiede un passaggio di backward
    // dedicato: le posizioni mascherate hanno probs == 0, quindi la
    // formula di softmaxBackward da' loro gradiente esattamente zero
    // (coerente con il non aver partecipato al forward).

    // --- backward attraverso Q@K^T (batchizzato): il fattore di scala
    // e' gia' applicato dentro batchedQKBackward (coerente con
    // batchedQK, che lo applica nello stesso kernel del prodotto,
    // invece di un passaggio scale() separato) ---
    BatchedQKGrad qkGrad = batchedQKBackward(dMaskedScores, q, k, numHeads, scaleFactor);

    DeviceTensor dQ = std::move(qkGrad.dQ);
    DeviceTensor dK = std::move(qkGrad.dK);
    DeviceTensor dV = std::move(pvGrad.dV);

    // --- backward attraverso le proiezioni Q/K/V ---
    MatmulGrad qBackward = matmulBackward(normedFlat, wq, flatten2D(dQ));
    MatmulGrad kBackward = matmulBackward(normedFlat, wk, flatten2D(dK));
    MatmulGrad vBackward = matmulBackward(normedFlat, wv, flatten2D(dV));

    DeviceTensor dNormedFlat = add(add(qBackward.dA, kBackward.dA), vBackward.dA);
    DeviceTensor dNormed = std::move(dNormedFlat).reshaped({batch, seq, dim});

    // --- backward attraverso rmsnorm, poi residual ---
    DeviceTensor dInputFromBranch = rmsnormBackward(input, dNormed);
    DeviceTensor dInput = add(gradOutput, dInputFromBranch);

    return SelfAttentionGrad{std::move(dInput), std::move(qBackward.dB), std::move(kBackward.dB),
                              std::move(vBackward.dB), std::move(woutGrad.dB)};
}

}  // namespace blackforge::backend::cuda
