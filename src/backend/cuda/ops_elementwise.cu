#include <cmath>
#include <stdexcept>
#include <string>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/backend/cuda/ops.hpp"

namespace blackforge::backend::cuda {

namespace {

constexpr int kBlockSize = 256;

int gridSizeFor(std::size_t n) { return static_cast<int>((n + kBlockSize - 1) / kBlockSize); }

__global__ void addKernel(float* out, const float* a, const float* b, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

__global__ void addBiasKernel(float* out, const float* input, const float* bias, std::size_t batch,
                               std::size_t features) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = batch * features;
    if (i < total) {
        std::size_t col = i % features;
        out[i] = input[i] + bias[col];
    }
}

// Le formule seguenti sono identiche, elemento per elemento, a quelle
// del backend CPU di riferimento (src/backend/cpu/ops.cpp): serve a
// poter confrontare direttamente i risultati CPU/GPU nei test.

__global__ void siluKernel(float* out, const float* input, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        out[i] = x / (1.0F + expf(-x));
    }
}

__global__ void reluKernel(float* out, const float* input, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        out[i] = x > 0.0F ? x : 0.0F;
    }
}

__global__ void geluKernel(float* out, const float* input, std::size_t n) {
    constexpr float kGeluCoeff = 0.044715F;
    constexpr float kSqrt2OverPi = 0.7978845608F;

    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        float inner = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
        out[i] = 0.5F * x * (1.0F + tanhf(inner));
    }
}

constexpr float kRmsNormEps = 1e-6F;  // deve coincidere con src/backend/cpu/ops.cpp

// Un blocco per riga del batch: ogni thread somma i quadrati di una
// porzione della riga, poi una riduzione in shared memory calcola la
// somma totale (e quindi la RMS) prima che tutti i thread normalizzino
// la propria porzione.
__global__ void rmsnormKernel(float* out, const float* input, std::size_t batch, std::size_t features) {
    extern __shared__ float partialSums[];

    std::size_t row = blockIdx.x;
    if (row >= batch) {
        return;
    }
    const float* rowIn = input + row * features;
    float* rowOut = out + row * features;

    float localSum = 0.0F;
    for (std::size_t col = threadIdx.x; col < features; col += blockDim.x) {
        float x = rowIn[col];
        localSum += x * x;
    }
    partialSums[threadIdx.x] = localSum;
    __syncthreads();

    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    __shared__ float rms;
    if (threadIdx.x == 0) {
        float meanSquare = partialSums[0] / static_cast<float>(features);
        rms = sqrtf(meanSquare + kRmsNormEps);
    }
    __syncthreads();

    for (std::size_t col = threadIdx.x; col < features; col += blockDim.x) {
        rowOut[col] = rowIn[col] / rms;
    }
}

void requireSameShape(const DeviceTensor& a, const DeviceTensor& b, const char* opName) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(std::string(opName) + ": forme incompatibili sul device");
    }
}

}  // namespace

DeviceTensor add(const DeviceTensor& a, const DeviceTensor& b) {
    requireSameShape(a, b, "add");
    DeviceTensor result(a.shape());
    std::size_t n = a.elementCount();
    if (n > 0) {
        addKernel<<<gridSizeFor(n), kBlockSize>>>(result.data(), a.data(), b.data(), n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor addBias(const DeviceTensor& input, const DeviceTensor& bias) {
    // Generalizzato a rango >= 2 (vedi il commento equivalente in
    // src/backend/cpu/ops.cpp/addBias): 'features' e' sempre l'ultima
    // dimensione, tutte le altre sono trattate come righe indipendenti.
    if (input.rank() < 2 || bias.rank() != 1 || input.shape().back() != bias.dim(0)) {
        throw std::invalid_argument("addBias: attesi input [..., features] e bias [features] con features "
                                     "corrispondenti");
    }

    std::size_t features = input.shape().back();
    std::size_t rows = input.elementCount() / features;
    DeviceTensor result(input.shape());
    std::size_t n = input.elementCount();
    if (n > 0) {
        addBiasKernel<<<gridSizeFor(n), kBlockSize>>>(result.data(), input.data(), bias.data(), rows, features);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor silu(const DeviceTensor& input) {
    DeviceTensor result(input.shape());
    std::size_t n = input.elementCount();
    if (n > 0) {
        siluKernel<<<gridSizeFor(n), kBlockSize>>>(result.data(), input.data(), n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor relu(const DeviceTensor& input) {
    DeviceTensor result(input.shape());
    std::size_t n = input.elementCount();
    if (n > 0) {
        reluKernel<<<gridSizeFor(n), kBlockSize>>>(result.data(), input.data(), n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor gelu(const DeviceTensor& input) {
    DeviceTensor result(input.shape());
    std::size_t n = input.elementCount();
    if (n > 0) {
        geluKernel<<<gridSizeFor(n), kBlockSize>>>(result.data(), input.data(), n);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

// Un blocco per riga: prima riduzione per il massimo (stabilita'
// numerica), poi riduzione per la somma degli esponenziali, infine
// normalizzazione. Tre passaggi sincronizzati con __syncthreads() tra
// blocco e blocco, stessa struttura del kernel rmsnorm sopra.
__global__ void softmaxKernel(float* out, const float* input, std::size_t batch, std::size_t features) {
    extern __shared__ float shared[];

    std::size_t row = blockIdx.x;
    if (row >= batch) {
        return;
    }
    const float* rowIn = input + row * features;
    float* rowOut = out + row * features;

    float localMax = -INFINITY;
    for (std::size_t col = threadIdx.x; col < features; col += blockDim.x) {
        localMax = fmaxf(localMax, rowIn[col]);
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

    float localSum = 0.0F;
    for (std::size_t col = threadIdx.x; col < features; col += blockDim.x) {
        float e = expf(rowIn[col] - rowMax);
        rowOut[col] = e;
        localSum += e;
    }
    shared[threadIdx.x] = localSum;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    __shared__ float rowSum;
    if (threadIdx.x == 0) {
        rowSum = shared[0];
    }
    __syncthreads();

    for (std::size_t col = threadIdx.x; col < features; col += blockDim.x) {
        rowOut[col] /= rowSum;
    }
}

// Un thread per elemento di output (b,s,d): legge il token id per
// (b,s), lo arrotonda a intero, e copia la riga corrispondente della
// tabella.
__global__ void embeddingLookupKernel(float* out, const float* tokenIds, const float* table, std::size_t batch,
                                       std::size_t seq, std::size_t vocabSize, std::size_t dim) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = batch * seq * dim;
    if (idx >= total) {
        return;
    }
    std::size_t token = idx / dim;
    std::size_t d = idx % dim;

    long long tokenId = static_cast<long long>(lroundf(tokenIds[token]));
    if (tokenId < 0 || static_cast<std::size_t>(tokenId) >= vocabSize) {
        // Un kernel CUDA non puo' lanciare eccezioni: la validazione
        // del range avviene sull'host, in embeddingLookup(), PRIMA di
        // lanciare questo kernel (vedi sotto). Qui e' una difesa
        // ridondante che azzera silenziosamente, non dovrebbe mai
        // scattare in pratica.
        out[idx] = 0.0F;
        return;
    }
    out[idx] = table[static_cast<std::size_t>(tokenId) * dim + d];
}

__global__ void addPositionalEmbeddingKernel(float* out, const float* input, const float* table, std::size_t batch,
                                              std::size_t seq, std::size_t dim) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = batch * seq * dim;
    if (idx >= total) {
        return;
    }
    std::size_t withinBatch = idx % (seq * dim);  // (s,d) piatto
    out[idx] = input[idx] + table[withinBatch];
}

// Come addPositionalEmbeddingKernel, ma la riga della tabella usata per
// la posizione s e' 'offset + s' invece di 's' (vedi
// backend::cuda::addPositionalEmbeddingAt).
__global__ void addPositionalEmbeddingAtKernel(float* out, const float* input, const float* table, std::size_t batch,
                                                std::size_t seq, std::size_t dim, std::size_t offset) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = batch * seq * dim;
    if (idx >= total) {
        return;
    }
    std::size_t withinBatch = idx % (seq * dim);
    std::size_t s = withinBatch / dim;
    std::size_t d = withinBatch % dim;
    out[idx] = input[idx] + table[(offset + s) * dim + d];
}

DeviceTensor rmsnorm(const DeviceTensor& input) {
    // Generalizzato a rango >= 2, stessa idea di addBias sopra.
    if (input.rank() < 2) {
        throw std::invalid_argument("rmsnorm: richiede un tensore a rango >= 2 sul device");
    }

    std::size_t features = input.shape().back();
    std::size_t rows = input.elementCount() / features;
    DeviceTensor result(input.shape());
    if (rows > 0 && features > 0) {
        rmsnormKernel<<<static_cast<int>(rows), kBlockSize, kBlockSize * sizeof(float)>>>(
            result.data(), input.data(), rows, features);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor embeddingLookup(const DeviceTensor& tokenIds, const DeviceTensor& table) {
    if (tokenIds.rank() != 2) {
        throw std::invalid_argument("embeddingLookup: richiede token id a rango 2 [batch, seq] sul device");
    }
    if (table.rank() != 2) {
        throw std::invalid_argument("embeddingLookup: richiede una tabella a rango 2 [vocabSize, dim] sul device");
    }

    std::size_t batch = tokenIds.dim(0);
    std::size_t seq = tokenIds.dim(1);
    std::size_t vocabSize = table.dim(0);
    std::size_t dim = table.dim(1);

    // Un kernel CUDA non puo' lanciare eccezioni: la validazione del
    // range dei token id avviene qui sull'host (una copia di [batch,
    // seq] float, trascurabile), PRIMA di lanciare il kernel di
    // gather vero e proprio (che opera interamente su device).
    runtime::Tensor tokenIdsHost = tokenIds.toHost();
    for (std::size_t i = 0; i < tokenIdsHost.elementCount(); ++i) {
        auto tokenId = static_cast<long long>(std::lround(tokenIdsHost.at(i)));
        if (tokenId < 0 || static_cast<std::size_t>(tokenId) >= vocabSize) {
            throw std::invalid_argument("embeddingLookup: token id " + std::to_string(tokenId) +
                                         " fuori dal vocabolario [0, " + std::to_string(vocabSize) + ")");
        }
    }

    DeviceTensor result({batch, seq, dim});
    std::size_t total = batch * seq * dim;
    if (total > 0) {
        embeddingLookupKernel<<<gridSizeFor(total), kBlockSize>>>(result.data(), tokenIds.data(), table.data(),
                                                                   batch, seq, vocabSize, dim);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor addPositionalEmbedding(const DeviceTensor& input, const DeviceTensor& table) {
    if (input.rank() != 3) {
        throw std::invalid_argument("addPositionalEmbedding: richiede un input a rango 3 [batch, seq, dim] sul "
                                     "device");
    }
    if (table.rank() != 2 || table.dim(1) != input.dim(2)) {
        throw std::invalid_argument("addPositionalEmbedding: richiede una tabella [maxSeqLen, dim] con dim "
                                     "coerente con l'input sul device");
    }

    std::size_t batch = input.dim(0);
    std::size_t seq = input.dim(1);
    std::size_t dim = input.dim(2);
    std::size_t maxSeqLen = table.dim(0);
    if (seq > maxSeqLen) {
        throw std::invalid_argument("addPositionalEmbedding: la sequenza supera maxSeqLen della tabella");
    }

    DeviceTensor result(input.shape());
    std::size_t total = batch * seq * dim;
    if (total > 0) {
        addPositionalEmbeddingKernel<<<gridSizeFor(total), kBlockSize>>>(result.data(), input.data(), table.data(),
                                                                          batch, seq, dim);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor addPositionalEmbeddingAt(const DeviceTensor& input, const DeviceTensor& table, std::size_t offset) {
    if (input.rank() != 3) {
        throw std::invalid_argument("addPositionalEmbeddingAt: richiede un input a rango 3 [batch, seq, dim] sul "
                                     "device");
    }
    if (table.rank() != 2 || table.dim(1) != input.dim(2)) {
        throw std::invalid_argument("addPositionalEmbeddingAt: richiede una tabella [maxSeqLen, dim] con dim "
                                     "coerente con l'input sul device");
    }

    std::size_t batch = input.dim(0);
    std::size_t seq = input.dim(1);
    std::size_t dim = input.dim(2);
    std::size_t maxSeqLen = table.dim(0);
    if (offset + seq > maxSeqLen) {
        throw std::invalid_argument("addPositionalEmbeddingAt: la posizione assoluta supera maxSeqLen della "
                                     "tabella");
    }

    DeviceTensor result(input.shape());
    std::size_t total = batch * seq * dim;
    if (total > 0) {
        addPositionalEmbeddingAtKernel<<<gridSizeFor(total), kBlockSize>>>(result.data(), input.data(), table.data(),
                                                                            batch, seq, dim, offset);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor softmax(const DeviceTensor& input) {
    // Generalizzato a rango >= 2, stessa idea di addBias sopra.
    if (input.rank() < 2) {
        throw std::invalid_argument("softmax: richiede un tensore a rango >= 2 sul device");
    }

    std::size_t features = input.shape().back();
    std::size_t rows = input.elementCount() / features;
    DeviceTensor result(input.shape());
    if (rows > 0 && features > 0) {
        softmaxKernel<<<static_cast<int>(rows), kBlockSize, kBlockSize * sizeof(float)>>>(
            result.data(), input.data(), rows, features);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

}  // namespace blackforge::backend::cuda
