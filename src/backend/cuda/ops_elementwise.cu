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
    if (input.rank() != 2 || bias.rank() != 1 || input.dim(1) != bias.dim(0)) {
        throw std::invalid_argument("addBias: attesi input [batch, features] e bias [features] con features "
                                     "corrispondenti");
    }

    DeviceTensor result(input.shape());
    std::size_t n = input.elementCount();
    if (n > 0) {
        addBiasKernel<<<gridSizeFor(n), kBlockSize>>>(result.data(), input.data(), bias.data(), input.dim(0),
                                                       input.dim(1));
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

DeviceTensor rmsnorm(const DeviceTensor& input) {
    if (input.rank() != 2) {
        throw std::invalid_argument("rmsnorm: richiede un tensore a rango 2 [batch, features] sul device");
    }

    DeviceTensor result(input.shape());
    std::size_t batch = input.dim(0);
    std::size_t features = input.dim(1);
    if (batch > 0 && features > 0) {
        rmsnormKernel<<<static_cast<int>(batch), kBlockSize, kBlockSize * sizeof(float)>>>(
            result.data(), input.data(), batch, features);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor softmax(const DeviceTensor& input) {
    if (input.rank() != 2) {
        throw std::invalid_argument("softmax: richiede un tensore a rango 2 [batch, classi] sul device");
    }

    DeviceTensor result(input.shape());
    std::size_t batch = input.dim(0);
    std::size_t features = input.dim(1);
    if (batch > 0 && features > 0) {
        softmaxKernel<<<static_cast<int>(batch), kBlockSize, kBlockSize * sizeof(float)>>>(
            result.data(), input.data(), batch, features);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

}  // namespace blackforge::backend::cuda
