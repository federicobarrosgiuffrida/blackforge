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

}  // namespace blackforge::backend::cuda
