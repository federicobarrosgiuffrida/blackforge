#include "blackforge/backend/cuda/autodiff.hpp"

#include <stdexcept>

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

AddBiasGrad addBiasBackward(const DeviceTensor& gradOutput) {
    if (gradOutput.rank() != 2) {
        throw std::invalid_argument("addBiasBackward: atteso un gradiente a rango 2 [batch, features] sul device");
    }

    std::size_t batch = gradOutput.dim(0);
    std::size_t features = gradOutput.dim(1);

    // dInput e' un'identita' nel gradiente: si copia semplicemente il
    // buffer (device-to-device), non serve un kernel elementwise.
    DeviceTensor dInput(gradOutput.shape());
    if (gradOutput.elementCount() > 0) {
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(dInput.data(), gradOutput.data(), gradOutput.elementCount() * sizeof(float),
                                         cudaMemcpyDeviceToDevice));
    }

    DeviceTensor dBias({features});
    if (features > 0) {
        addBiasBackwardDBiasKernel<<<gridSizeFor(features), kBlockSize>>>(dBias.data(), gradOutput.data(), batch,
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
    if (input.rank() != 2) {
        throw std::invalid_argument("rmsnormBackward: richiede un tensore a rango 2 [batch, features] sul device");
    }

    DeviceTensor result(input.shape());
    std::size_t batch = input.dim(0);
    std::size_t features = input.dim(1);
    if (batch > 0 && features > 0) {
        std::size_t sharedBytes = 2 * kBlockSize * sizeof(float);
        rmsnormBackwardKernel<<<static_cast<int>(batch), kBlockSize, sharedBytes>>>(
            result.data(), input.data(), gradOutput.data(), batch, features);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

DeviceTensor softmaxBackward(const DeviceTensor& input, const DeviceTensor& gradOutput) {
    if (input.shape() != gradOutput.shape()) {
        throw std::invalid_argument("softmaxBackward: forme incompatibili sul device");
    }
    if (input.rank() != 2) {
        throw std::invalid_argument("softmaxBackward: richiede un tensore a rango 2 [batch, classi] sul device");
    }

    DeviceTensor y = softmax(input);
    DeviceTensor result(input.shape());
    std::size_t batch = input.dim(0);
    std::size_t features = input.dim(1);
    if (batch > 0 && features > 0) {
        std::size_t sharedBytes = kBlockSize * sizeof(float);
        softmaxBackwardKernel<<<static_cast<int>(batch), kBlockSize, sharedBytes>>>(
            result.data(), y.data(), gradOutput.data(), batch, features);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

}  // namespace blackforge::backend::cuda
