#include "blackforge/backend/cuda/loss.hpp"

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

}  // namespace blackforge::backend::cuda
