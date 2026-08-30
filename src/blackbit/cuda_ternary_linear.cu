#include "blackforge/blackbit/cuda_ternary_linear.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/blackbit/device_shared.hpp"

namespace blackforge::blackbit::cuda {

namespace {

constexpr int kBlockSize = 256;
constexpr std::size_t kCublasWorkspaceBytes = 4 * 1024 * 1024;

std::size_t rowsOf(const Tensor& tensor) {
    if (tensor.rank() < 2) {
        throw std::invalid_argument("CUDA TernaryLinear requires rank >= 2");
    }
    return tensor.elementCount() / tensor.shape().back();
}

void requireFeatures(const Tensor& tensor, std::size_t features, const std::string& name, const char* role) {
    if (tensor.rank() < 2 || tensor.shape().back() != features) {
        throw std::invalid_argument("CUDA TernaryLinear '" + name + "': invalid " + role + " shape");
    }
}

unsigned int gridFor(std::size_t count) {
    return static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
}

__global__ void floatToBf16Kernel(__nv_bfloat16* output, const float* input, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __float2bfloat16(input[index]);
    }
}

__global__ void decodeBf16Kernel(__nv_bfloat16* output, const std::uint32_t* packed, const float* scales,
                                 std::size_t firstRow, std::size_t rowCount, std::size_t rowLength,
                                 std::size_t wordsPerRow, std::size_t groupsPerRow, std::size_t groupSize) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = rowCount * rowLength;
    if (index >= count) {
        return;
    }
    const std::size_t localRow = index / rowLength;
    const std::size_t col = index % rowLength;
    const std::size_t row = firstRow + localRow;
    const std::uint32_t word = packed[row * wordsPerRow + col / kTritsPerWord];
    const std::size_t inWord = col % kTritsPerWord;
    const int trit = decodeTritAt(wordByte(word, static_cast<int>(inWord / kTritsPerByte)),
                                  static_cast<int>(inWord % kTritsPerByte));
    output[index] = __float2bfloat16(static_cast<float>(trit) * scales[row * groupsPerRow + col / groupSize]);
}

cublasLtHandle_t sharedHandle() {
    static std::unordered_map<int, cublasLtHandle_t> handles;
    int device = 0;
    BLACKFORGE_CUDA_CHECK(cudaGetDevice(&device));
    auto found = handles.find(device);
    if (found != handles.end()) {
        return found->second;
    }
    cublasLtHandle_t handle = nullptr;
    BLACKFORGE_CUBLAS_CHECK(cublasLtCreate(&handle));
    handles.emplace(device, handle);
    return handle;
}

Buffer& sharedWorkspace() {
    static Buffer workspace(kCublasWorkspaceBytes, MemoryArena::CublasWorkspace);
    return workspace;
}

struct LtDescriptors {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    ~LtDescriptors() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (operation) cublasLtMatmulDescDestroy(operation);
    }
};

void gemm(cublasOperation_t transA, cublasOperation_t transB, const __nv_bfloat16* a, std::size_t aRows,
          std::size_t aCols, std::size_t aLd, const __nv_bfloat16* b, std::size_t bRows, std::size_t bCols,
          std::size_t bLd, float* c, std::size_t cRows, std::size_t cCols, std::size_t cLd, float beta) {
    LtDescriptors descriptors;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulDescCreate(&descriptors.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                           &transA, sizeof(transA)));
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                                           &transB, sizeof(transB)));
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&descriptors.a, CUDA_R_16BF, aRows, aCols, aLd));
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&descriptors.b, CUDA_R_16BF, bRows, bCols, bLd));
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&descriptors.c, CUDA_R_32F, cRows, cCols, cLd));
    cublasLtOrder_t rowMajor = CUBLASLT_ORDER_ROW;
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(descriptors.a, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(descriptors.b, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(descriptors.c, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&descriptors.preference));
    const std::size_t workspaceBytes = sharedWorkspace().bytes();
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        descriptors.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes)));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(sharedHandle(), descriptors.operation, descriptors.a,
                                                            descriptors.b, descriptors.c, descriptors.c,
                                                            descriptors.preference, 1, &heuristic, &returned));
    if (returned == 0) {
        throw std::runtime_error("CUDA TernaryLinear: cuBLASLt found no BF16 algorithm for the tile shape");
    }
    const float alpha = 1.0F;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmul(sharedHandle(), descriptors.operation, &alpha, a, descriptors.a, b,
                                            descriptors.b, &beta, c, descriptors.c, c, descriptors.c,
                                            &heuristic.algo, sharedWorkspace().data(), heuristic.workspaceSize, nullptr));
}

class EventPair {
public:
    EventPair() {
        BLACKFORGE_CUDA_CHECK(cudaEventCreate(&start_));
        BLACKFORGE_CUDA_CHECK(cudaEventCreate(&end_));
    }
    ~EventPair() {
        cudaEventDestroy(end_);
        cudaEventDestroy(start_);
    }
    void start() { BLACKFORGE_CUDA_CHECK(cudaEventRecord(start_)); }
    float stop() {
        BLACKFORGE_CUDA_CHECK(cudaEventRecord(end_));
        BLACKFORGE_CUDA_CHECK(cudaEventSynchronize(end_));
        float milliseconds = 0.0F;
        BLACKFORGE_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, end_));
        return milliseconds;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t end_{};
};

}  // namespace

GradientLifetimeStats& gradientLifetimeStats() {
    static GradientLifetimeStats stats;
    return stats;
}

void resetGradientLifetimeStats() { gradientLifetimeStats() = GradientLifetimeStats{}; }

TernaryLinear::TernaryLinear(const blackforge::blackbit::TernaryLinear& cpuReference)
    : name_(cpuReference.name()),
      inFeatures_(cpuReference.inFeatures()),
      outFeatures_(cpuReference.outFeatures()),
      tileRows_(cpuReference.tileRows()),
      weight_(cpuReference.weight()) {}

TernaryLinear::TernaryLinear(std::string name, std::size_t inFeatures, std::size_t outFeatures,
                             blackforge::blackbit::TernaryTensor hostWeight, std::size_t tileRows)
    : name_(std::move(name)),
      inFeatures_(inFeatures),
      outFeatures_(outFeatures),
      tileRows_(std::min(tileRows, outFeatures)),
      weight_(hostWeight) {
    if (hostWeight.rows() != outFeatures_ || hostWeight.rowLength() != inFeatures_ || tileRows == 0) {
        throw std::invalid_argument("CUDA TernaryLinear: host weight or tile size does not match layer dimensions");
    }
}

Tensor TernaryLinear::forward(const Tensor& input) const {
    requireFeatures(input, inFeatures_, name_, "input");
    const std::size_t rows = rowsOf(input);
    std::vector<std::size_t> outputShape = input.shape();
    outputShape.back() = outFeatures_;
    Tensor output(std::move(outputShape), MemoryArena::Activations);
    Buffer inputBf16(input.elementCount() * sizeof(__nv_bfloat16), MemoryArena::Temporary);
    Buffer tile(tileRows_ * inFeatures_ * sizeof(__nv_bfloat16), MemoryArena::DequantizationTiles);

    EventPair totalTimer;
    totalTimer.start();
    floatToBf16Kernel<<<gridFor(input.elementCount()), kBlockSize>>>(inputBf16.as<__nv_bfloat16>(), input.data(),
                                                                    input.elementCount());
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());

    metrics_ = {};
    metrics_.dequantWorkspaceBytes = tile.bytes();
    metrics_.cublasWorkspaceBytes = sharedWorkspace().bytes();
    EventPair phaseTimer;
    for (std::size_t first = 0; first < outFeatures_; first += tileRows_) {
        const std::size_t count = std::min(tileRows_, outFeatures_ - first);
        phaseTimer.start();
        decodeBf16Kernel<<<gridFor(count * inFeatures_), kBlockSize>>>(
            tile.as<__nv_bfloat16>(), weight_.packedWords(), weight_.scales(), first, count, inFeatures_,
            weight_.wordsPerRow(), weight_.groupsPerRow(), weight_.groupSize());
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
        metrics_.decodeMs += phaseTimer.stop();

        phaseTimer.start();
        gemm(CUBLAS_OP_N, CUBLAS_OP_T, inputBf16.as<__nv_bfloat16>(), rows, inFeatures_, inFeatures_,
             tile.as<__nv_bfloat16>(), count, inFeatures_, inFeatures_, output.data() + first, rows, count,
             outFeatures_, 0.0F);
        metrics_.gemmMs += phaseTimer.stop();
        metrics_.decodedBytesWritten += count * inFeatures_ * sizeof(__nv_bfloat16);
        metrics_.packedBytesRead += count * weight_.wordsPerRow() * sizeof(std::uint32_t) +
                                    count * weight_.groupsPerRow() * sizeof(float);
    }
    metrics_.totalMs = totalTimer.stop();
    return output;
}

Tensor TernaryLinear::backward(const Tensor& input, const Tensor& gradOutput, GradientSink* sink) const {
    requireFeatures(input, inFeatures_, name_, "input");
    requireFeatures(gradOutput, outFeatures_, name_, "output gradient");
    const std::size_t rows = rowsOf(input);
    if (rowsOf(gradOutput) != rows) {
        throw std::invalid_argument("CUDA TernaryLinear: input and output gradient row counts differ");
    }

    Tensor gradInput = Tensor::zeros(input.shape(), MemoryArena::Activations);
    Buffer inputBf16(input.elementCount() * sizeof(__nv_bfloat16), MemoryArena::Temporary);
    Buffer gradOutputBf16(gradOutput.elementCount() * sizeof(__nv_bfloat16), MemoryArena::Temporary);
    Buffer tile(tileRows_ * inFeatures_ * sizeof(__nv_bfloat16), MemoryArena::DequantizationTiles);
    Buffer gradTile(tileRows_ * inFeatures_ * sizeof(float), MemoryArena::GradientTiles);
    floatToBf16Kernel<<<gridFor(input.elementCount()), kBlockSize>>>(inputBf16.as<__nv_bfloat16>(), input.data(),
                                                                    input.elementCount());
    floatToBf16Kernel<<<gridFor(gradOutput.elementCount()), kBlockSize>>>(
        gradOutputBf16.as<__nv_bfloat16>(), gradOutput.data(), gradOutput.elementCount());
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());

    GradientLifetimeStats& lifetime = gradientLifetimeStats();
    for (std::size_t first = 0; first < outFeatures_; first += tileRows_) {
        const std::size_t count = std::min(tileRows_, outFeatures_ - first);
        const std::size_t gradientBytes = count * inFeatures_ * sizeof(float);
        decodeBf16Kernel<<<gridFor(count * inFeatures_), kBlockSize>>>(
            tile.as<__nv_bfloat16>(), weight_.packedWords(), weight_.scales(), first, count, inFeatures_,
            weight_.wordsPerRow(), weight_.groupsPerRow(), weight_.groupSize());
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());

        // dX += dY_tile @ W_tile. gradOutput is strided by outFeatures.
        gemm(CUBLAS_OP_N, CUBLAS_OP_N, gradOutputBf16.as<__nv_bfloat16>() + first, rows, count, outFeatures_,
             tile.as<__nv_bfloat16>(), count, inFeatures_, inFeatures_, gradInput.data(), rows, inFeatures_,
             inFeatures_, first == 0 ? 0.0F : 1.0F);
        // dW_tile = dY_tile^T @ X, produced and consumed before the next tile.
        gemm(CUBLAS_OP_T, CUBLAS_OP_N, gradOutputBf16.as<__nv_bfloat16>() + first, rows, count, outFeatures_,
             inputBf16.as<__nv_bfloat16>(), rows, inFeatures_, inFeatures_, gradTile.as<float>(), count,
             inFeatures_, inFeatures_, 0.0F);

        lifetime.liveBytes = gradientBytes;
        lifetime.peakLiveBytes = std::max(lifetime.peakLiveBytes, gradientBytes);
        lifetime.cumulativeBytes += gradientBytes;
        ++lifetime.blocksProduced;
        if (sink != nullptr) {
            sink->consumeWeightGradientBlock(parameterId(), first, count, gradTile.as<float>());
        }
        lifetime.liveBytes = 0;
        ++lifetime.blocksReleased;
    }
    BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());
    return gradInput;
}

}  // namespace blackforge::blackbit::cuda
