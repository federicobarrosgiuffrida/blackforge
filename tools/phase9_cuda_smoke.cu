#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void affineKernel(float* values, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] = values[index] * 2.0F + 1.0F;
    }
}

}  // namespace

int main() {
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");

    std::size_t freeBefore = 0;
    std::size_t total = 0;
    check(cudaMemGetInfo(&freeBefore, &total), "cudaMemGetInfo(before)");

    constexpr int count = 4096;
    std::vector<float> host(count);
    for (int i = 0; i < count; ++i) {
        host[i] = static_cast<float>(i) * 0.25F;
    }

    float* device = nullptr;
    check(cudaMalloc(&device, host.size() * sizeof(float)), "cudaMalloc");
    check(cudaMemcpy(device, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy(host-to-device)");

    affineKernel<<<(count + 255) / 256, 256>>>(device, count);
    check(cudaGetLastError(), "affineKernel launch");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check(cudaMemcpy(host.data(), device, host.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy(device-to-host)");

    for (int i = 0; i < count; ++i) {
        const float expected = static_cast<float>(i) * 0.5F + 1.0F;
        if (std::fabs(host[i] - expected) > 1.0e-6F) {
            std::fprintf(stderr, "verification failed at %d: got %.9g, expected %.9g\n", i, host[i], expected);
            cudaFree(device);
            return EXIT_FAILURE;
        }
    }

    check(cudaFree(device), "cudaFree");
    std::size_t freeAfter = 0;
    check(cudaMemGetInfo(&freeAfter, &total), "cudaMemGetInfo(after)");

    std::printf("CUDA SMOKE PASS\n");
    std::printf("device: %s\n", properties.name);
    std::printf("compute capability: %d.%d\n", properties.major, properties.minor);
    std::printf("global memory: %zu MiB\n", total / (1024 * 1024));
    std::printf("free before: %zu MiB\n", freeBefore / (1024 * 1024));
    std::printf("free after: %zu MiB\n", freeAfter / (1024 * 1024));
    return EXIT_SUCCESS;
}
