#include "blackforge/backend/cuda/device_query.hpp"

#include <stdexcept>

#include <cuda_runtime.h>

namespace blackforge::backend::cuda {

std::vector<CudaDeviceInfo> enumerateDevices() {
    std::vector<CudaDeviceInfo> devices;

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return devices;  // nessun driver/GPU: elenco vuoto, non un errore
    }

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, i) != cudaSuccess) {
            continue;
        }
        devices.push_back(CudaDeviceInfo{
            i,
            std::string(props.name),
            props.major,
            props.minor,
            props.totalGlobalMem,
        });
    }

    return devices;
}

void setActiveDevice(int index) {
    cudaError_t result = cudaSetDevice(index);
    if (result != cudaSuccess) {
        throw std::runtime_error("dispositivo CUDA non valido 'cuda:" + std::to_string(index) +
                                  "': " + cudaGetErrorString(result));
    }
}

}  // namespace blackforge::backend::cuda
