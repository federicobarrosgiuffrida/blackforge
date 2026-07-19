#include "blackforge/backend/cuda/device_pool.hpp"

#include <unordered_map>
#include <vector>

#include "blackforge/backend/cuda/cuda_check.hpp"

namespace blackforge::backend::cuda {

namespace {

struct DevicePool {
    // Dimensione in byte -> pila di buffer liberi di quella dimensione,
    // ancora validi (mai cudaFree'd) sul device a cui questo DevicePool
    // e' associato.
    std::unordered_map<std::size_t, std::vector<void*>> freeBySize;
};

// Un DevicePool per device CUDA, creato pigramente al primo utilizzo di
// quel device (stesso pattern di sharedHandle()/sharedLtHandle() in
// ops_gemm.cu/ops_tensorcore.cu).
std::unordered_map<int, DevicePool>& poolRegistry() {
    static std::unordered_map<int, DevicePool> registry;
    return registry;
}

}  // namespace

void* devicePoolAcquire(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }

    int device = 0;
    BLACKFORGE_CUDA_CHECK(cudaGetDevice(&device));
    DevicePool& pool = poolRegistry()[device];

    auto it = pool.freeBySize.find(bytes);
    if (it != pool.freeBySize.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        return ptr;
    }

    void* ptr = nullptr;
    BLACKFORGE_CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

void devicePoolRelease(void* ptr, std::size_t bytes) {
    if (ptr == nullptr) {
        return;
    }

    // Chiamata da ~DeviceTensor()/~DeviceBf16Buffer(): un distruttore
    // non puo' lanciare eccezioni (stessa convenzione gia' documentata
    // per cudaFree in device_tensor.cu), quindi qui cudaGetDevice non
    // passa dalla macro che lancia — un fallimento (estremamente
    // improbabile) lascia semplicemente 'device' a 0 invece di
    // interrompere il processo.
    int device = 0;
    cudaGetDevice(&device);
    DevicePool& pool = poolRegistry()[device];
    pool.freeBySize[bytes].push_back(ptr);
}

}  // namespace blackforge::backend::cuda
