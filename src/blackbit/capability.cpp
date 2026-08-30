#include "blackforge/blackbit/capability.hpp"

#include <sstream>

#if BLACKFORGE_HAS_CUDA
#include "blackforge/backend/cuda/device_query.hpp"
#endif

namespace blackforge::blackbit {

const char* tensorCoreGenerationName(TensorCoreGeneration generation) {
    switch (generation) {
        case TensorCoreGeneration::None: return "nessuno";
        case TensorCoreGeneration::Volta: return "Volta";
        case TensorCoreGeneration::Turing: return "Turing";
        case TensorCoreGeneration::Ampere: return "Ampere";
        case TensorCoreGeneration::Hopper: return "Hopper";
        case TensorCoreGeneration::Blackwell: return "Blackwell";
    }
    return "?";
}

TensorCoreGeneration DeviceCapability::tensorCores() const {
    if (major >= 10) {
        // sm_100 (data center) e sm_120 (GeForce Blackwell, la RTX 5060
        // bersaglio) condividono la generazione di Tensor Core.
        return TensorCoreGeneration::Blackwell;
    }
    if (major == 9) {
        return TensorCoreGeneration::Hopper;
    }
    if (major == 8) {
        return TensorCoreGeneration::Ampere;
    }
    if (major == 7) {
        return minor >= 5 ? TensorCoreGeneration::Turing : TensorCoreGeneration::Volta;
    }
    return TensorCoreGeneration::None;
}

bool DeviceCapability::supportsBf16TensorCores() const {
    return tensorCores() >= TensorCoreGeneration::Ampere;
}

bool DeviceCapability::supportsFp8() const { return tensorCores() >= TensorCoreGeneration::Hopper; }

bool DeviceCapability::supportsFp4() const { return tensorCores() >= TensorCoreGeneration::Blackwell; }

std::string DeviceCapability::describe() const {
    std::ostringstream out;
    out << name << " (compute capability " << major << "." << minor << ", Tensor Core "
        << tensorCoreGenerationName(tensorCores()) << ", " << (totalMemoryBytes / (1024ULL * 1024ULL))
        << " MiB)";
    if (supportsFp4()) {
        out << " — percorso FP4 disponibile nell'hardware, non ancora implementato in BlackForge";
    }
    return out.str();
}

std::optional<DeviceCapability> detectDeviceCapability(int index) {
#if BLACKFORGE_HAS_CUDA
    const auto devices = backend::cuda::enumerateDevices();
    for (const auto& device : devices) {
        if (device.index == index) {
            DeviceCapability capability;
            capability.major = device.computeCapabilityMajor;
            capability.minor = device.computeCapabilityMinor;
            capability.name = device.name;
            capability.totalMemoryBytes = device.totalMemoryBytes;
            return capability;
        }
    }
    return std::nullopt;
#else
    (void)index;
    return std::nullopt;
#endif
}

ComputeDType preferredComputeDType(const std::optional<DeviceCapability>& capability) {
    if (!capability.has_value()) {
        return ComputeDType::FP32;
    }
    // Volutamente NON restituisce FP4 su Blackwell: non esiste ancora un
    // GEMM che lo implementi, e dichiararlo qui renderebbe falso ogni
    // rapporto che riporta il formato di calcolo usato.
    return capability->supportsBf16TensorCores() ? ComputeDType::BF16 : ComputeDType::FP32;
}

}  // namespace blackforge::blackbit
