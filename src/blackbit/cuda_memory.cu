#include "blackforge/blackbit/cuda_memory.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "blackforge/backend/cuda/cuda_check.hpp"

namespace blackforge::blackbit::cuda {

namespace {

std::size_t indexOf(MemoryArena arena) { return static_cast<std::size_t>(arena); }

std::string mib(std::size_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MiB";
    return out.str();
}

}  // namespace

const char* memoryArenaName(MemoryArena arena) {
    switch (arena) {
        case MemoryArena::PackedWeights: return "packed ternary weights";
        case MemoryArena::Scales: return "scales";
        case MemoryArena::DenseParameters: return "router/norm parameters";
        case MemoryArena::Optimizer: return "optimizer state";
        case MemoryArena::Activations: return "activations";
        case MemoryArena::GradientTiles: return "gradient tiles";
        case MemoryArena::DequantizationTiles: return "dequantization tiles";
        case MemoryArena::MoeDispatch: return "MoE dispatch buffers";
        case MemoryArena::Attention: return "attention buffers";
        case MemoryArena::CublasWorkspace: return "cuBLAS/cuBLASLt workspace";
        case MemoryArena::Temporary: return "temporary tensors";
    }
    return "unknown";
}

MemoryTelemetry& MemoryTelemetry::instance() {
    static MemoryTelemetry telemetry;
    return telemetry;
}

void MemoryTelemetry::initialize(int device, std::size_t maxDeviceBytes) {
    if (currentTotal() != 0 && initialized_ && device != device_) {
        throw std::runtime_error("CUDA BlackBit telemetry: cannot change device while tracked allocations are alive");
    }
    BLACKFORGE_CUDA_CHECK(cudaSetDevice(device));
    // Force runtime/context initialization before taking the baseline.
    BLACKFORGE_CUDA_CHECK(cudaFree(nullptr));
    device_ = device;
    maxDeviceBytes_ = maxDeviceBytes;
    initialized_ = true;
    sampleDeviceMemory();
    baselineDeviceUsedBytes_ = totalDeviceBytes_ - freeDeviceBytes_;
    peakDeviceUsedBytes_ = baselineDeviceUsedBytes_;
}

void MemoryTelemetry::ensureInitialized() {
    if (!initialized_) {
        initialize();
    }
}

void MemoryTelemetry::sampleDeviceMemory() {
    BLACKFORGE_CUDA_CHECK(cudaMemGetInfo(&freeDeviceBytes_, &totalDeviceBytes_));
    peakDeviceUsedBytes_ = std::max(peakDeviceUsedBytes_, totalDeviceBytes_ - freeDeviceBytes_);
}

void MemoryTelemetry::checkBeforeAllocation(MemoryArena arena, std::size_t bytes) {
    ensureInitialized();
    sampleDeviceMemory();
    const std::size_t used = totalDeviceBytes_ - freeDeviceBytes_;
    const bool overflow = bytes > std::numeric_limits<std::size_t>::max() - used;
    const std::size_t after = overflow ? std::numeric_limits<std::size_t>::max() : used + bytes;
    if (bytes <= freeDeviceBytes_ && (maxDeviceBytes_ == 0 || after <= maxDeviceBytes_)) {
        return;
    }

    std::ostringstream out;
    out << "CUDA BlackBit allocation refused for '" << memoryArenaName(arena) << "': requested " << mib(bytes)
        << ", device currently uses " << mib(used) << " with " << mib(freeDeviceBytes_) << " free";
    if (maxDeviceBytes_ != 0) {
        out << ", hard max_vram_mb ceiling is " << mib(maxDeviceBytes_)
            << " and the allocation would raise measured device use to " << mib(after);
    }
    out << ". BlackForge currently accounts for " << mib(currentTotal())
        << ". Reduce sequence length/microbatch, optimizer rank, or workspace size.";
    throw std::runtime_error(out.str());
}

void MemoryTelemetry::recordAllocation(MemoryArena arena, std::size_t bytes) {
    const std::size_t i = indexOf(arena);
    current_[i] += bytes;
    peak_[i] = std::max(peak_[i], current_[i]);
    ++allocations_[i];
    peakAccountedTotal_ = std::max(peakAccountedTotal_, currentTotal());
    sampleDeviceMemory();
}

void MemoryTelemetry::recordRelease(MemoryArena arena, std::size_t bytes) noexcept {
    const std::size_t i = indexOf(arena);
    if (bytes > current_[i]) {
        current_[i] = 0;
        ++inconsistencies_;
    } else {
        current_[i] -= bytes;
    }
    // cudaMemGetInfo is diagnostic here; destructors must never throw.
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) {
        freeDeviceBytes_ = freeBytes;
        totalDeviceBytes_ = totalBytes;
        peakDeviceUsedBytes_ = std::max(peakDeviceUsedBytes_, totalBytes - freeBytes);
    }
}

std::size_t MemoryTelemetry::current(MemoryArena arena) const { return current_[indexOf(arena)]; }
std::size_t MemoryTelemetry::peak(MemoryArena arena) const { return peak_[indexOf(arena)]; }
std::size_t MemoryTelemetry::allocationCount(MemoryArena arena) const { return allocations_[indexOf(arena)]; }

std::size_t MemoryTelemetry::currentTotal() const {
    std::size_t result = 0;
    for (std::size_t bytes : current_) {
        result += bytes;
    }
    return result;
}

DeviceMemorySnapshot MemoryTelemetry::snapshot() {
    ensureInitialized();
    sampleDeviceMemory();
    return DeviceMemorySnapshot{totalDeviceBytes_, freeDeviceBytes_, totalDeviceBytes_ - freeDeviceBytes_,
                                currentTotal(), peakAccountedTotal_, peakDeviceUsedBytes_, baselineDeviceUsedBytes_};
}

std::string MemoryTelemetry::report() {
    const DeviceMemorySnapshot state = snapshot();
    std::ostringstream out;
    for (std::size_t i = 0; i < kMemoryArenaCount; ++i) {
        const auto arena = static_cast<MemoryArena>(i);
        out << "  " << std::left << std::setw(28) << memoryArenaName(arena) << " current " << std::right
            << std::setw(12) << mib(current_[i]) << " peak " << std::setw(12) << mib(peak_[i])
            << " allocations " << allocations_[i] << '\n';
    }
    out << "  " << std::left << std::setw(28) << "BlackForge accounted" << " current " << std::right
        << std::setw(12) << mib(state.blackForgeAccountedBytes) << " peak " << std::setw(12)
        << mib(state.blackForgePeakBytes) << '\n';
    out << "  " << std::left << std::setw(28) << "actual NVIDIA device used" << " current " << std::right
        << std::setw(12) << mib(state.usedBytes) << " peak " << std::setw(12) << mib(state.devicePeakUsedBytes)
        << '\n';
    out << "  " << std::left << std::setw(28) << "context/external baseline" << " " << std::right
        << std::setw(20) << mib(state.baselineDeviceUsedBytes) << '\n';
    out << "  " << std::left << std::setw(28) << "actual device free" << " " << std::right << std::setw(20)
        << mib(state.freeBytes) << '\n';
    return out.str();
}

void MemoryTelemetry::resetAccounting() {
    if (currentTotal() != 0) {
        throw std::runtime_error("CUDA BlackBit telemetry: reset requested while tracked allocations are alive");
    }
    current_.fill(0);
    peak_.fill(0);
    allocations_.fill(0);
    inconsistencies_ = 0;
    peakAccountedTotal_ = 0;
    if (initialized_) {
        sampleDeviceMemory();
        baselineDeviceUsedBytes_ = totalDeviceBytes_ - freeDeviceBytes_;
        peakDeviceUsedBytes_ = baselineDeviceUsedBytes_;
    }
}

Buffer::Buffer(std::size_t bytes, MemoryArena arena) : bytes_(bytes), arena_(arena) {
    if (bytes_ == 0) {
        return;
    }
    MemoryTelemetry& telemetry = MemoryTelemetry::instance();
    telemetry.checkBeforeAllocation(arena_, bytes_);
    BLACKFORGE_CUDA_CHECK(cudaMalloc(&data_, bytes_));
    telemetry.recordAllocation(arena_, bytes_);
}

Buffer::~Buffer() { release(); }

Buffer::Buffer(Buffer&& other) noexcept : data_(other.data_), bytes_(other.bytes_), arena_(other.arena_) {
    other.data_ = nullptr;
    other.bytes_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        release();
        data_ = other.data_;
        bytes_ = other.bytes_;
        arena_ = other.arena_;
        other.data_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

void Buffer::release() noexcept {
    if (data_ == nullptr) {
        return;
    }
    // Record after cudaFree so the device sample describes the released state.
    cudaFree(data_);
    MemoryTelemetry::instance().recordRelease(arena_, bytes_);
    data_ = nullptr;
    bytes_ = 0;
}

}  // namespace blackforge::blackbit::cuda
