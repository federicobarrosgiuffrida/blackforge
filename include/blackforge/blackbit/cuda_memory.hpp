#pragma once

#include <array>
#include <cstddef>
#include <string>

namespace blackforge::blackbit::cuda {

enum class MemoryArena {
    PackedWeights,
    Scales,
    DenseParameters,
    Optimizer,
    Activations,
    GradientTiles,
    DequantizationTiles,
    MoeDispatch,
    Attention,
    CublasWorkspace,
    Temporary,
};

inline constexpr std::size_t kMemoryArenaCount = 11;

const char* memoryArenaName(MemoryArena arena);

struct DeviceMemorySnapshot {
    std::size_t totalBytes = 0;
    std::size_t freeBytes = 0;
    std::size_t usedBytes = 0;
    std::size_t blackForgeAccountedBytes = 0;
    std::size_t blackForgePeakBytes = 0;
    std::size_t devicePeakUsedBytes = 0;
    std::size_t baselineDeviceUsedBytes = 0;
};

class MemoryTelemetry {
public:
    static MemoryTelemetry& instance();

    // Forces creation of the CUDA runtime context and captures the device
    // baseline. maxDeviceBytes is a hard ceiling on total used device memory,
    // not merely on BlackForge-accounted allocations. Zero disables it.
    void initialize(int device = 0, std::size_t maxDeviceBytes = 7800ULL * 1024ULL * 1024ULL);
    void setMaxDeviceBytes(std::size_t bytes) { maxDeviceBytes_ = bytes; }

    [[nodiscard]] int device() const { return device_; }
    [[nodiscard]] std::size_t maxDeviceBytes() const { return maxDeviceBytes_; }
    [[nodiscard]] bool initialized() const { return initialized_; }

    void checkBeforeAllocation(MemoryArena arena, std::size_t bytes);
    void recordAllocation(MemoryArena arena, std::size_t bytes);
    void recordRelease(MemoryArena arena, std::size_t bytes) noexcept;

    [[nodiscard]] std::size_t current(MemoryArena arena) const;
    [[nodiscard]] std::size_t peak(MemoryArena arena) const;
    [[nodiscard]] std::size_t allocationCount(MemoryArena arena) const;
    [[nodiscard]] std::size_t currentTotal() const;
    [[nodiscard]] std::size_t peakTotal() const { return peakAccountedTotal_; }
    [[nodiscard]] std::size_t inconsistencies() const { return inconsistencies_; }

    DeviceMemorySnapshot snapshot();
    [[nodiscard]] std::string report();

    // Starts a new measurement window without disturbing live persistent
    // allocations (model/optimizer). Used after benchmark warmup.
    void resetPeaks();

    // Only valid when no tracked allocations are alive.
    void resetAccounting();

private:
    MemoryTelemetry() = default;
    void ensureInitialized();
    void sampleDeviceMemory();

    int device_ = 0;
    bool initialized_ = false;
    std::size_t maxDeviceBytes_ = 7800ULL * 1024ULL * 1024ULL;
    std::size_t totalDeviceBytes_ = 0;
    std::size_t freeDeviceBytes_ = 0;
    std::size_t baselineDeviceUsedBytes_ = 0;
    std::size_t peakDeviceUsedBytes_ = 0;
    std::size_t peakAccountedTotal_ = 0;
    std::size_t inconsistencies_ = 0;
    std::array<std::size_t, kMemoryArenaCount> current_{};
    std::array<std::size_t, kMemoryArenaCount> peak_{};
    std::array<std::size_t, kMemoryArenaCount> allocations_{};
};

class Buffer {
public:
    Buffer() = default;
    Buffer(std::size_t bytes, MemoryArena arena);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    [[nodiscard]] void* data() { return data_; }
    [[nodiscard]] const void* data() const { return data_; }

    template <typename T>
    [[nodiscard]] T* as() {
        return static_cast<T*>(data_);
    }

    template <typename T>
    [[nodiscard]] const T* as() const {
        return static_cast<const T*>(data_);
    }

    [[nodiscard]] std::size_t bytes() const { return bytes_; }
    [[nodiscard]] MemoryArena arena() const { return arena_; }

private:
    void release() noexcept;

    void* data_ = nullptr;
    std::size_t bytes_ = 0;
    MemoryArena arena_ = MemoryArena::Temporary;
};

}  // namespace blackforge::blackbit::cuda
