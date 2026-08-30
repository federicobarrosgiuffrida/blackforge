#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/blackbit/cuda_memory.hpp"
#include "blackforge/blackbit/cuda_ternary.hpp"
#include "blackforge/blackbit/ternary.hpp"

namespace bb = blackforge::blackbit;
namespace bbcuda = blackforge::blackbit::cuda;

namespace {

std::vector<float> copyTile(const bbcuda::DecodedTile& tile) {
    std::vector<float> host(tile.rows() * tile.cols());
    BLACKFORGE_CUDA_CHECK(cudaMemcpy(host.data(), tile.data(), host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return host;
}

std::vector<std::int8_t> patternedTrits(std::size_t count) {
    std::vector<std::int8_t> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = static_cast<std::int8_t>(static_cast<int>(i % 3) - 1);
    }
    return result;
}

}  // namespace

TEST(CudaBlackBitTernaryTest, UploadPackedAndDecodeMatchesCpuOnEdges) {
    const std::vector<std::size_t> shape{3, 43};  // row/group/5/tile boundaries all have tails
    const auto trits = patternedTrits(3 * 43);
    const std::vector<float> scales{0.25F, 0.5F, 0.75F, 1.0F, 1.25F, 1.5F, 2.0F, 2.5F, 3.0F};
    const bb::TernaryTensor cpu = bb::TernaryTensor::fromTrits(shape, trits, scales, 20);
    const bbcuda::TernaryTensor gpu(cpu);

    const auto tile = gpu.decodeRows(0, 3);
    const auto actual = copyTile(tile);
    const auto expected = cpu.dequantize().data();
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "logical index " << i;
    }
}

TEST(CudaBlackBitTernaryTest, GpuPackMatchesCanonicalCpuBytesAndScales) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> distribution(-1, 1);
    const std::vector<std::size_t> shape{5, 67};
    std::vector<std::int8_t> trits(5 * 67);
    for (auto& value : trits) {
        value = static_cast<std::int8_t>(distribution(rng));
    }
    std::vector<float> scales(5 * 4);
    for (std::size_t i = 0; i < scales.size(); ++i) {
        scales[i] = 0.03125F * static_cast<float>(i + 1);
    }

    const bb::TernaryTensor cpu = bb::TernaryTensor::fromTrits(shape, trits, scales, 20);
    bbcuda::TernaryTensor gpu = bbcuda::TernaryTensor::fromTrits(shape, trits, scales, 20);
    const bb::TernaryTensor roundTrip = gpu.download();

    EXPECT_EQ(roundTrip.packedWords(), cpu.packedWords());
    EXPECT_EQ(roundTrip.scales(), cpu.scales());
    EXPECT_EQ(roundTrip.packedByteCount(), bb::ternaryPackedBytes(5, 67));
}

TEST(CudaBlackBitTernaryTest, DecodesSubtilesWithoutMaterializingWholeMatrix) {
    const std::vector<std::size_t> shape{7, 41};
    const auto trits = patternedTrits(7 * 41);
    const std::vector<float> scales(7 * 3, 0.125F);
    const bb::TernaryTensor cpu = bb::TernaryTensor::fromTrits(shape, trits, scales, 20);
    const bbcuda::TernaryTensor gpu(cpu);

    const auto tile = gpu.decodeRows(2, 3);
    const auto actual = copyTile(tile);
    std::vector<float> expected(3 * 41);
    cpu.dequantizeRows(2, 3, expected.data());
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(tile.bytes(), 3 * 41 * sizeof(float));
    EXPECT_LT(tile.bytes(), cpu.elementCount() * sizeof(float));
}

TEST(CudaBlackBitMemoryTest, TracksAccountedAndActualDeviceMemory) {
    auto& telemetry = bbcuda::MemoryTelemetry::instance();
    telemetry.initialize(0, 7800ULL * 1024ULL * 1024ULL);
    if (telemetry.currentTotal() != 0) {
        GTEST_SKIP() << "another CUDA BlackBit allocation is alive";
    }
    telemetry.resetAccounting();
    const auto before = telemetry.snapshot();
    {
        bbcuda::Buffer packed(4 * 1024 * 1024, bbcuda::MemoryArena::PackedWeights);
        bbcuda::Buffer scales(1 * 1024 * 1024, bbcuda::MemoryArena::Scales);
        const auto during = telemetry.snapshot();
        EXPECT_EQ(during.blackForgeAccountedBytes, 5 * 1024 * 1024);
        EXPECT_GE(during.usedBytes, before.usedBytes);
        EXPECT_EQ(telemetry.current(bbcuda::MemoryArena::PackedWeights), 4 * 1024 * 1024);
        EXPECT_EQ(telemetry.current(bbcuda::MemoryArena::Scales), 1 * 1024 * 1024);
    }
    EXPECT_EQ(telemetry.currentTotal(), 0U);
    EXPECT_EQ(telemetry.inconsistencies(), 0U);
}

TEST(CudaBlackBitMemoryTest, HardLimitRefusesAllocationWithDiagnostic) {
    auto& telemetry = bbcuda::MemoryTelemetry::instance();
    telemetry.initialize(0, 7800ULL * 1024ULL * 1024ULL);
    if (telemetry.currentTotal() != 0) {
        GTEST_SKIP() << "another CUDA BlackBit allocation is alive";
    }
    const auto state = telemetry.snapshot();
    telemetry.setMaxDeviceBytes(state.usedBytes + 1024);
    try {
        EXPECT_THROW(
            { bbcuda::Buffer impossible(2 * 1024 * 1024, bbcuda::MemoryArena::GradientTiles); },
            std::runtime_error);
    } catch (...) {
        telemetry.setMaxDeviceBytes(7800ULL * 1024ULL * 1024ULL);
        throw;
    }
    telemetry.setMaxDeviceBytes(7800ULL * 1024ULL * 1024ULL);
}
