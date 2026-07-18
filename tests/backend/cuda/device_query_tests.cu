#include "blackforge/backend/cuda/device_query.hpp"

#include <gtest/gtest.h>

// Come tutti i test in questo file (tests/backend/cuda/), presuppone
// una GPU NVIDIA realmente presente e visibile al driver: l'intera
// suite CUDA richiede hardware reale per eseguire (cudaMalloc fallisce
// senza una GPU), quindi questa non e' un'assunzione aggiuntiva.
TEST(CudaDeviceQueryTest, TrovaAlmenoUnaGpu) {
    auto devices = blackforge::backend::cuda::enumerateDevices();

    ASSERT_FALSE(devices.empty());
    EXPECT_FALSE(devices.front().name.empty());
    EXPECT_GT(devices.front().totalMemoryBytes, static_cast<std::size_t>(0));
    EXPECT_GE(devices.front().computeCapabilityMajor, 0);
}
