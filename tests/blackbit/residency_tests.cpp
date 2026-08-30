#include "blackforge/blackbit/residency.hpp"

#include <gtest/gtest.h>

#include "blackforge/blackbit/capability.hpp"

using namespace blackforge;
using blackforge::blackbit::ResidencyPolicy;
using blackforge::blackbit::ResidencyState;

TEST(ResidencyTest, PerDefaultTuttoRestaResidente) {
    // E' cio' che BlackBit-9B fa oggi e deve continuare a fare: 1,9 GB
    // di pesi stanno in VRAM senza bisogno di paginare nulla.
    const auto plan = blackbit::planResidency(blackbit::blackBit9bA3b(), ResidencyPolicy{});

    EXPECT_EQ(plan.bytes(ResidencyState::CpuPinned), 0U);
    EXPECT_EQ(plan.bytes(ResidencyState::Paged), 0U);
    EXPECT_EQ(plan.residentBytes(), plan.total());
    EXPECT_EQ(plan.worstCaseTransferBytesPerToken, 0U);

    // ~1,9 GiB di parametri: il conto deve coincidere con quello del
    // modello reale.
    EXPECT_GT(plan.total(), 1800ULL * 1024ULL * 1024ULL);
    EXPECT_LT(plan.total(), 2100ULL * 1024ULL * 1024ULL);
}

TEST(ResidencyTest, TenereMenoEspertiCaldiRiduceLaMemoriaResidente) {
    const blackbit::BlackBitConfig config = blackbit::blackBit9bA3b();

    ResidencyPolicy hot2;
    hot2.hotExpertsPerLayer = 2;  // quanti ne servono per un token top-2
    const auto plan = blackbit::planResidency(config, hot2);
    const auto full = blackbit::planResidency(config, ResidencyPolicy{});

    EXPECT_LT(plan.residentBytes(), full.residentBytes());
    EXPECT_GT(plan.bytes(ResidencyState::CpuPinned), 0U);
    EXPECT_EQ(plan.total(), full.total()) << "i byte totali non cambiano, cambia solo dove stanno";

    // Il traffico che la politica genera deve essere riportato: con sei
    // esperti su otto fuori dalla VRAM, un token sfortunato ne chiede
    // due per ogni layer.
    EXPECT_GT(plan.worstCaseTransferBytesPerToken, 0U);
    EXPECT_FALSE(plan.report().empty());
}

TEST(ResidencyTest, LaParteCondivisaPuoRestareResidenteAnchePaginandoGliEsperti) {
    const blackbit::BlackBitConfig config = blackbit::blackBit9bA3b();

    ResidencyPolicy policy;
    policy.hotExpertsPerLayer = 0;  // tutti caldi
    policy.keepSharedResident = false;
    policy.coldExpertState = ResidencyState::Paged;

    const auto plan = blackbit::planResidency(config, policy);
    // Attention + embedding + norm/router non residenti: e' il 9 % dei
    // parametri, e questo test documenta che spostarli e' esprimibile
    // (anche se e' una cattiva idea, dato che servono a ogni token).
    EXPECT_GT(plan.bytes(ResidencyState::Paged), 0U);
}

TEST(CapabilityTest, LaGenerazioneDiTensorCoreSegueLaComputeCapability) {
    auto capability = [](int major, int minor) {
        blackbit::DeviceCapability value;
        value.major = major;
        value.minor = minor;
        return value;
    };

    EXPECT_EQ(capability(6, 1).tensorCores(), blackbit::TensorCoreGeneration::None);
    EXPECT_EQ(capability(7, 0).tensorCores(), blackbit::TensorCoreGeneration::Volta);
    EXPECT_EQ(capability(7, 5).tensorCores(), blackbit::TensorCoreGeneration::Turing);
    EXPECT_EQ(capability(8, 6).tensorCores(), blackbit::TensorCoreGeneration::Ampere);
    EXPECT_EQ(capability(9, 0).tensorCores(), blackbit::TensorCoreGeneration::Hopper);
    // sm_120: la RTX 5060 bersaglio.
    EXPECT_EQ(capability(12, 0).tensorCores(), blackbit::TensorCoreGeneration::Blackwell);

    EXPECT_FALSE(capability(8, 6).supportsFp4());
    EXPECT_TRUE(capability(12, 0).supportsFp4());
    EXPECT_TRUE(capability(12, 0).supportsBf16TensorCores());
}

TEST(CapabilityTest, IlFormatoDiCalcoloNonDichiaraFp4FinchNonEsisteUnKernel) {
    blackbit::DeviceCapability blackwell;
    blackwell.major = 12;
    blackwell.minor = 0;
    blackwell.name = "RTX 5060";

    // L'hardware lo supporta...
    EXPECT_TRUE(blackwell.supportsFp4());
    // ...ma il formato scelto resta BF16, perche' un GEMM FP4 non
    // esiste ancora in questo motore. Dichiararlo qui renderebbe falso
    // ogni rapporto che riporta il formato di calcolo usato.
    EXPECT_EQ(blackbit::preferredComputeDType(blackwell), blackbit::ComputeDType::BF16);

    blackbit::DeviceCapability oldGpu;
    oldGpu.major = 6;
    oldGpu.minor = 1;
    EXPECT_EQ(blackbit::preferredComputeDType(oldGpu), blackbit::ComputeDType::FP32);

    // Senza GPU (o senza CUDA nella build) si calcola in FP32.
    EXPECT_EQ(blackbit::preferredComputeDType(std::nullopt), blackbit::ComputeDType::FP32);
    EXPECT_NE(blackwell.describe().find("Blackwell"), std::string::npos);
}

TEST(CapabilityTest, IlRilevamentoNonLanciaSenzaGpu) {
    // L'assenza di una GPU e' un esito normale del rilevamento.
    EXPECT_NO_THROW((void)blackbit::detectDeviceCapability(0));
    EXPECT_FALSE(blackbit::detectDeviceCapability(999).has_value());
}
