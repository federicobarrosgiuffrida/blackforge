#include "blackforge/backend/cpu/quantize.hpp"

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

using blackforge::runtime::Tensor;
namespace cpu = blackforge::backend::cpu;
using blackforge::sema::DType;

TEST(QuantizeTest, Fp32EUnIdentita) {
    EXPECT_FLOAT_EQ(cpu::quantizeScalar(3.14159F, DType::FP32), 3.14159F);
    EXPECT_FLOAT_EQ(cpu::quantizeScalar(-0.00123F, DType::FP32), -0.00123F);
}

TEST(QuantizeTest, ZeroRestaZeroPerTuttiIFormati) {
    for (DType dtype : {DType::FP32, DType::TF32, DType::BF16, DType::FP16, DType::FP8_E4M3, DType::FP8_E5M2}) {
        EXPECT_FLOAT_EQ(cpu::quantizeScalar(0.0F, dtype), 0.0F);
    }
}

TEST(QuantizeTest, UnoENonNegativiCiSonoSempreRappresentabiliEsattamente) {
    // 1.0 = 1.000...0 * 2^0: la mantissa e' tutta zero, quindi
    // rappresentabile esattamente in qualunque formato, per quanto
    // stretto (anche fp8 con 2 soli bit di mantissa).
    for (DType dtype : {DType::FP32, DType::TF32, DType::BF16, DType::FP16, DType::FP8_E4M3, DType::FP8_E5M2}) {
        EXPECT_FLOAT_EQ(cpu::quantizeScalar(1.0F, dtype), 1.0F) << "dtype index " << static_cast<int>(dtype);
        EXPECT_FLOAT_EQ(cpu::quantizeScalar(-1.0F, dtype), -1.0F) << "dtype index " << static_cast<int>(dtype);
        EXPECT_FLOAT_EQ(cpu::quantizeScalar(2.0F, dtype), 2.0F) << "dtype index " << static_cast<int>(dtype);
    }
}

TEST(QuantizeTest, InfinitoENanPassanoInvariati) {
    float inf = std::numeric_limits<float>::infinity();
    float nan = std::numeric_limits<float>::quiet_NaN();
    for (DType dtype : {DType::BF16, DType::FP16, DType::FP8_E4M3, DType::FP8_E5M2, DType::TF32}) {
        EXPECT_EQ(cpu::quantizeScalar(inf, dtype), inf);
        EXPECT_TRUE(std::isnan(cpu::quantizeScalar(nan, dtype)));
    }
}

TEST(QuantizeTest, Bf16ArrotondaLaMantissaAMenoBit) {
    // Un valore con molti bit di mantissa significativi deve cambiare
    // dopo l'arrotondamento a bf16 (7 bit di mantissa), ma restare
    // vicino all'originale (bf16 ha comunque lo stesso range di fp32).
    float value = 1.0F / 3.0F;  // 0.333... non rappresentabile esattamente in nessun formato binario
    float quantized = cpu::quantizeScalar(value, DType::BF16);

    EXPECT_NE(quantized, value);
    EXPECT_NEAR(quantized, value, 0.01F);
}

TEST(QuantizeTest, Bf16EIdempotente) {
    float value = 12345.6789F;
    float once = cpu::quantizeScalar(value, DType::BF16);
    float twice = cpu::quantizeScalar(once, DType::BF16);
    EXPECT_FLOAT_EQ(once, twice);
}

TEST(QuantizeTest, Fp8SaturaOltreIlMassimoRappresentabile) {
    EXPECT_FLOAT_EQ(cpu::quantizeScalar(1000.0F, DType::FP8_E4M3), 448.0F);
    EXPECT_FLOAT_EQ(cpu::quantizeScalar(-1000.0F, DType::FP8_E4M3), -448.0F);
    EXPECT_FLOAT_EQ(cpu::quantizeScalar(1e6F, DType::FP8_E5M2), 57344.0F);
}

TEST(QuantizeTest, Fp16SaturaOltreIlMassimoRappresentabile) {
    EXPECT_FLOAT_EQ(cpu::quantizeScalar(100000.0F, DType::FP16), 65504.0F);
    EXPECT_FLOAT_EQ(cpu::quantizeScalar(-100000.0F, DType::FP16), -65504.0F);
}

TEST(QuantizeTest, Fp8HaUnaGranaPiuGrossaDiFp16) {
    // Con lo stesso valore non esattamente rappresentabile, l'errore
    // di arrotondamento di fp8 (2-3 bit di mantissa) deve essere
    // maggiore o uguale a quello di fp16 (10 bit di mantissa).
    float value = 100.0F / 7.0F;  // ~14.2857...

    float errFp8 = std::fabs(cpu::quantizeScalar(value, DType::FP8_E4M3) - value);
    float errFp16 = std::fabs(cpu::quantizeScalar(value, DType::FP16) - value);

    EXPECT_GT(errFp8, errFp16);
}

TEST(QuantizeTest, TensorApplicaLArrotondamentoElementoPerElementoEPreservaLaForma) {
    Tensor input({2, 2}, {1.0F / 3.0F, 2.0F / 3.0F, -1.5F, 4.0F});
    Tensor result = cpu::quantize(input, DType::BF16);

    EXPECT_EQ(result.shape(), input.shape());
    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_FLOAT_EQ(result.at(i), cpu::quantizeScalar(input.at(i), DType::BF16));
    }
}

TEST(QuantizeTest, TensorConFp32RestituisceValoriIdentici) {
    Tensor input({3}, {1.23456F, -7.89F, 0.0F});
    Tensor result = cpu::quantize(input, DType::FP32);
    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_FLOAT_EQ(result.at(i), input.at(i));
    }
}
