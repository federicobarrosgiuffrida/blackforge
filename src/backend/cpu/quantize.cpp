#include "blackforge/backend/cpu/quantize.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace blackforge::backend::cpu {

namespace {

// bf16: stessi 8 bit di esponente di fp32, mantissa troncata a 7 bit.
// Un bf16 e' semplicemente i 16 bit alti di un float32: arrotondare a
// bf16 significa arrotondare (round-to-nearest-even) quei 16 bit alti
// e azzerare i 16 bassi. E' l'algoritmo standard usato da TensorFlow,
// PyTorch ed Eigen per questa conversione.
float roundToBf16(float value) {
    if (!std::isfinite(value)) {
        return value;
    }

    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    std::uint32_t lsb = (bits >> 16) & 1U;
    std::uint32_t roundingBias = 0x00007FFFU + lsb;
    std::uint32_t rounded = (bits + roundingBias) & 0xFFFF0000U;

    float result = 0.0F;
    std::memcpy(&result, &rounded, sizeof(result));
    return result;
}

// Arrotonda 'value' a 'mantissaBits' bit di mantissa (oltre al bit
// implicito), saturando a +-maxMagnitude se il valore supera il range
// rappresentabile. Usato per fp16/tf32/fp8: formati per cui non si
// implementa qui una conversione bit-esatta (vedi nota in quantize.hpp).
float roundToMantissaBits(float value, int mantissaBits, float maxMagnitude) {
    if (value == 0.0F || !std::isfinite(value)) {
        return value;
    }

    float sign = value < 0.0F ? -1.0F : 1.0F;
    float magnitude = std::fabs(value);

    if (magnitude > maxMagnitude) {
        return sign * maxMagnitude;
    }

    int exponent = 0;
    float mantissa = std::frexp(magnitude, &exponent);  // magnitude == mantissa * 2^exponent, mantissa in [0.5, 1)

    // Porta la mantissa in [1, 2) (il bit implicito dei formati in
    // virgola mobile) prima di arrotondarla a 'mantissaBits' bit di
    // frazione.
    float normalizedMantissa = mantissa * 2.0F;
    float scale = static_cast<float>(1U << mantissaBits);
    float roundedMantissa = std::round(normalizedMantissa * scale) / scale;

    // L'arrotondamento puo' far scattare la mantissa a 2.0 esatto
    // (es. 1.111...1 arrotondato per eccesso): si rinormalizza
    // incrementando l'esponente.
    if (roundedMantissa >= 2.0F) {
        roundedMantissa *= 0.5F;
        exponent += 1;
    }

    float result = sign * std::ldexp(roundedMantissa * 0.5F, exponent);
    return std::fabs(result) > maxMagnitude ? sign * maxMagnitude : result;
}

}  // namespace

float quantizeScalar(float value, sema::DType dtype) {
    switch (dtype) {
        case sema::DType::FP32:
            return value;
        case sema::DType::TF32:
            // Stesso range di fp32 (8 bit di esponente), mantissa a 10
            // bit: si usa il massimo rappresentabile di fp32 come
            // limite di saturazione (di fatto mai raggiunto in
            // pratica).
            return roundToMantissaBits(value, 10, 3.4e38F);
        case sema::DType::BF16:
            return roundToBf16(value);
        case sema::DType::FP16:
            return roundToMantissaBits(value, 10, 65504.0F);
        case sema::DType::FP8_E4M3:
            return roundToMantissaBits(value, 3, 448.0F);
        case sema::DType::FP8_E5M2:
            return roundToMantissaBits(value, 2, 57344.0F);
    }
    return value;
}

runtime::Tensor quantize(const runtime::Tensor& input, sema::DType dtype) {
    if (dtype == sema::DType::FP32) {
        return input;  // identita': evita una copia+giro inutile
    }

    std::vector<float> result(input.elementCount());
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = quantizeScalar(input.at(i), dtype);
    }
    return runtime::Tensor(input.shape(), std::move(result));
}

}  // namespace blackforge::backend::cpu
