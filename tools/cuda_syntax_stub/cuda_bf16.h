#pragma once
// STUB — vedi cuda_runtime.h.
//
// Il tipo vero e' a 16 bit con 8 bit di esponente e 7 di mantissa. Qui
// interessa solo che occupi 2 byte (i calcoli di sizeof nelle
// allocazioni devono restare corretti) e che le conversioni esistano.
#include <cstdint>

struct __nv_bfloat16 {
    std::uint16_t raw = 0;
};
static_assert(sizeof(__nv_bfloat16) == 2, "stub bf16: deve occupare 2 byte come il tipo vero");

struct __nv_bfloat162 {
    __nv_bfloat16 x, y;
};

inline __nv_bfloat16 __float2bfloat16(float value) {
    std::uint32_t bits = 0;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    return __nv_bfloat16{static_cast<std::uint16_t>(bits >> 16)};
}

inline float __bfloat162float(__nv_bfloat16 value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value.raw) << 16;
    float result = 0.0F;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}
