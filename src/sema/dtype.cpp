#include "blackforge/sema/dtype.hpp"

#include <unordered_map>

namespace blackforge::sema {

namespace {

const std::unordered_map<std::string, DType>& dtypeTable() {
    static const std::unordered_map<std::string, DType> table = {
        {"fp8.e4m3", DType::FP8_E4M3}, {"fp8.e5m2", DType::FP8_E5M2}, {"fp16", DType::FP16},
        {"bf16", DType::BF16},         {"tf32", DType::TF32},         {"fp32", DType::FP32},
    };
    return table;
}

}  // namespace

std::optional<DType> parseDType(const ast::DottedName& name) {
    auto it = dtypeTable().find(name.toString());
    if (it == dtypeTable().end()) {
        return std::nullopt;
    }
    return it->second;
}

int dtypeSizeInBytes(DType dtype) {
    switch (dtype) {
        case DType::FP8_E4M3: return 1;
        case DType::FP8_E5M2: return 1;
        case DType::FP16: return 2;
        case DType::BF16: return 2;
        case DType::TF32: return 4;
        case DType::FP32: return 4;
    }
    return 0;
}

bool isValidForStorage(DType dtype) { return dtype != DType::TF32; }

std::string dtypeName(DType dtype) {
    switch (dtype) {
        case DType::FP8_E4M3: return "fp8.e4m3";
        case DType::FP8_E5M2: return "fp8.e5m2";
        case DType::FP16: return "fp16";
        case DType::BF16: return "bf16";
        case DType::TF32: return "tf32";
        case DType::FP32: return "fp32";
    }
    return "?";
}

}  // namespace blackforge::sema
