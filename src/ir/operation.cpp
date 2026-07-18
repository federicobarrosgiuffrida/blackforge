#include "blackforge/ir/operation.hpp"

#include <unordered_map>

namespace blackforge::ir {

namespace {

const std::unordered_map<std::string, OpKind>& opKindTable() {
    static const std::unordered_map<std::string, OpKind> table = {
        {"linear", OpKind::Linear},
        {"silu", OpKind::Silu},
        {"relu", OpKind::Relu},
        {"gelu", OpKind::Gelu},
    };
    return table;
}

}  // namespace

std::optional<OpKind> parseOpKind(const std::string& name) {
    auto it = opKindTable().find(name);
    if (it == opKindTable().end()) {
        return std::nullopt;
    }
    return it->second;
}

std::string opKindName(OpKind kind) {
    switch (kind) {
        case OpKind::Linear: return "linear";
        case OpKind::Silu: return "silu";
        case OpKind::Relu: return "relu";
        case OpKind::Gelu: return "gelu";
    }
    return "?";
}

}  // namespace blackforge::ir
