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
        {"rmsnorm", OpKind::RmsNorm},
        {"softmax", OpKind::Softmax},
        {"embedding", OpKind::Embedding},
        {"positional_embedding", OpKind::PositionalEmbedding},
        {"attention", OpKind::Attention},
        {"feedforward", OpKind::FeedForward},
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
        case OpKind::RmsNorm: return "rmsnorm";
        case OpKind::Softmax: return "softmax";
        case OpKind::Embedding: return "embedding";
        case OpKind::PositionalEmbedding: return "positional_embedding";
        case OpKind::Attention: return "attention";
        case OpKind::FeedForward: return "feedforward";
    }
    return "?";
}

}  // namespace blackforge::ir
