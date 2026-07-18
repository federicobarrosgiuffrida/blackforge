#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace blackforge::ir {

// Operazioni tensoriali riconosciute dalla IR. L'insieme cresce insieme
// al backend che le implementa davvero (milestone CPU/CUDA).
enum class OpKind { Linear, Silu, Relu, Gelu, RmsNorm };

// Risolve il nome testuale di una fase di pipeline (gia' validato
// dall'analisi semantica) nel corrispondente OpKind della IR.
std::optional<OpKind> parseOpKind(const std::string& name);

std::string opKindName(OpKind kind);

// Un nodo del grafo IR: applica 'kind' al valore 'input' e produce il
// valore 'output'. La pipeline BlackForge e' oggi una sequenza lineare
// di nodi, senza diramazioni ne' fusioni.
struct Operation {
    OpKind kind;
    std::size_t input;
    std::size_t output;
    long long linearOutFeatures = 0;  // valido solo se kind == OpKind::Linear
};

}  // namespace blackforge::ir
