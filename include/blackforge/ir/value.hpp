#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "blackforge/sema/dtype.hpp"

namespace blackforge::ir {

// Una dimensione nella IR: intera concreta o simbolica (es. "batch"),
// mutuata dalla forma dichiarata nell'input del modello.
struct Dim {
    bool isSymbolic;
    std::string symbolicName;
    long long literalValue = 0;

    [[nodiscard]] std::string toString() const;
};

// Un valore tensoriale nel grafo IR: formato numerico + forma.
// Identificato da un id univoco all'interno del ModelIR che lo
// contiene (0 e' sempre l'input del modello).
struct Value {
    std::size_t id;
    sema::DType dtype;
    std::vector<Dim> shape;
};

// Rappresentazione testuale del tipo di un valore, es. "bf16[batch, 4096]".
std::string typeString(const Value& value);

}  // namespace blackforge::ir
