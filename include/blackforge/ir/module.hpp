#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "blackforge/ir/operation.hpp"
#include "blackforge/ir/value.hpp"

namespace blackforge::ir {

// Catena di operazioni applicate a partire da un valore sorgente
// (l'input del modello). La pipeline BlackForge e' oggi una sequenza
// lineare: non supporta ancora diramazioni o fusioni multiple.
struct Pipeline {
    std::size_t sourceValue;
    std::vector<Operation> operations;

    // Id del valore finale prodotto dalla pipeline (l'input stesso se
    // la pipeline non ha operazioni).
    [[nodiscard]] std::size_t outputValue() const {
        return operations.empty() ? sourceValue : operations.back().output;
    }
};

// Rappresentazione interna di un singolo modello BlackForge: l'insieme
// dei valori tensoriali (nodi) che appaiono nel suo grafo di calcolo e
// le pipeline che li collegano.
struct ModelIR {
    std::string name;
    std::vector<Value> values;  // values[i].id == i
    std::size_t inputValue = 0;
    std::vector<Pipeline> pipelines;

    [[nodiscard]] const Value& valueById(std::size_t id) const { return values.at(id); }
};

// Rappresentazione interna dell'intero programma: target hardware
// (se dichiarato) e i modelli definiti.
struct Module {
    std::optional<std::string> target;
    std::vector<ModelIR> models;
};

// Stampa una rappresentazione testuale indentata della IR, usata dal
// comando `blackforge check --print-ir` e utile per il debug.
std::string dump(const Module& module);

}  // namespace blackforge::ir
