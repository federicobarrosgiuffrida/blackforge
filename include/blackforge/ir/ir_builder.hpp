#pragma once

#include <optional>

#include "blackforge/ast/ast.hpp"
#include "blackforge/diagnostics/diagnostic.hpp"
#include "blackforge/ir/module.hpp"

namespace blackforge::ir {

// Costruisce la rappresentazione interna (IR) a partire da un Program
// gia' analizzato sintatticamente.
//
// A differenza dell'analisi semantica di base (che valida solo le forme
// dichiarate esplicitamente), l'IRBuilder attraversa davvero le
// pipeline e propaga forma e formato numerico da un'operazione
// all'altra: e' qui che un errore come "linear applicato a un tensore
// senza dimensioni" verrebbe rilevato.
//
// L'IRBuilder e' utilizzabile anche su programmi che l'analisi
// semantica non ha ancora validato: e' difensivo e riporta le proprie
// diagnostiche, ma per un uso normale (CLI) va eseguito dopo
// SemanticAnalyzer.
class IRBuilder {
public:
    Module build(const ast::Program& program);

    [[nodiscard]] const DiagnosticList& diagnostics() const { return diagnostics_; }

private:
    std::optional<ModelIR> buildModel(const ast::ModelDecl& decl);
    void buildPipeline(const ast::PipelineStmt& stmt, ModelIR& model, std::size_t& nextValueId);

    DiagnosticList diagnostics_;
};

}  // namespace blackforge::ir
