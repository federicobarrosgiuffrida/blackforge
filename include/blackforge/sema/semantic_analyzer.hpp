#pragma once

#include <set>
#include <string>

#include "blackforge/ast/ast.hpp"
#include "blackforge/diagnostics/diagnostic.hpp"

namespace blackforge::sema {

// Analizzatore semantico di BlackForge.
//
// Valida un Program gia' analizzato sintatticamente: target hardware
// riconosciuto, formati numerici validi nel blocco precision, forme
// tensoriali con dimensioni positive, operazioni di pipeline note con
// il numero di argomenti corretto. Non modifica l'AST: produce solo
// diagnostiche, con recupero (un errore non ferma l'analisi del resto
// del programma).
//
// NOTA: l'inferenza completa delle forme lungo la pipeline (es. capire
// che l'output di 'linear(4096)' ha effettivamente forma [batch, 4096])
// richiede la rappresentazione interna (IR) e non e' ancora
// implementata qui: questa fase controlla solo la validita' locale
// delle forme dichiarate esplicitamente.
class SemanticAnalyzer {
public:
    void analyze(const ast::Program& program);

    [[nodiscard]] const DiagnosticList& diagnostics() const { return diagnostics_; }

private:
    void analyzeTarget(const ast::TargetDecl& decl);
    void analyzePrecision(const ast::PrecisionDecl& decl);
    void analyzeModel(const ast::ModelDecl& decl);
    void analyzeDataset(const ast::DatasetDecl& decl);
    void analyzeTrain(const ast::TrainDecl& decl);
    void analyzeForecast(const ast::ForecastDecl& decl);

    void analyzeTensorType(const ast::TensorType& type);
    void analyzePipelineStmt(const ast::PipelineStmt& stmt, bool modelHasInput, const std::string& modelName);
    void analyzeStage(const ast::PipelineStage& stage);

    DiagnosticList diagnostics_;
    int targetCount_ = 0;
    int precisionCount_ = 0;
    std::set<std::string> modelNames_;
    std::set<std::string> datasetNames_;
};

}  // namespace blackforge::sema
