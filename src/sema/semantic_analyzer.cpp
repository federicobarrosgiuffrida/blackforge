#include "blackforge/sema/semantic_analyzer.hpp"

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "blackforge/sema/dtype.hpp"

namespace blackforge::sema {

namespace {

const std::unordered_set<std::string>& knownTargets() {
    static const std::unordered_set<std::string> targets = {
        "nvidia.blackwell", "nvidia.hopper", "nvidia.ampere", "nvidia.ada", "cpu",
    };
    return targets;
}

// Operazioni di pipeline riconosciute e numero di argomenti attesi.
// L'insieme e' volutamente piccolo: crescera' insieme al backend che le
// implementa davvero (milestone del backend CPU/CUDA).
const std::unordered_map<std::string, int>& knownOperations() {
    static const std::unordered_map<std::string, int> operations = {
        {"linear", 1},
        {"silu", 0},
        {"relu", 0},
        {"gelu", 0},
    };
    return operations;
}

std::string precisionFieldName(ast::PrecisionFieldKind kind) {
    switch (kind) {
        case ast::PrecisionFieldKind::Storage: return "storage";
        case ast::PrecisionFieldKind::Compute: return "compute";
        case ast::PrecisionFieldKind::Accumulate: return "accumulate";
        case ast::PrecisionFieldKind::Parameters: return "parameters";
        case ast::PrecisionFieldKind::Forward: return "forward";
        case ast::PrecisionFieldKind::Backward: return "backward";
    }
    return "?";
}

bool isStorageField(ast::PrecisionFieldKind kind) {
    return kind == ast::PrecisionFieldKind::Storage || kind == ast::PrecisionFieldKind::Parameters;
}

}  // namespace

void SemanticAnalyzer::analyze(const ast::Program& program) {
    for (const auto& decl : program.declarations) {
        std::visit(
            [&](const auto& node) {
                using T = std::decay_t<decltype(node)>;
                if constexpr (std::is_same_v<T, ast::TargetDecl>) {
                    analyzeTarget(node);
                } else if constexpr (std::is_same_v<T, ast::PrecisionDecl>) {
                    analyzePrecision(node);
                } else if constexpr (std::is_same_v<T, ast::ModelDecl>) {
                    analyzeModel(node);
                }
            },
            decl);
    }
}

void SemanticAnalyzer::analyzeTarget(const ast::TargetDecl& decl) {
    ++targetCount_;
    if (targetCount_ > 1) {
        diagnostics_.addError(decl.location, "dichiarazione 'target' duplicata: un programma puo' averne una sola");
    }

    if (knownTargets().find(decl.target.toString()) == knownTargets().end()) {
        diagnostics_.addError(decl.target.location,
                               "target sconosciuto '" + decl.target.toString() +
                                   "'. Target supportati: nvidia.blackwell, nvidia.hopper, nvidia.ampere, "
                                   "nvidia.ada, cpu");
    }
}

void SemanticAnalyzer::analyzePrecision(const ast::PrecisionDecl& decl) {
    ++precisionCount_;
    if (precisionCount_ > 1) {
        diagnostics_.addError(decl.location, "blocco 'precision' duplicato: un programma puo' averne uno solo");
    }

    std::set<ast::PrecisionFieldKind> seen;
    for (const auto& field : decl.fields) {
        if (!seen.insert(field.kind).second) {
            diagnostics_.addError(field.location,
                                   "campo '" + precisionFieldName(field.kind) + "' duplicato nel blocco 'precision'");
        }

        std::optional<DType> dtype = parseDType(field.value);
        if (!dtype.has_value()) {
            diagnostics_.addError(field.value.location, "formato numerico sconosciuto '" + field.value.toString() +
                                                              "'. Formati supportati: fp8.e4m3, fp8.e5m2, fp16, "
                                                              "bf16, tf32, fp32");
            continue;
        }

        if (isStorageField(field.kind) && !isValidForStorage(*dtype)) {
            diagnostics_.addError(field.value.location,
                                   "'" + dtypeName(*dtype) + "' non e' un formato di memorizzazione valido per '" +
                                       precisionFieldName(field.kind) +
                                       "': tf32 e' solo una modalita' di calcolo, non un formato di storage");
        }
    }
}

void SemanticAnalyzer::analyzeTensorType(const ast::TensorType& type) {
    if (!parseDType(type.dtype).has_value()) {
        diagnostics_.addError(type.dtype.location, "formato numerico sconosciuto '" + type.dtype.toString() +
                                                         "'. Formati supportati: fp8.e4m3, fp8.e5m2, fp16, bf16, "
                                                         "tf32, fp32");
    }

    if (type.shape.empty()) {
        diagnostics_.addError(type.location, "un tensore deve avere almeno una dimensione");
    }

    for (const auto& dim : type.shape) {
        if (!dim.isSymbolic && dim.literalValue <= 0) {
            diagnostics_.addError(dim.location,
                                   "dimensione del tensore non valida: " + std::to_string(dim.literalValue) +
                                       " (deve essere un intero positivo)");
        }
    }
}

void SemanticAnalyzer::analyzeStage(const ast::PipelineStage& stage) {
    auto it = knownOperations().find(stage.name);
    if (it == knownOperations().end()) {
        diagnostics_.addError(stage.location, "operazione sconosciuta '" + stage.name +
                                                   "'. Operazioni supportate: linear, silu, relu, gelu");
        return;
    }

    int expectedArgs = it->second;
    if (static_cast<int>(stage.args.size()) != expectedArgs) {
        diagnostics_.addError(stage.location, "'" + stage.name + "' richiede " + std::to_string(expectedArgs) +
                                                   " argomento/i, trovati " + std::to_string(stage.args.size()));
        return;
    }

    if (stage.name == "linear") {
        const ast::Expr& arg = stage.args.front();
        if (arg.kind != ast::ExprKind::IntegerLiteral) {
            diagnostics_.addError(arg.location, "'linear' richiede un numero intero di feature in output, trovato "
                                                     "un " +
                                                     std::string(arg.kind == ast::ExprKind::FloatLiteral ? "float"
                                                                 : arg.kind == ast::ExprKind::StringLiteral
                                                                     ? "stringa"
                                                                     : "identificatore"));
        } else {
            try {
                if (std::stoll(arg.text) <= 0) {
                    diagnostics_.addError(arg.location,
                                           "'linear' richiede un numero di feature in output positivo, trovato " +
                                               arg.text);
                }
            } catch (const std::out_of_range&) {
                diagnostics_.addError(arg.location, "valore intero troppo grande per 'linear': " + arg.text);
            }
        }
    }
}

void SemanticAnalyzer::analyzePipelineStmt(const ast::PipelineStmt& stmt, bool modelHasInput,
                                            const std::string& modelName) {
    if (stmt.source.kind == ast::PipelineSourceKind::Input) {
        if (!modelHasInput) {
            diagnostics_.addError(stmt.source.location, "la pipeline usa 'input' ma il modello '" + modelName +
                                                              "' non dichiara alcun input");
        }
    } else {
        // Il linguaggio non supporta ancora binding con nome per i
        // risultati intermedi: al momento l'unica sorgente valida di
        // una pipeline e' la parola chiave 'input'.
        diagnostics_.addError(stmt.source.location, "identificatore '" + stmt.source.identifierName +
                                                          "' non definito: i binding con nome per risultati "
                                                          "intermedi non sono ancora supportati, usa 'input'");
    }

    for (const auto& stage : stmt.stages) {
        analyzeStage(stage);
    }
}

void SemanticAnalyzer::analyzeModel(const ast::ModelDecl& decl) {
    if (!modelNames_.insert(decl.name).second) {
        diagnostics_.addError(decl.location, "modello '" + decl.name + "' dichiarato piu' volte");
    }

    int inputCount = 0;
    for (const auto& statement : decl.body) {
        if (std::holds_alternative<ast::InputDecl>(statement)) {
            ++inputCount;
            if (inputCount > 1) {
                diagnostics_.addError(std::get<ast::InputDecl>(statement).location,
                                       "il modello '" + decl.name + "' dichiara piu' di un input");
            }
            analyzeTensorType(std::get<ast::InputDecl>(statement).type);
        }
    }

    bool hasInput = inputCount > 0;
    for (const auto& statement : decl.body) {
        if (std::holds_alternative<ast::PipelineStmt>(statement)) {
            analyzePipelineStmt(std::get<ast::PipelineStmt>(statement), hasInput, decl.name);
        }
    }

    if (!hasInput) {
        diagnostics_.addError(decl.location, "il modello '" + decl.name + "' non dichiara alcun input");
    }
}

}  // namespace blackforge::sema
