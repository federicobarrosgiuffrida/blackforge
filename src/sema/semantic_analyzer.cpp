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
        {"rmsnorm", 0},
        {"softmax", 0},
        {"embedding", 2},
        {"positional_embedding", 1},
        {"attention", 1},
        {"feedforward", 1},
    };
    return operations;
}

// Vero per le operazioni che richiedono TUTTI i loro argomenti come
// interi positivi (embedding/positional_embedding/attention/feedforward,
// come 'linear'): evita di ripetere lo stesso controllo quattro volte.
bool requiresPositiveIntegerArgs(const std::string& opName) {
    return opName == "linear" || opName == "embedding" || opName == "positional_embedding" ||
           opName == "attention" || opName == "feedforward";
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

// Loss/optimizer implementati dal motore CPU: l'elenco crescera' insieme
// al motore stesso. 'mse' e' pensata per la regressione (forecasting
// incluso); 'cross_entropy' per la classificazione multiclasse (softmax
// applicata internamente, target one-hot o soft della stessa forma
// dell'uscita del modello, vedi loss.hpp); 'cross_entropy_sparse' e'
// matematicamente identica ma con un target di indici di classe invece
// di un vettore one-hot denso (una dimensione in meno rispetto
// all'uscita del modello) — essenziale per vocabolari grandi, dove il
// target denso sprecherebbe memoria proporzionale al vocabolario (vedi
// backend::cpu::softmaxCrossEntropySparse).
const std::unordered_set<std::string>& knownLossNames() {
    static const std::unordered_set<std::string> names = {"mse", "cross_entropy", "cross_entropy_sparse"};
    return names;
}

const std::unordered_set<std::string>& knownOptimizerNames() {
    static const std::unordered_set<std::string> names = {"sgd", "adamw"};
    return names;
}

// Schedule del learning rate riconosciuti. 'cosine' e' cosine
// annealing (Loshchilov & Hutter, "SGDR", 2016): decade da
// learning_rate a 0 seguendo mezza onda di coseno lungo le epoche.
const std::unordered_set<std::string>& knownLrScheduleNames() {
    static const std::unordered_set<std::string> names = {"cosine"};
    return names;
}

}  // namespace

void SemanticAnalyzer::analyze(const ast::Program& program) {
    // 'train' puo' fare riferimento a 'model'/'dataset' dichiarati in
    // qualunque punto del programma: prima si analizzano (e si
    // registrano i nomi di) tutte le altre dichiarazioni, poi si
    // validano i 'train' che le referenziano.
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
                } else if constexpr (std::is_same_v<T, ast::DatasetDecl>) {
                    analyzeDataset(node);
                }
            },
            decl);
    }

    for (const auto& decl : program.declarations) {
        if (std::holds_alternative<ast::TrainDecl>(decl)) {
            analyzeTrain(std::get<ast::TrainDecl>(decl));
        } else if (std::holds_alternative<ast::ForecastDecl>(decl)) {
            analyzeForecast(std::get<ast::ForecastDecl>(decl));
        }
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
                                                   "'. Operazioni supportate: linear, silu, relu, gelu, rmsnorm, "
                                                   "softmax, embedding, positional_embedding, attention, "
                                                   "feedforward");
        return;
    }

    int expectedArgs = it->second;
    if (static_cast<int>(stage.args.size()) != expectedArgs) {
        diagnostics_.addError(stage.location, "'" + stage.name + "' richiede " + std::to_string(expectedArgs) +
                                                   " argomento/i, trovati " + std::to_string(stage.args.size()));
        return;
    }

    if (requiresPositiveIntegerArgs(stage.name)) {
        for (const ast::Expr& arg : stage.args) {
            if (arg.kind != ast::ExprKind::IntegerLiteral) {
                diagnostics_.addError(arg.location, "'" + stage.name +
                                                          "' richiede argomenti interi, trovato un " +
                                                          std::string(arg.kind == ast::ExprKind::FloatLiteral ? "float"
                                                                      : arg.kind == ast::ExprKind::StringLiteral
                                                                          ? "stringa"
                                                                          : "identificatore"));
                continue;
            }
            try {
                if (std::stoll(arg.text) <= 0) {
                    diagnostics_.addError(arg.location, "'" + stage.name +
                                                              "' richiede argomenti interi positivi, trovato " +
                                                              arg.text);
                }
            } catch (const std::out_of_range&) {
                diagnostics_.addError(arg.location,
                                       "valore intero troppo grande per '" + stage.name + "': " + arg.text);
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

void SemanticAnalyzer::analyzeDataset(const ast::DatasetDecl& decl) {
    if (!datasetNames_.insert(decl.name).second) {
        diagnostics_.addError(decl.location, "dataset '" + decl.name + "' dichiarato piu' volte");
    }

    int pathCount = 0;
    int inputCount = 0;
    int labelsCount = 0;

    for (const auto& field : decl.fields) {
        std::visit(
            [&](const auto& node) {
                using T = std::decay_t<decltype(node)>;
                if constexpr (std::is_same_v<T, ast::DatasetPathField>) {
                    ++pathCount;
                    if (pathCount > 1) {
                        diagnostics_.addError(node.location, "campo 'path' duplicato nel dataset '" + decl.name +
                                                                   "'");
                    }
                } else if constexpr (std::is_same_v<T, ast::DatasetInputField>) {
                    ++inputCount;
                    if (inputCount > 1) {
                        diagnostics_.addError(node.location, "campo 'input' duplicato nel dataset '" + decl.name +
                                                                   "'");
                    }
                    analyzeTensorType(node.type);
                } else if constexpr (std::is_same_v<T, ast::DatasetLabelsField>) {
                    ++labelsCount;
                    if (labelsCount > 1) {
                        diagnostics_.addError(node.location, "campo 'labels' duplicato nel dataset '" + decl.name +
                                                                   "'");
                    }
                    analyzeTensorType(node.type);
                }
            },
            field);
    }

    if (pathCount == 0) {
        diagnostics_.addError(decl.location, "il dataset '" + decl.name + "' non dichiara un campo 'path'");
    }
    if (inputCount == 0) {
        diagnostics_.addError(decl.location, "il dataset '" + decl.name + "' non dichiara un campo 'input'");
    }
    if (labelsCount == 0) {
        diagnostics_.addError(decl.location, "il dataset '" + decl.name + "' non dichiara un campo 'labels'");
    }
}

void SemanticAnalyzer::analyzeTrain(const ast::TrainDecl& decl) {
    int modelCount = 0;
    int datasetCount = 0;
    int lossCount = 0;
    int optimizerCount = 0;
    int epochsCount = 0;
    int batchSizeCount = 0;
    int learningRateCount = 0;
    int lrScheduleCount = 0;
    int loraCount = 0;

    for (const auto& field : decl.fields) {
        std::visit(
            [&](const auto& node) {
                using T = std::decay_t<decltype(node)>;
                if constexpr (std::is_same_v<T, ast::TrainModelField>) {
                    ++modelCount;
                    if (modelCount > 1) {
                        diagnostics_.addError(node.location, "campo 'model' duplicato nel blocco 'train'");
                    }
                    if (modelNames_.find(node.name) == modelNames_.end()) {
                        diagnostics_.addError(node.location, "modello '" + node.name + "' non definito");
                    }
                } else if constexpr (std::is_same_v<T, ast::TrainDatasetField>) {
                    ++datasetCount;
                    if (datasetCount > 1) {
                        diagnostics_.addError(node.location, "campo 'dataset' duplicato nel blocco 'train'");
                    }
                    if (datasetNames_.find(node.name) == datasetNames_.end()) {
                        diagnostics_.addError(node.location, "dataset '" + node.name + "' non definito");
                    }
                } else if constexpr (std::is_same_v<T, ast::TrainLossField>) {
                    ++lossCount;
                    if (lossCount > 1) {
                        diagnostics_.addError(node.location, "campo 'loss' duplicato nel blocco 'train'");
                    }
                    if (knownLossNames().find(node.name) == knownLossNames().end()) {
                        diagnostics_.addError(
                            node.location,
                            "loss sconosciuta '" + node.name +
                                "'. Loss supportate: mse, cross_entropy, cross_entropy_sparse");
                    }
                } else if constexpr (std::is_same_v<T, ast::TrainOptimizerField>) {
                    ++optimizerCount;
                    if (optimizerCount > 1) {
                        diagnostics_.addError(node.location, "campo 'optimizer' duplicato nel blocco 'train'");
                    }
                    if (knownOptimizerNames().find(node.name) == knownOptimizerNames().end()) {
                        diagnostics_.addError(
                            node.location, "optimizer sconosciuto '" + node.name + "'. Optimizer supportati: sgd, adamw");
                    }
                } else if constexpr (std::is_same_v<T, ast::TrainEpochsField>) {
                    ++epochsCount;
                    if (epochsCount > 1) {
                        diagnostics_.addError(node.location, "campo 'epochs' duplicato nel blocco 'train'");
                    }
                    if (node.value <= 0) {
                        diagnostics_.addError(node.location, "'epochs' deve essere un intero positivo, trovato " +
                                                                   std::to_string(node.value));
                    }
                } else if constexpr (std::is_same_v<T, ast::TrainBatchSizeField>) {
                    ++batchSizeCount;
                    if (batchSizeCount > 1) {
                        diagnostics_.addError(node.location, "campo 'batch_size' duplicato nel blocco 'train'");
                    }
                    if (node.value <= 0) {
                        diagnostics_.addError(node.location,
                                               "'batch_size' deve essere un intero positivo, trovato " +
                                                   std::to_string(node.value));
                    }
                } else if constexpr (std::is_same_v<T, ast::TrainLearningRateField>) {
                    ++learningRateCount;
                    if (learningRateCount > 1) {
                        diagnostics_.addError(node.location, "campo 'learning_rate' duplicato nel blocco 'train'");
                    }
                    if (node.value <= 0.0) {
                        diagnostics_.addError(node.location, "'learning_rate' deve essere positivo");
                    }
                } else if constexpr (std::is_same_v<T, ast::TrainLrScheduleField>) {
                    ++lrScheduleCount;
                    if (lrScheduleCount > 1) {
                        diagnostics_.addError(node.location, "campo 'lr_schedule' duplicato nel blocco 'train'");
                    }
                    if (knownLrScheduleNames().find(node.name) == knownLrScheduleNames().end()) {
                        diagnostics_.addError(node.location, "schedule sconosciuto '" + node.name +
                                                                   "'. Schedule supportati: cosine");
                    }
                } else if constexpr (std::is_same_v<T, ast::TrainLoraField>) {
                    ++loraCount;
                    if (loraCount > 1) {
                        diagnostics_.addError(node.location, "campo 'lora' duplicato nel blocco 'train'");
                    }
                    if (node.rank <= 0) {
                        diagnostics_.addError(node.location, "'rank' deve essere un intero positivo, trovato " +
                                                                   std::to_string(node.rank));
                    }
                    if (node.alpha <= 0.0) {
                        diagnostics_.addError(node.location, "'alpha' deve essere positivo");
                    }
                }
            },
            field);
    }

    if (modelCount == 0) {
        diagnostics_.addError(decl.location, "il blocco 'train' non specifica un 'model'");
    }
    if (datasetCount == 0) {
        diagnostics_.addError(decl.location, "il blocco 'train' non specifica un 'dataset'");
    }
    if (lossCount == 0) {
        diagnostics_.addError(decl.location, "il blocco 'train' non specifica una 'loss'");
    }
    if (optimizerCount == 0) {
        diagnostics_.addError(decl.location, "il blocco 'train' non specifica un 'optimizer'");
    }
    if (epochsCount == 0) {
        diagnostics_.addError(decl.location, "il blocco 'train' non specifica 'epochs'");
    }
    if (batchSizeCount == 0) {
        diagnostics_.addError(decl.location, "il blocco 'train' non specifica 'batch_size'");
    }
}

void SemanticAnalyzer::analyzeForecast(const ast::ForecastDecl& decl) {
    int modelCount = 0;
    int horizonCount = 0;

    for (const auto& field : decl.fields) {
        std::visit(
            [&](const auto& node) {
                using T = std::decay_t<decltype(node)>;
                if constexpr (std::is_same_v<T, ast::ForecastModelField>) {
                    ++modelCount;
                    if (modelCount > 1) {
                        diagnostics_.addError(node.location, "campo 'model' duplicato nel blocco 'forecast'");
                    }
                    if (modelNames_.find(node.name) == modelNames_.end()) {
                        diagnostics_.addError(node.location, "modello '" + node.name + "' non definito");
                    }
                } else if constexpr (std::is_same_v<T, ast::ForecastHorizonField>) {
                    ++horizonCount;
                    if (horizonCount > 1) {
                        diagnostics_.addError(node.location, "campo 'horizon' duplicato nel blocco 'forecast'");
                    }
                    if (node.value <= 0) {
                        diagnostics_.addError(node.location, "'horizon' deve essere un intero positivo, trovato " +
                                                                   std::to_string(node.value));
                    }
                }
            },
            field);
    }

    if (modelCount == 0) {
        diagnostics_.addError(decl.location, "il blocco 'forecast' non specifica un 'model'");
    }
    if (horizonCount == 0) {
        diagnostics_.addError(decl.location, "il blocco 'forecast' non specifica 'horizon'");
    }
}

}  // namespace blackforge::sema
