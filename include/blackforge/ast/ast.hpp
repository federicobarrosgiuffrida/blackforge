#pragma once

#include <string>
#include <variant>
#include <vector>

#include "blackforge/diagnostics/diagnostic.hpp"

namespace blackforge::ast {

// Nome eventualmente composto da piu' segmenti separati da '.'
// (es. "nvidia.blackwell", "fp8.e4m3"). Non e' ancora risolto
// semanticamente: la risoluzione (target valido? dtype valido?)
// avviene nella fase di analisi semantica.
struct DottedName {
    std::vector<std::string> segments;
    SourceLocation location;

    [[nodiscard]] std::string toString() const;
};

// --- Espressioni argomento (usate negli argomenti degli stadi di pipeline) ---

enum class ExprKind { IntegerLiteral, FloatLiteral, StringLiteral, Identifier };

struct Expr {
    ExprKind kind;
    std::string text;  // testo originale del letterale/identificatore
    SourceLocation location;
};

// --- target ---

struct TargetDecl {
    DottedName target;
    SourceLocation location;
};

// --- precision { ... } ---

enum class PrecisionFieldKind { Storage, Compute, Accumulate, Parameters, Forward, Backward };

struct PrecisionField {
    PrecisionFieldKind kind;
    DottedName value;
    SourceLocation location;
};

struct PrecisionDecl {
    std::vector<PrecisionField> fields;
    SourceLocation location;
};

// --- model { ... } ---

// Una dimensione di forma tensoriale: o un intero concreto o un nome
// simbolico (es. "batch"), risolto in fase di analisi semantica.
struct ShapeDim {
    bool isSymbolic;
    std::string symbolicName;    // valido se isSymbolic
    long long literalValue = 0;  // valido se !isSymbolic
    SourceLocation location;
};

struct TensorType {
    DottedName dtype;
    std::vector<ShapeDim> shape;
    SourceLocation location;
};

struct InputDecl {
    TensorType type;
    SourceLocation location;
};

struct PipelineStage {
    std::string name;
    std::vector<Expr> args;
    SourceLocation location;
};

enum class PipelineSourceKind { Input, Identifier };

// Sorgente di una pipeline: la parola chiave 'input' oppure il nome di
// un risultato intermedio dichiarato in precedenza nel modello.
struct PipelineSource {
    PipelineSourceKind kind;
    std::string identifierName;  // valido se kind == Identifier
    SourceLocation location;
};

struct PipelineStmt {
    PipelineSource source;
    std::vector<PipelineStage> stages;
    SourceLocation location;
};

using ModelStatement = std::variant<InputDecl, PipelineStmt>;

struct ModelDecl {
    std::string name;
    std::vector<ModelStatement> body;
    SourceLocation location;
};

// --- dataset { ... } ---

struct DatasetPathField {
    std::string path;
    SourceLocation location;
};

struct DatasetInputField {
    TensorType type;
    SourceLocation location;
};

struct DatasetLabelsField {
    TensorType type;
    SourceLocation location;
};

using DatasetField = std::variant<DatasetPathField, DatasetInputField, DatasetLabelsField>;

struct DatasetDecl {
    std::string name;
    std::vector<DatasetField> fields;
    SourceLocation location;
};

// --- train { ... } ---

struct TrainModelField {
    std::string name;
    SourceLocation location;
};

struct TrainDatasetField {
    std::string name;
    SourceLocation location;
};

struct TrainLossField {
    std::string name;  // es. "mse"
    SourceLocation location;
};

struct TrainOptimizerField {
    std::string name;  // es. "sgd", "adamw"
    SourceLocation location;
};

struct TrainEpochsField {
    long long value;
    SourceLocation location;
};

struct TrainBatchSizeField {
    long long value;
    SourceLocation location;
};

struct TrainLearningRateField {
    double value;
    SourceLocation location;
};

// 'lr_schedule cosine': fa decadere il learning_rate lungo le epoche
// secondo la formula standard di cosine annealing (Loshchilov & Hutter,
// "SGDR", 2016), invece di restare costante. Opzionale: se assente, il
// learning rate resta costante per l'intero addestramento (comportamento
// invariato rispetto a prima).
struct TrainLrScheduleField {
    std::string name;  // es. "cosine"
    SourceLocation location;
};

// 'lora { rank N [alpha F] }' dentro un blocco 'train': quando presente,
// invece di allenare i pesi originali del modello, addestra un adapter
// a basso rango (matrici A/B) applicato a ogni layer 'linear', con i
// pesi originali congelati. Richiede un modello pre-allenato (vedi
// --from-checkpoint nella CLI): non ha senso applicare un adapter a
// pesi casuali.
struct TrainLoraField {
    long long rank;
    double alpha;
    SourceLocation location;
};

using TrainField = std::variant<TrainModelField, TrainDatasetField, TrainLossField, TrainOptimizerField,
                                 TrainEpochsField, TrainBatchSizeField, TrainLearningRateField,
                                 TrainLrScheduleField, TrainLoraField>;

struct TrainDecl {
    std::vector<TrainField> fields;
    SourceLocation location;
};

// --- forecast { ... } ---

struct ForecastModelField {
    std::string name;
    SourceLocation location;
};

struct ForecastHorizonField {
    long long value;
    SourceLocation location;
};

using ForecastField = std::variant<ForecastModelField, ForecastHorizonField>;

// Descrive una sessione di forecasting autoregressivo: a partire da un
// input iniziale, il modello referenziato viene applicato ripetutamente
// 'horizon' volte, usando ogni output come input del passo successivo.
// Richiede che l'ultima dimensione della forma di input e di output del
// modello coincidano (controllato a runtime, non qui: dipende dalla IR).
struct ForecastDecl {
    std::vector<ForecastField> fields;
    SourceLocation location;
};

// --- Programma ---

using Decl = std::variant<TargetDecl, PrecisionDecl, ModelDecl, DatasetDecl, TrainDecl, ForecastDecl>;

struct Program {
    std::vector<Decl> declarations;
};

// Stampa una rappresentazione testuale indentata dell'AST, usata dal
// comando `blackforge check --print-ast` e utile per il debug.
std::string dump(const Program& program);

}  // namespace blackforge::ast
