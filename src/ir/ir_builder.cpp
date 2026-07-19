#include "blackforge/ir/ir_builder.hpp"

#include <stdexcept>

#include "blackforge/sema/dtype.hpp"

namespace blackforge::ir {

namespace {

Dim toIRDim(const ast::ShapeDim& dim) {
    if (dim.isSymbolic) {
        return Dim{true, dim.symbolicName, 0};
    }
    return Dim{false, "", dim.literalValue};
}

// Legge stage.args[index] come intero letterale positivo. Ritorna
// nullopt (avendo gia' segnalato l'errore) se l'argomento non e' un
// intero letterale positivo valido: la stessa validazione di forma e'
// gia' fatta dall'analisi semantica, ma ir_builder puo' essere invocato
// anche indipendentemente da essa (es. nei test), quindi ripete il
// controllo qui invece di assumerlo.
std::optional<long long> readPositiveIntArg(const ast::PipelineStage& stage, std::size_t index,
                                             DiagnosticList& diagnostics) {
    const ast::Expr& arg = stage.args.at(index);
    if (arg.kind != ast::ExprKind::IntegerLiteral) {
        diagnostics.addError(arg.location, "'" + stage.name + "' richiede argomenti interi");
        return std::nullopt;
    }
    long long value = 0;
    try {
        value = std::stoll(arg.text);
    } catch (const std::out_of_range&) {
        diagnostics.addError(arg.location, "valore intero troppo grande per '" + stage.name + "': " + arg.text);
        return std::nullopt;
    }
    if (value <= 0) {
        diagnostics.addError(arg.location, "'" + stage.name + "' richiede argomenti interi positivi, trovato " +
                                                arg.text);
        return std::nullopt;
    }
    return value;
}

}  // namespace

Module IRBuilder::build(const ast::Program& program) {
    Module module;

    for (const auto& decl : program.declarations) {
        if (std::holds_alternative<ast::TargetDecl>(decl)) {
            const auto& target = std::get<ast::TargetDecl>(decl);
            if (module.target.has_value()) {
                diagnostics_.addError(target.location, "dichiarazione 'target' duplicata");
            } else {
                module.target = target.target.toString();
            }
        } else if (std::holds_alternative<ast::PrecisionDecl>(decl)) {
            const auto& precisionDecl = std::get<ast::PrecisionDecl>(decl);
            if (module.precision.has_value()) {
                diagnostics_.addError(precisionDecl.location, "blocco 'precision' duplicato");
                continue;
            }

            PrecisionPolicy policy;
            for (const auto& field : precisionDecl.fields) {
                std::optional<sema::DType> dtype = sema::parseDType(field.value);
                if (!dtype.has_value()) {
                    // Formato sconosciuto: gia' segnalato dall'analisi
                    // semantica, qui si ignora silenziosamente il campo
                    // (mantiene fp32, il default sicuro).
                    continue;
                }
                switch (field.kind) {
                    case ast::PrecisionFieldKind::Storage:
                    case ast::PrecisionFieldKind::Parameters:
                        policy.storage = *dtype;
                        break;
                    case ast::PrecisionFieldKind::Compute:
                    case ast::PrecisionFieldKind::Forward:
                        policy.compute = *dtype;
                        break;
                    case ast::PrecisionFieldKind::Accumulate:
                        policy.accumulate = *dtype;
                        break;
                    case ast::PrecisionFieldKind::Backward:
                        // La precisione del backward pass non e' ancora
                        // gestita separatamente da quella del forward:
                        // l'autodiff opera sempre alla precisione delle
                        // attivazioni memorizzate (storage) e accumula
                        // sempre in fp32.
                        break;
                }
            }
            module.precision = policy;
        }
    }

    for (const auto& decl : program.declarations) {
        if (std::holds_alternative<ast::ModelDecl>(decl)) {
            std::optional<ModelIR> model = buildModel(std::get<ast::ModelDecl>(decl));
            if (model.has_value()) {
                module.models.push_back(std::move(*model));
            }
        }
    }

    return module;
}

std::optional<ModelIR> IRBuilder::buildModel(const ast::ModelDecl& decl) {
    const ast::InputDecl* inputDecl = nullptr;
    for (const auto& statement : decl.body) {
        if (std::holds_alternative<ast::InputDecl>(statement)) {
            if (inputDecl != nullptr) {
                diagnostics_.addError(std::get<ast::InputDecl>(statement).location,
                                       "il modello '" + decl.name + "' dichiara piu' di un input");
                continue;
            }
            inputDecl = &std::get<ast::InputDecl>(statement);
        }
    }

    if (inputDecl == nullptr) {
        diagnostics_.addError(decl.location, "il modello '" + decl.name + "' non dichiara alcun input: impossibile "
                                              "costruire la rappresentazione interna");
        return std::nullopt;
    }

    std::optional<sema::DType> dtype = sema::parseDType(inputDecl->type.dtype);
    if (!dtype.has_value()) {
        diagnostics_.addError(inputDecl->type.dtype.location,
                               "formato numerico sconosciuto '" + inputDecl->type.dtype.toString() + "'");
        return std::nullopt;
    }

    if (inputDecl->type.shape.empty()) {
        diagnostics_.addError(inputDecl->type.location, "l'input del modello '" + decl.name +
                                                              "' deve avere almeno una dimensione");
        return std::nullopt;
    }

    ModelIR model;
    model.name = decl.name;

    std::vector<Dim> inputShape;
    inputShape.reserve(inputDecl->type.shape.size());
    for (const auto& dim : inputDecl->type.shape) {
        inputShape.push_back(toIRDim(dim));
    }

    model.inputValue = 0;
    model.values.push_back(Value{0, *dtype, std::move(inputShape)});

    std::size_t nextValueId = 1;
    for (const auto& statement : decl.body) {
        if (std::holds_alternative<ast::PipelineStmt>(statement)) {
            buildPipeline(std::get<ast::PipelineStmt>(statement), model, nextValueId);
        }
    }

    return model;
}

void IRBuilder::buildPipeline(const ast::PipelineStmt& stmt, ModelIR& model, std::size_t& nextValueId) {
    if (stmt.source.kind != ast::PipelineSourceKind::Input) {
        diagnostics_.addError(stmt.source.location, "identificatore '" + stmt.source.identifierName +
                                                          "' non definito: i binding con nome per risultati "
                                                          "intermedi non sono ancora supportati");
        return;
    }

    Pipeline pipeline;
    pipeline.sourceValue = model.inputValue;

    std::size_t currentValueId = model.inputValue;

    for (const auto& stage : stmt.stages) {
        std::optional<OpKind> kind = parseOpKind(stage.name);
        if (!kind.has_value()) {
            diagnostics_.addError(stage.location, "operazione sconosciuta '" + stage.name + "'");
            return;
        }

        const Value& current = model.valueById(currentValueId);
        std::vector<Dim> outputShape = current.shape;
        long long linearOutFeatures = 0;
        long long embeddingVocabSize = 0;
        long long embeddingDim = 0;
        long long positionalMaxSeqLen = 0;
        long long attentionNumHeads = 0;
        long long feedForwardHiddenDim = 0;

        if (*kind == OpKind::Linear) {
            if (stage.args.size() != 1 || stage.args[0].kind != ast::ExprKind::IntegerLiteral) {
                diagnostics_.addError(stage.location, "'linear' richiede un singolo argomento intero");
                return;
            }
            try {
                linearOutFeatures = std::stoll(stage.args[0].text);
            } catch (const std::out_of_range&) {
                diagnostics_.addError(stage.args[0].location, "valore intero troppo grande per 'linear'");
                return;
            }
            if (linearOutFeatures <= 0) {
                diagnostics_.addError(stage.args[0].location, "'linear' richiede un numero di feature positivo");
                return;
            }
            if (outputShape.empty()) {
                diagnostics_.addError(stage.location,
                                       "'linear' non puo' essere applicato a un tensore senza dimensioni");
                return;
            }
            // Un layer lineare rimpiazza l'ultima dimensione (le feature)
            // con il numero di feature in output; le altre dimensioni
            // (es. 'batch') sono preservate cosi' come sono.
            outputShape.back() = Dim{false, "", linearOutFeatures};
        } else if (*kind == OpKind::Embedding) {
            if (stage.args.size() != 2) {
                diagnostics_.addError(stage.location, "'embedding' richiede due argomenti interi (vocabolario, dim)");
                return;
            }
            std::optional<long long> vocabSize = readPositiveIntArg(stage, 0, diagnostics_);
            std::optional<long long> dim = readPositiveIntArg(stage, 1, diagnostics_);
            if (!vocabSize.has_value() || !dim.has_value()) {
                return;
            }
            embeddingVocabSize = *vocabSize;
            embeddingDim = *dim;
            // Il lookup di embedding aggiunge una nuova dimensione finale
            // (dim): [batch, seq] token id -> [batch, seq, dim].
            outputShape.push_back(Dim{false, "", embeddingDim});
        } else if (*kind == OpKind::PositionalEmbedding) {
            if (stage.args.size() != 1) {
                diagnostics_.addError(stage.location, "'positional_embedding' richiede un singolo argomento intero "
                                                        "(lunghezza massima di sequenza)");
                return;
            }
            std::optional<long long> maxSeqLen = readPositiveIntArg(stage, 0, diagnostics_);
            if (!maxSeqLen.has_value()) {
                return;
            }
            positionalMaxSeqLen = *maxSeqLen;
            // Additivo, stessa forma dell'ingresso (la dimensione 'dim'
            // usata per la tabella e' dedotta a runtime dall'ultima
            // dimensione dell'ingresso, non e' un argomento separato).
        } else if (*kind == OpKind::Attention) {
            if (stage.args.size() != 1) {
                diagnostics_.addError(stage.location, "'attention' richiede un singolo argomento intero (numero di "
                                                        "teste)");
                return;
            }
            std::optional<long long> numHeads = readPositiveIntArg(stage, 0, diagnostics_);
            if (!numHeads.has_value()) {
                return;
            }
            attentionNumHeads = *numHeads;
            // Pre-norm + residual interni all'operazione: forma invariata.
        } else if (*kind == OpKind::FeedForward) {
            if (stage.args.size() != 1) {
                diagnostics_.addError(stage.location, "'feedforward' richiede un singolo argomento intero "
                                                        "(dimensione nascosta)");
                return;
            }
            std::optional<long long> hiddenDim = readPositiveIntArg(stage, 0, diagnostics_);
            if (!hiddenDim.has_value()) {
                return;
            }
            feedForwardHiddenDim = *hiddenDim;
            // Pre-norm + residual interni all'operazione: forma invariata.
        }
        // silu/relu/gelu/rmsnorm/softmax non cambiano la forma ne' il
        // formato (rmsnorm/softmax operano lungo l'ultima dimensione a
        // runtime, ma non la alterano).

        std::size_t outputId = nextValueId++;
        model.values.push_back(Value{outputId, current.dtype, outputShape});

        Operation operation;
        operation.kind = *kind;
        operation.input = currentValueId;
        operation.output = outputId;
        operation.linearOutFeatures = linearOutFeatures;
        operation.embeddingVocabSize = embeddingVocabSize;
        operation.embeddingDim = embeddingDim;
        operation.positionalMaxSeqLen = positionalMaxSeqLen;
        operation.attentionNumHeads = attentionNumHeads;
        operation.feedForwardHiddenDim = feedForwardHiddenDim;
        pipeline.operations.push_back(operation);

        currentValueId = outputId;
    }

    model.pipelines.push_back(std::move(pipeline));
}

}  // namespace blackforge::ir
