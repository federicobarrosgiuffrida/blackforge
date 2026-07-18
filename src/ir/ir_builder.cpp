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
        }
        // silu/relu/gelu sono elementwise: forma e formato invariati.

        std::size_t outputId = nextValueId++;
        model.values.push_back(Value{outputId, current.dtype, outputShape});

        Operation operation;
        operation.kind = *kind;
        operation.input = currentValueId;
        operation.output = outputId;
        operation.linearOutFeatures = linearOutFeatures;
        pipeline.operations.push_back(operation);

        currentValueId = outputId;
    }

    model.pipelines.push_back(std::move(pipeline));
}

}  // namespace blackforge::ir
