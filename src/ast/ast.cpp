#include "blackforge/ast/ast.hpp"

#include <sstream>

namespace blackforge::ast {

namespace {

std::string joinSegments(const std::vector<std::string>& segments) {
    std::string result;
    for (std::size_t i = 0; i < segments.size(); ++i) {
        if (i > 0) {
            result += '.';
        }
        result += segments[i];
    }
    return result;
}

std::string precisionFieldKindName(PrecisionFieldKind kind) {
    switch (kind) {
        case PrecisionFieldKind::Storage: return "storage";
        case PrecisionFieldKind::Compute: return "compute";
        case PrecisionFieldKind::Accumulate: return "accumulate";
        case PrecisionFieldKind::Parameters: return "parameters";
        case PrecisionFieldKind::Forward: return "forward";
        case PrecisionFieldKind::Backward: return "backward";
    }
    return "?";
}

std::string exprKindName(ExprKind kind) {
    switch (kind) {
        case ExprKind::IntegerLiteral: return "intero";
        case ExprKind::FloatLiteral: return "float";
        case ExprKind::StringLiteral: return "stringa";
        case ExprKind::Identifier: return "identificatore";
    }
    return "?";
}

void indent(std::ostringstream& out, int depth) {
    for (int i = 0; i < depth; ++i) {
        out << "  ";
    }
}

void dumpShapeDim(std::ostringstream& out, int depth, const ShapeDim& dim) {
    indent(out, depth);
    if (dim.isSymbolic) {
        out << "ShapeDim(simbolica) " << dim.symbolicName << "\n";
    } else {
        out << "ShapeDim(letterale) " << dim.literalValue << "\n";
    }
}

void dumpTensorType(std::ostringstream& out, int depth, const TensorType& type) {
    indent(out, depth);
    out << "TensorType " << type.dtype.toString() << "\n";
    for (const auto& dim : type.shape) {
        dumpShapeDim(out, depth + 1, dim);
    }
}

void dumpExpr(std::ostringstream& out, int depth, const Expr& expr) {
    indent(out, depth);
    out << "Expr(" << exprKindName(expr.kind) << ") " << expr.text << "\n";
}

void dumpInputDecl(std::ostringstream& out, int depth, const InputDecl& decl) {
    indent(out, depth);
    out << "InputDecl\n";
    dumpTensorType(out, depth + 1, decl.type);
}

void dumpPipelineStmt(std::ostringstream& out, int depth, const PipelineStmt& stmt) {
    indent(out, depth);
    out << "PipelineStmt sorgente=";
    if (stmt.source.kind == PipelineSourceKind::Input) {
        out << "input";
    } else {
        out << stmt.source.identifierName;
    }
    out << "\n";

    for (const auto& stage : stmt.stages) {
        indent(out, depth + 1);
        out << "Stage " << stage.name << "\n";
        for (const auto& arg : stage.args) {
            dumpExpr(out, depth + 2, arg);
        }
    }
}

void dumpModelStatement(std::ostringstream& out, int depth, const ModelStatement& statement) {
    std::visit(
        [&](const auto& node) {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, InputDecl>) {
                dumpInputDecl(out, depth, node);
            } else if constexpr (std::is_same_v<T, PipelineStmt>) {
                dumpPipelineStmt(out, depth, node);
            }
        },
        statement);
}

void dumpTargetDecl(std::ostringstream& out, int depth, const TargetDecl& decl) {
    indent(out, depth);
    out << "TargetDecl " << decl.target.toString() << "\n";
}

void dumpPrecisionDecl(std::ostringstream& out, int depth, const PrecisionDecl& decl) {
    indent(out, depth);
    out << "PrecisionDecl\n";
    for (const auto& field : decl.fields) {
        indent(out, depth + 1);
        out << precisionFieldKindName(field.kind) << " = " << field.value.toString() << "\n";
    }
}

void dumpModelDecl(std::ostringstream& out, int depth, const ModelDecl& decl) {
    indent(out, depth);
    out << "ModelDecl " << decl.name << "\n";
    for (const auto& statement : decl.body) {
        dumpModelStatement(out, depth + 1, statement);
    }
}

void dumpDecl(std::ostringstream& out, int depth, const Decl& decl) {
    std::visit(
        [&](const auto& node) {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, TargetDecl>) {
                dumpTargetDecl(out, depth, node);
            } else if constexpr (std::is_same_v<T, PrecisionDecl>) {
                dumpPrecisionDecl(out, depth, node);
            } else if constexpr (std::is_same_v<T, ModelDecl>) {
                dumpModelDecl(out, depth, node);
            }
        },
        decl);
}

}  // namespace

std::string DottedName::toString() const { return joinSegments(segments); }

std::string dump(const Program& program) {
    std::ostringstream out;
    out << "Program\n";
    for (const auto& decl : program.declarations) {
        dumpDecl(out, 1, decl);
    }
    return out.str();
}

}  // namespace blackforge::ast
