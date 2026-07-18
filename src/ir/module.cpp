#include "blackforge/ir/module.hpp"

#include <sstream>

namespace blackforge::ir {

namespace {

void indent(std::ostringstream& out, int depth) {
    for (int i = 0; i < depth; ++i) {
        out << "  ";
    }
}

void dumpOperation(std::ostringstream& out, int depth, const Operation& op, const ModelIR& model) {
    indent(out, depth);
    out << opKindName(op.kind);
    if (op.kind == OpKind::Linear) {
        out << "(" << op.linearOutFeatures << ")";
    }
    out << " -> " << typeString(model.valueById(op.output)) << "  (id=" << op.output << ")\n";
}

void dumpPipeline(std::ostringstream& out, int depth, const Pipeline& pipeline, const ModelIR& model) {
    indent(out, depth);
    out << "pipeline da id=" << pipeline.sourceValue << ":\n";
    for (const auto& op : pipeline.operations) {
        dumpOperation(out, depth + 1, op, model);
    }
}

void dumpModel(std::ostringstream& out, int depth, const ModelIR& model) {
    indent(out, depth);
    out << "Model " << model.name << "\n";

    indent(out, depth + 1);
    out << "input: " << typeString(model.valueById(model.inputValue)) << "  (id=" << model.inputValue << ")\n";

    for (const auto& pipeline : model.pipelines) {
        dumpPipeline(out, depth + 1, pipeline, model);
    }
}

}  // namespace

std::string dump(const Module& module) {
    std::ostringstream out;
    out << "Module";
    if (module.target.has_value()) {
        out << " target=" << *module.target;
    }
    out << "\n";

    for (const auto& model : module.models) {
        dumpModel(out, 1, model);
    }

    return out.str();
}

}  // namespace blackforge::ir
