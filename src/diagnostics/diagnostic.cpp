#include "blackforge/diagnostics/diagnostic.hpp"

namespace blackforge {

std::string formatDiagnostic(const Diagnostic& diagnostic) {
    const char* label = diagnostic.severity == DiagnosticSeverity::Error ? "errore" : "avviso";

    return diagnostic.location.file + ":" + std::to_string(diagnostic.location.line) + ":" +
           std::to_string(diagnostic.location.column) + ": " + label + ": " + diagnostic.message;
}

}  // namespace blackforge
