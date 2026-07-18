#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace blackforge {

// Posizione di un elemento all'interno di un file sorgente BlackForge.
struct SourceLocation {
    std::string file;
    std::size_t line = 1;
    std::size_t column = 1;
};

enum class DiagnosticSeverity {
    Error,
    Warning,
};

// Un errore o un avviso prodotto durante una fase del compilatore
// (lexer, parser, analisi semantica, ...).
struct Diagnostic {
    DiagnosticSeverity severity;
    SourceLocation location;
    std::string message;
};

// Contenitore di diagnostiche raccolte durante una fase di compilazione.
// Permette il recupero dagli errori: la fase continua a lavorare per
// riportare quanti più problemi possibile in un'unica esecuzione.
class DiagnosticList {
public:
    void addError(SourceLocation location, std::string message) {
        diagnostics_.push_back(Diagnostic{DiagnosticSeverity::Error, std::move(location), std::move(message)});
    }

    void addWarning(SourceLocation location, std::string message) {
        diagnostics_.push_back(Diagnostic{DiagnosticSeverity::Warning, std::move(location), std::move(message)});
    }

    [[nodiscard]] bool hasErrors() const {
        for (const auto& diagnostic : diagnostics_) {
            if (diagnostic.severity == DiagnosticSeverity::Error) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] const std::vector<Diagnostic>& all() const { return diagnostics_; }

private:
    std::vector<Diagnostic> diagnostics_;
};

// Formatta una diagnostica come "file:riga:colonna: errore: messaggio".
std::string formatDiagnostic(const Diagnostic& diagnostic);

}  // namespace blackforge
