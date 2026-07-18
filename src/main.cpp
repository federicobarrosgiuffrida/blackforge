#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "blackforge/ast/ast.hpp"
#include "blackforge/diagnostics/diagnostic.hpp"
#include "blackforge/frontend/lexer.hpp"
#include "blackforge/frontend/parser.hpp"
#include "blackforge/frontend/token.hpp"
#include "blackforge/ir/ir_builder.hpp"
#include "blackforge/ir/module.hpp"
#include "blackforge/sema/semantic_analyzer.hpp"

namespace {

void printUsage() {
    std::cout << "Uso: blackforge <comando> [opzioni] <file.bf>\n\n"
              << "Comandi:\n"
              << "  check <file>       Analizza il file e riporta gli errori (lessicali, sintattici, semantici)\n"
              << "  --help, -h         Mostra questo messaggio\n"
              << "  --version, -v      Mostra la versione del compilatore\n\n"
              << "Opzioni:\n"
              << "  --verbose          Mostra i token riconosciuti\n"
              << "  --print-ast        Mostra l'albero sintattico (AST) prodotto dal parser\n"
              << "  --print-ir         Mostra la rappresentazione interna (IR) del programma\n";
}

void printVersion() { std::cout << "blackforge " << BLACKFORGE_VERSION << "\n"; }

// Legge l'intero contenuto di un file in una stringa. Restituisce false
// se il file non esiste o non e' leggibile.
bool readFile(const std::string& path, std::string& outContent) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file) {
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    outContent = buffer.str();
    return true;
}

int runCheck(const std::string& path, bool verbose, bool printAst, bool printIr) {
    std::string source;
    if (!readFile(path, source)) {
        std::cerr << "errore: impossibile aprire il file '" << path << "'\n";
        return 2;
    }

    blackforge::Lexer lexer(source, path);
    std::vector<blackforge::Token> tokens = lexer.tokenize();

    if (verbose) {
        for (const auto& token : tokens) {
            std::cout << token.location.file << ":" << token.location.line << ":" << token.location.column << ": "
                      << blackforge::tokenKindName(token.kind);
            if (!token.lexeme.empty()) {
                std::cout << " '" << token.lexeme << "'";
            }
            std::cout << "\n";
        }
    }

    blackforge::Parser parser(tokens);
    blackforge::ast::Program program = parser.parseProgram();

    if (printAst) {
        std::cout << blackforge::ast::dump(program);
    }

    blackforge::sema::SemanticAnalyzer analyzer;
    analyzer.analyze(program);

    bool hasErrors = false;
    for (const auto& diagnostic : lexer.diagnostics().all()) {
        std::cerr << blackforge::formatDiagnostic(diagnostic) << "\n";
    }
    hasErrors = hasErrors || lexer.diagnostics().hasErrors();

    for (const auto& diagnostic : parser.diagnostics().all()) {
        std::cerr << blackforge::formatDiagnostic(diagnostic) << "\n";
    }
    hasErrors = hasErrors || parser.diagnostics().hasErrors();

    for (const auto& diagnostic : analyzer.diagnostics().all()) {
        std::cerr << blackforge::formatDiagnostic(diagnostic) << "\n";
    }
    hasErrors = hasErrors || analyzer.diagnostics().hasErrors();

    // La IR viene costruita solo se le fasi precedenti non hanno gia'
    // trovato errori: su un programma invalido l'albero potrebbe essere
    // incompleto e la IR risulterebbe fuorviante.
    if (!hasErrors) {
        blackforge::ir::IRBuilder irBuilder;
        blackforge::ir::Module module = irBuilder.build(program);

        if (printIr) {
            std::cout << blackforge::ir::dump(module);
        }

        for (const auto& diagnostic : irBuilder.diagnostics().all()) {
            std::cerr << blackforge::formatDiagnostic(diagnostic) << "\n";
        }
        hasErrors = hasErrors || irBuilder.diagnostics().hasErrors();
    }

    if (hasErrors) {
        std::cerr << path << ": analisi fallita\n";
        return 1;
    }

    std::cout << path << ": nessun errore (" << tokens.size() << " token, " << program.declarations.size()
              << " dichiarazioni top-level)\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);

    if (args.empty()) {
        printUsage();
        return 2;
    }

    bool verbose = false;
    bool printAst = false;
    bool printIr = false;
    std::vector<std::string> positional;
    std::string command = args.front();

    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--verbose") {
            verbose = true;
        } else if (args[i] == "--print-ast") {
            printAst = true;
        } else if (args[i] == "--print-ir") {
            printIr = true;
        } else {
            positional.push_back(args[i]);
        }
    }

    if (command == "--help" || command == "-h") {
        printUsage();
        return 0;
    }
    if (command == "--version" || command == "-v") {
        printVersion();
        return 0;
    }
    if (command == "check") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'check' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runCheck(positional.front(), verbose, printAst, printIr);
    }

    std::cerr << "errore: comando sconosciuto '" << command << "'\n";
    printUsage();
    return 2;
}
