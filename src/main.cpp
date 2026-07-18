#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "blackforge/ast/ast.hpp"
#include "blackforge/backend/cpu/executor.hpp"
#include "blackforge/diagnostics/diagnostic.hpp"
#include "blackforge/frontend/lexer.hpp"
#include "blackforge/frontend/parser.hpp"
#include "blackforge/frontend/token.hpp"
#include "blackforge/ir/ir_builder.hpp"
#include "blackforge/ir/module.hpp"
#include "blackforge/runtime/tensor.hpp"
#include "blackforge/sema/semantic_analyzer.hpp"

namespace {

void printUsage() {
    std::cout << "Uso: blackforge <comando> [opzioni] <file.bf>\n\n"
              << "Comandi:\n"
              << "  check <file>       Analizza il file e riporta gli errori (lessicali, sintattici, semantici)\n"
              << "  run <file>         Esegue il primo modello del file sul backend CPU di riferimento\n"
              << "  --help, -h         Mostra questo messaggio\n"
              << "  --version, -v      Mostra la versione del compilatore\n\n"
              << "Opzioni:\n"
              << "  --verbose          Mostra i token riconosciuti (solo 'check')\n"
              << "  --print-ast        Mostra l'albero sintattico (AST) prodotto dal parser (solo 'check')\n"
              << "  --print-ir         Mostra la rappresentazione interna (IR) del programma (solo 'check')\n"
              << "  --batch N          Dimensione di batch usata per risolvere le dimensioni simboliche "
                 "dell'input (solo 'run', default 1)\n";
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

enum class CompileStatus { FileNotFound, AnalysisFailed, Ok };

struct CompileOutput {
    CompileStatus status = CompileStatus::FileNotFound;
    blackforge::ast::Program program;
    blackforge::ir::Module module;
    std::size_t tokenCount = 0;
};

// Esegue lexer, parser, analisi semantica e costruzione della IR su un
// file, stampando tutte le diagnostiche su stderr. La IR viene
// costruita solo se le fasi precedenti non hanno trovato errori.
CompileOutput compile(const std::string& path, bool verbose, bool printAst, bool printIr) {
    CompileOutput result;

    std::string source;
    if (!readFile(path, source)) {
        std::cerr << "errore: impossibile aprire il file '" << path << "'\n";
        return result;
    }

    blackforge::Lexer lexer(source, path);
    std::vector<blackforge::Token> tokens = lexer.tokenize();
    result.tokenCount = tokens.size();

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
    result.program = parser.parseProgram();

    if (printAst) {
        std::cout << blackforge::ast::dump(result.program);
    }

    blackforge::sema::SemanticAnalyzer analyzer;
    analyzer.analyze(result.program);

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

    if (!hasErrors) {
        blackforge::ir::IRBuilder irBuilder;
        result.module = irBuilder.build(result.program);

        if (printIr) {
            std::cout << blackforge::ir::dump(result.module);
        }

        for (const auto& diagnostic : irBuilder.diagnostics().all()) {
            std::cerr << blackforge::formatDiagnostic(diagnostic) << "\n";
        }
        hasErrors = hasErrors || irBuilder.diagnostics().hasErrors();
    }

    result.status = hasErrors ? CompileStatus::AnalysisFailed : CompileStatus::Ok;
    return result;
}

int runCheck(const std::string& path, bool verbose, bool printAst, bool printIr) {
    CompileOutput result = compile(path, verbose, printAst, printIr);

    switch (result.status) {
        case CompileStatus::FileNotFound:
            return 2;
        case CompileStatus::AnalysisFailed:
            std::cerr << path << ": analisi fallita\n";
            return 1;
        case CompileStatus::Ok:
            std::cout << path << ": nessun errore (" << result.tokenCount << " token, "
                       << result.program.declarations.size() << " dichiarazioni top-level)\n";
            return 0;
    }
    return 1;
}

int runRun(const std::string& path, std::size_t batchSize) {
    CompileOutput result = compile(path, /*verbose=*/false, /*printAst=*/false, /*printIr=*/false);

    if (result.status == CompileStatus::FileNotFound) {
        return 2;
    }
    if (result.status == CompileStatus::AnalysisFailed) {
        std::cerr << path << ": impossibile eseguire, l'analisi ha trovato errori\n";
        return 1;
    }
    if (result.module.models.empty()) {
        std::cerr << "errore: il programma non definisce alcun modello da eseguire\n";
        return 1;
    }

    const blackforge::ir::ModelIR& model = result.module.models.front();

    if (result.module.target.has_value() && *result.module.target != "cpu") {
        std::cout << "nota: target dichiarato '" << *result.module.target
                   << "', ma il backend CUDA non e' ancora implementato: eseguo sul backend CPU di riferimento\n";
    }

    try {
        blackforge::backend::cpu::Executor executor;
        blackforge::runtime::Tensor input =
            executor.makeSyntheticInput(model.valueById(model.inputValue), batchSize);
        blackforge::runtime::Tensor output = executor.run(model, input);

        std::cout << "Eseguito modello '" << model.name << "' sul backend CPU di riferimento\n";
        std::cout << "  input:  " << input.shapeToString() << "\n";
        std::cout << "  output: " << output.shapeToString() << "  min=" << output.min() << " max=" << output.max()
                   << " mean=" << output.mean() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "errore di esecuzione: " << e.what() << "\n";
        return 1;
    }

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
    std::size_t batchSize = 1;
    std::vector<std::string> positional;
    std::string command = args.front();

    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--verbose") {
            verbose = true;
        } else if (args[i] == "--print-ast") {
            printAst = true;
        } else if (args[i] == "--print-ir") {
            printIr = true;
        } else if (args[i] == "--batch") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--batch' richiede un valore\n";
                return 2;
            }
            batchSize = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
            if (batchSize == 0) {
                std::cerr << "errore: '--batch' richiede un intero positivo\n";
                return 2;
            }
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
    if (command == "run") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'run' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runRun(positional.front(), batchSize);
    }

    std::cerr << "errore: comando sconosciuto '" << command << "'\n";
    printUsage();
    return 2;
}
