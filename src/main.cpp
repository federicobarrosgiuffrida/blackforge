#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "blackforge/ast/ast.hpp"
#include "blackforge/backend/cpu/executor.hpp"
#include "blackforge/backend/cpu/forecast_runner.hpp"
#include "blackforge/backend/cpu/train_runner.hpp"
#include "blackforge/diagnostics/diagnostic.hpp"
#include "blackforge/frontend/lexer.hpp"
#include "blackforge/frontend/parser.hpp"
#include "blackforge/frontend/token.hpp"
#include "blackforge/ir/ir_builder.hpp"
#include "blackforge/ir/module.hpp"
#include "blackforge/runtime/tensor.hpp"
#include "blackforge/sema/semantic_analyzer.hpp"

#if BLACKFORGE_HAS_CUDA
#include "blackforge/backend/cuda/device_query.hpp"
#include "blackforge/backend/cuda/executor.hpp"
#endif

namespace {

void printUsage() {
    std::cout << "Uso: blackforge <comando> [opzioni] <file.bf>\n\n"
              << "Comandi:\n"
              << "  check <file>       Analizza il file e riporta gli errori (lessicali, sintattici, semantici)\n"
              << "  run <file>         Esegue il primo modello del file\n"
              << "  train <file>       Addestra il modello descritto dal blocco 'train' del file (CPU)\n"
              << "  forecast <file>    Genera 'horizon' passi autoregressivi dal blocco 'forecast' del file (CPU)\n"
              << "  devices            Elenca i dispositivi di calcolo disponibili (CPU e GPU CUDA)\n"
              << "  --help, -h         Mostra questo messaggio\n"
              << "  --version, -v      Mostra la versione del compilatore\n\n"
              << "Opzioni:\n"
              << "  --verbose          Mostra i token riconosciuti (solo 'check')\n"
              << "  --print-ast        Mostra l'albero sintattico (AST) prodotto dal parser (solo 'check')\n"
              << "  --print-ir         Mostra la rappresentazione interna (IR) del programma (solo 'check')\n"
              << "  --batch N          Dimensione di batch usata per risolvere le dimensioni simboliche "
                 "dell'input (solo 'run', default 1)\n"
              << "  --device cpu|cuda  Dispositivo su cui eseguire (solo 'run', default cpu)\n"
              << "  --from-checkpoint <file>  Pesi di partenza: fine-tuning/LoRA per 'train', "
                 "obbligatorio per 'forecast'\n"
              << "  --save-checkpoint <file>  Dove salvare i pesi al termine dell'addestramento (solo 'train')\n";
}

void printDevices() {
    std::cout << "cpu: sempre disponibile (backend di riferimento)\n";

#if BLACKFORGE_HAS_CUDA
    auto devices = blackforge::backend::cuda::enumerateDevices();
    if (devices.empty()) {
        std::cout << "cuda: nessuna GPU NVIDIA rilevata (compilato con supporto CUDA, ma nessun dispositivo "
                     "visibile al driver)\n";
    }
    for (const auto& device : devices) {
        std::cout << "cuda:" << device.index << ": " << device.name << " (compute capability "
                   << device.computeCapabilityMajor << "." << device.computeCapabilityMinor << ", "
                   << (device.totalMemoryBytes / (1024ULL * 1024ULL)) << " MiB)\n";
    }
#else
    std::cout << "cuda: non disponibile (questa build e' stata compilata senza supporto CUDA)\n";
#endif
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

int runRun(const std::string& path, std::size_t batchSize, const std::string& device) {
    if (device != "cpu" && device != "cuda") {
        std::cerr << "errore: dispositivo sconosciuto '" << device << "' (valori validi: cpu, cuda)\n";
        return 2;
    }

    if (device == "cuda" && !BLACKFORGE_HAS_CUDA) {
        std::cerr << "errore: '--device cuda' richiesto ma questa build di blackforge e' stata compilata senza "
                     "supporto CUDA (BLACKFORGE_ENABLE_CUDA=OFF in fase di compilazione)\n";
        return 1;
    }

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

    if (device == "cpu" && result.module.target.has_value() && *result.module.target != "cpu") {
        std::cout << "nota: target dichiarato '" << *result.module.target
                   << "'; sto eseguendo su CPU. Usa '--device cuda' per eseguire sulla GPU (se disponibile).\n";
    }

    try {
        blackforge::runtime::Tensor input;
        blackforge::runtime::Tensor output;

        if (device == "cpu") {
            blackforge::backend::cpu::Executor executor;
            input = executor.makeSyntheticInput(model.valueById(model.inputValue), batchSize);
            output = executor.run(model, input);
        } else {
#if BLACKFORGE_HAS_CUDA
            blackforge::backend::cuda::Executor executor;
            input = executor.makeSyntheticInput(model.valueById(model.inputValue), batchSize);
            output = executor.run(model, input);
#endif
        }

        std::cout << "Eseguito modello '" << model.name << "' sul backend " << device << "\n";
        std::cout << "  input:  " << input.shapeToString() << "\n";
        std::cout << "  output: " << output.shapeToString() << "  min=" << output.min() << " max=" << output.max()
                   << " mean=" << output.mean() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "errore di esecuzione: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

int runTrain(const std::string& path, const std::string& fromCheckpoint, const std::string& saveCheckpointPath) {
    CompileOutput result = compile(path, /*verbose=*/false, /*printAst=*/false, /*printIr=*/false);

    if (result.status == CompileStatus::FileNotFound) {
        return 2;
    }
    if (result.status == CompileStatus::AnalysisFailed) {
        std::cerr << path << ": impossibile addestrare, l'analisi ha trovato errori\n";
        return 1;
    }

    try {
        blackforge::backend::cpu::runTraining(result.program, result.module, fromCheckpoint, saveCheckpointPath,
                                               &std::cout);
    } catch (const std::exception& e) {
        std::cerr << "errore di addestramento: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

int runForecast(const std::string& path, const std::string& fromCheckpoint, std::size_t batchSize) {
    CompileOutput result = compile(path, /*verbose=*/false, /*printAst=*/false, /*printIr=*/false);

    if (result.status == CompileStatus::FileNotFound) {
        return 2;
    }
    if (result.status == CompileStatus::AnalysisFailed) {
        std::cerr << path << ": impossibile eseguire il forecasting, l'analisi ha trovato errori\n";
        return 1;
    }

    try {
        blackforge::backend::cpu::runForecast(result.program, result.module, fromCheckpoint, batchSize, &std::cout);
    } catch (const std::exception& e) {
        std::cerr << "errore di forecasting: " << e.what() << "\n";
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
    std::string device = "cpu";
    std::string fromCheckpoint;
    std::string saveCheckpointPath;
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
        } else if (args[i] == "--device") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--device' richiede un valore (cpu o cuda)\n";
                return 2;
            }
            device = args[++i];
        } else if (args[i] == "--from-checkpoint") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--from-checkpoint' richiede un percorso\n";
                return 2;
            }
            fromCheckpoint = args[++i];
        } else if (args[i] == "--save-checkpoint") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--save-checkpoint' richiede un percorso\n";
                return 2;
            }
            saveCheckpointPath = args[++i];
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
        return runRun(positional.front(), batchSize, device);
    }
    if (command == "train") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'train' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runTrain(positional.front(), fromCheckpoint, saveCheckpointPath);
    }
    if (command == "forecast") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'forecast' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runForecast(positional.front(), fromCheckpoint, batchSize);
    }
    if (command == "devices") {
        printDevices();
        return 0;
    }

    std::cerr << "errore: comando sconosciuto '" << command << "'\n";
    printUsage();
    return 2;
}
