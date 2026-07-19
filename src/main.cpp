#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "blackforge/ast/ast.hpp"
#include "blackforge/backend/cpu/benchmark.hpp"
#include "blackforge/backend/cpu/checkpoint.hpp"
#include "blackforge/backend/cpu/executor.hpp"
#include "blackforge/backend/cpu/forecast_runner.hpp"
#include "blackforge/backend/cpu/model.hpp"
#include "blackforge/backend/cpu/train_runner.hpp"
#include "blackforge/diagnostics/diagnostic.hpp"
#include "blackforge/frontend/lexer.hpp"
#include "blackforge/frontend/parser.hpp"
#include "blackforge/frontend/token.hpp"
#include "blackforge/ir/ir_builder.hpp"
#include "blackforge/ir/module.hpp"
#include "blackforge/runtime/tensor.hpp"
#include "blackforge/sema/dtype.hpp"
#include "blackforge/sema/semantic_analyzer.hpp"
#include "blackforge/tokenizer/dataset_prep.hpp"
#include "blackforge/tokenizer/tokenizer.hpp"

#if BLACKFORGE_HAS_CUDA
#include "blackforge/backend/cuda/checkpoint.hpp"
#include "blackforge/backend/cuda/device_query.hpp"
#include "blackforge/backend/cuda/executor.hpp"
#include "blackforge/backend/cuda/model.hpp"
#include "blackforge/backend/cuda/train_runner.hpp"
#endif

namespace {

void printUsage() {
    std::cout << "Uso: blackforge <comando> [opzioni] <file.bf>\n\n"
              << "Comandi:\n"
              << "  check <file>       Analizza il file e riporta gli errori (lessicali, sintattici, semantici)\n"
              << "  build <file>       Compila e costruisce ogni modello (alloca i parametri), senza eseguirlo\n"
              << "  run <file>         Esegue il primo modello del file\n"
              << "  train <file>       Addestra il modello descritto dal blocco 'train' del file (CPU o CUDA)\n"
              << "  forecast <file>    Genera 'horizon' passi autoregressivi dal blocco 'forecast' del file (CPU)\n"
              << "  benchmark <file>   Misura tempo/throughput/memoria del primo modello del file\n"
              << "  inspect <file>     Mostra un riepilogo dei modelli (input, pipeline, numero di parametri)\n"
              << "  devices            Elenca i dispositivi di calcolo disponibili (CPU e GPU CUDA)\n"
              << "  tokenizer-train <corpus.txt>   Addestra un tokenizer BPE byte-level sul corpus dato\n"
              << "                                  (richiede --vocab-size e --output)\n"
              << "  tokenizer-encode <tok.bftok> <testo.txt>   Codifica un file di testo, stampa gli id di token\n"
              << "  dataset-build <corpus.txt> <tok.bftok>   Tokenizza un corpus e costruisce un dataset .bfdata\n"
              << "                                  per l'addestramento di un modello linguistico (richiede\n"
              << "                                  --seq-len e --output)\n"
              << "  --help, -h         Mostra questo messaggio\n"
              << "  --version, -v      Mostra la versione del compilatore\n\n"
              << "Opzioni:\n"
              << "  --verbose          Mostra i token riconosciuti (solo 'check')\n"
              << "  --print-ast        Mostra l'albero sintattico (AST) prodotto dal parser (solo 'check')\n"
              << "  --print-ir         Mostra la rappresentazione interna (IR) del programma (solo 'check')\n"
              << "  --batch N          Dimensione di batch usata per risolvere le dimensioni simboliche "
                 "dell'input (default 1)\n"
              << "  --device cpu|cuda|cuda:N  Dispositivo su cui eseguire ('run'/'train'/'benchmark', default "
                 "cpu). Su 'train', il backend CUDA supporta 'loss mse' e 'cross_entropy' ma non ancora 'lora' "
                 "(errore esplicito se richiesto); checkpoint supportati su entrambi i device, stesso formato\n"
              << "  --from-checkpoint <file>  Pesi da caricare: esecuzione con pesi allenati per 'run', "
                 "fine-tuning/LoRA per 'train' (LoRA solo CPU), obbligatorio per 'forecast' (solo CPU)\n"
              << "  --save-checkpoint <file>  Dove salvare i pesi al termine dell'addestramento (solo 'train')\n"
              << "  --warmup N         Iterazioni di riscaldamento scartate (solo 'benchmark', default 5)\n"
              << "  --iterations N     Iterazioni misurate (solo 'benchmark', default 20)\n"
              << "  --vocab-size N     Dimensione del vocabolario da apprendere (solo 'tokenizer-train'; deve "
                 "superare " << blackforge::tokenizer::Tokenizer::kFirstMergeId << ")\n"
              << "  --seq-len N        Lunghezza di sequenza di ogni esempio (solo 'dataset-build')\n"
              << "  --output <file>    Percorso di output (solo 'tokenizer-train'/'dataset-build')\n";
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

struct DeviceSpec {
    bool isCuda = false;
    int cudaIndex = 0;
};

// Analizza '--device cpu' | 'cuda' | 'cuda:N'. Restituisce false e
// riempie 'error' se la stringa non e' un dispositivo valido.
bool parseDeviceSpec(const std::string& device, DeviceSpec& outSpec, std::string& outError) {
    if (device == "cpu") {
        outSpec = DeviceSpec{false, 0};
        return true;
    }
    if (device == "cuda") {
        outSpec = DeviceSpec{true, 0};
        return true;
    }
    if (device.rfind("cuda:", 0) == 0) {
        std::string indexPart = device.substr(5);
        bool allDigits = !indexPart.empty() && std::all_of(indexPart.begin(), indexPart.end(), [](unsigned char c) {
                              return std::isdigit(c) != 0;
                          });
        if (!allDigits) {
            outError = "indice GPU non valido in '" + device + "'";
            return false;
        }
        outSpec = DeviceSpec{true, std::stoi(indexPart)};
        return true;
    }
    outError = "dispositivo sconosciuto '" + device + "' (valori validi: cpu, cuda, cuda:N)";
    return false;
}

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

int runRun(const std::string& path, std::size_t batchSize, const std::string& device,
           const std::string& fromCheckpoint) {
    DeviceSpec spec;
    std::string deviceError;
    if (!parseDeviceSpec(device, spec, deviceError)) {
        std::cerr << "errore: " << deviceError << "\n";
        return 2;
    }
    if (spec.isCuda && !BLACKFORGE_HAS_CUDA) {
        std::cerr << "errore: '--device " << device << "' richiesto ma questa build di blackforge e' stata "
                     "compilata senza supporto CUDA (BLACKFORGE_ENABLE_CUDA=OFF in fase di compilazione)\n";
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

    if (!spec.isCuda && result.module.target.has_value() && *result.module.target != "cpu") {
        std::cout << "nota: target dichiarato '" << *result.module.target
                   << "'; sto eseguendo su CPU. Usa '--device cuda' per eseguire sulla GPU (se disponibile).\n";
    }

    try {
        blackforge::runtime::Tensor input;
        blackforge::runtime::Tensor output;

        if (!spec.isCuda) {
            blackforge::backend::cpu::Executor executor;
            input = executor.makeSyntheticInput(model, batchSize);

            if (fromCheckpoint.empty()) {
                output = executor.run(model, input, result.module.precision);
            } else {
                // A differenza di Executor (pesi casuali rigenerati ad ogni
                // chiamata), Model possiede i propri parametri: e' quello che
                // serve per eseguire davvero con pesi allenati caricati da
                // un checkpoint, non solo per verificare che la pipeline
                // funzioni.
                blackforge::backend::cpu::Model loadedModel(model, /*seed=*/42, /*lora=*/std::nullopt,
                                                              result.module.precision);
                blackforge::backend::cpu::loadCheckpoint(loadedModel, fromCheckpoint);
                std::cout << "Pesi caricati da '" << fromCheckpoint << "'\n";
                output = loadedModel.forward(input);
            }
        } else {
#if BLACKFORGE_HAS_CUDA
            blackforge::backend::cuda::setActiveDevice(spec.cudaIndex);
            blackforge::backend::cuda::Executor executor;
            input = executor.makeSyntheticInput(model, batchSize);

            if (fromCheckpoint.empty()) {
                output = executor.run(model, input);
            } else {
                blackforge::backend::cuda::Model loadedModel(model);
                blackforge::backend::cuda::loadCheckpoint(loadedModel, fromCheckpoint);
                std::cout << "Pesi caricati da '" << fromCheckpoint << "'\n";
                blackforge::backend::cuda::DeviceTensor inputDevice =
                    blackforge::backend::cuda::DeviceTensor::fromHost(input);
                output = loadedModel.forward(inputDevice).toHost();
            }
            if (result.module.precision.has_value()) {
                std::cout << "nota: un blocco 'precision' e' dichiarato, ma il backend CUDA non applica ancora "
                             "la quantizzazione simulata (solo il backend CPU lo fa per ora): l'esecuzione "
                             "sulla GPU e' in piena precisione float32.\n";
            }
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

int runTrain(const std::string& path, const std::string& fromCheckpoint, const std::string& saveCheckpointPath,
             const std::string& device) {
    DeviceSpec spec;
    std::string deviceError;
    if (!parseDeviceSpec(device, spec, deviceError)) {
        std::cerr << "errore: " << deviceError << "\n";
        return 2;
    }
    if (spec.isCuda && !BLACKFORGE_HAS_CUDA) {
        std::cerr << "errore: '--device " << device << "' richiesto ma questa build di blackforge e' stata "
                     "compilata senza supporto CUDA (BLACKFORGE_ENABLE_CUDA=OFF in fase di compilazione)\n";
        return 1;
    }

    CompileOutput result = compile(path, /*verbose=*/false, /*printAst=*/false, /*printIr=*/false);

    if (result.status == CompileStatus::FileNotFound) {
        return 2;
    }
    if (result.status == CompileStatus::AnalysisFailed) {
        std::cerr << path << ": impossibile addestrare, l'analisi ha trovato errori\n";
        return 1;
    }

    try {
        if (!spec.isCuda) {
            blackforge::backend::cpu::runTraining(result.program, result.module, fromCheckpoint, saveCheckpointPath,
                                                   &std::cout);
        } else {
#if BLACKFORGE_HAS_CUDA
            blackforge::backend::cuda::setActiveDevice(spec.cudaIndex);
            blackforge::backend::cuda::runTraining(result.program, result.module, fromCheckpoint, saveCheckpointPath,
                                                    &std::cout);
#endif
        }
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

int runTokenizerTrain(const std::string& corpusPath, std::size_t vocabSize, const std::string& outputPath) {
    std::string corpus;
    if (!readFile(corpusPath, corpus)) {
        std::cerr << "errore: impossibile leggere il corpus '" << corpusPath << "'\n";
        return 2;
    }
    if (vocabSize == 0) {
        std::cerr << "errore: '--vocab-size' richiede un intero positivo (deve superare "
                   << blackforge::tokenizer::Tokenizer::kFirstMergeId << ", i token base + speciali)\n";
        return 2;
    }
    if (outputPath.empty()) {
        std::cerr << "errore: comando 'tokenizer-train' richiede '--output <file.bftok>'\n";
        return 2;
    }

    try {
        blackforge::tokenizer::Tokenizer tok;
        tok.train(corpus, vocabSize);
        blackforge::tokenizer::saveTokenizer(tok, outputPath);
        std::cout << "Tokenizer addestrato: " << tok.vocabSize() << " token totali (" << tok.merges().size()
                   << " merge appresi da un corpus di " << corpus.size() << " byte)\n";
        std::cout << "Salvato in '" << outputPath << "'\n";
    } catch (const std::exception& e) {
        std::cerr << "errore di training del tokenizer: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

int runTokenizerEncode(const std::string& tokenizerPath, const std::string& textPath) {
    std::string text;
    if (!readFile(textPath, text)) {
        std::cerr << "errore: impossibile leggere il file di testo '" << textPath << "'\n";
        return 2;
    }

    try {
        blackforge::tokenizer::Tokenizer tok = blackforge::tokenizer::loadTokenizer(tokenizerPath);
        std::vector<std::uint32_t> ids = tok.encode(text);

        std::cout << ids.size() << " token (da " << text.size() << " byte, vocabolario " << tok.vocabSize()
                   << "):\n";
        for (std::size_t i = 0; i < ids.size(); ++i) {
            std::cout << ids[i] << (i + 1 < ids.size() ? " " : "\n");
        }

        std::string roundTrip = tok.decode(ids);
        if (roundTrip != text) {
            std::cerr << "attenzione: decode(encode(testo)) non coincide con il testo originale (non dovrebbe mai "
                         "accadere per un tokenizer byte-level corretto: possibile bug)\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "errore di codifica: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

int runDatasetBuild(const std::string& corpusPath, const std::string& tokenizerPath, std::size_t seqLen,
                     const std::string& outputPath) {
    std::string corpus;
    if (!readFile(corpusPath, corpus)) {
        std::cerr << "errore: impossibile leggere il corpus '" << corpusPath << "'\n";
        return 2;
    }
    if (seqLen == 0) {
        std::cerr << "errore: '--seq-len' richiede un intero positivo\n";
        return 2;
    }
    if (outputPath.empty()) {
        std::cerr << "errore: comando 'dataset-build' richiede '--output <file.bfdata>'\n";
        return 2;
    }

    try {
        blackforge::tokenizer::Tokenizer tok = blackforge::tokenizer::loadTokenizer(tokenizerPath);
        std::size_t numExamples = blackforge::tokenizer::buildLanguageModelDataset(tok, corpus, seqLen, outputPath);
        std::cout << "Dataset costruito: " << numExamples << " esempi (seqLen=" << seqLen
                   << ", vocabolario=" << tok.vocabSize() << ")\n";
        std::cout << "Salvato in '" << outputPath << "'\n";
    } catch (const std::exception& e) {
        std::cerr << "errore di costruzione del dataset: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

std::string formatShape(const std::vector<std::size_t>& shape) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << shape[i];
    }
    out << "]";
    return out.str();
}

int runBenchmark(const std::string& path, const std::string& device, std::size_t batchSize,
                  std::size_t warmupIterations, std::size_t measuredIterations) {
    DeviceSpec spec;
    std::string deviceError;
    if (!parseDeviceSpec(device, spec, deviceError)) {
        std::cerr << "errore: " << deviceError << "\n";
        return 2;
    }
    if (spec.isCuda && !BLACKFORGE_HAS_CUDA) {
        std::cerr << "errore: '--device " << device << "' richiesto ma questa build di blackforge e' stata "
                     "compilata senza supporto CUDA (BLACKFORGE_ENABLE_CUDA=OFF in fase di compilazione)\n";
        return 1;
    }

    CompileOutput result = compile(path, /*verbose=*/false, /*printAst=*/false, /*printIr=*/false);
    if (result.status == CompileStatus::FileNotFound) {
        return 2;
    }
    if (result.status == CompileStatus::AnalysisFailed) {
        std::cerr << path << ": impossibile eseguire il benchmark, l'analisi ha trovato errori\n";
        return 1;
    }
    if (result.module.models.empty()) {
        std::cerr << "errore: il programma non definisce alcun modello da misurare\n";
        return 1;
    }

    const blackforge::ir::ModelIR& model = result.module.models.front();
    std::string dtypeName = blackforge::sema::dtypeName(model.valueById(model.inputValue).dtype);

    try {
        blackforge::backend::cpu::BenchmarkResult cpuResult = blackforge::backend::cpu::runBenchmark(
            model, batchSize, warmupIterations, measuredIterations, result.module.precision);

        std::cout << "Benchmark di '" << model.name << "'\n";
        std::cout << "  hardware:          CPU (backend di riferimento; nome del processore non rilevato)\n";
        if (result.module.precision.has_value()) {
            std::cout << "  precisione:        " << dtypeName << " dichiarata; quantizzazione simulata applicata "
                       << "(storage=" << blackforge::sema::dtypeName(result.module.precision->storage)
                       << " compute=" << blackforge::sema::dtypeName(result.module.precision->compute)
                       << "), accumulo interno sempre in float32\n";
        } else {
            std::cout << "  precisione:        " << dtypeName
                       << " dichiarata (nessun blocco 'precision': calcolo in piena float32)\n";
        }
        std::cout << "  forma input:       " << formatShape(cpuResult.inputShape) << "\n";
        std::cout << "  warmup:            " << cpuResult.warmupIterations << " iterazioni\n";
        std::cout << "  iterazioni:        " << cpuResult.measuredIterations << "\n";
        std::cout << "  tempo medio:       " << cpuResult.meanMilliseconds << " ms\n";
        std::cout << "  throughput:        " << cpuResult.throughputSamplesPerSecond << " esempi/s\n";
        std::cout << "  memoria (stimata): " << (static_cast<double>(cpuResult.estimatedMemoryBytes) / 1024.0 / 1024.0)
                   << " MiB (float32; e' una stima teorica, non memoria di processo misurata)\n";
        std::cout << "  per operazione (CPU, ciascuna misurata separatamente sul proprio input reale):\n";
        for (const auto& op : cpuResult.perOperation) {
            std::cout << "    [" << op.operationIndex << "] " << op.operationName << ": " << op.meanMilliseconds
                       << " ms\n";
        }

        if (spec.isCuda) {
#if BLACKFORGE_HAS_CUDA
            blackforge::backend::cuda::setActiveDevice(spec.cudaIndex);

            std::string gpuName = "GPU sconosciuta";
            for (const auto& d : blackforge::backend::cuda::enumerateDevices()) {
                if (d.index == spec.cudaIndex) {
                    gpuName = d.name;
                }
            }

            blackforge::backend::cuda::Executor cudaExecutor;
            blackforge::runtime::Tensor cudaInput =
                cudaExecutor.makeSyntheticInput(model, batchSize);

            for (std::size_t i = 0; i < warmupIterations; ++i) {
                blackforge::runtime::Tensor warm = cudaExecutor.run(model, cudaInput);
                (void)warm;
            }

            blackforge::runtime::Tensor gpuOutput;
            auto start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < measuredIterations; ++i) {
                gpuOutput = cudaExecutor.run(model, cudaInput);
            }
            auto end = std::chrono::steady_clock::now();

            double totalMillis = std::chrono::duration<double, std::milli>(end - start).count();
            double gpuMeanMillis = totalMillis / static_cast<double>(measuredIterations);
            double gpuThroughput = gpuMeanMillis > 0.0 ? (static_cast<double>(batchSize) * 1000.0) / gpuMeanMillis
                                                         : 0.0;

            std::cout << "\n  hardware GPU:      " << gpuName << " (cuda:" << spec.cudaIndex << ")\n";
            std::cout << "  tempo medio GPU:   " << gpuMeanMillis << " ms\n";
            std::cout << "  throughput GPU:    " << gpuThroughput << " esempi/s\n";
            std::cout << "  speedup vs CPU:    " << (cpuResult.meanMilliseconds / gpuMeanMillis) << "x\n";

            // Confronto con la modalita' di riferimento (CPU): stesso
            // seme, quindi stesso input e stessi pesi iniziali.
            blackforge::backend::cpu::Executor cpuExecutor;
            blackforge::runtime::Tensor cpuInput =
                cpuExecutor.makeSyntheticInput(model, batchSize);
            blackforge::runtime::Tensor cpuOutput = cpuExecutor.run(model, cpuInput);

            float maxAbsDiff = 0.0F;
            for (std::size_t i = 0; i < cpuOutput.elementCount() && i < gpuOutput.elementCount(); ++i) {
                maxAbsDiff = std::max(maxAbsDiff, std::fabs(cpuOutput.at(i) - gpuOutput.at(i)));
            }
            std::cout << "  scarto massimo vs CPU (riferimento): " << maxAbsDiff << "\n";
#endif
        }
    } catch (const std::exception& e) {
        std::cerr << "errore di benchmark: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

int runInspect(const std::string& path) {
    CompileOutput result = compile(path, /*verbose=*/false, /*printAst=*/false, /*printIr=*/false);
    if (result.status == CompileStatus::FileNotFound) {
        return 2;
    }
    if (result.status == CompileStatus::AnalysisFailed) {
        std::cerr << path << ": impossibile ispezionare, l'analisi ha trovato errori\n";
        return 1;
    }

    std::cout << path << "\n";
    if (result.module.target.has_value()) {
        std::cout << "target: " << *result.module.target << "\n";
    }
    if (result.module.precision.has_value()) {
        std::cout << "precision: storage=" << blackforge::sema::dtypeName(result.module.precision->storage)
                   << " compute=" << blackforge::sema::dtypeName(result.module.precision->compute)
                   << " accumulate=" << blackforge::sema::dtypeName(result.module.precision->accumulate)
                   << " (applicata da 'run'/'forecast'/'benchmark' su CPU; non ancora su CUDA; ignorata durante "
                      "'train')\n";
    }
    std::cout << result.module.models.size() << " modello/i\n";

    for (const auto& model : result.module.models) {
        std::cout << "\nmodello '" << model.name << "'\n";
        std::cout << "  input: " << blackforge::ir::typeString(model.valueById(model.inputValue)) << "\n";

        try {
            blackforge::backend::cpu::Model built(model);
            std::size_t totalParams = 0;
            for (blackforge::backend::cpu::Parameter* param : built.allParameters()) {
                totalParams += param->value.elementCount();
            }
            std::cout << "  parametri: " << totalParams << " ("
                       << (static_cast<double>(totalParams * sizeof(float)) / 1024.0 / 1024.0)
                       << " MiB come float32)\n";
        } catch (const std::exception& e) {
            std::cout << "  parametri: non calcolabili (" << e.what() << ")\n";
        }

        for (const auto& pipeline : model.pipelines) {
            std::cout << "  pipeline:\n";
            for (const auto& op : pipeline.operations) {
                std::cout << "    " << blackforge::ir::opKindName(op.kind);
                if (op.kind == blackforge::ir::OpKind::Linear) {
                    std::cout << "(" << op.linearOutFeatures << ")";
                }
                std::cout << " -> " << blackforge::ir::typeString(model.valueById(op.output)) << "\n";
            }
        }
    }

    return 0;
}

int runBuild(const std::string& path) {
    CompileOutput result = compile(path, /*verbose=*/false, /*printAst=*/false, /*printIr=*/false);
    if (result.status == CompileStatus::FileNotFound) {
        return 2;
    }
    if (result.status == CompileStatus::AnalysisFailed) {
        std::cerr << path << ": build fallita, l'analisi ha trovato errori\n";
        return 1;
    }

    // A differenza di 'check' (che valida solo AST/IR), 'build' prova
    // davvero a costruire ogni modello (allocando i suoi parametri):
    // rileva errori che richiedono la costruzione effettiva, come una
    // dimensione delle feature ancora simbolica (impossibile allocare
    // pesi concreti).
    bool ok = true;
    for (const auto& model : result.module.models) {
        try {
            blackforge::backend::cpu::Model built(model);
            std::size_t totalParams = 0;
            for (blackforge::backend::cpu::Parameter* param : built.allParameters()) {
                totalParams += param->value.elementCount();
            }
            std::cout << "  modello '" << model.name << "': OK (" << totalParams << " parametri)\n";
        } catch (const std::exception& e) {
            std::cerr << "  modello '" << model.name << "': errore di costruzione: " << e.what() << "\n";
            ok = false;
        }
    }

    if (!ok) {
        std::cerr << path << ": build fallita\n";
        return 1;
    }

    std::cout << path << ": build riuscita (" << result.module.models.size()
               << " modello/i pronti per l'esecuzione)\n";
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
    std::size_t warmupIterations = 5;
    std::size_t measuredIterations = 20;
    std::size_t vocabSize = 0;
    std::size_t seqLen = 0;
    std::string outputPath;
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
        } else if (args[i] == "--warmup") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--warmup' richiede un valore\n";
                return 2;
            }
            warmupIterations = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--iterations") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--iterations' richiede un valore\n";
                return 2;
            }
            measuredIterations = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
            if (measuredIterations == 0) {
                std::cerr << "errore: '--iterations' richiede un intero positivo\n";
                return 2;
            }
        } else if (args[i] == "--vocab-size") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--vocab-size' richiede un valore\n";
                return 2;
            }
            vocabSize = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--seq-len") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--seq-len' richiede un valore\n";
                return 2;
            }
            seqLen = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--output") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--output' richiede un percorso\n";
                return 2;
            }
            outputPath = args[++i];
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
    if (command == "build") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'build' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runBuild(positional.front());
    }
    if (command == "run") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'run' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runRun(positional.front(), batchSize, device, fromCheckpoint);
    }
    if (command == "train") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'train' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runTrain(positional.front(), fromCheckpoint, saveCheckpointPath, device);
    }
    if (command == "forecast") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'forecast' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runForecast(positional.front(), fromCheckpoint, batchSize);
    }
    if (command == "benchmark") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'benchmark' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runBenchmark(positional.front(), device, batchSize, warmupIterations, measuredIterations);
    }
    if (command == "inspect") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'inspect' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runInspect(positional.front());
    }
    if (command == "devices") {
        printDevices();
        return 0;
    }
    if (command == "tokenizer-train") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'tokenizer-train' richiede il percorso di un corpus di testo\n";
            return 2;
        }
        return runTokenizerTrain(positional.front(), vocabSize, outputPath);
    }
    if (command == "tokenizer-encode") {
        if (positional.size() < 2) {
            std::cerr << "errore: comando 'tokenizer-encode' richiede '<tokenizer.bftok> <testo.txt>'\n";
            return 2;
        }
        return runTokenizerEncode(positional[0], positional[1]);
    }
    if (command == "dataset-build") {
        if (positional.size() < 2) {
            std::cerr << "errore: comando 'dataset-build' richiede '<corpus.txt> <tokenizer.bftok>'\n";
            return 2;
        }
        return runDatasetBuild(positional[0], positional[1], seqLen, outputPath);
    }

    std::cerr << "errore: comando sconosciuto '" << command << "'\n";
    printUsage();
    return 2;
}
