#include "blackforge/sema/semantic_analyzer.hpp"

#include <gtest/gtest.h>

#include "blackforge/frontend/lexer.hpp"
#include "blackforge/frontend/parser.hpp"

using namespace blackforge;

namespace {

sema::SemanticAnalyzer analyze(const std::string& source) {
    Lexer lexer(source, "test.bf");
    auto tokens = lexer.tokenize();
    EXPECT_FALSE(lexer.diagnostics().hasErrors());

    Parser parser(std::move(tokens));
    ast::Program program = parser.parseProgram();
    EXPECT_FALSE(parser.diagnostics().hasErrors());

    sema::SemanticAnalyzer analyzer;
    analyzer.analyze(program);
    return analyzer;
}

}  // namespace

TEST(SemanticAnalyzerTest, AccettaIlModelloTinyModelCompleto) {
    auto analyzer = analyze(
        "target nvidia.blackwell\n"
        "\n"
        "precision {\n"
        "    storage bf16\n"
        "    compute fp8.e4m3\n"
        "    accumulate fp32\n"
        "}\n"
        "\n"
        "model TinyModel {\n"
        "    input bf16[batch, 4096]\n"
        "\n"
        "    input\n"
        "        |> linear(4096)\n"
        "        |> silu\n"
        "        |> linear(4096)\n"
        "}\n");

    EXPECT_FALSE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaTargetSconosciuto) {
    auto analyzer = analyze("target amd.mi300");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaFormatoNumericoSconosciutoInPrecision) {
    auto analyzer = analyze(
        "precision {\n"
        "    storage int8\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaTf32ComeFormatoDiStorage) {
    auto analyzer = analyze(
        "precision {\n"
        "    storage tf32\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, AccettaTf32ComeFormatoDiCompute) {
    auto analyzer = analyze(
        "precision {\n"
        "    compute tf32\n"
        "}\n");
    EXPECT_FALSE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaCampoDiPrecisionDuplicato) {
    auto analyzer = analyze(
        "precision {\n"
        "    storage bf16\n"
        "    storage fp32\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaDimensioneTensorialeNonPositiva) {
    auto analyzer = analyze(
        "model M {\n"
        "    input bf16[0]\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaModelloSenzaInput) {
    auto analyzer = analyze(
        "model M {\n"
        "    input |> silu\n"
        "}\n");
    // 'input' senza una dichiarazione 'input bf16[...]' e' un errore
    // (nessun input dichiarato nel modello).
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaOperazioneSconosciuta) {
    auto analyzer = analyze(
        "model M {\n"
        "    input bf16[4096]\n"
        "    input |> softmax\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaLinearConNumeroArgomentiErrato) {
    auto analyzer = analyze(
        "model M {\n"
        "    input bf16[4096]\n"
        "    input |> linear(4096, 2048)\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaLinearConFeatureNonPositive) {
    auto analyzer = analyze(
        "model M {\n"
        "    input bf16[4096]\n"
        "    input |> linear(0)\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, AccettaRmsnorm) {
    auto analyzer = analyze(
        "model M {\n"
        "    input bf16[batch, 4096]\n"
        "    input |> rmsnorm |> linear(2048)\n"
        "}\n");
    EXPECT_FALSE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaRmsnormConArgomenti) {
    auto analyzer = analyze(
        "model M {\n"
        "    input bf16[4096]\n"
        "    input |> rmsnorm(4096)\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaIdentificatoreNonDefinitoComeSorgentePipeline) {
    auto analyzer = analyze(
        "model M {\n"
        "    input bf16[4096]\n"
        "    hidden |> silu\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaModelloDuplicato) {
    auto analyzer = analyze(
        "model M {\n"
        "    input bf16[4096]\n"
        "}\n"
        "model M {\n"
        "    input bf16[4096]\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaTargetDuplicato) {
    auto analyzer = analyze("target cpu\ntarget cpu\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

namespace {

// Programma valido riusato dai test di dataset/train: un modello con
// input compatibile, un dataset con path/input/labels completi e un
// blocco train che li referenzia entrambi.
const char* kValidTrainProgram =
    "model M {\n"
    "    input bf16[batch, 4]\n"
    "    input |> linear(2)\n"
    "}\n"
    "dataset D {\n"
    "    path \"data/train.bin\"\n"
    "    input bf16[batch, 4]\n"
    "    labels bf16[batch, 2]\n"
    "}\n"
    "train {\n"
    "    model M\n"
    "    dataset D\n"
    "    loss mse\n"
    "    optimizer adamw\n"
    "    epochs 10\n"
    "    batch_size 32\n"
    "}\n";

}  // namespace

TEST(SemanticAnalyzerTest, AccettaDatasetETrainValidi) {
    auto analyzer = analyze(kValidTrainProgram);
    EXPECT_FALSE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaDatasetSenzaPath) {
    auto analyzer = analyze(
        "dataset D {\n"
        "    input bf16[4]\n"
        "    labels bf16[2]\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaDatasetSenzaLabels) {
    auto analyzer = analyze(
        "dataset D {\n"
        "    path \"data/train.bin\"\n"
        "    input bf16[4]\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaDatasetDuplicato) {
    auto analyzer = analyze(
        "dataset D {\n"
        "    path \"a.bin\"\n"
        "    input bf16[4]\n"
        "    labels bf16[2]\n"
        "}\n"
        "dataset D {\n"
        "    path \"b.bin\"\n"
        "    input bf16[4]\n"
        "    labels bf16[2]\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaTrainConModelloNonDefinito) {
    auto analyzer = analyze(
        "dataset D {\n"
        "    path \"a.bin\"\n"
        "    input bf16[4]\n"
        "    labels bf16[2]\n"
        "}\n"
        "train {\n"
        "    model NonEsiste\n"
        "    dataset D\n"
        "    loss mse\n"
        "    optimizer sgd\n"
        "    epochs 1\n"
        "    batch_size 1\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaTrainConDatasetNonDefinito) {
    auto analyzer = analyze(
        "model M {\n"
        "    input bf16[4]\n"
        "}\n"
        "train {\n"
        "    model M\n"
        "    dataset NonEsiste\n"
        "    loss mse\n"
        "    optimizer sgd\n"
        "    epochs 1\n"
        "    batch_size 1\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaLossSconosciuta) {
    std::string source = kValidTrainProgram;
    auto pos = source.find("loss mse");
    source.replace(pos, std::string("loss mse").size(), "loss huber");
    auto analyzer = analyze(source);
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, AccettaLossCrossEntropy) {
    std::string source = kValidTrainProgram;
    auto pos = source.find("loss mse");
    source.replace(pos, std::string("loss mse").size(), "loss cross_entropy");
    auto analyzer = analyze(source);
    EXPECT_FALSE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaOptimizerSconosciuto) {
    std::string source = kValidTrainProgram;
    auto pos = source.find("optimizer adamw");
    source.replace(pos, std::string("optimizer adamw").size(), "optimizer rmsprop");
    auto analyzer = analyze(source);
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaEpochsNonPositivo) {
    std::string source = kValidTrainProgram;
    auto pos = source.find("epochs 10");
    source.replace(pos, std::string("epochs 10").size(), "epochs 0");
    auto analyzer = analyze(source);
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaBatchSizeNonPositivo) {
    std::string source = kValidTrainProgram;
    auto pos = source.find("batch_size 32");
    source.replace(pos, std::string("batch_size 32").size(), "batch_size 0");
    auto analyzer = analyze(source);
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaTrainSenzaOptimizer) {
    auto analyzer = analyze(
        "model M {\n"
        "    input bf16[4]\n"
        "}\n"
        "dataset D {\n"
        "    path \"a.bin\"\n"
        "    input bf16[4]\n"
        "    labels bf16[2]\n"
        "}\n"
        "train {\n"
        "    model M\n"
        "    dataset D\n"
        "    loss mse\n"
        "    epochs 1\n"
        "    batch_size 1\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, AccettaTrainConLoraValido) {
    std::string source = kValidTrainProgram;
    auto pos = source.find("}\n", source.find("train {"));
    // Inserisce il blocco lora prima della chiusura di 'train'.
    source.insert(pos, "    lora {\n        rank 4\n        alpha 8.0\n    }\n");
    auto analyzer = analyze(source);
    EXPECT_FALSE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaLoraConRankNonPositivo) {
    std::string source = kValidTrainProgram;
    auto pos = source.find("}\n", source.find("train {"));
    source.insert(pos, "    lora {\n        rank 0\n    }\n");
    auto analyzer = analyze(source);
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, AccettaForecastValido) {
    auto analyzer = analyze(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(4)\n"
        "}\n"
        "forecast {\n"
        "    model M\n"
        "    horizon 10\n"
        "}\n");
    EXPECT_FALSE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaForecastConModelloNonDefinito) {
    auto analyzer = analyze(
        "forecast {\n"
        "    model NonEsiste\n"
        "    horizon 10\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaForecastConHorizonNonPositivo) {
    auto analyzer = analyze(
        "model M {\n"
        "    input bf16[4]\n"
        "}\n"
        "forecast {\n"
        "    model M\n"
        "    horizon 0\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}

TEST(SemanticAnalyzerTest, RifiutaForecastSenzaHorizon) {
    auto analyzer = analyze(
        "model M {\n"
        "    input bf16[4]\n"
        "}\n"
        "forecast {\n"
        "    model M\n"
        "}\n");
    EXPECT_TRUE(analyzer.diagnostics().hasErrors());
}
