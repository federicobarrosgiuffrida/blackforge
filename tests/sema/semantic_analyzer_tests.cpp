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
