#include "blackforge/frontend/parser.hpp"

#include <gtest/gtest.h>

#include "blackforge/ast/ast.hpp"
#include "blackforge/frontend/lexer.hpp"

using namespace blackforge;

namespace {

struct ParseResult {
    ast::Program program;
    DiagnosticList diagnostics;
};

ParseResult parse(const std::string& source) {
    Lexer lexer(source, "test.bf");
    auto tokens = lexer.tokenize();
    EXPECT_FALSE(lexer.diagnostics().hasErrors());

    Parser parser(std::move(tokens));
    ast::Program program = parser.parseProgram();
    return ParseResult{std::move(program), parser.diagnostics()};
}

}  // namespace

TEST(ParserTest, ParsaDichiarazioneTarget) {
    ParseResult result = parse("target nvidia.blackwell");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    ASSERT_EQ(result.program.declarations.size(), 1u);

    const auto& target = std::get<ast::TargetDecl>(result.program.declarations[0]);
    EXPECT_EQ(target.target.toString(), "nvidia.blackwell");
}

TEST(ParserTest, ParsaDichiarazionePrecisionConTuttiICampi) {
    ParseResult result = parse(
        "precision {\n"
        "    storage bf16\n"
        "    compute fp8.e4m3\n"
        "    accumulate fp32\n"
        "}");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    ASSERT_EQ(result.program.declarations.size(), 1u);

    const auto& precision = std::get<ast::PrecisionDecl>(result.program.declarations[0]);
    ASSERT_EQ(precision.fields.size(), 3u);

    EXPECT_EQ(precision.fields[0].kind, ast::PrecisionFieldKind::Storage);
    EXPECT_EQ(precision.fields[0].value.toString(), "bf16");

    EXPECT_EQ(precision.fields[1].kind, ast::PrecisionFieldKind::Compute);
    EXPECT_EQ(precision.fields[1].value.toString(), "fp8.e4m3");

    EXPECT_EQ(precision.fields[2].kind, ast::PrecisionFieldKind::Accumulate);
    EXPECT_EQ(precision.fields[2].value.toString(), "fp32");
}

TEST(ParserTest, ParsaModelloTinyModelCompleto) {
    ParseResult result = parse(
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

    ASSERT_FALSE(result.diagnostics.hasErrors());
    ASSERT_EQ(result.program.declarations.size(), 3u);

    const auto& model = std::get<ast::ModelDecl>(result.program.declarations[2]);
    EXPECT_EQ(model.name, "TinyModel");
    ASSERT_EQ(model.body.size(), 2u);

    const auto& inputDecl = std::get<ast::InputDecl>(model.body[0]);
    EXPECT_EQ(inputDecl.type.dtype.toString(), "bf16");
    ASSERT_EQ(inputDecl.type.shape.size(), 2u);
    EXPECT_TRUE(inputDecl.type.shape[0].isSymbolic);
    EXPECT_EQ(inputDecl.type.shape[0].symbolicName, "batch");
    EXPECT_FALSE(inputDecl.type.shape[1].isSymbolic);
    EXPECT_EQ(inputDecl.type.shape[1].literalValue, 4096);

    const auto& pipeline = std::get<ast::PipelineStmt>(model.body[1]);
    EXPECT_EQ(pipeline.source.kind, ast::PipelineSourceKind::Input);
    ASSERT_EQ(pipeline.stages.size(), 3u);

    EXPECT_EQ(pipeline.stages[0].name, "linear");
    ASSERT_EQ(pipeline.stages[0].args.size(), 1u);
    EXPECT_EQ(pipeline.stages[0].args[0].kind, ast::ExprKind::IntegerLiteral);
    EXPECT_EQ(pipeline.stages[0].args[0].text, "4096");

    EXPECT_EQ(pipeline.stages[1].name, "silu");
    EXPECT_TRUE(pipeline.stages[1].args.empty());

    EXPECT_EQ(pipeline.stages[2].name, "linear");
}

TEST(ParserTest, SegnalaErroreQuandoMancaParentesiGraffa) {
    ParseResult result = parse("precision storage bf16 }");

    EXPECT_TRUE(result.diagnostics.hasErrors());
}

TEST(ParserTest, RecuperaDaErroreEContinuaConLaDichiarazioneSuccessiva) {
    // La prima dichiarazione e' invalida (manca il nome del target),
    // ma la seconda ('precision { ... }') deve comunque essere analizzata.
    ParseResult result = parse(
        "target\n"
        "precision {\n"
        "    storage bf16\n"
        "}\n");

    EXPECT_TRUE(result.diagnostics.hasErrors());
    ASSERT_EQ(result.program.declarations.size(), 1u);
    EXPECT_TRUE(std::holds_alternative<ast::PrecisionDecl>(result.program.declarations[0]));
}

TEST(ParserTest, SegnalaPosizioneDellErroreSintattico) {
    ParseResult result = parse("model {}");  // manca il nome del modello

    ASSERT_TRUE(result.diagnostics.hasErrors());
    const auto& diags = result.diagnostics.all();
    EXPECT_EQ(diags[0].location.line, 1u);
    EXPECT_EQ(diags[0].location.column, 7u);
}

TEST(ParserTest, ParsaPipelineConSorgenteIdentificatore) {
    ParseResult result = parse(
        "model M {\n"
        "    hidden |> silu\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    const auto& model = std::get<ast::ModelDecl>(result.program.declarations[0]);
    const auto& pipeline = std::get<ast::PipelineStmt>(model.body[0]);

    EXPECT_EQ(pipeline.source.kind, ast::PipelineSourceKind::Identifier);
    EXPECT_EQ(pipeline.source.identifierName, "hidden");
}

TEST(ParserTest, DumpAstNonVuotoPerProgrammaValido) {
    ParseResult result = parse("target nvidia.blackwell");
    std::string text = ast::dump(result.program);

    EXPECT_NE(text.find("TargetDecl"), std::string::npos);
    EXPECT_NE(text.find("nvidia.blackwell"), std::string::npos);
}

TEST(ParserTest, ParsaDatasetDeclCompleto) {
    ParseResult result = parse(
        "dataset MyData {\n"
        "    path \"data/train.bin\"\n"
        "    input bf16[batch, 4]\n"
        "    labels bf16[batch, 2]\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    ASSERT_EQ(result.program.declarations.size(), 1u);

    const auto& dataset = std::get<ast::DatasetDecl>(result.program.declarations[0]);
    EXPECT_EQ(dataset.name, "MyData");
    ASSERT_EQ(dataset.fields.size(), 3u);

    const auto& pathField = std::get<ast::DatasetPathField>(dataset.fields[0]);
    EXPECT_EQ(pathField.path, "data/train.bin");

    const auto& inputField = std::get<ast::DatasetInputField>(dataset.fields[1]);
    EXPECT_EQ(inputField.type.dtype.toString(), "bf16");

    const auto& labelsField = std::get<ast::DatasetLabelsField>(dataset.fields[2]);
    EXPECT_EQ(labelsField.type.dtype.toString(), "bf16");
}

TEST(ParserTest, ParsaTrainDeclCompleto) {
    ParseResult result = parse(
        "train {\n"
        "    model TinyModel\n"
        "    dataset MyData\n"
        "    loss mse\n"
        "    optimizer adamw\n"
        "    epochs 10\n"
        "    batch_size 32\n"
        "    learning_rate 0.001\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    ASSERT_EQ(result.program.declarations.size(), 1u);

    const auto& train = std::get<ast::TrainDecl>(result.program.declarations[0]);
    ASSERT_EQ(train.fields.size(), 7u);

    EXPECT_EQ(std::get<ast::TrainModelField>(train.fields[0]).name, "TinyModel");
    EXPECT_EQ(std::get<ast::TrainDatasetField>(train.fields[1]).name, "MyData");
    EXPECT_EQ(std::get<ast::TrainLossField>(train.fields[2]).name, "mse");
    EXPECT_EQ(std::get<ast::TrainOptimizerField>(train.fields[3]).name, "adamw");
    EXPECT_EQ(std::get<ast::TrainEpochsField>(train.fields[4]).value, 10);
    EXPECT_EQ(std::get<ast::TrainBatchSizeField>(train.fields[5]).value, 32);
    EXPECT_DOUBLE_EQ(std::get<ast::TrainLearningRateField>(train.fields[6]).value, 0.001);
}

TEST(ParserTest, SegnalaErroreSuCampoDatasetSconosciuto) {
    ParseResult result = parse(
        "dataset MyData {\n"
        "    foo \"bar\"\n"
        "}\n");
    EXPECT_TRUE(result.diagnostics.hasErrors());
}

TEST(ParserTest, SegnalaErroreSuCampoTrainSconosciuto) {
    ParseResult result = parse(
        "train {\n"
        "    foo bar\n"
        "}\n");
    EXPECT_TRUE(result.diagnostics.hasErrors());
}

TEST(ParserTest, ParsaTrainDeclConLora) {
    ParseResult result = parse(
        "train {\n"
        "    model M\n"
        "    dataset D\n"
        "    loss mse\n"
        "    optimizer adamw\n"
        "    epochs 5\n"
        "    batch_size 8\n"
        "    lora {\n"
        "        rank 4\n"
        "        alpha 8.0\n"
        "    }\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    const auto& train = std::get<ast::TrainDecl>(result.program.declarations[0]);
    ASSERT_EQ(train.fields.size(), 7u);

    const auto& lora = std::get<ast::TrainLoraField>(train.fields[6]);
    EXPECT_EQ(lora.rank, 4);
    EXPECT_DOUBLE_EQ(lora.alpha, 8.0);
}

TEST(ParserTest, ParsaTrainDeclConLoraSenzaAlphaUsaRankComeDefault) {
    ParseResult result = parse(
        "train {\n"
        "    model M\n"
        "    dataset D\n"
        "    loss mse\n"
        "    optimizer sgd\n"
        "    epochs 1\n"
        "    batch_size 1\n"
        "    lora {\n"
        "        rank 6\n"
        "    }\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    const auto& train = std::get<ast::TrainDecl>(result.program.declarations[0]);
    const auto& lora = std::get<ast::TrainLoraField>(train.fields[6]);
    EXPECT_EQ(lora.rank, 6);
    EXPECT_DOUBLE_EQ(lora.alpha, 6.0);
}

TEST(ParserTest, SegnalaErroreSeLoraNonHaRank) {
    ParseResult result = parse(
        "train {\n"
        "    model M\n"
        "    dataset D\n"
        "    loss mse\n"
        "    optimizer sgd\n"
        "    epochs 1\n"
        "    batch_size 1\n"
        "    lora {\n"
        "        alpha 8.0\n"
        "    }\n"
        "}\n");
    EXPECT_TRUE(result.diagnostics.hasErrors());
}

TEST(ParserTest, ParsaForecastDeclCompleto) {
    ParseResult result = parse(
        "forecast {\n"
        "    model M\n"
        "    horizon 10\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    ASSERT_EQ(result.program.declarations.size(), 1u);

    const auto& forecast = std::get<ast::ForecastDecl>(result.program.declarations[0]);
    ASSERT_EQ(forecast.fields.size(), 2u);
    EXPECT_EQ(std::get<ast::ForecastModelField>(forecast.fields[0]).name, "M");
    EXPECT_EQ(std::get<ast::ForecastHorizonField>(forecast.fields[1]).value, 10);
}

TEST(ParserTest, SegnalaErroreSuCampoForecastSconosciuto) {
    ParseResult result = parse(
        "forecast {\n"
        "    foo bar\n"
        "}\n");
    EXPECT_TRUE(result.diagnostics.hasErrors());
}
