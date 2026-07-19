#include "blackforge/ir/ir_builder.hpp"

#include <gtest/gtest.h>

#include "blackforge/frontend/lexer.hpp"
#include "blackforge/frontend/parser.hpp"

using namespace blackforge;

namespace {

struct BuildResult {
    ir::Module module;
    DiagnosticList diagnostics;
};

BuildResult buildIR(const std::string& source) {
    Lexer lexer(source, "test.bf");
    auto tokens = lexer.tokenize();
    EXPECT_FALSE(lexer.diagnostics().hasErrors());

    Parser parser(std::move(tokens));
    ast::Program program = parser.parseProgram();
    EXPECT_FALSE(parser.diagnostics().hasErrors());

    ir::IRBuilder builder;
    ir::Module module = builder.build(program);
    return BuildResult{std::move(module), builder.diagnostics()};
}

}  // namespace

TEST(IRBuilderTest, CostruisceModuloConTargetEModello) {
    BuildResult result = buildIR(
        "target nvidia.blackwell\n"
        "model M {\n"
        "    input bf16[batch, 4096]\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    ASSERT_TRUE(result.module.target.has_value());
    EXPECT_EQ(*result.module.target, "nvidia.blackwell");

    ASSERT_EQ(result.module.models.size(), 1u);
    const ir::ModelIR& model = result.module.models[0];
    EXPECT_EQ(model.name, "M");

    const ir::Value& input = model.valueById(model.inputValue);
    EXPECT_EQ(input.dtype, sema::DType::BF16);
    ASSERT_EQ(input.shape.size(), 2u);
    EXPECT_TRUE(input.shape[0].isSymbolic);
    EXPECT_EQ(input.shape[0].symbolicName, "batch");
    EXPECT_FALSE(input.shape[1].isSymbolic);
    EXPECT_EQ(input.shape[1].literalValue, 4096);
}

TEST(IRBuilderTest, PropagaLaFormaAttraversoLaPipeline) {
    BuildResult result = buildIR(
        "model TinyModel {\n"
        "    input bf16[batch, 4096]\n"
        "\n"
        "    input\n"
        "        |> linear(2048)\n"
        "        |> silu\n"
        "        |> linear(512)\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    ASSERT_EQ(result.module.models.size(), 1u);
    const ir::ModelIR& model = result.module.models[0];

    ASSERT_EQ(model.pipelines.size(), 1u);
    const ir::Pipeline& pipeline = model.pipelines[0];
    ASSERT_EQ(pipeline.operations.size(), 3u);

    // linear(2048): [batch, 4096] -> [batch, 2048]
    const ir::Value& afterLinear1 = model.valueById(pipeline.operations[0].output);
    EXPECT_EQ(pipeline.operations[0].kind, ir::OpKind::Linear);
    EXPECT_EQ(pipeline.operations[0].linearOutFeatures, 2048);
    ASSERT_EQ(afterLinear1.shape.size(), 2u);
    EXPECT_TRUE(afterLinear1.shape[0].isSymbolic);
    EXPECT_EQ(afterLinear1.shape[1].literalValue, 2048);

    // silu: forma invariata
    const ir::Value& afterSilu = model.valueById(pipeline.operations[1].output);
    EXPECT_EQ(pipeline.operations[1].kind, ir::OpKind::Silu);
    EXPECT_EQ(afterSilu.shape[1].literalValue, 2048);

    // linear(512): [batch, 2048] -> [batch, 512]
    const ir::Value& afterLinear2 = model.valueById(pipeline.operations[2].output);
    EXPECT_EQ(afterLinear2.shape[1].literalValue, 512);
    EXPECT_EQ(pipeline.outputValue(), afterLinear2.id);

    // Il formato numerico si propaga invariato lungo tutta la pipeline.
    EXPECT_EQ(afterLinear1.dtype, sema::DType::BF16);
    EXPECT_EQ(afterSilu.dtype, sema::DType::BF16);
    EXPECT_EQ(afterLinear2.dtype, sema::DType::BF16);
}

TEST(IRBuilderTest, PropagaLaFormaAttraversoUnaPipelineDaModelloLinguistico) {
    BuildResult result = buildIR(
        "model TinyLM {\n"
        "    input bf16[batch, 8]\n"
        "\n"
        "    input\n"
        "        |> embedding(100, 16)\n"
        "        |> positional_embedding(8)\n"
        "        |> attention(4)\n"
        "        |> feedforward(32)\n"
        "        |> linear(100)\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    const ir::ModelIR& model = result.module.models[0];
    const ir::Pipeline& pipeline = model.pipelines[0];
    ASSERT_EQ(pipeline.operations.size(), 5u);

    // embedding(100, 16): [batch, 8] -> [batch, 8, 16] (aggiunge una dimensione).
    const ir::Operation& embeddingOp = pipeline.operations[0];
    EXPECT_EQ(embeddingOp.kind, ir::OpKind::Embedding);
    EXPECT_EQ(embeddingOp.embeddingVocabSize, 100);
    EXPECT_EQ(embeddingOp.embeddingDim, 16);
    const ir::Value& afterEmbedding = model.valueById(embeddingOp.output);
    ASSERT_EQ(afterEmbedding.shape.size(), 3u);
    EXPECT_TRUE(afterEmbedding.shape[0].isSymbolic);
    EXPECT_EQ(afterEmbedding.shape[1].literalValue, 8);
    EXPECT_EQ(afterEmbedding.shape[2].literalValue, 16);

    // positional_embedding(8): forma invariata.
    const ir::Operation& posOp = pipeline.operations[1];
    EXPECT_EQ(posOp.kind, ir::OpKind::PositionalEmbedding);
    EXPECT_EQ(posOp.positionalMaxSeqLen, 8);
    const ir::Value& afterPositional = model.valueById(posOp.output);
    ASSERT_EQ(afterPositional.shape.size(), 3u);
    EXPECT_EQ(afterPositional.shape[1].literalValue, 8);
    EXPECT_EQ(afterPositional.shape[2].literalValue, 16);

    // attention(4): forma invariata (residual + pre-norm interni).
    const ir::Operation& attnOp = pipeline.operations[2];
    EXPECT_EQ(attnOp.kind, ir::OpKind::Attention);
    EXPECT_EQ(attnOp.attentionNumHeads, 4);
    const ir::Value& afterAttention = model.valueById(attnOp.output);
    ASSERT_EQ(afterAttention.shape.size(), 3u);
    EXPECT_EQ(afterAttention.shape[1].literalValue, 8);
    EXPECT_EQ(afterAttention.shape[2].literalValue, 16);

    // feedforward(32): forma invariata.
    const ir::Operation& ffOp = pipeline.operations[3];
    EXPECT_EQ(ffOp.kind, ir::OpKind::FeedForward);
    EXPECT_EQ(ffOp.feedForwardHiddenDim, 32);
    const ir::Value& afterFeedForward = model.valueById(ffOp.output);
    ASSERT_EQ(afterFeedForward.shape.size(), 3u);
    EXPECT_EQ(afterFeedForward.shape[1].literalValue, 8);
    EXPECT_EQ(afterFeedForward.shape[2].literalValue, 16);

    // linear(100): rimpiazza l'ultima dimensione con le feature in uscita.
    const ir::Operation& linearOp = pipeline.operations[4];
    EXPECT_EQ(linearOp.kind, ir::OpKind::Linear);
    const ir::Value& afterLinear = model.valueById(linearOp.output);
    ASSERT_EQ(afterLinear.shape.size(), 3u);
    EXPECT_EQ(afterLinear.shape[1].literalValue, 8);
    EXPECT_EQ(afterLinear.shape[2].literalValue, 100);
}

TEST(IRBuilderTest, BidirectionalAttentionNonAlteraFormaEHaIlProprioOpKind) {
    // Stesso principio di 'attention': forma invariata, residual/pre-norm
    // interni — ma OpKind distinto (BidirectionalAttention, non
    // Attention), cosi' backend::cpu::Model sa quale forward/backward
    // (con o senza maschera causale) dispatchare.
    BuildResult result = buildIR(
        "model TinyMLM {\n"
        "    input bf16[batch, 8]\n"
        "    input |> embedding(100, 16) |> bidirectional_attention(4)\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    const ir::ModelIR& model = result.module.models[0];
    const ir::Pipeline& pipeline = model.pipelines[0];
    ASSERT_EQ(pipeline.operations.size(), 2u);

    const ir::Operation& bidiOp = pipeline.operations[1];
    EXPECT_EQ(bidiOp.kind, ir::OpKind::BidirectionalAttention);
    EXPECT_EQ(bidiOp.attentionNumHeads, 4);

    const ir::Value& afterEmbedding = model.valueById(pipeline.operations[0].output);
    const ir::Value& afterBidi = model.valueById(bidiOp.output);
    ASSERT_EQ(afterBidi.shape.size(), 3u);
    EXPECT_EQ(afterBidi.shape[1].literalValue, afterEmbedding.shape[1].literalValue);
    EXPECT_EQ(afterBidi.shape[2].literalValue, afterEmbedding.shape[2].literalValue);
}

TEST(IRBuilderTest, RmsnormNonAlteraFormaNeFormato) {
    BuildResult result = buildIR(
        "model TinyModel {\n"
        "    input bf16[batch, 4096]\n"
        "    input |> rmsnorm |> linear(2048)\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    const ir::ModelIR& model = result.module.models[0];
    const ir::Pipeline& pipeline = model.pipelines[0];
    ASSERT_EQ(pipeline.operations.size(), 2u);

    const ir::Value& afterRmsnorm = model.valueById(pipeline.operations[0].output);
    EXPECT_EQ(pipeline.operations[0].kind, ir::OpKind::RmsNorm);
    ASSERT_EQ(afterRmsnorm.shape.size(), 2u);
    EXPECT_TRUE(afterRmsnorm.shape[0].isSymbolic);
    EXPECT_EQ(afterRmsnorm.shape[1].literalValue, 4096);
    EXPECT_EQ(afterRmsnorm.dtype, sema::DType::BF16);
}

TEST(IRBuilderTest, SoftmaxNonAlteraFormaNeFormato) {
    BuildResult result = buildIR(
        "model TinyModel {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(3) |> softmax\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    const ir::ModelIR& model = result.module.models[0];
    const ir::Pipeline& pipeline = model.pipelines[0];
    ASSERT_EQ(pipeline.operations.size(), 2u);

    const ir::Value& afterSoftmax = model.valueById(pipeline.operations[1].output);
    EXPECT_EQ(pipeline.operations[1].kind, ir::OpKind::Softmax);
    ASSERT_EQ(afterSoftmax.shape.size(), 2u);
    EXPECT_EQ(afterSoftmax.shape[1].literalValue, 3);
    EXPECT_EQ(afterSoftmax.dtype, sema::DType::BF16);
}

TEST(IRBuilderTest, PipelineSenzaOperazioniHaOutputUgualeAllInput) {
    BuildResult result = buildIR(
        "model M {\n"
        "    input bf16[4096]\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    const ir::ModelIR& model = result.module.models[0];
    EXPECT_TRUE(model.pipelines.empty());
    EXPECT_EQ(model.values.size(), 1u);
}

TEST(IRBuilderTest, SegnalaModelloSenzaInput) {
    // Costruito a mano (bypassando il parser) non e' necessario: un
    // modello senza input non e' esprimibile con una pipeline valida,
    // quindi verifichiamo che l'IRBuilder lo rifiuti comunque quando
    // interrogato direttamente su un programma senza dichiarazioni utili.
    BuildResult result = buildIR("model M {}\n");
    EXPECT_TRUE(result.diagnostics.hasErrors());
    EXPECT_TRUE(result.module.models.empty());
}

TEST(IRBuilderTest, RifiutaSorgentePipelineNonInput) {
    BuildResult result = buildIR(
        "model M {\n"
        "    input bf16[4096]\n"
        "    hidden |> silu\n"
        "}\n");

    EXPECT_TRUE(result.diagnostics.hasErrors());
}

TEST(IRBuilderTest, ModuloSenzaBloccoPrecisionNonHaPolicy) {
    BuildResult result = buildIR(
        "model M {\n"
        "    input bf16[4096]\n"
        "}\n");
    ASSERT_FALSE(result.diagnostics.hasErrors());
    EXPECT_FALSE(result.module.precision.has_value());
}

TEST(IRBuilderTest, EstraeLaPrecisionPolicyDalBloccoPrecision) {
    BuildResult result = buildIR(
        "precision {\n"
        "    storage bf16\n"
        "    compute fp8.e4m3\n"
        "    accumulate fp32\n"
        "}\n"
        "model M {\n"
        "    input bf16[4096]\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    ASSERT_TRUE(result.module.precision.has_value());
    EXPECT_EQ(result.module.precision->storage, sema::DType::BF16);
    EXPECT_EQ(result.module.precision->compute, sema::DType::FP8_E4M3);
    EXPECT_EQ(result.module.precision->accumulate, sema::DType::FP32);
}

TEST(IRBuilderTest, ParametersEForwardSonoAliasDiStorageECompute) {
    BuildResult result = buildIR(
        "precision {\n"
        "    parameters bf16\n"
        "    forward fp8.e5m2\n"
        "}\n"
        "model M {\n"
        "    input bf16[4096]\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    ASSERT_TRUE(result.module.precision.has_value());
    EXPECT_EQ(result.module.precision->storage, sema::DType::BF16);
    EXPECT_EQ(result.module.precision->compute, sema::DType::FP8_E5M2);
}

TEST(IRBuilderTest, DumpModuloContieneNomeModelloEForme) {
    BuildResult result = buildIR(
        "model TinyModel {\n"
        "    input bf16[batch, 4096]\n"
        "    input |> linear(2048)\n"
        "}\n");

    ASSERT_FALSE(result.diagnostics.hasErrors());
    std::string text = ir::dump(result.module);

    EXPECT_NE(text.find("TinyModel"), std::string::npos);
    EXPECT_NE(text.find("bf16[batch, 4096]"), std::string::npos);
    EXPECT_NE(text.find("bf16[batch, 2048]"), std::string::npos);
}
