#include "blackforge/backend/cpu/executor.hpp"

#include <gtest/gtest.h>

#include "blackforge/frontend/lexer.hpp"
#include "blackforge/frontend/parser.hpp"
#include "blackforge/ir/ir_builder.hpp"

using namespace blackforge;

namespace {

ir::Module buildModule(const std::string& source) {
    Lexer lexer(source, "test.bf");
    auto tokens = lexer.tokenize();
    EXPECT_FALSE(lexer.diagnostics().hasErrors());

    Parser parser(std::move(tokens));
    ast::Program program = parser.parseProgram();
    EXPECT_FALSE(parser.diagnostics().hasErrors());

    ir::IRBuilder builder;
    ir::Module module = builder.build(program);
    EXPECT_FALSE(builder.diagnostics().hasErrors());
    return module;
}

}  // namespace

TEST(CpuExecutorTest, EseguePipelineEProduceLaFormaAttesa) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 16]\n"
        "    input |> linear(8) |> silu |> linear(4)\n"
        "}\n");

    const ir::ModelIR& model = module.models.front();
    backend::cpu::Executor executor;

    runtime::Tensor input = executor.makeSyntheticInput(model.valueById(model.inputValue), /*batchSize=*/5);
    EXPECT_EQ(input.shape(), (std::vector<std::size_t>{5, 16}));

    runtime::Tensor output = executor.run(model, input);
    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{5, 4}));
}

TEST(CpuExecutorTest, EsecuzioneDeterministicaAParitaDiSeme) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 8]\n"
        "    input |> linear(4)\n"
        "}\n");

    const ir::ModelIR& model = module.models.front();

    backend::cpu::Executor executorA(/*seed=*/7);
    backend::cpu::Executor executorB(/*seed=*/7);

    runtime::Tensor inputA = executorA.makeSyntheticInput(model.valueById(model.inputValue), 2);
    runtime::Tensor inputB = executorB.makeSyntheticInput(model.valueById(model.inputValue), 2);

    runtime::Tensor outputA = executorA.run(model, inputA);
    runtime::Tensor outputB = executorB.run(model, inputB);

    ASSERT_EQ(outputA.elementCount(), outputB.elementCount());
    for (std::size_t i = 0; i < outputA.elementCount(); ++i) {
        EXPECT_FLOAT_EQ(outputA.at(i), outputB.at(i));
    }
}

TEST(CpuExecutorTest, SemiDiversiProduconoPesiDiversi) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 8]\n"
        "    input |> linear(4)\n"
        "}\n");

    const ir::ModelIR& model = module.models.front();

    backend::cpu::Executor executorA(/*seed=*/1);
    backend::cpu::Executor executorB(/*seed=*/2);

    runtime::Tensor inputA = executorA.makeSyntheticInput(model.valueById(model.inputValue), 2);
    runtime::Tensor inputB = executorB.makeSyntheticInput(model.valueById(model.inputValue), 2);

    runtime::Tensor outputA = executorA.run(model, inputA);
    runtime::Tensor outputB = executorB.run(model, inputB);

    bool allEqual = true;
    for (std::size_t i = 0; i < outputA.elementCount(); ++i) {
        if (outputA.at(i) != outputB.at(i)) {
            allEqual = false;
            break;
        }
    }
    EXPECT_FALSE(allEqual);
}

TEST(CpuExecutorTest, EseguePipelineConRmsnorm) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 8]\n"
        "    input |> rmsnorm |> linear(4)\n"
        "}\n");

    const ir::ModelIR& model = module.models.front();
    backend::cpu::Executor executor;
    runtime::Tensor input = executor.makeSyntheticInput(model.valueById(model.inputValue), 3);

    runtime::Tensor output = executor.run(model, input);
    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{3, 4}));
}

TEST(CpuExecutorTest, LanciaSeIlModelloNonHaPipeline) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[4]\n"
        "}\n");

    const ir::ModelIR& model = module.models.front();
    backend::cpu::Executor executor;
    runtime::Tensor input = executor.makeSyntheticInput(model.valueById(model.inputValue), 1);

    EXPECT_THROW((void)executor.run(model, input), std::invalid_argument);
}

TEST(CpuExecutorTest, SenzaPrecisionPolicyIlRisultatoENonQuantizzato) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 8]\n"
        "    input |> linear(4)\n"
        "}\n");

    const ir::ModelIR& model = module.models.front();
    backend::cpu::Executor executor;
    runtime::Tensor input = executor.makeSyntheticInput(model.valueById(model.inputValue), 2);

    runtime::Tensor withoutPolicy = executor.run(model, input);
    runtime::Tensor withFp32Policy = executor.run(model, input, ir::PrecisionPolicy{});

    // Una policy tutta fp32 (il default di PrecisionPolicy) e' un
    //'identita': stesso risultato di non passare alcuna policy.
    ASSERT_EQ(withoutPolicy.elementCount(), withFp32Policy.elementCount());
    for (std::size_t i = 0; i < withoutPolicy.elementCount(); ++i) {
        EXPECT_FLOAT_EQ(withoutPolicy.at(i), withFp32Policy.at(i));
    }
}

TEST(CpuExecutorTest, UnaPrecisionPolicyRidottaCambiaDavveroIlRisultato) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 8]\n"
        "    input |> linear(4) |> silu |> linear(2)\n"
        "}\n");

    const ir::ModelIR& model = module.models.front();
    backend::cpu::Executor executor;
    runtime::Tensor input = executor.makeSyntheticInput(model.valueById(model.inputValue), 2);

    ir::PrecisionPolicy fp8Policy{sema::DType::FP8_E4M3, sema::DType::FP8_E4M3, sema::DType::FP32};

    runtime::Tensor full = executor.run(model, input);
    runtime::Tensor quantized = executor.run(model, input, fp8Policy);

    ASSERT_EQ(full.elementCount(), quantized.elementCount());
    bool anyDifferent = false;
    for (std::size_t i = 0; i < full.elementCount(); ++i) {
        if (full.at(i) != quantized.at(i)) {
            anyDifferent = true;
            break;
        }
    }
    EXPECT_TRUE(anyDifferent) << "una policy fp8 dovrebbe produrre numeri diversi da fp32 pieno";
}
