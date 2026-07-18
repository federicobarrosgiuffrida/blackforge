#include "blackforge/backend/cuda/executor.hpp"

#include <gtest/gtest.h>

#include "blackforge/backend/cpu/executor.hpp"
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

TEST(CudaExecutorTest, EseguePipelineEProduceLaFormaAttesa) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 16]\n"
        "    input |> linear(8) |> silu |> linear(4)\n"
        "}\n");

    const ir::ModelIR& model = module.models.front();
    backend::cuda::Executor executor;

    runtime::Tensor input = executor.makeSyntheticInput(model.valueById(model.inputValue), /*batchSize=*/5);
    EXPECT_EQ(input.shape(), (std::vector<std::size_t>{5, 16}));

    runtime::Tensor output = executor.run(model, input);
    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{5, 4}));
}

TEST(CudaExecutorTest, ProduceLoStessoRisultatoDelBackendCpuAParitaDiSeme) {
    // Il test di parita' piu' importante di questa milestone: stesso
    // modello, stesso seme (quindi stessi pesi e stesso input
    // sintetico) su CPU e su GPU devono produrre risultati
    // numericamente equivalenti. Piccole differenze sono attese per
    // via del diverso ordine delle somme in virgola mobile tra i loop
    // CPU e cuBLAS/i kernel GPU.
    ir::Module module = buildModule(
        "model TinyModel {\n"
        "    input bf16[batch, 32]\n"
        "    input |> rmsnorm |> linear(16) |> silu |> linear(8) |> relu |> linear(4)\n"
        "}\n");

    const ir::ModelIR& model = module.models.front();

    backend::cpu::Executor cpuExecutor(/*seed=*/7);
    backend::cuda::Executor cudaExecutor(/*seed=*/7);

    runtime::Tensor cpuInput = cpuExecutor.makeSyntheticInput(model.valueById(model.inputValue), 3);
    runtime::Tensor cudaInput = cudaExecutor.makeSyntheticInput(model.valueById(model.inputValue), 3);

    // Stesso seme => stesso input sintetico: prerequisito del confronto.
    ASSERT_EQ(cpuInput.elementCount(), cudaInput.elementCount());
    for (std::size_t i = 0; i < cpuInput.elementCount(); ++i) {
        ASSERT_FLOAT_EQ(cpuInput.at(i), cudaInput.at(i));
    }

    runtime::Tensor cpuOutput = cpuExecutor.run(model, cpuInput);
    runtime::Tensor cudaOutput = cudaExecutor.run(model, cudaInput);

    ASSERT_EQ(cpuOutput.shape(), cudaOutput.shape());
    for (std::size_t i = 0; i < cpuOutput.elementCount(); ++i) {
        EXPECT_NEAR(cpuOutput.at(i), cudaOutput.at(i), 1e-2F) << "indice " << i;
    }
}

TEST(CudaExecutorTest, LanciaSeIlModelloNonHaPipeline) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[4]\n"
        "}\n");

    const ir::ModelIR& model = module.models.front();
    backend::cuda::Executor executor;
    runtime::Tensor input = executor.makeSyntheticInput(model.valueById(model.inputValue), 1);

    EXPECT_THROW((void)executor.run(model, input), std::invalid_argument);
}
