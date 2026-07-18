#include "blackforge/backend/cpu/forecast_runner.hpp"

#include <cstdio>
#include <filesystem>

#include <gtest/gtest.h>

#include "blackforge/backend/cpu/checkpoint.hpp"
#include "blackforge/backend/cpu/model.hpp"
#include "blackforge/frontend/lexer.hpp"
#include "blackforge/frontend/parser.hpp"
#include "blackforge/ir/ir_builder.hpp"

using namespace blackforge;

namespace {

struct TempFile {
    std::string path;

    explicit TempFile(const std::string& name)
        : path((std::filesystem::temp_directory_path() / name).string()) {}

    ~TempFile() { std::remove(path.c_str()); }
};

struct Compiled {
    ast::Program program;
    ir::Module module;
};

Compiled compile(const std::string& source) {
    Lexer lexer(source, "test.bf");
    auto tokens = lexer.tokenize();
    EXPECT_FALSE(lexer.diagnostics().hasErrors());

    Parser parser(std::move(tokens));
    ast::Program program = parser.parseProgram();
    EXPECT_FALSE(parser.diagnostics().hasErrors());

    ir::IRBuilder builder;
    ir::Module module = builder.build(program);
    EXPECT_FALSE(builder.diagnostics().hasErrors());

    return Compiled{std::move(program), std::move(module)};
}

}  // namespace

TEST(ForecastRunnerTest, GeneraIlNumeroDiPassiRichiesto) {
    // Input e output hanno entrambi ultima dimensione 4: il modello e'
    // "autoregressivo-compatibile" (l'output puo' diventare il prossimo
    // input).
    Compiled compiled = compile(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(4)\n"
        "}\n"
        "forecast {\n"
        "    model M\n"
        "    horizon 5\n"
        "}\n");

    TempFile checkpointFile("blackforge_test_forecast_checkpoint.bfckpt");
    backend::cpu::Model model(compiled.module.models.front());
    backend::cpu::saveCheckpoint(model, checkpointFile.path);

    backend::cpu::ForecastRunResult result =
        backend::cpu::runForecast(compiled.program, compiled.module, checkpointFile.path, /*batchSize=*/2);

    ASSERT_EQ(result.steps.size(), 5u);
    EXPECT_EQ(result.modelName, "M");
    for (const auto& step : result.steps) {
        EXPECT_EQ(step.shape(), (std::vector<std::size_t>{2, 4}));
    }
}

TEST(ForecastRunnerTest, LanciaSenzaCheckpoint) {
    Compiled compiled = compile(
        "model M {\n"
        "    input bf16[4]\n"
        "    input |> linear(4)\n"
        "}\n"
        "forecast {\n"
        "    model M\n"
        "    horizon 3\n"
        "}\n");

    EXPECT_THROW((void)backend::cpu::runForecast(compiled.program, compiled.module, "", 1), std::runtime_error);
}

TEST(ForecastRunnerTest, LanciaSeInputEOutputNonCoincidono) {
    // linear(2) produce un output a 2 feature partendo da un input a 4:
    // non e' possibile riusare l'output come input del passo successivo.
    Compiled compiled = compile(
        "model M {\n"
        "    input bf16[4]\n"
        "    input |> linear(2)\n"
        "}\n"
        "forecast {\n"
        "    model M\n"
        "    horizon 3\n"
        "}\n");

    TempFile checkpointFile("blackforge_test_forecast_mismatch.bfckpt");
    backend::cpu::Model model(compiled.module.models.front());
    backend::cpu::saveCheckpoint(model, checkpointFile.path);

    EXPECT_THROW((void)backend::cpu::runForecast(compiled.program, compiled.module, checkpointFile.path, 1),
                 std::runtime_error);
}

TEST(ForecastRunnerTest, LanciaSeIlProgrammaNonHaBloccoForecast) {
    Compiled compiled = compile(
        "model M {\n"
        "    input bf16[4]\n"
        "}\n");

    EXPECT_THROW((void)backend::cpu::runForecast(compiled.program, compiled.module, "qualcosa", 1),
                 std::runtime_error);
}

TEST(ForecastRunnerTest, IPassiSuccessiviUsanoLoutputPrecedenteComeInput) {
    // Con un layer lineare identita' (pesi = matrice identita', bias =
    // 0), l'output di ogni passo deve essere identico all'input di quel
    // passo: verifica che il rollout autoregressivo alimenti davvero
    // l'output del passo N come input del passo N+1.
    Compiled compiled = compile(
        "model M {\n"
        "    input bf16[batch, 2]\n"
        "    input |> linear(2)\n"
        "}\n"
        "forecast {\n"
        "    model M\n"
        "    horizon 3\n"
        "}\n");

    backend::cpu::Model model(compiled.module.models.front());
    auto params = model.allParameters();  // [weight, bias]
    // Sovrascrive i pesi con l'identita' e il bias con zero.
    params[0]->value = runtime::Tensor({2, 2}, {1.0F, 0.0F, 0.0F, 1.0F});
    params[1]->value = runtime::Tensor({2}, {0.0F, 0.0F});

    TempFile checkpointFile("blackforge_test_forecast_identity.bfckpt");
    backend::cpu::saveCheckpoint(model, checkpointFile.path);

    backend::cpu::ForecastRunResult result =
        backend::cpu::runForecast(compiled.program, compiled.module, checkpointFile.path, /*batchSize=*/1);

    ASSERT_EQ(result.steps.size(), 3u);
    // Con un layer identita' ogni passo restituisce esattamente lo
    // stesso tensore del passo precedente (compreso il primo, che parte
    // dall'input sintetico iniziale).
    for (std::size_t i = 1; i < result.steps.size(); ++i) {
        for (std::size_t j = 0; j < result.steps[i].elementCount(); ++j) {
            EXPECT_FLOAT_EQ(result.steps[i].at(j), result.steps[i - 1].at(j));
        }
    }
}
