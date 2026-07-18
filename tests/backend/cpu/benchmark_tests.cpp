#include "blackforge/backend/cpu/benchmark.hpp"

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

TEST(BenchmarkTest, ProduceStatisticheCoerenti) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 8]\n"
        "    input |> linear(4) |> silu |> linear(2)\n"
        "}\n");

    backend::cpu::BenchmarkResult result =
        backend::cpu::runBenchmark(module.models.front(), /*batchSize=*/4, /*warmup=*/2, /*measured=*/5);

    EXPECT_EQ(result.inputShape, (std::vector<std::size_t>{4, 8}));
    EXPECT_EQ(result.warmupIterations, 2u);
    EXPECT_EQ(result.measuredIterations, 5u);
    EXPECT_GT(result.meanMilliseconds, 0.0);
    EXPECT_GT(result.throughputSamplesPerSecond, 0.0);
    // input (4*8) + attivazioni (4*4 + 4*4 + 4*2) + pesi/bias
    // (8*4+4 per il primo linear, 4*2+2 per il secondo) = 32+16+16+8+36+10 = 118 elementi.
    EXPECT_EQ(result.estimatedMemoryBytes, 118u * sizeof(float));
}

TEST(BenchmarkTest, LanciaSeIlModelloNonHaPipeline) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[4]\n"
        "}\n");
    EXPECT_THROW((void)backend::cpu::runBenchmark(module.models.front(), 1, 0, 1), std::invalid_argument);
}

TEST(BenchmarkTest, LanciaSeMeasuredIterationsEZero) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[4]\n"
        "    input |> linear(2)\n"
        "}\n");
    EXPECT_THROW((void)backend::cpu::runBenchmark(module.models.front(), 1, 0, 0), std::invalid_argument);
}
