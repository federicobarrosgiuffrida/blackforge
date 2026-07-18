#include "blackforge/backend/cpu/model.hpp"

#include <algorithm>

#include <gtest/gtest.h>

#include "blackforge/backend/cpu/loss.hpp"
#include "blackforge/backend/cpu/optimizer.hpp"
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

TEST(ModelTest, ForwardProduceLaFormaAttesa) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(3) |> silu |> linear(2)\n"
        "}\n");
    backend::cpu::Model model(module.models.front());

    runtime::Tensor input({2, 4}, {0.1F, 0.2F, 0.3F, 0.4F, -0.1F, -0.2F, -0.3F, -0.4F});
    runtime::Tensor output = model.forward(input);
    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{2, 2}));
}

TEST(ModelTest, ParametersRestituisceUnaCoppiaPesoBiasPerOgniLinear) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(3) |> linear(2)\n"
        "}\n");
    backend::cpu::Model model(module.models.front());

    auto params = model.parameters();
    ASSERT_EQ(params.size(), 4u);  // weight+bias per ciascuno dei 2 layer linear
}

TEST(ModelTest, LanciaSeLaDimensioneDelleFeatureENonConcreta) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[features]\n"
        "    input |> linear(2)\n"
        "}\n");
    // 'features' e' l'unica dimensione ed e' simbolica: impossibile
    // allocare pesi concreti per il layer linear.
    EXPECT_THROW(backend::cpu::Model(module.models.front()), std::invalid_argument);
}

TEST(ModelTest, BackwardCorrispondeAllaDerivataNumericaDelLoss) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 3]\n"
        "    input |> linear(4) |> silu |> linear(2)\n"
        "}\n");

    backend::cpu::Model model(module.models.front());
    runtime::Tensor input({2, 3}, {0.2F, -0.1F, 0.4F, -0.3F, 0.5F, 0.1F});
    runtime::Tensor target({2, 2}, {0.0F, 1.0F, 1.0F, 0.0F});

    model.zeroGrad();
    runtime::Tensor output = model.forward(input);
    backend::cpu::LossResult loss = backend::cpu::meanSquaredError(output, target);
    model.backward(loss.grad);

    auto lossOf = [&]() {
        runtime::Tensor out = model.forward(input);
        return backend::cpu::meanSquaredError(out, target).value;
    };

    auto params = model.parameters();
    ASSERT_FALSE(params.empty());

    backend::cpu::Parameter* param = params.front();
    float eps = 1e-3F;
    for (std::size_t i = 0; i < std::min<std::size_t>(5, param->value.elementCount()); ++i) {
        float original = param->value.at(i);

        param->value.at(i) = original + eps;
        float plus = lossOf();
        param->value.at(i) = original - eps;
        float minus = lossOf();
        param->value.at(i) = original;

        float numeric = (plus - minus) / (2.0F * eps);
        EXPECT_NEAR(param->grad.at(i), numeric, 1e-2F) << "parametro " << param->name << " indice " << i;
    }
}

namespace {

// Piccolo problema di regressione risolvibile esattamente da un solo
// layer lineare: verifica che un ciclo di addestramento completo
// (forward -> loss -> backward -> optimizer.step) riduca davvero la
// loss, non solo che le singole parti compilino.
struct ToyProblem {
    runtime::Tensor input;
    runtime::Tensor target;
};

ToyProblem makeToyProblem() {
    runtime::Tensor input({4, 4}, {
                                       1.0F, 0.0F, 0.0F, 0.0F,
                                       0.0F, 1.0F, 0.0F, 0.0F,
                                       0.0F, 0.0F, 1.0F, 0.0F,
                                       0.0F, 0.0F, 0.0F, 1.0F,
                                   });
    runtime::Tensor target({4, 2}, {
                                        1.0F, 0.0F,
                                        0.0F, 1.0F,
                                        1.0F, 1.0F,
                                        0.0F, 0.0F,
                                    });
    return ToyProblem{input, target};
}

}  // namespace

TEST(ModelTest, TrainingLoopRiduceLaLossConSgd) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(2)\n"
        "}\n");

    backend::cpu::Model model(module.models.front());
    ToyProblem problem = makeToyProblem();
    backend::cpu::SGD optimizer(0.5F);

    float firstLoss = 0.0F;
    float lastLoss = 0.0F;
    for (int step = 0; step < 200; ++step) {
        model.zeroGrad();
        runtime::Tensor output = model.forward(problem.input);
        backend::cpu::LossResult loss = backend::cpu::meanSquaredError(output, problem.target);
        if (step == 0) {
            firstLoss = loss.value;
        }
        lastLoss = loss.value;
        model.backward(loss.grad);
        optimizer.step(model.parameters());
    }

    EXPECT_LT(lastLoss, firstLoss * 0.1F);
}

TEST(ModelTest, TrainingLoopRiduceLaLossConAdamW) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(2)\n"
        "}\n");

    backend::cpu::Model model(module.models.front());
    ToyProblem problem = makeToyProblem();
    backend::cpu::AdamW optimizer(/*learningRate=*/0.1F, /*beta1=*/0.9F, /*beta2=*/0.999F, /*eps=*/1e-8F,
                                   /*weightDecay=*/0.0F);

    float firstLoss = 0.0F;
    float lastLoss = 0.0F;
    for (int step = 0; step < 200; ++step) {
        model.zeroGrad();
        runtime::Tensor output = model.forward(problem.input);
        backend::cpu::LossResult loss = backend::cpu::meanSquaredError(output, problem.target);
        if (step == 0) {
            firstLoss = loss.value;
        }
        lastLoss = loss.value;
        model.backward(loss.grad);
        optimizer.step(model.parameters());
    }

    EXPECT_LT(lastLoss, firstLoss * 0.1F);
}
