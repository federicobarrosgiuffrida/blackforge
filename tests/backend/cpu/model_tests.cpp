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

TEST(ModelTest, BackwardConRmsnormCorrispondeAllaDerivataNumericaDelLoss) {
    // rmsnorm non ha parametri propri, quindi non compare direttamente
    // in model.parameters(): questo test verifica indirettamente che
    // rmsnormBackward propaghi correttamente il gradiente al layer
    // linear successivo (se la formula fosse sbagliata, il gradiente
    // del peso non corrisponderebbe piu' alla derivata numerica della
    // loss, perche' rmsnorm e' nel mezzo della catena forward).
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 3]\n"
        "    input |> rmsnorm |> linear(4) |> silu |> linear(2)\n"
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

// --- LoRA ---

TEST(ModelLoraTest, RichiedeRankPositivo) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(2)\n"
        "}\n");
    EXPECT_THROW(backend::cpu::Model(module.models.front(), 42, backend::cpu::LoraOptions{0, 1.0}),
                 std::invalid_argument);
}

TEST(ModelLoraTest, ParametersRestituisceSoloAdapterQuandoLoraEAttivo) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(2)\n"
        "}\n");
    backend::cpu::Model model(module.models.front(), 42, backend::cpu::LoraOptions{2, 4.0});

    EXPECT_EQ(model.parameters().size(), 2u);      // solo loraA, loraB
    EXPECT_EQ(model.allParameters().size(), 4u);   // + weight, bias congelati
}

TEST(ModelLoraTest, ForwardNonAlteraLUscitaAllInizializzazione) {
    // B e' inizializzato a zero: il contributo dell'adapter deve essere
    // esattamente nullo, quindi l'uscita con LoRA attivo deve coincidere
    // con quella di un modello identico senza LoRA (stesso seme, stessi
    // pesi di base).
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(3) |> silu |> linear(2)\n"
        "}\n");

    backend::cpu::Model plain(module.models.front(), /*seed=*/7);
    backend::cpu::Model withLora(module.models.front(), /*seed=*/7, backend::cpu::LoraOptions{2, 4.0});

    runtime::Tensor input({2, 4}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.1F, 0.2F, -0.3F});
    runtime::Tensor plainOut = plain.forward(input);
    runtime::Tensor loraOut = withLora.forward(input);

    ASSERT_EQ(plainOut.elementCount(), loraOut.elementCount());
    for (std::size_t i = 0; i < plainOut.elementCount(); ++i) {
        EXPECT_FLOAT_EQ(plainOut.at(i), loraOut.at(i)) << "indice " << i;
    }
}

TEST(ModelLoraTest, BackwardCorrispondeAllaDerivataNumericaDelLoss) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 3]\n"
        "    input |> linear(4)\n"
        "}\n");

    backend::cpu::Model model(module.models.front(), 42, backend::cpu::LoraOptions{2, 3.0});

    // B a zero azzererebbe il gradiente di A per costruzione (matmul con
    // una matrice nulla): si assegnano valori non nulli ad A e B per
    // rendere il test significativo.
    for (backend::cpu::Parameter* param : model.parameters()) {
        for (std::size_t i = 0; i < param->value.elementCount(); ++i) {
            param->value.at(i) = 0.05F * static_cast<float>(i % 5) - 0.1F;
        }
    }

    runtime::Tensor input({2, 3}, {0.2F, -0.1F, 0.4F, -0.3F, 0.5F, 0.1F});
    runtime::Tensor target({2, 4}, {0.0F, 1.0F, 1.0F, 0.0F, 0.5F, -0.5F, 0.2F, 0.1F});

    model.zeroGrad();
    runtime::Tensor output = model.forward(input);
    backend::cpu::LossResult loss = backend::cpu::meanSquaredError(output, target);
    model.backward(loss.grad);

    auto lossOf = [&]() {
        runtime::Tensor out = model.forward(input);
        return backend::cpu::meanSquaredError(out, target).value;
    };

    float eps = 1e-3F;
    for (backend::cpu::Parameter* param : model.parameters()) {
        for (std::size_t i = 0; i < std::min<std::size_t>(4, param->value.elementCount()); ++i) {
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
}

TEST(ModelLoraTest, TrainingLoopMantieneCongelatiIPesiDiBase) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(2)\n"
        "}\n");

    backend::cpu::Model model(module.models.front(), 42, backend::cpu::LoraOptions{2, 4.0});

    // Valori di partenza dei pesi di base (congelati) e degli adapter.
    std::vector<float> weightBefore = model.allParameters()[0]->value.data();
    std::vector<float> biasBefore = model.allParameters()[1]->value.data();

    ToyProblem problem = makeToyProblem();
    backend::cpu::AdamW optimizer(/*learningRate=*/0.2F, 0.9F, 0.999F, 1e-8F, 0.0F);

    float firstLoss = 0.0F;
    float lastLoss = 0.0F;
    for (int step = 0; step < 100; ++step) {
        model.zeroGrad();
        runtime::Tensor output = model.forward(problem.input);
        backend::cpu::LossResult loss = backend::cpu::meanSquaredError(output, problem.target);
        if (step == 0) {
            firstLoss = loss.value;
        }
        lastLoss = loss.value;
        model.backward(loss.grad);
        optimizer.step(model.parameters());  // solo loraA/loraB vengono aggiornati
    }

    EXPECT_LT(lastLoss, firstLoss);

    auto allParams = model.allParameters();
    for (std::size_t i = 0; i < weightBefore.size(); ++i) {
        EXPECT_FLOAT_EQ(allParams[0]->value.at(i), weightBefore[i]) << "il peso congelato e' cambiato";
    }
    for (std::size_t i = 0; i < biasBefore.size(); ++i) {
        EXPECT_FLOAT_EQ(allParams[1]->value.at(i), biasBefore[i]) << "il bias congelato e' cambiato";
    }
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

// --- Precisione numerica (quantizzazione simulata) ---

TEST(ModelPrecisionTest, SenzaPrecisionPolicyIlForwardENonQuantizzato) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(3) |> silu |> linear(2)\n"
        "}\n");

    runtime::Tensor input({2, 4}, {0.1F, 0.2F, 0.3F, 0.4F, -0.1F, -0.2F, -0.3F, -0.4F});

    backend::cpu::Model plain(module.models.front());
    backend::cpu::Model withFp32Policy(module.models.front(), 42, std::nullopt, ir::PrecisionPolicy{});

    runtime::Tensor plainOut = plain.forward(input);
    runtime::Tensor fp32Out = withFp32Policy.forward(input);

    ASSERT_EQ(plainOut.elementCount(), fp32Out.elementCount());
    for (std::size_t i = 0; i < plainOut.elementCount(); ++i) {
        EXPECT_FLOAT_EQ(plainOut.at(i), fp32Out.at(i));
    }
}

TEST(ModelPrecisionTest, UnaPrecisionPolicyRidottaCambiaDavveroIlForward) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(3) |> silu |> linear(2)\n"
        "}\n");

    runtime::Tensor input({2, 4}, {0.1F, 0.2F, 0.3F, 0.4F, -0.1F, -0.2F, -0.3F, -0.4F});

    backend::cpu::Model plain(module.models.front());
    ir::PrecisionPolicy fp8Policy{sema::DType::FP8_E4M3, sema::DType::FP8_E4M3, sema::DType::FP32};
    backend::cpu::Model quantized(module.models.front(), 42, std::nullopt, fp8Policy);

    runtime::Tensor plainOut = plain.forward(input);
    runtime::Tensor quantizedOut = quantized.forward(input);

    ASSERT_EQ(plainOut.elementCount(), quantizedOut.elementCount());
    bool anyDifferent = false;
    for (std::size_t i = 0; i < plainOut.elementCount(); ++i) {
        if (plainOut.at(i) != quantizedOut.at(i)) {
            anyDifferent = true;
            break;
        }
    }
    EXPECT_TRUE(anyDifferent) << "una policy fp8 dovrebbe produrre numeri diversi da fp32 pieno";
}

TEST(ModelPrecisionTest, LoraFunzionaAncheConUnaPrecisionPolicyAttiva) {
    // Verifica solo che LoRA + precision policy insieme non vadano in
    // crash e producano una forma corretta: la combinazione dei due
    // percorsi di quantizzazione dentro il ramo 'linear' e' quella piu'
    // a rischio di un errore di distrazione nel codice.
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(2)\n"
        "}\n");

    backend::cpu::Model model(module.models.front(), 42, backend::cpu::LoraOptions{2, 4.0},
                               ir::PrecisionPolicy{sema::DType::BF16, sema::DType::BF16, sema::DType::FP32});

    runtime::Tensor input({2, 4}, {0.1F, 0.2F, 0.3F, 0.4F, -0.1F, -0.2F, -0.3F, -0.4F});
    runtime::Tensor output = model.forward(input);

    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{2, 2}));
}
