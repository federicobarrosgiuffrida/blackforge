#include "blackforge/backend/cuda/model.hpp"

#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>

#include "blackforge/backend/cpu/loss.hpp"
#include "blackforge/backend/cpu/model.hpp"
#include "blackforge/backend/cpu/optimizer.hpp"
#include "blackforge/backend/cuda/device_tensor.hpp"
#include "blackforge/backend/cuda/loss.hpp"
#include "blackforge/backend/cuda/optimizer.hpp"
#include "blackforge/frontend/lexer.hpp"
#include "blackforge/frontend/parser.hpp"
#include "blackforge/ir/ir_builder.hpp"

using namespace blackforge;
using blackforge::runtime::Tensor;
namespace cpu = blackforge::backend::cpu;
namespace cuda = blackforge::backend::cuda;

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

TEST(CudaModelTest, ForwardProduceLaFormaAttesa) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(3) |> silu |> linear(2)\n"
        "}\n");
    cuda::Model model(module.models.front());

    Tensor input({2, 4}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.1F, 0.2F, -0.3F});
    Tensor output = model.forward(cuda::DeviceTensor::fromHost(input)).toHost();

    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{2, 2}));
}

TEST(CudaModelTest, ForwardCorrispondeAllaVersioneCpuAParitaDiSeme) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> rmsnorm |> linear(3) |> silu |> linear(2)\n"
        "}\n");

    cpu::Model cpuModel(module.models.front(), /*seed=*/7);
    cuda::Model cudaModel(module.models.front(), /*seed=*/7);

    Tensor input({2, 4}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.1F, 0.2F, -0.3F});
    Tensor cpuOut = cpuModel.forward(input);
    Tensor cudaOut = cudaModel.forward(cuda::DeviceTensor::fromHost(input)).toHost();

    ASSERT_EQ(cpuOut.elementCount(), cudaOut.elementCount());
    for (std::size_t i = 0; i < cpuOut.elementCount(); ++i) {
        EXPECT_NEAR(cpuOut.at(i), cudaOut.at(i), 1e-4F) << "indice " << i;
    }
}

TEST(CudaModelTest, LanciaSeLaDimensioneDelleFeatureENonConcreta) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, batch2]\n"
        "    input |> linear(2)\n"
        "}\n");
    EXPECT_THROW((cuda::Model(module.models.front())), std::invalid_argument);
}

TEST(CudaModelTest, LanciaUnErroreEsplicitoSeLaPipelineContieneBidirectionalAttention) {
    // 'bidirectional_attention' (MLM) non e' ancora implementato su
    // CUDA: la costruzione deve fallire con un errore chiaro, non
    // ignorare silenziosamente il layer (che produrrebbe un risultato
    // quietamente sbagliato — l'input passerebbe attraverso invariato).
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> bidirectional_attention(2)\n"
        "}\n");
    EXPECT_THROW((cuda::Model(module.models.front())), std::invalid_argument);
}

TEST(CudaModelTest, BackwardCorrispondeAllaDerivataNumericaDelLoss) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 3]\n"
        "    input |> linear(4) |> silu |> linear(2)\n"
        "}\n");

    cuda::Model model(module.models.front());
    Tensor input({2, 3}, {0.2F, -0.1F, 0.4F, -0.3F, 0.5F, 0.1F});
    Tensor target({2, 2}, {0.0F, 1.0F, 1.0F, 0.0F});
    cuda::DeviceTensor inputDevice = cuda::DeviceTensor::fromHost(input);
    cuda::DeviceTensor targetDevice = cuda::DeviceTensor::fromHost(target);

    model.zeroGrad();
    cuda::DeviceTensor output = model.forward(inputDevice);
    cuda::LossResult loss = cuda::meanSquaredError(output, targetDevice);
    model.backward(loss.grad);

    auto lossOf = [&]() {
        cuda::DeviceTensor out = model.forward(inputDevice);
        return cuda::meanSquaredError(out, targetDevice).value;
    };

    auto params = model.parameters();
    ASSERT_FALSE(params.empty());

    cuda::Parameter* param = params.front();
    Tensor hostValue = param->value.toHost();
    Tensor hostGrad = param->grad.toHost();

    float eps = 1e-3F;
    for (std::size_t i = 0; i < std::min<std::size_t>(5, hostValue.elementCount()); ++i) {
        float original = hostValue.at(i);

        hostValue.at(i) = original + eps;
        param->value = cuda::DeviceTensor::fromHost(hostValue);
        float plus = lossOf();

        hostValue.at(i) = original - eps;
        param->value = cuda::DeviceTensor::fromHost(hostValue);
        float minus = lossOf();

        hostValue.at(i) = original;
        param->value = cuda::DeviceTensor::fromHost(hostValue);

        float numeric = (plus - minus) / (2.0F * eps);
        EXPECT_NEAR(hostGrad.at(i), numeric, 1e-2F) << "parametro " << param->name << " indice " << i;
    }
}

TEST(CudaModelTest, BackwardConSoftmaxCorrispondeAllaDerivataNumericaDelLoss) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 3]\n"
        "    input |> linear(4) |> softmax\n"
        "}\n");

    cuda::Model model(module.models.front());
    Tensor input({2, 3}, {0.2F, -0.1F, 0.4F, -0.3F, 0.5F, 0.1F});
    Tensor target({2, 4}, {0.25F, 0.25F, 0.25F, 0.25F, 1.0F, 0.0F, 0.0F, 0.0F});
    cuda::DeviceTensor inputDevice = cuda::DeviceTensor::fromHost(input);
    cuda::DeviceTensor targetDevice = cuda::DeviceTensor::fromHost(target);

    model.zeroGrad();
    cuda::DeviceTensor output = model.forward(inputDevice);
    cuda::LossResult loss = cuda::meanSquaredError(output, targetDevice);
    model.backward(loss.grad);

    auto lossOf = [&]() {
        cuda::DeviceTensor out = model.forward(inputDevice);
        return cuda::meanSquaredError(out, targetDevice).value;
    };

    auto params = model.parameters();
    ASSERT_FALSE(params.empty());

    cuda::Parameter* param = params.front();
    Tensor hostValue = param->value.toHost();
    Tensor hostGrad = param->grad.toHost();

    float eps = 1e-3F;
    for (std::size_t i = 0; i < std::min<std::size_t>(5, hostValue.elementCount()); ++i) {
        float original = hostValue.at(i);

        hostValue.at(i) = original + eps;
        param->value = cuda::DeviceTensor::fromHost(hostValue);
        float plus = lossOf();

        hostValue.at(i) = original - eps;
        param->value = cuda::DeviceTensor::fromHost(hostValue);
        float minus = lossOf();

        hostValue.at(i) = original;
        param->value = cuda::DeviceTensor::fromHost(hostValue);

        float numeric = (plus - minus) / (2.0F * eps);
        EXPECT_NEAR(hostGrad.at(i), numeric, 1e-2F) << "parametro " << param->name << " indice " << i;
    }
}

TEST(CudaModelTest, TrainingLoopRiduceLaLossConSgd) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(2)\n"
        "}\n");

    cuda::Model model(module.models.front());
    Tensor input({4, 4}, {
                             1.0F, 0.0F, 0.0F, 0.0F,
                             0.0F, 1.0F, 0.0F, 0.0F,
                             0.0F, 0.0F, 1.0F, 0.0F,
                             0.0F, 0.0F, 0.0F, 1.0F,
                         });
    Tensor target({4, 2}, {
                              1.0F, 0.0F,
                              0.0F, 1.0F,
                              1.0F, 1.0F,
                              0.0F, 0.0F,
                          });
    cuda::DeviceTensor inputDevice = cuda::DeviceTensor::fromHost(input);
    cuda::DeviceTensor targetDevice = cuda::DeviceTensor::fromHost(target);

    cuda::SGD optimizer(0.5F);

    float firstLoss = 0.0F;
    float lastLoss = 0.0F;
    for (int step = 0; step < 200; ++step) {
        model.zeroGrad();
        cuda::DeviceTensor output = model.forward(inputDevice);
        cuda::LossResult loss = cuda::meanSquaredError(output, targetDevice);
        if (step == 0) {
            firstLoss = loss.value;
        }
        lastLoss = loss.value;
        model.backward(loss.grad);
        optimizer.step(model.parameters());
    }

    EXPECT_LT(lastLoss, firstLoss * 0.1F);
}

TEST(CudaModelTest, TrainingLoopRiduceLaLossConAdamW) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(2)\n"
        "}\n");

    cuda::Model model(module.models.front());
    Tensor input({4, 4}, {
                             1.0F, 0.0F, 0.0F, 0.0F,
                             0.0F, 1.0F, 0.0F, 0.0F,
                             0.0F, 0.0F, 1.0F, 0.0F,
                             0.0F, 0.0F, 0.0F, 1.0F,
                         });
    Tensor target({4, 2}, {
                              1.0F, 0.0F,
                              0.0F, 1.0F,
                              1.0F, 1.0F,
                              0.0F, 0.0F,
                          });
    cuda::DeviceTensor inputDevice = cuda::DeviceTensor::fromHost(input);
    cuda::DeviceTensor targetDevice = cuda::DeviceTensor::fromHost(target);

    cuda::AdamW optimizer(/*learningRate=*/0.1F);

    float firstLoss = 0.0F;
    float lastLoss = 0.0F;
    for (int step = 0; step < 200; ++step) {
        model.zeroGrad();
        cuda::DeviceTensor output = model.forward(inputDevice);
        cuda::LossResult loss = cuda::meanSquaredError(output, targetDevice);
        if (step == 0) {
            firstLoss = loss.value;
        }
        lastLoss = loss.value;
        model.backward(loss.grad);
        optimizer.step(model.parameters());
    }

    EXPECT_LT(lastLoss, firstLoss * 0.1F);
}

// --- Tensor Core BF16 reale (precision { compute bf16 }) ---

TEST(CudaModelTest, ForwardConPrecisionBf16CorrispondeApprossimativamenteAllaVersioneFp32) {
    ir::Module fp32Module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(3) |> silu |> linear(2)\n"
        "}\n");
    ir::Module bf16Module = buildModule(
        "precision {\n"
        "    compute bf16\n"
        "}\n"
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(3) |> silu |> linear(2)\n"
        "}\n");
    ASSERT_TRUE(bf16Module.precision.has_value());
    ASSERT_EQ(bf16Module.precision->compute, sema::DType::BF16);

    cuda::Model fp32Model(fp32Module.models.front(), /*seed=*/7);
    cuda::Model bf16Model(bf16Module.models.front(), /*seed=*/7, bf16Module.precision);

    Tensor input({2, 4}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.1F, 0.2F, -0.3F});
    Tensor fp32Out = fp32Model.forward(cuda::DeviceTensor::fromHost(input)).toHost();
    Tensor bf16Out = bf16Model.forward(cuda::DeviceTensor::fromHost(input)).toHost();

    ASSERT_EQ(fp32Out.elementCount(), bf16Out.elementCount());
    for (std::size_t i = 0; i < fp32Out.elementCount(); ++i) {
        float tolerance = std::max(0.05F, std::abs(fp32Out.at(i)) * 0.1F);
        EXPECT_NEAR(bf16Out.at(i), fp32Out.at(i), tolerance) << "indice " << i;
    }
}

TEST(CudaModelTest, TrainingLoopConPrecisionBf16RiduceLaLoss) {
    // Prova che il percorso Tensor Core sia genuinamente allenabile
    // (forward E backward via matmulBf16/matmulBf16Backward), non solo
    // utilizzabile per l'inferenza: un ciclo di addestramento completo
    // con 'precision { compute bf16 }' deve far scendere la loss quasi
    // quanto la controparte FP32 (non esattamente, per via
    // dell'arrotondamento BF16 nel prodotto matriciale).
    ir::Module module = buildModule(
        "precision {\n"
        "    compute bf16\n"
        "}\n"
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(2)\n"
        "}\n");

    cuda::Model model(module.models.front(), /*seed=*/42, module.precision);
    Tensor input({4, 4}, {
                             1.0F, 0.0F, 0.0F, 0.0F,
                             0.0F, 1.0F, 0.0F, 0.0F,
                             0.0F, 0.0F, 1.0F, 0.0F,
                             0.0F, 0.0F, 0.0F, 1.0F,
                         });
    Tensor target({4, 2}, {
                              1.0F, 0.0F,
                              0.0F, 1.0F,
                              1.0F, 1.0F,
                              0.0F, 0.0F,
                          });
    cuda::DeviceTensor inputDevice = cuda::DeviceTensor::fromHost(input);
    cuda::DeviceTensor targetDevice = cuda::DeviceTensor::fromHost(target);

    cuda::AdamW optimizer(/*learningRate=*/0.1F);

    float firstLoss = 0.0F;
    float lastLoss = 0.0F;
    for (int step = 0; step < 200; ++step) {
        model.zeroGrad();
        cuda::DeviceTensor output = model.forward(inputDevice);
        cuda::LossResult loss = cuda::meanSquaredError(output, targetDevice);
        if (step == 0) {
            firstLoss = loss.value;
        }
        lastLoss = loss.value;
        model.backward(loss.grad);
        optimizer.step(model.parameters());
    }

    EXPECT_LT(lastLoss, firstLoss * 0.1F);
}

// --- Modello linguistico minimale (embedding/positional_embedding/attention/feedforward) ---

namespace {

ir::Module buildTinyLmModule() {
    return buildModule(
        "model TinyLM {\n"
        "    input bf16[batch, 3]\n"
        "    input |> embedding(6, 4) |> positional_embedding(4) |> attention(2) |> feedforward(8) |> linear(6)\n"
        "}\n");
}

}  // namespace

TEST(CudaModelTest, ForwardDiUnModelloLinguisticoCorrispondeAllaVersioneCpuAParitaDiSeme) {
    ir::Module module = buildTinyLmModule();

    cpu::Model cpuModel(module.models.front(), /*seed=*/11);
    cuda::Model cudaModel(module.models.front(), /*seed=*/11);

    Tensor tokenIds({2, 3}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
    Tensor cpuOut = cpuModel.forward(tokenIds);
    Tensor cudaOut = cudaModel.forward(cuda::DeviceTensor::fromHost(tokenIds)).toHost();

    ASSERT_EQ(cpuOut.shape(), (std::vector<std::size_t>{2, 3, 6}));
    ASSERT_EQ(cpuOut.elementCount(), cudaOut.elementCount());
    for (std::size_t i = 0; i < cpuOut.elementCount(); ++i) {
        EXPECT_NEAR(cpuOut.at(i), cudaOut.at(i), 1e-3F) << "indice " << i;
    }
}

TEST(CudaModelTest, TrainingLoopDiUnModelloLinguisticoCorrispondeAllaVersioneCpuAParitaDiSeme) {
    // Stesso principio del test di parita' generico piu' sotto, ma sulla
    // pipeline completa di un modello linguistico (embedding, attention
    // causale multi-testa, feedforward), allenata con cross-entropy:
    // verifica che l'intera catena backward (inclusi
    // embeddingLookupBackward, addPositionalEmbeddingBackward,
    // selfAttentionBackward, feedForwardBackward) sia numericamente
    // equivalente tra CPU e GPU dopo piu' step di addestramento.
    ir::Module module = buildTinyLmModule();

    cpu::Model cpuModel(module.models.front(), /*seed=*/5);
    cuda::Model cudaModel(module.models.front(), /*seed=*/5);

    Tensor tokenIds({2, 3}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
    cuda::DeviceTensor tokenIdsDevice = cuda::DeviceTensor::fromHost(tokenIds);

    constexpr std::size_t vocab = 6;
    std::vector<float> targetData(2 * 3 * vocab, 0.0F);
    for (std::size_t b = 0; b < 2; ++b) {
        for (std::size_t s = 0; s < 3; ++s) {
            auto tokenId = static_cast<std::size_t>(tokenIds.at(b * 3 + s));
            std::size_t nextToken = (tokenId + 1) % vocab;
            targetData[(b * 3 + s) * vocab + nextToken] = 1.0F;
        }
    }
    Tensor target({2, 3, vocab}, targetData);
    cuda::DeviceTensor targetDevice = cuda::DeviceTensor::fromHost(target);

    cpu::AdamW cpuOptimizer(/*learningRate=*/0.05F);
    cuda::AdamW cudaOptimizer(/*learningRate=*/0.05F);

    float cpuLoss = 0.0F;
    float cudaLoss = 0.0F;
    for (int step = 0; step < 30; ++step) {
        cpuModel.zeroGrad();
        Tensor cpuOutput = cpuModel.forward(tokenIds);
        cpu::LossResult cpuLossResult = cpu::softmaxCrossEntropy(cpuOutput, target);
        cpuLoss = cpuLossResult.value;
        cpuModel.backward(cpuLossResult.grad);
        cpuOptimizer.step(cpuModel.parameters());

        cudaModel.zeroGrad();
        cuda::DeviceTensor cudaOutput = cudaModel.forward(tokenIdsDevice);
        cuda::LossResult cudaLossResult = cuda::softmaxCrossEntropy(cudaOutput, targetDevice);
        cudaLoss = cudaLossResult.value;
        cudaModel.backward(cudaLossResult.grad);
        cudaOptimizer.step(cudaModel.parameters());
    }

    EXPECT_NEAR(cpuLoss, cudaLoss, 5e-2F);
}

// --- Generazione incrementale con cache K/V (mirror CUDA di
// ModelTest.ForwardIncrementalCorrispondeAForwardCompletoTokenPerToken /
// ResetGenerationStateAzzeraLaCacheELaPosizione in tests/backend/cpu/model_tests.cpp) ---

TEST(CudaModelTest, ForwardIncrementalCorrispondeAForwardCompletoTokenPerToken) {
    // Stesso test di correttezza fondamentale del KV-cache della
    // controparte CPU: genera una sequenza di 4 token uno alla volta
    // con forwardIncremental() (prompt iniziale di 2 token, poi 2
    // chiamate con un solo token nuovo ciascuna) e verifica che
    // l'uscita per OGNI posizione coincida con quella che forward()
    // produrrebbe ricalcolando l'intera sottosequenza da capo.
    ir::Module module = buildTinyLmModule();
    cuda::Model incrementalModel(module.models.front(), /*seed=*/11);
    cuda::Model referenceModel(module.models.front(), /*seed=*/11);

    // buildTinyLmModule() usa positional_embedding(4): 4 e' la
    // lunghezza massima di sequenza supportata dal modello.
    std::vector<float> fullSequence = {0.0F, 3.0F, 1.0F, 4.0F};
    incrementalModel.resetGenerationState();

    Tensor prompt({1, 2}, {fullSequence[0], fullSequence[1]});
    Tensor incOut = incrementalModel.forwardIncremental(cuda::DeviceTensor::fromHost(prompt)).toHost();

    Tensor refPrompt({1, 2}, {fullSequence[0], fullSequence[1]});
    Tensor refOut = referenceModel.forward(cuda::DeviceTensor::fromHost(refPrompt)).toHost();

    ASSERT_EQ(incOut.shape(), refOut.shape());
    for (std::size_t i = 0; i < incOut.elementCount(); ++i) {
        EXPECT_NEAR(incOut.at(i), refOut.at(i), 1e-3F) << "prompt, indice " << i;
    }

    for (std::size_t step = 2; step < fullSequence.size(); ++step) {
        Tensor newToken({1, 1}, {fullSequence[step]});
        Tensor incStepOut = incrementalModel.forwardIncremental(cuda::DeviceTensor::fromHost(newToken)).toHost();

        std::vector<float> subsequence(fullSequence.begin(),
                                        fullSequence.begin() + static_cast<std::ptrdiff_t>(step) + 1);
        Tensor refSub({1, subsequence.size()}, subsequence);
        Tensor refSubOut = referenceModel.forward(cuda::DeviceTensor::fromHost(refSub)).toHost();

        std::size_t vocab = incStepOut.dim(2);
        ASSERT_EQ(incStepOut.shape(), (std::vector<std::size_t>{1, 1, vocab}));
        for (std::size_t v = 0; v < vocab; ++v) {
            float incValue = incStepOut.at(v);
            float refValue = refSubOut.at(step * vocab + v);  // ultima posizione di refSubOut
            EXPECT_NEAR(incValue, refValue, 1e-3F) << "step " << step << " colonna " << v;
        }
    }
}

TEST(CudaModelTest, ForwardIncrementalCorrispondeAllaVersioneCpuAParitaDiSeme) {
    // Parita' diretta CPU/GPU sulla generazione incrementale: stesso
    // seme (stessi pesi), stesso prompt, stessa sequenza di token nuovi
    // -> stesso output ad ogni step, entro la tolleranza numerica
    // attesa tra somme in virgola mobile CPU e cuBLAS/kernel GPU.
    ir::Module module = buildTinyLmModule();
    cpu::Model cpuModel(module.models.front(), /*seed=*/7);
    cuda::Model cudaModel(module.models.front(), /*seed=*/7);

    cpuModel.resetGenerationState();
    cudaModel.resetGenerationState();

    Tensor prompt({1, 2}, {0.0F, 2.0F});
    Tensor cpuOut = cpuModel.forwardIncremental(prompt);
    Tensor cudaOut = cudaModel.forwardIncremental(cuda::DeviceTensor::fromHost(prompt)).toHost();

    ASSERT_EQ(cpuOut.shape(), cudaOut.shape());
    for (std::size_t i = 0; i < cpuOut.elementCount(); ++i) {
        EXPECT_NEAR(cpuOut.at(i), cudaOut.at(i), 1e-3F) << "prompt, indice " << i;
    }

    Tensor nextToken({1, 1}, {1.0F});
    Tensor cpuStepOut = cpuModel.forwardIncremental(nextToken);
    Tensor cudaStepOut = cudaModel.forwardIncremental(cuda::DeviceTensor::fromHost(nextToken)).toHost();

    ASSERT_EQ(cpuStepOut.shape(), cudaStepOut.shape());
    for (std::size_t i = 0; i < cpuStepOut.elementCount(); ++i) {
        EXPECT_NEAR(cpuStepOut.at(i), cudaStepOut.at(i), 1e-3F) << "step, indice " << i;
    }
}

TEST(CudaModelTest, ResetGenerationStateAzzeraLaCacheELaPosizione) {
    // Dopo resetGenerationState(), una nuova sessione di generazione
    // deve comportarsi esattamente come la prima (stessa cache vuota,
    // stessa posizione 0).
    ir::Module module = buildTinyLmModule();
    cuda::Model model(module.models.front(), /*seed=*/3);

    Tensor prompt({1, 2}, {1.0F, 2.0F});

    model.resetGenerationState();
    Tensor firstSessionOut = model.forwardIncremental(cuda::DeviceTensor::fromHost(prompt)).toHost();

    model.resetGenerationState();
    Tensor secondSessionOut = model.forwardIncremental(cuda::DeviceTensor::fromHost(prompt)).toHost();

    ASSERT_EQ(firstSessionOut.shape(), secondSessionOut.shape());
    for (std::size_t i = 0; i < firstSessionOut.elementCount(); ++i) {
        EXPECT_FLOAT_EQ(firstSessionOut.at(i), secondSessionOut.at(i)) << "indice " << i;
    }
}

TEST(CudaModelTest, TrainingLoopCorrispondeAllaVersioneCpuAParitaDiSeme) {
    // Il test di parita' piu' importante di questa milestone: lo stesso
    // ciclo di addestramento (stesso seme, stessi dati, stesso
    // optimizer), eseguito indipendentemente su CPU e su GPU, deve
    // arrivare a una loss finale numericamente equivalente. Se un
    // singolo kernel di backward avesse una formula sbagliata, le due
    // traiettorie di addestramento divergerebbero rapidamente (l'errore
    // si accumula ad ogni step), quindi questo test e' molto piu'
    // sensibile di un confronto sul solo primo forward/backward.
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> rmsnorm |> linear(3) |> silu |> linear(2)\n"
        "}\n");

    cpu::Model cpuModel(module.models.front(), /*seed=*/3);
    cuda::Model cudaModel(module.models.front(), /*seed=*/3);

    Tensor input({4, 4}, {
                             1.0F, 0.0F, 0.0F, 0.0F,
                             0.0F, 1.0F, 0.0F, 0.0F,
                             0.0F, 0.0F, 1.0F, 0.0F,
                             0.0F, 0.0F, 0.0F, 1.0F,
                         });
    Tensor target({4, 2}, {
                              1.0F, 0.0F,
                              0.0F, 1.0F,
                              1.0F, 1.0F,
                              0.0F, 0.0F,
                          });
    cuda::DeviceTensor inputDevice = cuda::DeviceTensor::fromHost(input);
    cuda::DeviceTensor targetDevice = cuda::DeviceTensor::fromHost(target);

    cpu::AdamW cpuOptimizer(/*learningRate=*/0.1F);
    cuda::AdamW cudaOptimizer(/*learningRate=*/0.1F);

    float cpuLoss = 0.0F;
    float cudaLoss = 0.0F;
    for (int step = 0; step < 50; ++step) {
        cpuModel.zeroGrad();
        Tensor cpuOutput = cpuModel.forward(input);
        cpu::LossResult cpuLossResult = cpu::meanSquaredError(cpuOutput, target);
        cpuLoss = cpuLossResult.value;
        cpuModel.backward(cpuLossResult.grad);
        cpuOptimizer.step(cpuModel.parameters());

        cudaModel.zeroGrad();
        cuda::DeviceTensor cudaOutput = cudaModel.forward(inputDevice);
        cuda::LossResult cudaLossResult = cuda::meanSquaredError(cudaOutput, targetDevice);
        cudaLoss = cudaLossResult.value;
        cudaModel.backward(cudaLossResult.grad);
        cudaOptimizer.step(cudaModel.parameters());
    }

    EXPECT_NEAR(cpuLoss, cudaLoss, 1e-2F);

    auto cpuParams = cpuModel.parameters();
    auto cudaParams = cudaModel.parameters();
    ASSERT_EQ(cpuParams.size(), cudaParams.size());
    for (std::size_t p = 0; p < cpuParams.size(); ++p) {
        Tensor cudaValue = cudaParams[p]->value.toHost();
        ASSERT_EQ(cpuParams[p]->value.elementCount(), cudaValue.elementCount());
        for (std::size_t i = 0; i < cpuParams[p]->value.elementCount(); ++i) {
            EXPECT_NEAR(cpuParams[p]->value.at(i), cudaValue.at(i), 5e-2F)
                << "parametro " << p << " indice " << i;
        }
    }
}
