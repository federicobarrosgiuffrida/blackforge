#include "blackforge/backend/cuda/train_runner.hpp"

#include <cstdio>
#include <filesystem>

#include <gtest/gtest.h>

#include "blackforge/backend/cpu/checkpoint.hpp"
#include "blackforge/backend/cpu/model.hpp"
#include "blackforge/backend/cuda/checkpoint.hpp"
#include "blackforge/backend/cuda/device_tensor.hpp"
#include "blackforge/backend/cuda/model.hpp"
#include "blackforge/data/dataset.hpp"
#include "blackforge/frontend/lexer.hpp"
#include "blackforge/frontend/parser.hpp"
#include "blackforge/ir/ir_builder.hpp"

using namespace blackforge;
using blackforge::runtime::Tensor;
namespace cuda = blackforge::backend::cuda;

namespace {

struct TempFile {
    std::string path;

    explicit TempFile(const std::string& name) : path((std::filesystem::temp_directory_path() / name).string()) {}

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

void writeToyDataset(const std::string& path) {
    std::vector<std::size_t> inputShape{4};
    std::vector<std::size_t> targetShape{2};
    std::vector<float> inputs{
        1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F,
    };
    std::vector<float> targets{
        1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F,
    };
    data::saveDataset(path, inputShape, targetShape, inputs, targets, 4);
}

std::string toForwardSlashes(const std::string& path) {
    std::string result = path;
    for (char& c : result) {
        if (c == '\\') {
            c = '/';
        }
    }
    return result;
}

std::string toyProgram(const std::string& datasetPath, int epochs, const std::string& optimizerName,
                        const std::string& lossName = "mse") {
    return "model M {\n"
           "    input bf16[batch, 4]\n"
           "    input |> linear(2)\n"
           "}\n"
           "dataset D {\n"
           "    path \"" +
           toForwardSlashes(datasetPath) +
           "\"\n"
           "    input bf16[batch, 4]\n"
           "    labels bf16[batch, 2]\n"
           "}\n"
           "train {\n"
           "    model M\n"
           "    dataset D\n"
           "    loss " +
           lossName +
           "\n"
           "    optimizer " +
           optimizerName +
           "\n"
           "    epochs " +
           std::to_string(epochs) +
           "\n"
           "    batch_size 4\n"
           "    learning_rate 0.3\n"
           "}\n";
}

}  // namespace

TEST(CudaTrainRunnerTest, RiduceLaLossAttraversoLeEpoche) {
    TempFile datasetFile("blackforge_cuda_test_train_dataset.bfdata");
    writeToyDataset(datasetFile.path);

    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/50, "adamw"));
    backend::cuda::TrainRunResult result = backend::cuda::runTraining(compiled.program, compiled.module, "", "");

    ASSERT_EQ(result.epochLosses.size(), 50u);
    EXPECT_EQ(result.modelName, "M");
    EXPECT_LT(result.epochLosses.back(), result.epochLosses.front() * 0.1);
}

TEST(CudaTrainRunnerTest, FunzionaAncheConSgd) {
    TempFile datasetFile("blackforge_cuda_test_train_dataset_sgd.bfdata");
    writeToyDataset(datasetFile.path);

    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/50, "sgd"));
    backend::cuda::TrainRunResult result = backend::cuda::runTraining(compiled.program, compiled.module, "", "");

    ASSERT_EQ(result.epochLosses.size(), 50u);
    EXPECT_LT(result.epochLosses.back(), result.epochLosses.front());
}

TEST(CudaTrainRunnerTest, LanciaSeIlProgrammaNonHaBloccoTrain) {
    Compiled compiled = compile(
        "model M {\n"
        "    input bf16[4]\n"
        "}\n");
    EXPECT_THROW((void)backend::cuda::runTraining(compiled.program, compiled.module, "", ""), std::runtime_error);
}

TEST(CudaTrainRunnerTest, LanciaSeIlDatasetNonEsiste) {
    Compiled compiled = compile(toyProgram("blackforge_percorso_inesistente.bfdata", 1, "sgd"));
    EXPECT_THROW((void)backend::cuda::runTraining(compiled.program, compiled.module, "", ""), std::runtime_error);
}

TEST(CudaTrainRunnerTest, LanciaSeLaLossNonEMse) {
    // cross-entropy su CUDA non e' ancora supportata: deve fallire con
    // un errore chiaro, non silenziosamente su MSE o su CPU.
    TempFile datasetFile("blackforge_cuda_test_train_dataset_ce.bfdata");
    writeToyDataset(datasetFile.path);

    Compiled compiled = compile(toyProgram(datasetFile.path, 1, "sgd", "cross_entropy"));
    EXPECT_THROW((void)backend::cuda::runTraining(compiled.program, compiled.module, "", ""), std::runtime_error);
}

TEST(CudaTrainRunnerTest, SalvaERicaricaUnCheckpointPerIlFineTuning) {
    TempFile datasetFile("blackforge_cuda_test_train_dataset_ckpt.bfdata");
    writeToyDataset(datasetFile.path);
    TempFile checkpointFile("blackforge_cuda_test_train_checkpoint.bfckpt");

    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/30, "adamw"));

    backend::cuda::TrainRunResult first =
        backend::cuda::runTraining(compiled.program, compiled.module, "", checkpointFile.path);
    ASSERT_TRUE(std::filesystem::exists(checkpointFile.path));

    // Seconda sessione ("fine-tuning"): riparte dal checkpoint salvato,
    // la loss della prima epoca deve partire gia' bassa (vicina a dove
    // si era fermato il primo addestramento), non da zero come un
    // modello con pesi casuali.
    backend::cuda::TrainRunResult second =
        backend::cuda::runTraining(compiled.program, compiled.module, checkpointFile.path, "");

    ASSERT_FALSE(second.epochLosses.empty());
    EXPECT_LT(second.epochLosses.front(), first.epochLosses.front());
}

TEST(CudaTrainRunnerTest, UnCheckpointSalvatoDaCudaECaricabileDaCpuEViceversa) {
    // Il formato binario e' identico tra i due backend (vedi
    // backend::cuda::checkpoint.hpp): questo test verifica che
    // l'interoperabilita' dichiarata sia vera, non solo teorica.
    TempFile datasetFile("blackforge_cuda_test_train_dataset_interop.bfdata");
    writeToyDataset(datasetFile.path);
    TempFile checkpointFile("blackforge_cuda_test_train_checkpoint_interop.bfckpt");

    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/20, "adamw"));
    backend::cuda::runTraining(compiled.program, compiled.module, "", checkpointFile.path);

    backend::cpu::Model cpuModel(compiled.module.models.front());
    backend::cpu::loadCheckpoint(cpuModel, checkpointFile.path);

    backend::cuda::Model cudaModel(compiled.module.models.front());
    backend::cuda::loadCheckpoint(cudaModel, checkpointFile.path);

    Tensor input({4, 4}, {
                             1.0F, 0.0F, 0.0F, 0.0F,
                             0.0F, 1.0F, 0.0F, 0.0F,
                             0.0F, 0.0F, 1.0F, 0.0F,
                             0.0F, 0.0F, 0.0F, 1.0F,
                         });
    Tensor cpuOutput = cpuModel.forward(input);
    Tensor cudaOutput = cudaModel.forward(cuda::DeviceTensor::fromHost(input)).toHost();

    ASSERT_EQ(cpuOutput.elementCount(), cudaOutput.elementCount());
    for (std::size_t i = 0; i < cpuOutput.elementCount(); ++i) {
        EXPECT_NEAR(cpuOutput.at(i), cudaOutput.at(i), 1e-4F) << "indice " << i;
    }
}
