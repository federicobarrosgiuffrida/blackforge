#include "blackforge/backend/cuda/multi_gpu_train_runner.hpp"

#include <cstdio>
#include <filesystem>

#include <gtest/gtest.h>

#include "blackforge/backend/cuda/checkpoint.hpp"
#include "blackforge/backend/cuda/device_tensor.hpp"
#include "blackforge/backend/cuda/model.hpp"
#include "blackforge/backend/cuda/train_runner.hpp"
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

// Stesso dataset di tests/backend/cuda/train_runner_tests.cu (8 esempi
// invece di 4: serve poter dividere batch_size per 2 O per 3 repliche
// senza resto in vari test di questo file).
void writeToyDataset(const std::string& path) {
    std::vector<std::size_t> inputShape{4};
    std::vector<std::size_t> targetShape{2};
    std::vector<float> inputs{
        1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F,
        1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F,
    };
    std::vector<float> targets{
        1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F,
    };
    data::saveDataset(path, inputShape, targetShape, inputs, targets, 8);
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

std::string toyProgram(const std::string& datasetPath, int epochs, std::size_t batchSize,
                        const std::string& optimizerName = "adamw") {
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
           "    loss mse\n"
           "    optimizer " +
           optimizerName +
           "\n"
           "    epochs " +
           std::to_string(epochs) +
           "\n"
           "    batch_size " +
           std::to_string(batchSize) +
           "\n"
           "    learning_rate 0.3\n"
           "}\n";
}

}  // namespace

TEST(CudaMultiGpuTrainRunnerTest, RiduceLaLossAttraversoLeEpocheConDueRepliche) {
    // deviceIndices = {0, 0}: caso limite deliberato (vedi il commento
    // in multi_gpu_train_runner.hpp) che esercita l'intera logica di
    // sharding/all-reduce su una macchina con una sola GPU fisica.
    TempFile datasetFile("blackforge_cuda_test_multigpu_dataset.bfdata");
    writeToyDataset(datasetFile.path);

    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/50, /*batchSize=*/8));
    backend::cuda::TrainRunResult result =
        backend::cuda::runMultiGpuTraining(compiled.program, compiled.module, {0, 0}, "", "");

    ASSERT_EQ(result.epochLosses.size(), 50u);
    EXPECT_EQ(result.modelName, "M");
    EXPECT_LT(result.epochLosses.back(), result.epochLosses.front() * 0.1);
}

TEST(CudaMultiGpuTrainRunnerTest, FunzionaAncheConTreRepliche) {
    TempFile datasetFile("blackforge_cuda_test_multigpu_dataset_3rep.bfdata");
    writeToyDataset(datasetFile.path);

    // batch_size 6 e' divisibile per 3 repliche (shard di 2 esempi
    // ciascuna); il dataset ha 8 esempi, quindi 1 batch/epoca.
    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/50, /*batchSize=*/6));
    backend::cuda::TrainRunResult result =
        backend::cuda::runMultiGpuTraining(compiled.program, compiled.module, {0, 0, 0}, "", "");

    ASSERT_EQ(result.epochLosses.size(), 50u);
    EXPECT_LT(result.epochLosses.back(), result.epochLosses.front() * 0.1);
}

TEST(CudaMultiGpuTrainRunnerTest, ProduceUnaLossFinaleEquivalenteAllaVersioneSingolaGpu) {
    // Il test di correttezza fondamentale di questa milestone: dato che
    // ogni op del modello (linear/rmsnorm/softmax/attention/...) opera
    // per-esempio (nessuna dipendenza incrociata dentro il batch),
    // dividere un batch in N shard di uguale dimensione, calcolare il
    // gradiente medio di ciascuno indipendentemente e poi MEDIARE quei
    // gradienti deve essere matematicamente equivalente a calcolare il
    // gradiente medio sull'intero batch in un colpo solo — non
    // un'approssimazione. Stesso seme (stessi pesi iniziali), stesso
    // dataset, stesso numero di epoche: la loss finale a singola GPU e
    // quella a 2 repliche (degeneri, stesso device fisico) devono
    // coincidere entro la tolleranza numerica attesa da un diverso
    // ordine di somma in virgola mobile.
    TempFile datasetFile("blackforge_cuda_test_multigpu_dataset_parity.bfdata");
    writeToyDataset(datasetFile.path);

    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/30, /*batchSize=*/8));

    backend::cuda::TrainRunResult singleGpu = backend::cuda::runTraining(compiled.program, compiled.module, "", "");
    backend::cuda::TrainRunResult multiGpu =
        backend::cuda::runMultiGpuTraining(compiled.program, compiled.module, {0, 0}, "", "");

    ASSERT_EQ(singleGpu.epochLosses.size(), multiGpu.epochLosses.size());
    for (std::size_t epoch = 0; epoch < singleGpu.epochLosses.size(); ++epoch) {
        EXPECT_NEAR(singleGpu.epochLosses[epoch], multiGpu.epochLosses[epoch], 1e-3)
            << "epoca " << (epoch + 1);
    }
}

TEST(CudaMultiGpuTrainRunnerTest, LanciaSeMenoDiDueIndiciDiGpu) {
    TempFile datasetFile("blackforge_cuda_test_multigpu_dataset_toofew.bfdata");
    writeToyDataset(datasetFile.path);

    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/1, /*batchSize=*/8));
    EXPECT_THROW((void)backend::cuda::runMultiGpuTraining(compiled.program, compiled.module, {0}, "", ""),
                 std::invalid_argument);
}

TEST(CudaMultiGpuTrainRunnerTest, LanciaSeBatchSizeNonEDivisibilePerIlNumeroDiGpu) {
    TempFile datasetFile("blackforge_cuda_test_multigpu_dataset_uneven.bfdata");
    writeToyDataset(datasetFile.path);

    // batch_size 8 non e' divisibile per 3 repliche.
    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/1, /*batchSize=*/8));
    EXPECT_THROW((void)backend::cuda::runMultiGpuTraining(compiled.program, compiled.module, {0, 0, 0}, "", ""),
                 std::runtime_error);
}

TEST(CudaMultiGpuTrainRunnerTest, SalvaERicaricaUnCheckpointPerIlFineTuning) {
    TempFile datasetFile("blackforge_cuda_test_multigpu_dataset_ckpt.bfdata");
    writeToyDataset(datasetFile.path);
    TempFile checkpointFile("blackforge_cuda_test_multigpu_checkpoint.bfckpt");

    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/20, /*batchSize=*/8));

    backend::cuda::TrainRunResult first =
        backend::cuda::runMultiGpuTraining(compiled.program, compiled.module, {0, 0}, "", checkpointFile.path);
    ASSERT_TRUE(std::filesystem::exists(checkpointFile.path));

    backend::cuda::TrainRunResult second =
        backend::cuda::runMultiGpuTraining(compiled.program, compiled.module, {0, 0}, checkpointFile.path, "");

    ASSERT_FALSE(second.epochLosses.empty());
    EXPECT_LT(second.epochLosses.front(), first.epochLosses.front());
}

TEST(CudaMultiGpuTrainRunnerTest, IlCheckpointSalvatoEUtilizzabileConIlBackendASingolaGpu) {
    // Il checkpoint salvato dalla replica 0 e' un checkpoint CUDA
    // ordinario (stesso formato binario, vedi backend::cuda::checkpoint.hpp):
    // deve essere caricabile da un cuda::Model normale, non solo da
    // un'altra sessione multi-GPU.
    TempFile datasetFile("blackforge_cuda_test_multigpu_dataset_interop.bfdata");
    writeToyDataset(datasetFile.path);
    TempFile checkpointFile("blackforge_cuda_test_multigpu_checkpoint_interop.bfckpt");

    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/20, /*batchSize=*/8));
    backend::cuda::runMultiGpuTraining(compiled.program, compiled.module, {0, 0}, "", checkpointFile.path);

    cuda::Model model(compiled.module.models.front());
    cuda::loadCheckpoint(model, checkpointFile.path);

    Tensor input({2, 4}, {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F});
    cuda::DeviceTensor output = model.forward(cuda::DeviceTensor::fromHost(input));
    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{2, 2}));
}

TEST(CudaMultiGpuTrainRunnerTest, ParitaConPiuBatchPerEpocaEPiuEpoche) {
    // Estende ProduceUnaLossFinaleEquivalenteAllaVersioneSingolaGpu
    // (che ha un solo batch/epoca) al caso con PIU' batch/epoca (2, dato
    // che batch_size=4 su 8 esempi) e molte piu' epoche (150 invece di
    // 30): verifica che la parita' single-GPU/multi-GPU regga anche
    // quando lo shuffling e la lettura di shard consecutivi dallo
    // stesso dataset rimescolato avvengono piu' volte per epoca, non
    // solo una.
    TempFile datasetFile("blackforge_cuda_test_multigpu_dataset_multibatch.bfdata");
    writeToyDataset(datasetFile.path);

    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/150, /*batchSize=*/4));

    backend::cuda::TrainRunResult singleGpu = backend::cuda::runTraining(compiled.program, compiled.module, "", "");
    backend::cuda::TrainRunResult multiGpu =
        backend::cuda::runMultiGpuTraining(compiled.program, compiled.module, {0, 0}, "", "");

    ASSERT_EQ(singleGpu.epochLosses.size(), multiGpu.epochLosses.size());
    for (std::size_t epoch = 0; epoch < singleGpu.epochLosses.size(); ++epoch) {
        EXPECT_NEAR(singleGpu.epochLosses[epoch], multiGpu.epochLosses[epoch], 1e-3)
            << "epoca " << (epoch + 1);
    }
}
