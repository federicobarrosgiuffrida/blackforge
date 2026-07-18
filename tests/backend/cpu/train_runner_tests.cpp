#include "blackforge/backend/cpu/train_runner.hpp"

#include <cstdio>
#include <filesystem>

#include <gtest/gtest.h>

#include "blackforge/backend/cpu/checkpoint.hpp"
#include "blackforge/backend/cpu/model.hpp"
#include "blackforge/data/dataset.hpp"
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

// Un piccolo problema di regressione lineare risolvibile esattamente:
// 4 esempi one-hot in ingresso (4 feature), target a 2 valori.
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

// Un piccolo problema di classificazione a 2 classi, linearmente
// separabile: 4 esempi one-hot in ingresso (4 feature), target one-hot
// a 2 classi (ogni riga somma a 1, coerente con softmaxCrossEntropy).
void writeToyClassificationDataset(const std::string& path) {
    std::vector<std::size_t> inputShape{4};
    std::vector<std::size_t> targetShape{2};
    std::vector<float> inputs{
        1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F,
    };
    std::vector<float> targets{
        1.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F,
    };
    data::saveDataset(path, inputShape, targetShape, inputs, targets, 4);
}

// I letterali stringa di BlackForge interpretano '\' come inizio di
// escape (come C/C++/Python): un percorso Windows con backslash va
// convertito in forward slash prima di essere incorporato in un
// sorgente .bf, altrimenti il lexer segnala escape sconosciuti.
std::string toForwardSlashes(const std::string& path) {
    std::string result = path;
    for (char& c : result) {
        if (c == '\\') {
            c = '/';
        }
    }
    return result;
}

std::string toyProgram(const std::string& datasetPath, int epochs, const std::string& optimizerName) {
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
           "    batch_size 4\n"
           "    learning_rate 0.3\n"
           "}\n";
}

std::string toyProgramCrossEntropy(const std::string& datasetPath, int epochs) {
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
           "    loss cross_entropy\n"
           "    optimizer adamw\n"
           "    epochs " +
           std::to_string(epochs) +
           "\n"
           "    batch_size 4\n"
           "    learning_rate 0.3\n"
           "}\n";
}

std::string toyProgramWithLrSchedule(const std::string& datasetPath, int epochs) {
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
           "    optimizer adamw\n"
           "    epochs " +
           std::to_string(epochs) +
           "\n"
           "    batch_size 4\n"
           "    learning_rate 0.3\n"
           "    lr_schedule cosine\n"
           "}\n";
}

std::string toyProgramWithLora(const std::string& datasetPath, int epochs, long long rank) {
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
           "    optimizer adamw\n"
           "    epochs " +
           std::to_string(epochs) +
           "\n"
           "    batch_size 4\n"
           "    learning_rate 0.3\n"
           "    lora {\n"
           "        rank " +
           std::to_string(rank) +
           "\n"
           "        alpha 4.0\n"
           "    }\n"
           "}\n";
}

}  // namespace

TEST(TrainRunnerTest, RiduceLaLossAttraversoLeEpoche) {
    TempFile datasetFile("blackforge_test_train_dataset.bfdata");
    writeToyDataset(datasetFile.path);

    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/50, "adamw"));

    backend::cpu::TrainRunResult result = backend::cpu::runTraining(compiled.program, compiled.module, "", "");

    ASSERT_EQ(result.epochLosses.size(), 50u);
    EXPECT_EQ(result.modelName, "M");
    EXPECT_LT(result.epochLosses.back(), result.epochLosses.front() * 0.1);
}

TEST(TrainRunnerTest, FunzionaAncheConSgd) {
    TempFile datasetFile("blackforge_test_train_dataset_sgd.bfdata");
    writeToyDataset(datasetFile.path);

    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/50, "sgd"));
    backend::cpu::TrainRunResult result = backend::cpu::runTraining(compiled.program, compiled.module, "", "");

    ASSERT_EQ(result.epochLosses.size(), 50u);
    EXPECT_LT(result.epochLosses.back(), result.epochLosses.front());
}

TEST(TrainRunnerTest, RiduceLaLossConCrossEntropy) {
    TempFile datasetFile("blackforge_test_train_dataset_ce.bfdata");
    writeToyClassificationDataset(datasetFile.path);

    Compiled compiled = compile(toyProgramCrossEntropy(datasetFile.path, /*epochs=*/50));
    backend::cpu::TrainRunResult result = backend::cpu::runTraining(compiled.program, compiled.module, "", "");

    ASSERT_EQ(result.epochLosses.size(), 50u);
    EXPECT_LT(result.epochLosses.back(), result.epochLosses.front() * 0.1);
}

TEST(TrainRunnerTest, RiduceLaLossConLrScheduleCosine) {
    TempFile datasetFile("blackforge_test_train_dataset_lrsched.bfdata");
    writeToyDataset(datasetFile.path);

    Compiled compiled = compile(toyProgramWithLrSchedule(datasetFile.path, /*epochs=*/50));
    backend::cpu::TrainRunResult result = backend::cpu::runTraining(compiled.program, compiled.module, "", "");

    ASSERT_EQ(result.epochLosses.size(), 50u);
    EXPECT_LT(result.epochLosses.back(), result.epochLosses.front());
}

TEST(TrainRunnerTest, ConvergeAncheConPiuDiUnBatchPerEpocaEShufflingAttivo) {
    // writeToyDataset ha 4 esempi: con batch_size=2 ci sono 2 batch per
    // epoca, quindi lo shuffling (che riordina gli esempi prima di
    // costruire i batch) cambia davvero quali esempi finiscono in quale
    // batch da un'epoca all'altra (a differenza degli altri test di
    // questo file, che usano batch_size=numEsempi: un solo batch per
    // epoca, dove l'ordine non ha alcun effetto). Verifica che la
    // riduzione della loss non sia stata rotta dallo shuffling.
    TempFile datasetFile("blackforge_test_train_dataset_multibatch.bfdata");
    writeToyDataset(datasetFile.path);

    std::string program = toyProgram(datasetFile.path, /*epochs=*/50, "adamw");
    auto pos = program.find("batch_size 4");
    ASSERT_NE(pos, std::string::npos);
    program.replace(pos, std::string("batch_size 4").size(), "batch_size 2");

    Compiled compiled = compile(program);
    backend::cpu::TrainRunResult result = backend::cpu::runTraining(compiled.program, compiled.module, "", "");

    ASSERT_EQ(result.epochLosses.size(), 50u);
    EXPECT_LT(result.epochLosses.back(), result.epochLosses.front() * 0.5);
}

TEST(TrainRunnerTest, LanciaSeIlProgrammaNonHaBloccoTrain) {
    Compiled compiled = compile(
        "model M {\n"
        "    input bf16[4]\n"
        "}\n");

    EXPECT_THROW((void)backend::cpu::runTraining(compiled.program, compiled.module, "", ""), std::runtime_error);
}

TEST(TrainRunnerTest, LanciaSeIlDatasetNonEsiste) {
    Compiled compiled = compile(toyProgram("blackforge_percorso_inesistente.bfdata", 1, "sgd"));
    EXPECT_THROW((void)backend::cpu::runTraining(compiled.program, compiled.module, "", ""), std::runtime_error);
}

TEST(TrainRunnerTest, SalvaERicaricaUnCheckpointPerIlFineTuning) {
    TempFile datasetFile("blackforge_test_train_dataset_ft.bfdata");
    writeToyDataset(datasetFile.path);
    TempFile checkpointFile("blackforge_test_train_checkpoint.bfckpt");

    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/30, "adamw"));

    // Prima sessione: addestra da zero e salva un checkpoint.
    backend::cpu::TrainRunResult first =
        backend::cpu::runTraining(compiled.program, compiled.module, "", checkpointFile.path);
    ASSERT_TRUE(std::filesystem::exists(checkpointFile.path));

    // Seconda sessione ("fine-tuning"): riparte dal checkpoint salvato.
    // La loss della prima epoca dovrebbe partire gia' bassa (vicina a
    // dove si era fermato il primo addestramento), non da zero come un
    // modello nuovo.
    backend::cpu::TrainRunResult second =
        backend::cpu::runTraining(compiled.program, compiled.module, checkpointFile.path, "");

    ASSERT_FALSE(second.epochLosses.empty());
    EXPECT_LT(second.epochLosses.front(), first.epochLosses.front());
}

TEST(TrainRunnerTest, LoraRichiedeUnCheckpointDiPartenza) {
    TempFile datasetFile("blackforge_test_train_dataset_lora_nockpt.bfdata");
    writeToyDataset(datasetFile.path);

    Compiled compiled = compile(toyProgramWithLora(datasetFile.path, 5, 2));

    // 'lora' senza --from-checkpoint non ha senso (allenerebbe un
    // adapter su pesi casuali): deve fallire con un errore chiaro,
    // non silenziosamente.
    EXPECT_THROW((void)backend::cpu::runTraining(compiled.program, compiled.module, "", ""), std::runtime_error);
}

TEST(TrainRunnerTest, LoraRiduceLaLossEMantieneCongelatiIPesiDiBase) {
    TempFile datasetFile("blackforge_test_train_dataset_lora.bfdata");
    writeToyDataset(datasetFile.path);
    TempFile baseCheckpoint("blackforge_test_train_base.bfckpt");
    TempFile loraCheckpoint("blackforge_test_train_lora.bfckpt");

    // Prima sessione: pretraining normale, checkpoint di base.
    Compiled compiled = compile(toyProgram(datasetFile.path, /*epochs=*/30, "adamw"));
    backend::cpu::runTraining(compiled.program, compiled.module, "", baseCheckpoint.path);

    // Seconda sessione: fine-tuning LoRA a partire da quel checkpoint.
    Compiled compiledLora = compile(toyProgramWithLora(datasetFile.path, /*epochs=*/30, /*rank=*/2));
    backend::cpu::TrainRunResult loraResult = backend::cpu::runTraining(
        compiledLora.program, compiledLora.module, baseCheckpoint.path, loraCheckpoint.path);

    ASSERT_FALSE(loraResult.epochLosses.empty());
    EXPECT_LT(loraResult.epochLosses.back(), loraResult.epochLosses.front());

    // I pesi di base salvati nel checkpoint LoRA devono coincidere
    // esattamente con quelli del checkpoint pre-allenato originale:
    // devono essere rimasti congelati durante il fine-tuning LoRA.
    backend::cpu::Model baseModel(compiled.module.models.front());
    backend::cpu::loadCheckpoint(baseModel, baseCheckpoint.path);

    backend::cpu::Model loraModel(compiledLora.module.models.front(), 42, backend::cpu::LoraOptions{2, 4.0});
    backend::cpu::loadCheckpoint(loraModel, loraCheckpoint.path);

    auto baseParams = baseModel.allParameters();  // weight, bias
    auto loraParams = loraModel.allParameters();  // weight, bias, loraA, loraB

    ASSERT_GE(loraParams.size(), baseParams.size());
    for (std::size_t i = 0; i < baseParams.size(); ++i) {
        ASSERT_EQ(baseParams[i]->value.elementCount(), loraParams[i]->value.elementCount());
        for (std::size_t j = 0; j < baseParams[i]->value.elementCount(); ++j) {
            EXPECT_FLOAT_EQ(baseParams[i]->value.at(j), loraParams[i]->value.at(j))
                << "parametro " << i << " indice " << j;
        }
    }
}
