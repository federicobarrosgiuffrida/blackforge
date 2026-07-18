#include "blackforge/backend/cuda/train_runner.hpp"

#include <memory>
#include <stdexcept>
#include <variant>

#include "blackforge/backend/cuda/checkpoint.hpp"
#include "blackforge/backend/cuda/device_tensor.hpp"
#include "blackforge/backend/cuda/loss.hpp"
#include "blackforge/backend/cuda/model.hpp"
#include "blackforge/backend/cuda/optimizer.hpp"
#include "blackforge/data/dataset.hpp"

namespace blackforge::backend::cuda {

namespace {

template <typename FieldType, typename VariantType>
const FieldType* findField(const std::vector<VariantType>& fields) {
    for (const auto& field : fields) {
        if (std::holds_alternative<FieldType>(field)) {
            return &std::get<FieldType>(field);
        }
    }
    return nullptr;
}

const ir::ModelIR* findModelIR(const ir::Module& module, const std::string& name) {
    for (const auto& model : module.models) {
        if (model.name == name) {
            return &model;
        }
    }
    return nullptr;
}

const ast::DatasetDecl* findDatasetDecl(const ast::Program& program, const std::string& name) {
    for (const auto& decl : program.declarations) {
        if (std::holds_alternative<ast::DatasetDecl>(decl)) {
            const auto& dataset = std::get<ast::DatasetDecl>(decl);
            if (dataset.name == name) {
                return &dataset;
            }
        }
    }
    return nullptr;
}

}  // namespace

TrainRunResult runTraining(const ast::Program& program, const ir::Module& module,
                            const std::string& fromCheckpointPath, const std::string& saveCheckpointPath,
                            std::ostream* progressOutput) {
    const ast::TrainDecl* trainDecl = nullptr;
    for (const auto& decl : program.declarations) {
        if (std::holds_alternative<ast::TrainDecl>(decl)) {
            trainDecl = &std::get<ast::TrainDecl>(decl);
            break;
        }
    }
    if (trainDecl == nullptr) {
        throw std::runtime_error("il programma non contiene un blocco 'train'");
    }

    const auto* modelField = findField<ast::TrainModelField>(trainDecl->fields);
    const auto* datasetField = findField<ast::TrainDatasetField>(trainDecl->fields);
    const auto* lossField = findField<ast::TrainLossField>(trainDecl->fields);
    const auto* optimizerField = findField<ast::TrainOptimizerField>(trainDecl->fields);
    const auto* epochsField = findField<ast::TrainEpochsField>(trainDecl->fields);
    const auto* batchSizeField = findField<ast::TrainBatchSizeField>(trainDecl->fields);
    const auto* learningRateField = findField<ast::TrainLearningRateField>(trainDecl->fields);
    const auto* loraField = findField<ast::TrainLoraField>(trainDecl->fields);

    if (modelField == nullptr || datasetField == nullptr || lossField == nullptr || optimizerField == nullptr ||
        epochsField == nullptr || batchSizeField == nullptr) {
        throw std::runtime_error("il blocco 'train' non e' completo (manca model/dataset/loss/optimizer/epochs/"
                                  "batch_size): l'analisi semantica avrebbe dovuto rifiutarlo");
    }
    if (loraField != nullptr) {
        throw std::runtime_error("'blackforge train --device cuda' non supporta ancora 'lora': nessun adapter a "
                                  "basso rango implementato su cuda::Model. Usa '--device cpu' per LoRA.");
    }
    if (lossField->name != "mse") {
        throw std::runtime_error("'blackforge train --device cuda' supporta solo 'loss mse' per ora (dichiarata: '" +
                                  lossField->name +
                                  "'): cross-entropy su GPU e' lavoro futuro. Usa '--device cpu' per "
                                  "'cross_entropy'.");
    }

    const ir::ModelIR* modelIR = findModelIR(module, modelField->name);
    if (modelIR == nullptr) {
        throw std::runtime_error("modello '" + modelField->name + "' non trovato nella rappresentazione interna");
    }

    const ast::DatasetDecl* datasetDecl = findDatasetDecl(program, datasetField->name);
    if (datasetDecl == nullptr) {
        throw std::runtime_error("dataset '" + datasetField->name + "' non trovato");
    }
    const auto* pathField = findField<ast::DatasetPathField>(datasetDecl->fields);
    if (pathField == nullptr) {
        throw std::runtime_error("il dataset '" + datasetField->name + "' non ha un campo 'path'");
    }

    data::Dataset dataset = data::loadDataset(pathField->path);

    Model model(*modelIR);

    if (!fromCheckpointPath.empty()) {
        loadCheckpoint(model, fromCheckpointPath);
        if (progressOutput != nullptr) {
            *progressOutput << "Pesi caricati da '" << fromCheckpointPath << "' (fine-tuning)\n";
        }
    }

    double learningRate = (learningRateField != nullptr) ? learningRateField->value : 1e-3;
    std::unique_ptr<Optimizer> optimizer;
    if (optimizerField->name == "sgd") {
        optimizer = std::make_unique<SGD>(static_cast<float>(learningRate));
    } else {
        optimizer = std::make_unique<AdamW>(static_cast<float>(learningRate));
    }

    auto epochs = static_cast<long long>(epochsField->value);
    auto batchSize = static_cast<std::size_t>(batchSizeField->value);

    if (dataset.numExamples() < batchSize) {
        throw std::runtime_error("il dataset ha " + std::to_string(dataset.numExamples()) +
                                  " esempi, meno del batch_size richiesto (" + std::to_string(batchSize) + ")");
    }

    std::size_t batchesPerEpoch = dataset.numExamples() / batchSize;

    if (progressOutput != nullptr) {
        *progressOutput << "Addestramento CUDA di '" << modelIR->name << "' su " << dataset.numExamples()
                         << " esempi (" << batchesPerEpoch << " batch/epoca, batch_size=" << batchSize
                         << ", loss=" << lossField->name << ", optimizer=" << optimizerField->name
                         << ", learning_rate=" << learningRate << ")\n";
    }

    TrainRunResult result;
    result.modelName = modelIR->name;

    for (long long epoch = 1; epoch <= epochs; ++epoch) {
        double epochLossSum = 0.0;
        for (std::size_t b = 0; b < batchesPerEpoch; ++b) {
            data::Dataset::Batch batch = dataset.batch(b * batchSize, batchSize);
            DeviceTensor inputDevice = DeviceTensor::fromHost(batch.input);
            DeviceTensor targetDevice = DeviceTensor::fromHost(batch.target);

            model.zeroGrad();
            DeviceTensor output = model.forward(inputDevice);
            LossResult loss = meanSquaredError(output, targetDevice);
            model.backward(loss.grad);
            optimizer->step(model.parameters());

            epochLossSum += loss.value;
        }

        double avgLoss = epochLossSum / static_cast<double>(batchesPerEpoch);
        result.epochLosses.push_back(avgLoss);

        if (progressOutput != nullptr) {
            *progressOutput << "  epoca " << epoch << "/" << epochs << "  loss=" << avgLoss << "\n";
        }
    }

    if (!saveCheckpointPath.empty()) {
        saveCheckpoint(model, saveCheckpointPath);
        if (progressOutput != nullptr) {
            *progressOutput << "Pesi salvati in '" << saveCheckpointPath << "'\n";
        }
    }

    return result;
}

}  // namespace blackforge::backend::cuda
