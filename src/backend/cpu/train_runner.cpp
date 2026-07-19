#include "blackforge/backend/cpu/train_runner.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <variant>

#include "blackforge/backend/cpu/checkpoint.hpp"
#include "blackforge/backend/cpu/loss.hpp"
#include "blackforge/backend/cpu/model.hpp"
#include "blackforge/backend/cpu/optimizer.hpp"
#include "blackforge/backend/lr_schedule.hpp"
#include "blackforge/data/dataset.hpp"

namespace blackforge::backend::cpu {

namespace {

// Cerca il primo campo di tipo FieldType in un elenco di campi variant
// (usato sia per ast::TrainField sia per ast::DatasetField).
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

    // Un blocco 'train' semanticamente valido (precondizione di questa
    // funzione) garantisce la presenza di questi campi.
    const auto* modelField = findField<ast::TrainModelField>(trainDecl->fields);
    const auto* datasetField = findField<ast::TrainDatasetField>(trainDecl->fields);
    const auto* lossField = findField<ast::TrainLossField>(trainDecl->fields);
    const auto* optimizerField = findField<ast::TrainOptimizerField>(trainDecl->fields);
    const auto* epochsField = findField<ast::TrainEpochsField>(trainDecl->fields);
    const auto* batchSizeField = findField<ast::TrainBatchSizeField>(trainDecl->fields);
    const auto* learningRateField = findField<ast::TrainLearningRateField>(trainDecl->fields);
    const auto* lrScheduleField = findField<ast::TrainLrScheduleField>(trainDecl->fields);
    const auto* loraField = findField<ast::TrainLoraField>(trainDecl->fields);

    if (modelField == nullptr || datasetField == nullptr || lossField == nullptr || optimizerField == nullptr ||
        epochsField == nullptr || batchSizeField == nullptr) {
        throw std::runtime_error("il blocco 'train' non e' completo (manca model/dataset/loss/optimizer/epochs/"
                                  "batch_size): l'analisi semantica avrebbe dovuto rifiutarlo");
    }
    if (loraField != nullptr && fromCheckpointPath.empty()) {
        throw std::runtime_error("il blocco 'train' usa 'lora' ma non e' stato fornito un checkpoint di partenza "
                                  "(--from-checkpoint): allenare un adapter a basso rango su pesi casuali non ha "
                                  "senso, serve un modello gia' pre-allenato");
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

    std::optional<LoraOptions> loraOptions;
    if (loraField != nullptr) {
        loraOptions = LoraOptions{loraField->rank, loraField->alpha};
    }
    Model model(*modelIR, /*seed=*/42, loraOptions);

    if (!fromCheckpointPath.empty()) {
        loadCheckpoint(model, fromCheckpointPath);
        if (progressOutput != nullptr) {
            *progressOutput << "Pesi caricati da '" << fromCheckpointPath << "' ("
                             << (loraOptions.has_value() ? "base per LoRA" : "fine-tuning") << ")\n";
        }
    }

    if (loraOptions.has_value() && progressOutput != nullptr) {
        *progressOutput << "LoRA attivo: rank=" << loraOptions->rank << " alpha=" << loraOptions->alpha
                         << " (" << model.parameters().size() << " parametri allenabili, pesi di base congelati)\n";
    }

    std::function<LossResult(const runtime::Tensor&, const runtime::Tensor&)> lossFn;
    if (lossField->name == "cross_entropy") {
        lossFn = softmaxCrossEntropy;
    } else if (lossField->name == "cross_entropy_sparse") {
        lossFn = softmaxCrossEntropySparse;
    } else {
        lossFn = meanSquaredError;
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
        *progressOutput << "Addestramento di '" << modelIR->name << "' su " << dataset.numExamples() << " esempi ("
                         << batchesPerEpoch << " batch/epoca, batch_size=" << batchSize
                         << ", loss=" << lossField->name << ", optimizer=" << optimizerField->name
                         << ", learning_rate=" << learningRate
                         << (lrScheduleField != nullptr ? " (schedule: " + lrScheduleField->name + ")" : "")
                         << ")\n";
    }

    TrainRunResult result;
    result.modelName = modelIR->name;

    for (long long epoch = 1; epoch <= epochs; ++epoch) {
        // Rimescola l'ordine degli esempi ad ogni epoca (seme derivato
        // dal numero di epoca: riproducibile, ma diverso epoca per
        // epoca) cosi' che SGD/AdamW non vedano sempre gli stessi batch
        // nello stesso ordine.
        dataset.shuffle(static_cast<unsigned int>(epoch));

        if (lrScheduleField != nullptr) {
            float scheduledLr = cosineAnnealingLearningRate(epoch, epochs, static_cast<float>(learningRate));
            optimizer->setLearningRate(scheduledLr);
        }

        double epochLossSum = 0.0;
        for (std::size_t b = 0; b < batchesPerEpoch; ++b) {
            data::Dataset::Batch batch = dataset.batch(b * batchSize, batchSize);

            model.zeroGrad();
            runtime::Tensor output = model.forward(batch.input);
            LossResult loss = lossFn(output, batch.target);
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

}  // namespace blackforge::backend::cpu
