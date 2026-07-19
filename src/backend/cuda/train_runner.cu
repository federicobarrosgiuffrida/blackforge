#include "blackforge/backend/cuda/train_runner.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

#include "blackforge/backend/cuda/checkpoint.hpp"
#include "blackforge/backend/cuda/device_tensor.hpp"
#include "blackforge/backend/cuda/loss.hpp"
#include "blackforge/backend/cuda/model.hpp"
#include "blackforge/backend/cuda/optimizer.hpp"
#include "blackforge/backend/lr_schedule.hpp"
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

// Lancia std::invalid_argument se un elemento di 'hostTensor' e' fuori
// da [0, bound) — stessa validazione fatta storicamente da
// embeddingLookup()/softmaxCrossEntropySparse() sul device, ma eseguita
// qui direttamente sui dati host PRIMA di caricarli su device (vedi
// Model::forward()'s 'inputRangeTrusted' e le varianti "Accumulate" in
// loss.hpp): stessi valori, zero round-trip device->host->device per
// ogni singolo step di addestramento. Duplicata deliberatamente in
// multi_gpu_train_runner.cu (stessa idea, diversa unita' di traduzione).
void validateHostIndexRange(const runtime::Tensor& hostTensor, std::size_t bound, const char* errorLabel) {
    for (std::size_t i = 0; i < hostTensor.elementCount(); ++i) {
        auto index = static_cast<long long>(std::lround(hostTensor.at(i)));
        if (index < 0 || static_cast<std::size_t>(index) >= bound) {
            throw std::invalid_argument(std::string(errorLabel) + ": indice " + std::to_string(index) +
                                         " fuori da [0, " + std::to_string(bound) + ")");
        }
    }
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
    const auto* lrScheduleField = findField<ast::TrainLrScheduleField>(trainDecl->fields);
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

    Model model(*modelIR, /*seed=*/42, module.precision);

    // Varianti "Accumulate" (vedi loss.hpp): calcolano il gradiente
    // subito, ma sommano la loss di riga in un accumulatore device
    // invece di leggerla sull'host ad ogni chiamata — una sola cudaMemcpy
    // a fine epoca invece di una per step (vedi il ciclo sotto).
    bool isSparseCrossEntropy = (lossField->name == "cross_entropy_sparse");
    std::function<DeviceTensor(const DeviceTensor&, const DeviceTensor&, DeviceTensor&)> lossAccumulateFn;
    if (lossField->name == "cross_entropy") {
        lossAccumulateFn = softmaxCrossEntropyAccumulate;
    } else if (isSparseCrossEntropy) {
        lossAccumulateFn = softmaxCrossEntropySparseAccumulate;
    } else {
        lossAccumulateFn = meanSquaredErrorAccumulate;
    }

    // Se il primo layer del modello e' 'embedding', i token id in
    // ingresso possono essere validati direttamente sui dati host del
    // batch (vedi validateHostIndexRange sopra), PRIMA di caricarli su
    // device — stessi valori che embeddingLookup()/embeddingLookupBackward()
    // validerebbero comunque, ma senza il round-trip device->host->device
    // che farebbero se richiamati con la validazione interna attiva (vedi
    // Model::forward()/backward()).
    std::optional<std::size_t> firstLayerVocabSize = model.firstLayerEmbeddingVocabSize();

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
                         << ", learning_rate=" << learningRate
                         << (lrScheduleField != nullptr ? " (schedule: " + lrScheduleField->name + ")" : "")
                         << ")\n";
    }

    TrainRunResult result;
    result.modelName = modelIR->name;

    for (long long epoch = 1; epoch <= epochs; ++epoch) {
        // Stesso schema del train runner CPU: rimescola gli esempi ad
        // ogni epoca (seme derivato dal numero di epoca) e, se
        // richiesto, aggiorna il learning rate secondo lo schedule.
        dataset.shuffle(static_cast<unsigned int>(epoch));

        if (lrScheduleField != nullptr) {
            float scheduledLr = cosineAnnealingLearningRate(epoch, epochs, static_cast<float>(learningRate));
            optimizer->setLearningRate(scheduledLr);
        }

        // Accumulatore device della loss dell'intera epoca: una singola
        // cudaMemcpy dopo l'ultimo batch legge il totale, invece di
        // sincronizzare la pipeline GPU una volta per ogni singolo step
        // (vedi le varianti "Accumulate" in loss.hpp).
        DeviceTensor lossAccumulator = DeviceTensor::zeros({1});
        std::size_t lossDivisor = 0;  // righe per chiamata * batchesPerEpoca, noto dopo il primo batch

        for (std::size_t b = 0; b < batchesPerEpoch; ++b) {
            data::Dataset::Batch batch = dataset.batch(b * batchSize, batchSize);

            if (firstLayerVocabSize.has_value()) {
                validateHostIndexRange(batch.input, *firstLayerVocabSize, "embeddingLookup");
            }

            DeviceTensor inputDevice = DeviceTensor::fromHost(batch.input);

            model.zeroGrad();
            DeviceTensor output = model.forward(inputDevice, firstLayerVocabSize.has_value());

            if (isSparseCrossEntropy) {
                // batch.target e' ancora sull'host qui: stessi valori che
                // softmaxCrossEntropySparse() validerebbe sul device, zero
                // round-trip aggiuntivo (vedi softmaxCrossEntropySparseAccumulate).
                validateHostIndexRange(batch.target, output.shape().back(), "softmaxCrossEntropySparse");
            }
            DeviceTensor targetDevice = DeviceTensor::fromHost(batch.target);

            std::size_t rowsPerCall =
                isSparseCrossEntropy || lossField->name == "cross_entropy"
                    ? output.elementCount() / output.shape().back()
                    : output.elementCount();
            lossDivisor += rowsPerCall;

            DeviceTensor grad = lossAccumulateFn(output, targetDevice, lossAccumulator);
            model.backward(grad, firstLayerVocabSize.has_value());
            optimizer->step(model.parameters());
        }

        float lossSum = lossAccumulator.toHost().at(0);
        double avgLoss = static_cast<double>(lossSum) / static_cast<double>(lossDivisor);
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
