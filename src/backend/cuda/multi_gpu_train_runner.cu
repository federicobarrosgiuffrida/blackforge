#include "blackforge/backend/cuda/multi_gpu_train_runner.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

#include "blackforge/backend/cpu/ops.hpp"
#include "blackforge/backend/cuda/checkpoint.hpp"
#include "blackforge/backend/cuda/device_query.hpp"
#include "blackforge/backend/cuda/device_tensor.hpp"
#include "blackforge/backend/cuda/loss.hpp"
#include "blackforge/backend/cuda/model.hpp"
#include "blackforge/backend/cuda/optimizer.hpp"
#include "blackforge/backend/lr_schedule.hpp"
#include "blackforge/data/dataset.hpp"

namespace blackforge::backend::cuda {

namespace {

// Duplicati deliberatamente da train_runner.cu (stessa idea, diversa
// unita' di traduzione): helper minuscoli per estrarre i campi di un
// blocco 'train'/'dataset' dall'AST.
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

// Duplicata deliberatamente da train_runner.cu (stessa idea, diversa
// unita' di traduzione): vedi il commento li' per il razionale completo.
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

TrainRunResult runMultiGpuTraining(const ast::Program& program, const ir::Module& module,
                                    const std::vector<int>& deviceIndices, const std::string& fromCheckpointPath,
                                    const std::string& saveCheckpointPath, std::ostream* progressOutput) {
    if (deviceIndices.size() < 2) {
        throw std::invalid_argument("runMultiGpuTraining richiede almeno 2 indici di GPU (per una sola GPU usa "
                                     "runTraining)");
    }
    std::size_t numDevices = deviceIndices.size();

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
        throw std::runtime_error("'blackforge train --devices' non supporta ancora 'lora': nessun adapter a "
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

    auto batchSize = static_cast<std::size_t>(batchSizeField->value);
    if (batchSize % numDevices != 0) {
        throw std::runtime_error("runMultiGpuTraining: batch_size (" + std::to_string(batchSize) +
                                  ") deve essere divisibile per il numero di GPU (" + std::to_string(numDevices) +
                                  "), cosi' ogni replica riceve uno shard della stessa dimensione");
    }
    std::size_t shardSize = batchSize / numDevices;

    if (dataset.numExamples() < batchSize) {
        throw std::runtime_error("il dataset ha " + std::to_string(dataset.numExamples()) +
                                  " esempi, meno del batch_size richiesto (" + std::to_string(batchSize) + ")");
    }

    // Una replica completa del modello per GPU, stesso seme => pesi
    // iniziali identici su tutte le repliche (vedi backend::cpu::
    // random_init.hpp per il generatore deterministico condiviso da
    // ogni backend/Model/Executor di questo progetto).
    std::vector<Model> models;
    models.reserve(numDevices);
    for (std::size_t i = 0; i < numDevices; ++i) {
        setActiveDevice(deviceIndices[i]);
        models.emplace_back(*modelIR, /*seed=*/42, module.precision);
        if (!fromCheckpointPath.empty()) {
            loadCheckpoint(models[i], fromCheckpointPath);
        }
    }
    if (!fromCheckpointPath.empty() && progressOutput != nullptr) {
        *progressOutput << "Pesi caricati da '" << fromCheckpointPath << "' su " << numDevices
                         << " GPU (fine-tuning)\n";
    }

    // Vedi il commento equivalente in train_runner.cu: le varianti
    // "Accumulate" evitano una cudaMemcpy per ogni replica ad ogni step.
    bool isSparseCrossEntropy = (lossField->name == "cross_entropy_sparse");
    std::function<DeviceTensor(const DeviceTensor&, const DeviceTensor&, DeviceTensor&)> lossAccumulateFn;
    if (lossField->name == "cross_entropy") {
        lossAccumulateFn = softmaxCrossEntropyAccumulate;
    } else if (isSparseCrossEntropy) {
        lossAccumulateFn = softmaxCrossEntropySparseAccumulate;
    } else {
        lossAccumulateFn = meanSquaredErrorAccumulate;
    }

    // Stesso vocabolario per tutte le repliche (stesso modelIR): basta
    // interrogare la prima.
    std::optional<std::size_t> firstLayerVocabSize = models[0].firstLayerEmbeddingVocabSize();

    double learningRate = (learningRateField != nullptr) ? learningRateField->value : 1e-3;
    std::vector<std::unique_ptr<Optimizer>> optimizers;
    optimizers.reserve(numDevices);
    for (std::size_t i = 0; i < numDevices; ++i) {
        if (optimizerField->name == "sgd") {
            optimizers.push_back(std::make_unique<SGD>(static_cast<float>(learningRate)));
        } else {
            optimizers.push_back(std::make_unique<AdamW>(static_cast<float>(learningRate)));
        }
    }

    auto epochs = static_cast<long long>(epochsField->value);
    std::size_t batchesPerEpoch = dataset.numExamples() / batchSize;

    if (progressOutput != nullptr) {
        *progressOutput << "Addestramento CUDA multi-GPU di '" << modelIR->name << "' su " << numDevices
                         << " GPU (indici: ";
        for (std::size_t i = 0; i < numDevices; ++i) {
            *progressOutput << deviceIndices[i] << (i + 1 < numDevices ? "," : "");
        }
        *progressOutput << "), " << dataset.numExamples() << " esempi (" << batchesPerEpoch
                         << " batch/epoca, batch_size=" << batchSize << " -> shard=" << shardSize
                         << " per GPU, loss=" << lossField->name << ", optimizer=" << optimizerField->name
                         << ", learning_rate=" << learningRate
                         << (lrScheduleField != nullptr ? " (schedule: " + lrScheduleField->name + ")" : "")
                         << ")\n";
    }

    TrainRunResult result;
    result.modelName = modelIR->name;

    for (long long epoch = 1; epoch <= epochs; ++epoch) {
        // Un solo rimescolamento condiviso: ogni replica legge poi uno
        // shard DISGIUNTO dello stesso batch rimescolato (vedi sotto),
        // cosi' l'intero batch_size di esempi e' coperto esattamente
        // una volta per batch, come nel training a singola GPU.
        dataset.shuffle(static_cast<unsigned int>(epoch));

        if (lrScheduleField != nullptr) {
            float scheduledLr = cosineAnnealingLearningRate(epoch, epochs, static_cast<float>(learningRate));
            for (auto& optimizer : optimizers) {
                optimizer->setLearningRate(scheduledLr);
            }
        }

        // Un accumulatore device della loss dell'epoca per replica (vedi
        // il commento equivalente in train_runner.cu): una singola
        // cudaMemcpy per GPU dopo l'ultimo batch, invece di sincronizzare
        // ogni replica una volta per step.
        std::vector<DeviceTensor> lossAccumulators;
        lossAccumulators.reserve(numDevices);
        for (std::size_t i = 0; i < numDevices; ++i) {
            setActiveDevice(deviceIndices[i]);
            lossAccumulators.push_back(DeviceTensor::zeros({1}));
        }
        std::size_t lossDivisor = 0;

        for (std::size_t b = 0; b < batchesPerEpoch; ++b) {
            std::vector<std::vector<Parameter*>> paramsPerDevice(numDevices);

            // Forward + backward indipendente per replica, ognuna sul
            // proprio shard disgiunto del batch corrente e sulla
            // propria GPU (setActiveDevice prima di ogni operazione
            // device-side: allocazioni/kernel CUDA agiscono sempre sul
            // device "corrente" del contesto CUDA, non su quello
            // passato esplicitamente come parametro).
            for (std::size_t i = 0; i < numDevices; ++i) {
                setActiveDevice(deviceIndices[i]);
                data::Dataset::Batch shard = dataset.batch(b * batchSize + i * shardSize, shardSize);

                if (firstLayerVocabSize.has_value()) {
                    validateHostIndexRange(shard.input, *firstLayerVocabSize, "embeddingLookup");
                }

                DeviceTensor inputDevice = DeviceTensor::fromHost(shard.input);

                models[i].zeroGrad();
                DeviceTensor output = models[i].forward(inputDevice, firstLayerVocabSize.has_value());

                if (isSparseCrossEntropy) {
                    validateHostIndexRange(shard.target, output.shape().back(), "softmaxCrossEntropySparse");
                }
                DeviceTensor targetDevice = DeviceTensor::fromHost(shard.target);

                std::size_t rowsPerCall = isSparseCrossEntropy || lossField->name == "cross_entropy"
                                               ? output.elementCount() / output.shape().back()
                                               : output.elementCount();
                lossDivisor += rowsPerCall;

                DeviceTensor grad = lossAccumulateFn(output, targetDevice, lossAccumulators[i]);
                models[i].backward(grad, firstLayerVocabSize.has_value());

                paramsPerDevice[i] = models[i].parameters();
            }

            // All-reduce dei gradienti via staging host (vedi il
            // commento in multi_gpu_train_runner.hpp per il perche':
            // NCCL non e' disponibile su Windows, questo funziona su
            // qualunque combinazione di GPU indipendentemente dal
            // supporto P2P/NVLink). L'ordine dei parametri e'
            // identico su ogni replica: stessa pipeline, stessa
            // costruzione deterministica di Model.
            std::size_t numParams = paramsPerDevice[0].size();
            for (std::size_t p = 0; p < numParams; ++p) {
                setActiveDevice(deviceIndices[0]);
                runtime::Tensor accumulated = paramsPerDevice[0][p]->grad.toHost();
                for (std::size_t i = 1; i < numDevices; ++i) {
                    setActiveDevice(deviceIndices[i]);
                    runtime::Tensor deviceGrad = paramsPerDevice[i][p]->grad.toHost();
                    accumulated = blackforge::backend::cpu::add(accumulated, deviceGrad);
                }
                accumulated = blackforge::backend::cpu::scale(accumulated, 1.0F / static_cast<float>(numDevices));

                for (std::size_t i = 0; i < numDevices; ++i) {
                    setActiveDevice(deviceIndices[i]);
                    paramsPerDevice[i][p]->grad = DeviceTensor::fromHost(accumulated);
                }
            }

            // Ogni replica applica lo STESSO gradiente mediato: pesi e
            // stato dell'optimizer (momenti di AdamW) restano
            // sincronizzati tra le repliche ad ogni step, per
            // costruzione (partono identici e ricevono sempre lo
            // stesso aggiornamento).
            for (std::size_t i = 0; i < numDevices; ++i) {
                setActiveDevice(deviceIndices[i]);
                optimizers[i]->step(models[i].parameters());
            }

        }

        // Somma delle loss accumulate su ogni GPU: una cudaMemcpy per
        // replica (numDevices in tutto, invece di numDevices*batchesPerEpoca).
        double lossSum = 0.0;
        for (std::size_t i = 0; i < numDevices; ++i) {
            setActiveDevice(deviceIndices[i]);
            lossSum += static_cast<double>(lossAccumulators[i].toHost().at(0));
        }
        double avgLoss = lossSum / static_cast<double>(lossDivisor);
        result.epochLosses.push_back(avgLoss);

        if (progressOutput != nullptr) {
            *progressOutput << "  epoca " << epoch << "/" << epochs << "  loss=" << avgLoss << "\n";
        }
    }

    if (!saveCheckpointPath.empty()) {
        // Tutte le repliche restano sincronizzate ad ogni step (stessi
        // pesi iniziali, stesso gradiente mediato ad ogni step): basta
        // salvare la replica 0.
        setActiveDevice(deviceIndices[0]);
        saveCheckpoint(models[0], saveCheckpointPath);
        if (progressOutput != nullptr) {
            *progressOutput << "Pesi salvati in '" << saveCheckpointPath << "'\n";
        }
    }

    return result;
}

}  // namespace blackforge::backend::cuda
