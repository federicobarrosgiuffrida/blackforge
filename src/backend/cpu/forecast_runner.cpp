#include "blackforge/backend/cpu/forecast_runner.hpp"

#include <stdexcept>
#include <variant>

#include "blackforge/backend/cpu/checkpoint.hpp"
#include "blackforge/backend/cpu/model.hpp"
#include "blackforge/backend/cpu/random_init.hpp"

namespace blackforge::backend::cpu {

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

}  // namespace

ForecastRunResult runForecast(const ast::Program& program, const ir::Module& module,
                               const std::string& fromCheckpointPath, std::size_t batchSize,
                               std::ostream* progressOutput) {
    const ast::ForecastDecl* forecastDecl = nullptr;
    for (const auto& decl : program.declarations) {
        if (std::holds_alternative<ast::ForecastDecl>(decl)) {
            forecastDecl = &std::get<ast::ForecastDecl>(decl);
            break;
        }
    }
    if (forecastDecl == nullptr) {
        throw std::runtime_error("il programma non contiene un blocco 'forecast'");
    }

    const auto* modelField = findField<ast::ForecastModelField>(forecastDecl->fields);
    const auto* horizonField = findField<ast::ForecastHorizonField>(forecastDecl->fields);
    if (modelField == nullptr || horizonField == nullptr) {
        throw std::runtime_error("il blocco 'forecast' non e' completo (manca model/horizon): l'analisi "
                                  "semantica avrebbe dovuto rifiutarlo");
    }

    if (fromCheckpointPath.empty()) {
        throw std::runtime_error("il forecasting richiede un modello pre-allenato (--from-checkpoint): un "
                                  "modello con pesi casuali non produce previsioni sensate");
    }

    const ir::ModelIR* modelIR = findModelIR(module, modelField->name);
    if (modelIR == nullptr) {
        throw std::runtime_error("modello '" + modelField->name + "' non trovato nella rappresentazione interna");
    }
    if (modelIR->pipelines.empty()) {
        throw std::runtime_error("il modello '" + modelField->name + "' non ha una pipeline da eseguire");
    }

    const ir::Value& inputValue = modelIR->valueById(modelIR->inputValue);
    const ir::Value& outputValue = modelIR->valueById(modelIR->pipelines.front().outputValue());

    // Il forecasting autoregressivo richiede di poter riusare l'output
    // come input del passo successivo: serve che l'ultima dimensione
    // (le feature) coincida tra input e output del modello.
    if (inputValue.shape.empty() || outputValue.shape.empty() ||
        inputValue.shape.back().literalValue != outputValue.shape.back().literalValue ||
        inputValue.shape.back().isSymbolic || outputValue.shape.back().isSymbolic) {
        throw std::runtime_error("il modello '" + modelField->name +
                                  "' non e' adatto al forecasting autoregressivo: l'ultima dimensione "
                                  "dell'input e quella dell'output devono coincidere ed essere concrete");
    }

    Model model(*modelIR);
    loadCheckpoint(model, fromCheckpointPath);

    std::vector<std::size_t> shape;
    shape.reserve(inputValue.shape.size());
    for (const auto& dim : inputValue.shape) {
        shape.push_back(dim.isSymbolic ? batchSize : static_cast<std::size_t>(dim.literalValue));
    }
    runtime::Tensor current = randomTensor(shape, seedFor(42, 0, 0x9E3779B1U));

    if (progressOutput != nullptr) {
        *progressOutput << "Forecasting di '" << modelIR->name << "' per " << horizonField->value
                         << " passi (batch=" << batchSize << ")\n";
    }

    ForecastRunResult result;
    result.modelName = modelIR->name;

    for (long long step = 1; step <= horizonField->value; ++step) {
        current = model.forward(current);
        result.steps.push_back(current);

        if (progressOutput != nullptr) {
            *progressOutput << "  passo " << step << "/" << horizonField->value << "  " << current.shapeToString()
                             << "  min=" << current.min() << " max=" << current.max() << " mean=" << current.mean()
                             << "\n";
        }
    }

    return result;
}

}  // namespace blackforge::backend::cpu
