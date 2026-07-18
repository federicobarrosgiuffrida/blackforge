#pragma once

#include <optional>
#include <string>
#include <vector>

#include "blackforge/ir/module.hpp"
#include "blackforge/runtime/tensor.hpp"

namespace blackforge::backend::cpu {

// Un parametro allenabile: il suo valore corrente e il gradiente
// accumulato dall'ultima chiamata a Model::zeroGrad().
struct Parameter {
    std::string name;
    runtime::Tensor value;
    runtime::Tensor grad;
};

// Un modello CPU allenabile, costruito dalla pipeline di un ir::ModelIR.
//
// A differenza di Executor (milestone del backend CPU di riferimento,
// che rigenera pesi casuali ad ogni chiamata solo per ispezionare
// rapidamente un'esecuzione), Model possiede i propri parametri e li
// mantiene tra una forward() e la successiva: e' il tipo pensato per
// essere effettivamente allenato con un optimizer.
//
// NOTA: l'inizializzazione dei pesi resta deterministica ma non
// statisticamente valida (vedi random_init.hpp); una strategia di init
// seria (Xavier/Kaiming) e il caricamento di pesi pre-allenati sono
// lavoro futuro.
class Model {
public:
    // Lancia std::invalid_argument se il modello non ha pipeline, se ha
    // piu' di una dimensione di feature simbolica in ingresso a un
    // layer 'linear' (serve una dimensione concreta per allocare i
    // pesi).
    explicit Model(const ir::ModelIR& modelIR, unsigned int seed = 42);

    // Esegue la pipeline e salva le attivazioni intermedie necessarie a
    // backward(). Va chiamata prima di ogni backward().
    runtime::Tensor forward(const runtime::Tensor& input);

    // Calcola i gradienti dei parametri a partire dal gradiente
    // dell'uscita (dL/doutput), usando le attivazioni salvate
    // dall'ultima forward(). Li ACCUMULA (non li azzera): chiamare
    // zeroGrad() prima di ogni step di addestramento.
    void backward(const runtime::Tensor& outputGrad);

    void zeroGrad();

    [[nodiscard]] std::vector<Parameter*> parameters();

    [[nodiscard]] const std::string& name() const { return name_; }

private:
    struct LayerState {
        ir::OpKind kind;
        std::size_t operationOutputId;
        std::optional<Parameter> weight;  // valido solo se kind == Linear
        std::optional<Parameter> bias;    // valido solo se kind == Linear
        runtime::Tensor cachedInput;      // popolato da forward()
    };

    std::string name_;
    std::vector<LayerState> layers_;
};

}  // namespace blackforge::backend::cpu
