#pragma once

#include <cstddef>

#include "blackforge/ir/module.hpp"
#include "blackforge/runtime/tensor.hpp"

namespace blackforge::backend::cuda {

// Controparte GPU di blackforge::backend::cpu::Executor: esegue la
// prima pipeline di una ModelIR sul backend CUDA.
//
// Usa la STESSA inizializzazione dei pesi (stesso seme, stessa
// funzione host blackforge::backend::cpu::randomTensor) dell'Executor
// CPU: a parita' di seme, CudaExecutor e Executor devono produrre
// risultati numericamente equivalenti (a meno degli arrotondamenti
// dovuti a un ordine diverso delle somme in virgola mobile tra CPU e
// GPU/cuBLAS) — e' esattamente quello che i test di parita' CPU/GPU
// verificano.
class Executor {
public:
    explicit Executor(unsigned int seed = 42) : seed_(seed) {}

    [[nodiscard]] runtime::Tensor makeSyntheticInput(const ir::Value& inputValue, std::size_t batchSize) const;

    // Lancia std::invalid_argument se il modello non ha pipeline.
    [[nodiscard]] runtime::Tensor run(const ir::ModelIR& model, const runtime::Tensor& input) const;

private:
    unsigned int seed_;
};

}  // namespace blackforge::backend::cuda
