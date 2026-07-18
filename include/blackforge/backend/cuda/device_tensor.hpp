#pragma once

#include <cstddef>
#include <vector>

#include "blackforge/runtime/tensor.hpp"

namespace blackforge::backend::cuda {

// Tensore residente in memoria device (GPU).
//
// Possiede il buffer float* allocato con cudaMalloc e lo libera nel
// distruttore (RAII). Non copiabile (evita doppie cudaFree accidentali
// da una copia superficiale del puntatore), spostabile. Come il
// backend CPU di riferimento, memorizza sempre float32: l'emulazione
// dei formati ridotti (fp8/bf16/tf32 come precisione di calcolo reale,
// Tensor Core) e' lavoro futuro rispetto a questa prima versione del
// backend CUDA.
class DeviceTensor {
public:
    DeviceTensor() = default;

    // Alloca (senza inizializzare) un buffer device della forma data.
    explicit DeviceTensor(std::vector<std::size_t> shape);

    ~DeviceTensor();

    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    DeviceTensor(DeviceTensor&& other) noexcept;
    DeviceTensor& operator=(DeviceTensor&& other) noexcept;

    // Copia i dati di un runtime::Tensor host su un nuovo DeviceTensor.
    static DeviceTensor fromHost(const runtime::Tensor& host);

    // Copia i dati da device a un nuovo runtime::Tensor host.
    [[nodiscard]] runtime::Tensor toHost() const;

    [[nodiscard]] float* data() { return data_; }
    [[nodiscard]] const float* data() const { return data_; }

    [[nodiscard]] const std::vector<std::size_t>& shape() const { return shape_; }
    [[nodiscard]] std::size_t rank() const { return shape_.size(); }
    [[nodiscard]] std::size_t dim(std::size_t axis) const { return shape_.at(axis); }
    [[nodiscard]] std::size_t elementCount() const;

private:
    float* data_ = nullptr;
    std::vector<std::size_t> shape_;
};

}  // namespace blackforge::backend::cuda
