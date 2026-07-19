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

    // Alloca un buffer device della forma data, azzerato (cudaMemset):
    // usato per gradienti e stato dell'optimizer (momenti di AdamW), che
    // devono partire da zero prima di essere accumulati.
    static DeviceTensor zeros(std::vector<std::size_t> shape);

    // Copia i dati da device a un nuovo runtime::Tensor host.
    [[nodiscard]] runtime::Tensor toHost() const;

    // Copia device-to-device su un nuovo DeviceTensor indipendente:
    // serve dove servono due handle allo stesso contenuto che evolvono
    // in modo indipendente (es. un'attivazione cachata per il backward
    // mentre il forward continua a trasformarla), dato che DeviceTensor
    // non e' copiabile per evitare doppie cudaFree accidentali.
    [[nodiscard]] DeviceTensor clone() const;

    // Reinterpreta lo stesso buffer device con una nuova forma, senza
    // copiare (il numero di elementi deve coincidere): il layout flat
    // riga-maggiore di [batch, seq, features] e' identico, byte per
    // byte, a quello di [batch*seq, features], quindi non serve una
    // vera riorganizzazione dei dati per "appiattire" un tensore a
    // rango > 2 prima di passarlo a un primitivo puramente 2D come
    // matmul(). Qualificata su rvalue: consuma questo DeviceTensor
    // (ownership del buffer trasferita al risultato), per rendere
    // esplicito nel codice chiamante che non e' piu' utilizzabile con
    // la forma originale dopo la chiamata.
    [[nodiscard]] DeviceTensor reshaped(std::vector<std::size_t> newShape) &&;

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
