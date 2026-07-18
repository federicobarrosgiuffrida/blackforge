#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace blackforge::runtime {

// Tensore residente in memoria host (CPU).
//
// E' il tipo di valore usato dal backend CPU di riferimento. I dati
// sono sempre memorizzati come float a 32 bit indipendentemente dal
// formato numerico dichiarato nel programma BlackForge (bf16, fp8,
// ...): il backend CPU serve a verificare la correttezza funzionale,
// non a riprodurre fedelmente la precisione numerica dell'hardware.
// L'emulazione dei formati ridotti e i kernel a precisione reale
// arriveranno con il backend CUDA.
class Tensor {
public:
    Tensor() = default;
    Tensor(std::vector<std::size_t> shape, std::vector<float> data);

    static Tensor zeros(std::vector<std::size_t> shape);
    static Tensor filled(std::vector<std::size_t> shape, float value);

    [[nodiscard]] const std::vector<std::size_t>& shape() const { return shape_; }
    [[nodiscard]] std::size_t rank() const { return shape_.size(); }
    [[nodiscard]] std::size_t dim(std::size_t axis) const { return shape_.at(axis); }
    [[nodiscard]] std::size_t elementCount() const;

    [[nodiscard]] const std::vector<float>& data() const { return data_; }
    [[nodiscard]] std::vector<float>& data() { return data_; }

    [[nodiscard]] float at(std::size_t flatIndex) const { return data_.at(flatIndex); }
    float& at(std::size_t flatIndex) { return data_.at(flatIndex); }

    // Statistiche semplici, usate per riassumere un tensore senza
    // stamparne tutti i valori (es. nell'output di 'blackforge run').
    [[nodiscard]] float min() const;
    [[nodiscard]] float max() const;
    [[nodiscard]] float mean() const;

    [[nodiscard]] std::string shapeToString() const;

private:
    std::vector<std::size_t> shape_;
    std::vector<float> data_;
};

}  // namespace blackforge::runtime
