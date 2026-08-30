#include "blackforge/blackbit/gradient.hpp"

#include <stdexcept>

namespace blackforge::blackbit {

void DenseGradientCollector::consumeWeightGradientBlock(const ParameterId& id, std::size_t firstRow,
                                                         std::size_t rowCount, const float* block) {
    if (firstRow + rowCount > id.rows) {
        throw std::out_of_range("DenseGradientCollector: blocco fuori dalla forma del parametro '" + id.name + "'");
    }

    std::vector<float>& target = gradients_[id.name];
    if (target.empty()) {
        target.assign(id.rows * id.cols, 0.0F);
    } else if (target.size() != id.rows * id.cols) {
        throw std::invalid_argument("DenseGradientCollector: il parametro '" + id.name +
                                     "' ha ricevuto blocchi con forme diverse");
    }

    // ACCUMULA (non sovrascrive): un parametro condiviso, come
    // l'embedding legata alla proiezione di uscita, riceve contributi
    // da piu' punti del backward.
    for (std::size_t i = 0; i < rowCount * id.cols; ++i) {
        target[firstRow * id.cols + i] += block[i];
    }
}

void DenseGradientCollector::consumeDenseGradient(const ParameterId& id, const float* values, std::size_t count) {
    std::vector<float>& target = gradients_[id.name];
    if (target.empty()) {
        target.assign(count, 0.0F);
    } else if (target.size() != count) {
        throw std::invalid_argument("DenseGradientCollector: il parametro '" + id.name +
                                     "' ha ricevuto gradienti di dimensioni diverse");
    }
    for (std::size_t i = 0; i < count; ++i) {
        target[i] += values[i];
    }
}

const std::vector<float>& DenseGradientCollector::gradient(const std::string& name) const {
    auto it = gradients_.find(name);
    if (it == gradients_.end()) {
        throw std::out_of_range("DenseGradientCollector: nessun gradiente per il parametro '" + name + "'");
    }
    return it->second;
}

std::size_t DenseGradientCollector::heldBytes() const {
    std::size_t total = 0;
    for (const auto& entry : gradients_) {
        total += entry.second.size() * sizeof(float);
    }
    return total;
}

GradientLifetimeStats& gradientLifetimeStats() {
    static GradientLifetimeStats stats;
    return stats;
}

void resetGradientLifetimeStats() { gradientLifetimeStats() = GradientLifetimeStats{}; }

}  // namespace blackforge::blackbit
