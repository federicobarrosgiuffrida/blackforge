#include "blackforge/blackbit/ternary_update.hpp"

#include <cmath>
#include <stdexcept>

#include "blackforge/blackbit/stochastic_round.hpp"

namespace blackforge::blackbit {

std::uint64_t parameterNameHash(const std::string& name) {
    std::uint64_t hash = 1469598103934665603ULL;  // FNV-1a offset basis
    for (unsigned char c : name) {
        hash ^= c;
        hash *= 1099511628211ULL;  // FNV prime
    }
    return hash;
}

TernaryUpdateStats applyTernaryUpdateBlock(TernaryTensor& weight, std::size_t firstRow, std::size_t rowCount,
                                            const float* delta, std::uint64_t seed, std::uint64_t step,
                                            TernaryUpdateUnits units) {
    if (firstRow + rowCount > weight.rows()) {
        throw std::out_of_range("applyTernaryUpdateBlock: intervallo di righe fuori dal tensore");
    }
    if (delta == nullptr && rowCount != 0) {
        throw std::invalid_argument("applyTernaryUpdateBlock: buffer di aggiornamento nullo");
    }

    TernaryUpdateStats stats;
    const std::size_t cols = weight.rowLength();
    // Il seme del passo si mescola una volta sola: dentro il ciclo
    // resta solo la funzione (seme, indice) -> casuale, che e' cio' che
    // un kernel CUDA puo' valutare per thread senza stato condiviso.
    const std::uint64_t stepSeed = splitMix64(seed ^ (step * 0xD1B54A32D192ED03ULL));

    for (std::size_t r = 0; r < rowCount; ++r) {
        const std::size_t row = firstRow + r;
        for (std::size_t c = 0; c < cols; ++c) {
            const std::size_t flat = row * cols + c;
            const float scale = weight.scaleAt(flat);
            const int oldTrit = weight.tritAt(flat);

            // Il peso reale aggiornato, espresso in unita' di griglia:
            // e' il valore che la griglia ternaria deve rappresentare.
            const float gridStep =
                units == TernaryUpdateUnits::Grid ? delta[r * cols + c] : delta[r * cols + c] / scale;
            const float targetInUnits = static_cast<float>(oldTrit) + gridStep;
            const int newTrit = stochasticRoundToTrit(targetInUnits, stepSeed, flat);

            ++stats.elementsConsidered;
            if (newTrit != oldTrit) {
                weight.setTritAt(flat, newTrit);
                ++stats.flips;
            }
        }
    }
    return stats;
}

void TernarySgdSink::registerTernary(const std::string& name, TernaryTensor& weight) { ternary_[name] = &weight; }

void TernarySgdSink::registerDense(const std::string& name, std::vector<float>& values) { dense_[name] = &values; }

void TernarySgdSink::consumeWeightGradientBlock(const ParameterId& id, std::size_t firstRow, std::size_t rowCount,
                                                 const float* block) {
    auto it = ternary_.find(id.name);
    if (it == ternary_.end()) {
        throw std::invalid_argument("TernarySgdSink: nessun peso ternario registrato con nome '" + id.name + "'");
    }

    // Il delta viene calcolato nello stesso buffer temporaneo per tutto
    // il blocco: e' l'unica memoria che questo sink usa, ed e'
    // proporzionale al blocco, non al parametro.
    std::vector<float> delta(rowCount * id.cols);

    if (normalizeUpdates_) {
        // Radice del valore quadratico medio del blocco: rende il passo
        // indipendente dal modulo del gradiente (vedi il commento
        // nell'header).
        double squared = 0.0;
        for (std::size_t i = 0; i < delta.size(); ++i) {
            squared += static_cast<double>(block[i]) * static_cast<double>(block[i]);
        }
        const double rms = std::sqrt(squared / static_cast<double>(delta.size()));
        if (rms <= 0.0) {
            return;  // blocco a gradiente esattamente nullo: niente da fare
        }
        const float inverse = static_cast<float>(1.0 / rms);
        for (std::size_t i = 0; i < delta.size(); ++i) {
            delta[i] = -learningRate_ * block[i] * inverse;
        }
    } else {
        for (std::size_t i = 0; i < delta.size(); ++i) {
            delta[i] = -learningRate_ * block[i];
        }
    }

    stats_ += applyTernaryUpdateBlock(*it->second, firstRow, rowCount, delta.data(),
                                       seed_ ^ parameterNameHash(id.name), step_,
                                       normalizeUpdates_ ? TernaryUpdateUnits::Grid : TernaryUpdateUnits::Weight);
}

void TernarySgdSink::consumeDenseGradient(const ParameterId& id, const float* values, std::size_t count) {
    auto it = dense_.find(id.name);
    if (it == dense_.end()) {
        throw std::invalid_argument("TernarySgdSink: nessun parametro denso registrato con nome '" + id.name + "'");
    }
    std::vector<float>& target = *it->second;
    if (target.size() != count) {
        throw std::invalid_argument("TernarySgdSink: il parametro denso '" + id.name +
                                     "' ha una dimensione diversa dal suo gradiente");
    }
    // Nessun arrotondamento qui: i parametri densi (gamma delle
    // RMSNorm, matrici di routing) sono numericamente sensibili e
    // costano insieme meno di un millesimo del modello (vedi
    // requisito 15).
    //
    // Con gli aggiornamenti normalizzati attivi, il passo e' RELATIVO
    // all'ampiezza del parametro stesso (lr * rms(parametro) *
    // gradiente / rms(gradiente)): senza, lo stesso learning rate
    // significherebbe due cose diverse per un peso ternario (frazione
    // di passo della griglia) e per uno denso (unita' assolute), e
    // regolarne uno scombinerebbe l'altro.
    if (normalizeUpdates_) {
        double gradientSquared = 0.0;
        double parameterSquared = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            gradientSquared += static_cast<double>(values[i]) * static_cast<double>(values[i]);
            parameterSquared += static_cast<double>(target[i]) * static_cast<double>(target[i]);
        }
        const double gradientRms = std::sqrt(gradientSquared / static_cast<double>(count));
        const double parameterRms = std::sqrt(parameterSquared / static_cast<double>(count));
        if (gradientRms <= 0.0 || parameterRms <= 0.0) {
            return;
        }
        const float factor = static_cast<float>(static_cast<double>(learningRate_) * parameterRms / gradientRms);
        for (std::size_t i = 0; i < count; ++i) {
            target[i] -= factor * values[i];
        }
        return;
    }

    for (std::size_t i = 0; i < count; ++i) {
        target[i] -= learningRate_ * values[i];
    }
}

}  // namespace blackforge::blackbit
