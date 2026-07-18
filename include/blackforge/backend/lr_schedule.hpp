#pragma once

#include <cmath>

namespace blackforge::backend {

// Cosine annealing (Loshchilov & Hutter, "SGDR", 2016): fa decadere il
// learning rate da 'baseLearningRate' (all'epoca 1) a 0 (all'ultima
// epoca) seguendo mezza onda di coseno. 'epoch' e' 1-based. Condivisa
// tra i train runner CPU e CUDA per evitare che le due implementazioni
// del blocco 'lr_schedule cosine' divergano silenziosamente.
inline float cosineAnnealingLearningRate(long long epoch, long long totalEpochs, float baseLearningRate) {
    if (totalEpochs <= 1) {
        return baseLearningRate;
    }
    constexpr double kPi = 3.14159265358979323846;
    double progress = static_cast<double>(epoch - 1) / static_cast<double>(totalEpochs - 1);
    return static_cast<float>(baseLearningRate * 0.5 * (1.0 + std::cos(kPi * progress)));
}

}  // namespace blackforge::backend
