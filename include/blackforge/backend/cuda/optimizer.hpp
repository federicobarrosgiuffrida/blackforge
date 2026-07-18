#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "blackforge/backend/cuda/model.hpp"

namespace blackforge::backend::cuda {

// Interfaccia comune, usata dove l'optimizer va scelto a runtime (es.
// dal nome 'sgd'/'adamw' in un blocco 'train' di BlackForge). Stessa
// interfaccia della controparte CPU (backend::cpu::Optimizer), ma opera
// su Parameter residenti su device: l'aggiornamento dei pesi avviene
// interamente tramite kernel CUDA, senza mai copiare valore/gradiente
// sull'host.
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(const std::vector<Parameter*>& parameters) = 0;
};

// Discesa del gradiente stocastica, senza momento: param -= lr * grad.
class SGD : public Optimizer {
public:
    explicit SGD(float learningRate) : learningRate_(learningRate) {}

    void step(const std::vector<Parameter*>& parameters) override;

private:
    float learningRate_;
};

// AdamW (Loshchilov & Hutter, "Decoupled Weight Decay Regularization",
// 2019): stessa formula del backend CPU (backend::cpu::AdamW), con lo
// stato dei momenti (m, v) residente su device.
class AdamW : public Optimizer {
public:
    explicit AdamW(float learningRate = 1e-3F, float beta1 = 0.9F, float beta2 = 0.999F, float eps = 1e-8F,
                    float weightDecay = 0.01F);

    void step(const std::vector<Parameter*>& parameters) override;

private:
    struct MomentState {
        DeviceTensor m;
        DeviceTensor v;
    };

    float learningRate_;
    float beta1_;
    float beta2_;
    float eps_;
    float weightDecay_;
    std::size_t stepCount_ = 0;

    // Stato per parametro, indicizzato per nome: sopravvive tra una
    // step() e la successiva, come richiesto da Adam.
    std::unordered_map<std::string, MomentState> state_;
};

}  // namespace blackforge::backend::cuda
