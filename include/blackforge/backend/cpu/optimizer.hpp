#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "blackforge/backend/cpu/model.hpp"

namespace blackforge::backend::cpu {

// Interfaccia comune, usata dove l'optimizer va scelto a runtime (es.
// dal nome 'sgd'/'adamw' in un blocco 'train' di BlackForge).
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(const std::vector<Parameter*>& parameters) = 0;

    // Cambia il learning rate usato dagli step successivi, senza
    // toccare altro stato (per AdamW, i momenti accumulati e il
    // contatore di step restano invariati): serve a implementare uno
    // scheduler che decade il learning rate lungo le epoche senza
    // perdere lo stato dell'optimizer.
    virtual void setLearningRate(float learningRate) = 0;
};

// Discesa del gradiente stocastica, senza momento: param -= lr * grad.
class SGD : public Optimizer {
public:
    explicit SGD(float learningRate) : learningRate_(learningRate) {}

    void step(const std::vector<Parameter*>& parameters) override;
    void setLearningRate(float learningRate) override { learningRate_ = learningRate; }

private:
    float learningRate_;
};

// AdamW (Loshchilov & Hutter, "Decoupled Weight Decay Regularization",
// 2019): Adam con weight decay disaccoppiato dal gradiente.
class AdamW : public Optimizer {
public:
    explicit AdamW(float learningRate = 1e-3F, float beta1 = 0.9F, float beta2 = 0.999F, float eps = 1e-8F,
                    float weightDecay = 0.01F);

    void step(const std::vector<Parameter*>& parameters) override;
    void setLearningRate(float learningRate) override { learningRate_ = learningRate; }

private:
    struct MomentState {
        std::vector<float> m;
        std::vector<float> v;
    };

    float learningRate_;
    float beta1_;
    float beta2_;
    float eps_;
    float weightDecay_;
    std::size_t stepCount_ = 0;

    // Stato (primo/secondo momento) per parametro, indicizzato per nome:
    // sopravvive tra una step() e la successiva, come richiesto da Adam.
    std::unordered_map<std::string, MomentState> state_;
};

}  // namespace blackforge::backend::cpu
