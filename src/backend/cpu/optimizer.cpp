#include "blackforge/backend/cpu/optimizer.hpp"

#include <cmath>

namespace blackforge::backend::cpu {

void SGD::step(const std::vector<Parameter*>& parameters) {
    for (Parameter* param : parameters) {
        for (std::size_t i = 0; i < param->value.elementCount(); ++i) {
            param->value.at(i) -= learningRate_ * param->grad.at(i);
        }
    }
}

AdamW::AdamW(float learningRate, float beta1, float beta2, float eps, float weightDecay)
    : learningRate_(learningRate), beta1_(beta1), beta2_(beta2), eps_(eps), weightDecay_(weightDecay) {}

void AdamW::step(const std::vector<Parameter*>& parameters) {
    ++stepCount_;
    auto t = static_cast<float>(stepCount_);
    float bias1Correction = 1.0F - std::pow(beta1_, t);
    float bias2Correction = 1.0F - std::pow(beta2_, t);

    for (Parameter* param : parameters) {
        MomentState& moment = state_[param->name];
        if (moment.m.empty()) {
            moment.m.assign(param->value.elementCount(), 0.0F);
            moment.v.assign(param->value.elementCount(), 0.0F);
        }

        for (std::size_t i = 0; i < param->value.elementCount(); ++i) {
            float grad = param->grad.at(i);

            moment.m[i] = beta1_ * moment.m[i] + (1.0F - beta1_) * grad;
            moment.v[i] = beta2_ * moment.v[i] + (1.0F - beta2_) * grad * grad;

            float mHat = moment.m[i] / bias1Correction;
            float vHat = moment.v[i] / bias2Correction;

            // Weight decay disaccoppiato (AdamW): agisce direttamente sul
            // parametro, non sul gradiente come nel classico L2 regularization.
            param->value.at(i) -=
                learningRate_ * (mHat / (std::sqrt(vHat) + eps_) + weightDecay_ * param->value.at(i));
        }
    }
}

}  // namespace blackforge::backend::cpu
