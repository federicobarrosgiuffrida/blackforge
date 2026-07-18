#include "blackforge/backend/cuda/optimizer.hpp"

#include <cmath>

#include "blackforge/backend/cuda/cuda_check.hpp"

namespace blackforge::backend::cuda {

namespace {

constexpr int kBlockSize = 256;

int gridSizeFor(std::size_t n) { return static_cast<int>((n + kBlockSize - 1) / kBlockSize); }

__global__ void sgdStepKernel(float* value, const float* grad, float learningRate, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        value[i] -= learningRate * grad[i];
    }
}

__global__ void adamWStepKernel(float* value, float* m, float* v, const float* grad, float learningRate,
                                 float beta1, float beta2, float eps, float weightDecay, float bias1Correction,
                                 float bias2Correction, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float g = grad[i];
    m[i] = beta1 * m[i] + (1.0F - beta1) * g;
    v[i] = beta2 * v[i] + (1.0F - beta2) * g * g;

    float mHat = m[i] / bias1Correction;
    float vHat = v[i] / bias2Correction;

    // Weight decay disaccoppiato (AdamW): agisce direttamente sul
    // parametro, non sul gradiente come nel classico L2 regularization.
    value[i] -= learningRate * (mHat / (sqrtf(vHat) + eps) + weightDecay * value[i]);
}

}  // namespace

void SGD::step(const std::vector<Parameter*>& parameters) {
    for (Parameter* param : parameters) {
        std::size_t n = param->value.elementCount();
        if (n > 0) {
            sgdStepKernel<<<gridSizeFor(n), kBlockSize>>>(param->value.data(), param->grad.data(), learningRate_,
                                                           n);
            BLACKFORGE_CUDA_CHECK(cudaGetLastError());
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
        // shape() vuota == mai inizializzato (DeviceTensor di default):
        // elementCount() da solo non basta, perche' il prodotto di una
        // shape vuota vale 1 per convenzione (identita' moltiplicativa),
        // non 0.
        if (moment.m.shape().empty() && !param->value.shape().empty()) {
            moment.m = DeviceTensor::zeros(param->value.shape());
            moment.v = DeviceTensor::zeros(param->value.shape());
        }

        std::size_t n = param->value.elementCount();
        if (n > 0) {
            adamWStepKernel<<<gridSizeFor(n), kBlockSize>>>(param->value.data(), moment.m.data(), moment.v.data(),
                                                             param->grad.data(), learningRate_, beta1_, beta2_,
                                                             eps_, weightDecay_, bias1Correction, bias2Correction,
                                                             n);
            BLACKFORGE_CUDA_CHECK(cudaGetLastError());
        }
    }
}

}  // namespace blackforge::backend::cuda
