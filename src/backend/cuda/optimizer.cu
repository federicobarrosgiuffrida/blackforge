#include "blackforge/backend/cuda/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/backend/cuda/device_pool.hpp"

namespace blackforge::backend::cuda {

namespace {

constexpr int kBlockSize = 256;

int gridSizeFor(std::size_t n) { return static_cast<int>((n + kBlockSize - 1) / kBlockSize); }

// Varianti "multi-tensor" (stesso principio di ZeroTarget/zeroFused in
// model.cu): un solo lancio di kernel per TUTTI i parametri invece di
// uno per parametro (~68 per il modello del benchmark), blockIdx.y
// sceglie il parametro. L'array di metadati e' ricostruito e ricaricato
// su device ad ogni chiamata a step() (vedi zeroFused per il perche' e'
// sicuro: nessuna dipendenza da un risultato GPU pendente).

struct SgdTarget {
    float* value;
    const float* grad;
    std::size_t n;
};

__global__ void sgdStepFusedKernel(const SgdTarget* targets, float learningRate, std::size_t numTargets) {
    std::size_t paramIdx = blockIdx.y;
    if (paramIdx >= numTargets) {
        return;
    }
    SgdTarget target = targets[paramIdx];
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < target.n) {
        target.value[i] -= learningRate * target.grad[i];
    }
}

struct AdamWTarget {
    float* value;
    float* m;
    float* v;
    const float* grad;
    std::size_t n;
};

__global__ void adamWStepFusedKernel(const AdamWTarget* targets, float learningRate, float beta1, float beta2,
                                      float eps, float weightDecay, float bias1Correction, float bias2Correction,
                                      std::size_t numTargets) {
    std::size_t paramIdx = blockIdx.y;
    if (paramIdx >= numTargets) {
        return;
    }
    AdamWTarget target = targets[paramIdx];
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= target.n) {
        return;
    }

    float g = target.grad[i];
    target.m[i] = beta1 * target.m[i] + (1.0F - beta1) * g;
    target.v[i] = beta2 * target.v[i] + (1.0F - beta2) * g * g;

    float mHat = target.m[i] / bias1Correction;
    float vHat = target.v[i] / bias2Correction;

    target.value[i] -= learningRate * (mHat / (sqrtf(vHat) + eps) + weightDecay * target.value[i]);
}

}  // namespace

void SGD::step(const std::vector<Parameter*>& parameters) {
    std::vector<SgdTarget> targets;
    targets.reserve(parameters.size());
    std::size_t maxN = 0;
    for (Parameter* param : parameters) {
        std::size_t n = param->value.elementCount();
        targets.push_back({param->value.data(), param->grad.data(), n});
        maxN = std::max(maxN, n);
    }
    if (targets.empty() || maxN == 0) {
        return;
    }

    std::size_t bytes = targets.size() * sizeof(SgdTarget);
    auto* deviceTargets = static_cast<SgdTarget*>(devicePoolAcquire(bytes));
    BLACKFORGE_CUDA_CHECK(cudaMemcpy(deviceTargets, targets.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid(static_cast<unsigned int>(gridSizeFor(maxN)), static_cast<unsigned int>(targets.size()));
    sgdStepFusedKernel<<<grid, kBlockSize>>>(deviceTargets, learningRate_, targets.size());
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());

    devicePoolRelease(deviceTargets, bytes);
}

AdamW::AdamW(float learningRate, float beta1, float beta2, float eps, float weightDecay)
    : learningRate_(learningRate), beta1_(beta1), beta2_(beta2), eps_(eps), weightDecay_(weightDecay) {}

void AdamW::step(const std::vector<Parameter*>& parameters) {
    ++stepCount_;
    auto t = static_cast<float>(stepCount_);
    float bias1Correction = 1.0F - std::pow(beta1_, t);
    float bias2Correction = 1.0F - std::pow(beta2_, t);

    std::vector<AdamWTarget> targets;
    targets.reserve(parameters.size());
    std::size_t maxN = 0;
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
        targets.push_back({param->value.data(), moment.m.data(), moment.v.data(), param->grad.data(), n});
        maxN = std::max(maxN, n);
    }
    if (targets.empty() || maxN == 0) {
        return;
    }

    std::size_t bytes = targets.size() * sizeof(AdamWTarget);
    auto* deviceTargets = static_cast<AdamWTarget*>(devicePoolAcquire(bytes));
    BLACKFORGE_CUDA_CHECK(cudaMemcpy(deviceTargets, targets.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid(static_cast<unsigned int>(gridSizeFor(maxN)), static_cast<unsigned int>(targets.size()));
    adamWStepFusedKernel<<<grid, kBlockSize>>>(deviceTargets, learningRate_, beta1_, beta2_, eps_, weightDecay_,
                                                bias1Correction, bias2Correction, targets.size());
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());

    devicePoolRelease(deviceTargets, bytes);
}

}  // namespace blackforge::backend::cuda
