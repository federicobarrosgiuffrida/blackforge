#include "blackforge/backend/cuda/optimizer.hpp"

#include <cmath>

#include <gtest/gtest.h>

using blackforge::runtime::Tensor;
namespace cuda = blackforge::backend::cuda;

namespace {

cuda::Parameter makeParam(const std::string& name, std::vector<float> value, std::vector<float> grad) {
    std::vector<std::size_t> shape{value.size()};
    return cuda::Parameter{name, cuda::DeviceTensor::fromHost(Tensor(shape, std::move(value))),
                            cuda::DeviceTensor::fromHost(Tensor(shape, std::move(grad)))};
}

}  // namespace

TEST(CudaOptimizerTest, SgdAggiornaIlParametroSecondoLaFormulaEsatta) {
    cuda::Parameter param = makeParam("p", {1.0F, -2.0F}, {0.5F, 0.5F});
    cuda::SGD optimizer(0.1F);

    optimizer.step({&param});
    Tensor result = param.value.toHost();

    EXPECT_NEAR(result.at(0), 1.0F - 0.1F * 0.5F, 1e-6F);
    EXPECT_NEAR(result.at(1), -2.0F - 0.1F * 0.5F, 1e-6F);
}

TEST(CudaOptimizerTest, AdamWPrimoStepCorrispondeAllaFormulaConBiasCorrection) {
    cuda::Parameter param = makeParam("p", {1.0F}, {0.2F});
    cuda::AdamW optimizer(/*learningRate=*/0.1F, /*beta1=*/0.9F, /*beta2=*/0.999F, /*eps=*/1e-8F,
                           /*weightDecay=*/0.0F);

    optimizer.step({&param});
    Tensor result = param.value.toHost();

    float expected = 1.0F - 0.1F * (0.2F / (std::sqrt(0.04F) + 1e-8F));
    EXPECT_NEAR(result.at(0), expected, 1e-4F);
}

TEST(CudaOptimizerTest, AdamWMantieneStatoTraUnoStepEIlSuccessivo) {
    cuda::Parameter param = makeParam("p", {1.0F}, {0.2F});
    cuda::AdamW optimizer(0.1F);

    optimizer.step({&param});
    float afterFirst = param.value.toHost().at(0);

    param.grad = cuda::DeviceTensor::fromHost(Tensor({1}, {0.2F}));  // stesso gradiente al secondo step
    optimizer.step({&param});
    float afterSecond = param.value.toHost().at(0);

    // La bias correction e i momenti accumulati cambiano con lo step:
    // l'ampiezza del secondo aggiornamento non deve essere identica a
    // quella del primo.
    EXPECT_NE(afterFirst - 1.0F, afterSecond - afterFirst);
}

TEST(CudaOptimizerTest, WeightDecayDisaccoppiatoAgisceAncheAGradienteNullo) {
    cuda::Parameter param = makeParam("p", {1.0F}, {0.0F});
    cuda::AdamW optimizer(/*learningRate=*/0.1F, /*beta1=*/0.9F, /*beta2=*/0.999F, /*eps=*/1e-8F,
                           /*weightDecay=*/0.1F);

    optimizer.step({&param});
    Tensor result = param.value.toHost();

    EXPECT_NEAR(result.at(0), 1.0F - 0.1F * 0.1F * 1.0F, 1e-6F);
}
