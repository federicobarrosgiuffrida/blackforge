#include "blackforge/backend/cpu/random_init.hpp"

#include <random>

namespace blackforge::backend::cpu {

runtime::Tensor randomTensor(std::vector<std::size_t> shape, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.1F, 0.1F);

    std::size_t count = 1;
    for (std::size_t dim : shape) {
        count *= dim;
    }

    std::vector<float> data(count);
    for (float& value : data) {
        value = dist(rng);
    }

    return runtime::Tensor(std::move(shape), std::move(data));
}

unsigned int seedFor(unsigned int base, std::size_t valueId, unsigned int salt) {
    return base ^ (static_cast<unsigned int>(valueId) * salt);
}

}  // namespace blackforge::backend::cpu
