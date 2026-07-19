#include "blackforge/backend/cuda/device_tensor.hpp"

#include <numeric>
#include <utility>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/backend/cuda/device_pool.hpp"

namespace blackforge::backend::cuda {

namespace {

std::size_t product(const std::vector<std::size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies<>());
}

}  // namespace

DeviceTensor::DeviceTensor(std::vector<std::size_t> shape) : shape_(std::move(shape)) {
    std::size_t bytes = product(shape_) * sizeof(float);
    if (bytes > 0) {
        // Pool per-device invece di cudaMalloc diretto (vedi
        // device_pool.hpp): la stragrande maggioranza dei DeviceTensor
        // di questo backend sono intermedi di vita brevissima (un
        // singolo matmul/attivazione), riallocati con le STESSE forme
        // ad ogni step di un training loop.
        data_ = static_cast<float*>(devicePoolAcquire(bytes));
    }
}

DeviceTensor::~DeviceTensor() {
    if (data_ != nullptr) {
        // Torna al pool invece di cudaFree diretto: il distruttore non
        // puo' lanciare eccezioni, e devicePoolRelease() non ne lancia
        // (vedi device_pool.cu).
        devicePoolRelease(data_, elementCount() * sizeof(float));
    }
}

DeviceTensor::DeviceTensor(DeviceTensor&& other) noexcept
    : data_(other.data_), shape_(std::move(other.shape_)) {
    other.data_ = nullptr;
}

DeviceTensor& DeviceTensor::operator=(DeviceTensor&& other) noexcept {
    if (this != &other) {
        if (data_ != nullptr) {
            // Torna al pool invece di cudaFree diretto — stessa scelta
            // del distruttore (vedi sopra). Prima di questo fix,
            // l'assegnamento-spostamento (usato ad ogni 'current = op(...)'
            // in Model::forward()/backward() e ad ogni zeroGrad()) faceva
            // una cudaFree reale, bypassando completamente il pool di
            // memoria (device_pool.hpp): il buffer liberato qui non
            // tornava mai al free-list, quindi la successiva
            // devicePoolAcquire() per la stessa dimensione mancava sempre
            // la cache e doveva fare una nuova cudaMalloc — il pool
            // veniva svuotato in pratica ad ogni riassegnamento, il
            // percorso piu' comune nell'intero forward/backward.
            devicePoolRelease(data_, elementCount() * sizeof(float));
        }
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        other.data_ = nullptr;
    }
    return *this;
}

DeviceTensor DeviceTensor::fromHost(const runtime::Tensor& host) {
    DeviceTensor device(host.shape());
    if (device.elementCount() > 0) {
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(device.data_, host.data().data(), device.elementCount() * sizeof(float),
                                         cudaMemcpyHostToDevice));
    }
    return device;
}

DeviceTensor DeviceTensor::zeros(std::vector<std::size_t> shape) {
    DeviceTensor device(std::move(shape));
    if (device.elementCount() > 0) {
        // Un pattern di byte tutto a zero e' anche il float 0.0f valido
        // (rappresentazione IEEE 754), quindi cudaMemset(...,0,...) e'
        // corretto qui (non lo sarebbe per un valore diverso da zero).
        BLACKFORGE_CUDA_CHECK(cudaMemset(device.data_, 0, device.elementCount() * sizeof(float)));
    }
    return device;
}

runtime::Tensor DeviceTensor::toHost() const {
    std::vector<float> hostData(elementCount());
    if (!hostData.empty()) {
        BLACKFORGE_CUDA_CHECK(
            cudaMemcpy(hostData.data(), data_, hostData.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return runtime::Tensor(shape_, std::move(hostData));
}

DeviceTensor DeviceTensor::reshaped(std::vector<std::size_t> newShape) && {
    if (product(newShape) != elementCount()) {
        throw std::invalid_argument("DeviceTensor::reshaped: il numero di elementi non corrisponde");
    }
    DeviceTensor result;
    result.data_ = data_;
    result.shape_ = std::move(newShape);
    data_ = nullptr;
    return result;
}

DeviceTensor DeviceTensor::clone() const {
    DeviceTensor copy(shape_);
    if (copy.elementCount() > 0) {
        BLACKFORGE_CUDA_CHECK(
            cudaMemcpy(copy.data_, data_, copy.elementCount() * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    return copy;
}

std::size_t DeviceTensor::elementCount() const { return product(shape_); }

}  // namespace blackforge::backend::cuda
