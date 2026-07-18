#include "blackforge/runtime/device.hpp"

namespace blackforge::runtime {

std::string Device::toString() const {
    switch (kind_) {
        case DeviceKind::CPU: return "cpu";
        case DeviceKind::CUDA: return "cuda:" + std::to_string(index_);
    }
    return "?";
}

}  // namespace blackforge::runtime
