#pragma once

#include <string>

namespace blackforge::runtime {

// Famiglia di dispositivi di calcolo supportati da BlackForge.
enum class DeviceKind { CPU, CUDA };

// Rappresenta un dispositivo di calcolo target.
//
// Il backend CUDA non e' ancora implementato: per ora l'unico
// dispositivo costruibile e' la CPU. Device esiste comunque gia' come
// astrazione stabile in modo che il codice che la usa (executor, CLI)
// non debba cambiare quando il backend CUDA verra' aggiunto.
class Device {
public:
    static Device cpu() { return Device(DeviceKind::CPU, 0); }

    [[nodiscard]] DeviceKind kind() const { return kind_; }
    [[nodiscard]] int index() const { return index_; }
    [[nodiscard]] std::string toString() const;

private:
    Device(DeviceKind kind, int index) : kind_(kind), index_(index) {}

    DeviceKind kind_;
    int index_;
};

}  // namespace blackforge::runtime
