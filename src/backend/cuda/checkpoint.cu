#include "blackforge/backend/cuda/checkpoint.hpp"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace blackforge::backend::cuda {

namespace {

constexpr char kMagic[8] = {'B', 'F', 'C', 'K', 'P', 'T', '1', '\0'};

#if defined(_MSC_VER)
std::string describeErrno(int code) {
    char buffer[256];
    strerror_s(buffer, sizeof(buffer), code);
    return buffer;
}
#else
std::string describeErrno(int code) { return std::strerror(code); }
#endif

void writeU32(std::ostream& out, std::uint32_t value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }

void writeU64(std::ostream& out, std::uint64_t value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }

std::uint32_t readU32(std::istream& in) {
    std::uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

std::uint64_t readU64(std::istream& in) {
    std::uint64_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

void writeParameter(std::ostream& out, const Parameter& param) {
    writeU32(out, static_cast<std::uint32_t>(param.name.size()));
    out.write(param.name.data(), static_cast<std::streamsize>(param.name.size()));

    const auto& shape = param.value.shape();
    writeU32(out, static_cast<std::uint32_t>(shape.size()));
    for (std::size_t dim : shape) {
        writeU64(out, static_cast<std::uint64_t>(dim));
    }

    // I dati vivono su device: si copiano sull'host solo per la durata
    // della scrittura su file (l'I/O su disco non puo' avvenire in
    // altro modo), non e' un fallback della computazione.
    runtime::Tensor hostValue = param.value.toHost();
    const auto& data = hostValue.data();
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
}

}  // namespace

void saveCheckpoint(Model& model, const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("saveCheckpoint: impossibile scrivere il file '" + path +
                                  "' (errno=" + std::to_string(errno) + " " + describeErrno(errno) + ")");
    }

    out.write(kMagic, sizeof(kMagic));

    std::vector<Parameter*> params = model.parameters();
    writeU32(out, static_cast<std::uint32_t>(params.size()));
    for (const Parameter* param : params) {
        writeParameter(out, *param);
    }

    if (!out) {
        throw std::runtime_error("saveCheckpoint: scrittura fallita su '" + path + "'");
    }
}

void loadCheckpoint(Model& model, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("loadCheckpoint: impossibile aprire il file '" + path + "'");
    }

    char magic[8];
    in.read(magic, sizeof(magic));
    if (!in || std::string(magic, sizeof(magic)) != std::string(kMagic, sizeof(kMagic))) {
        throw std::runtime_error("loadCheckpoint: '" + path + "' non e' un checkpoint BlackForge valido");
    }

    std::unordered_map<std::string, Parameter*> byName;
    for (Parameter* param : model.parameters()) {
        byName[param->name] = param;
    }

    std::uint32_t count = readU32(in);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint32_t nameLength = readU32(in);
        std::string name(nameLength, '\0');
        in.read(name.data(), nameLength);

        std::uint32_t rank = readU32(in);
        std::vector<std::size_t> shape(rank);
        std::size_t elementCount = 1;
        for (std::uint32_t d = 0; d < rank; ++d) {
            shape[d] = static_cast<std::size_t>(readU64(in));
            elementCount *= shape[d];
        }

        std::vector<float> data(elementCount);
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(elementCount * sizeof(float)));

        if (!in) {
            throw std::runtime_error("loadCheckpoint: file '" + path + "' troncato o corrotto");
        }

        auto it = byName.find(name);
        if (it == byName.end()) {
            throw std::runtime_error("loadCheckpoint: il modello non ha un parametro chiamato '" + name + "'");
        }
        if (it->second->value.shape() != shape) {
            throw std::runtime_error("loadCheckpoint: forma incompatibile per '" + name + "'");
        }

        it->second->value = DeviceTensor::fromHost(runtime::Tensor(shape, std::move(data)));
    }
}

}  // namespace blackforge::backend::cuda
