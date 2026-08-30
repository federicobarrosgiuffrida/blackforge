#include "blackforge/blackbit/checkpoint.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace blackforge::blackbit {

namespace {

constexpr char kMagic[8] = {'B', 'F', 'B', 'I', 'T', '\0', '\0', '\0'};

enum class ParameterKind : std::uint8_t { Ternary = 0, Dense = 1 };

template <typename T>
void writeScalar(std::ostream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T readScalar(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("checkpoint BlackBit: file terminato prima del previsto");
    }
    return value;
}

void writeString(std::ostream& out, const std::string& value) {
    writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(value.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

std::string readString(std::istream& in) {
    const auto length = readScalar<std::uint32_t>(in);
    std::string value(length, '\0');
    in.read(value.data(), length);
    if (!in) {
        throw std::runtime_error("checkpoint BlackBit: file terminato prima del previsto");
    }
    return value;
}

// Raccoglie i parametri del modello riusando registerParameters(), che
// esiste gia' per gli ottimizzatori: nessun secondo elenco di parametri
// da tenere allineato a mano (e quindi nessun parametro che possa
// finire in un ottimizzatore ma non in un checkpoint, o viceversa).
struct ParameterCollector {
    std::vector<std::pair<std::string, TernaryTensor*>> ternary;
    std::vector<std::pair<std::string, std::vector<float>*>> dense;

    void registerTernary(const std::string& name, TernaryTensor& weight) { ternary.emplace_back(name, &weight); }
    void registerDense(const std::string& name, std::vector<float>& values) { dense.emplace_back(name, &values); }
};

ParameterCollector collect(BlackBitModel& model) {
    ParameterCollector collector;
    model.registerParameters(collector);
    return collector;
}

void readHeaderMagicAndVersion(std::istream& in, const std::string& path) {
    char magic[sizeof(kMagic)]{};
    in.read(magic, sizeof(kMagic));
    if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("checkpoint BlackBit: '" + path +
                                  "' non ha il magic atteso (non e' un checkpoint BlackBit)");
    }
    const auto version = readScalar<std::uint32_t>(in);
    if (version != kBlackBitCheckpointVersion) {
        throw std::runtime_error("checkpoint BlackBit: '" + path + "' e' di versione " + std::to_string(version) +
                                  ", questa build legge la versione " +
                                  std::to_string(kBlackBitCheckpointVersion));
    }
}

}  // namespace

void saveCheckpoint(const std::string& path, BlackBitModel& model, const BlackBitTrainingState& state,
                     LowRankProjectedOptimizer* optimizer) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("checkpoint BlackBit: impossibile scrivere '" + path + "'");
    }

    out.write(kMagic, sizeof(kMagic));
    writeScalar<std::uint32_t>(out, kBlackBitCheckpointVersion);
    writeString(out, toJson(model.config()));
    writeScalar<std::uint64_t>(out, state.step);
    writeScalar<std::uint64_t>(out, state.tokensSeen);
    writeScalar<float>(out, state.learningRate);
    writeScalar<std::uint64_t>(out, state.rngSeed);
    writeScalar<std::uint64_t>(out, state.optimizerStep);
    writeScalar<std::uint8_t>(out, optimizer != nullptr ? 1 : 0);

    const ParameterCollector collector = collect(model);
    writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(collector.ternary.size() + collector.dense.size()));

    for (const auto& entry : collector.ternary) {
        writeString(out, entry.first);
        writeScalar<std::uint8_t>(out, static_cast<std::uint8_t>(ParameterKind::Ternary));
        // Impacchettato, non espanso: e' il punto dell'intero formato.
        entry.second->serialize(out);
    }
    for (const auto& entry : collector.dense) {
        writeString(out, entry.first);
        writeScalar<std::uint8_t>(out, static_cast<std::uint8_t>(ParameterKind::Dense));
        writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(entry.second->size()));
        out.write(reinterpret_cast<const char*>(entry.second->data()),
                  static_cast<std::streamsize>(entry.second->size() * sizeof(float)));
    }

    if (optimizer != nullptr) {
        optimizer->serializeState(out);
    }

    if (!out) {
        throw std::runtime_error("checkpoint BlackBit: scrittura di '" + path + "' fallita");
    }
}

BlackBitConfig readCheckpointConfig(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("checkpoint BlackBit: impossibile leggere '" + path + "'");
    }
    readHeaderMagicAndVersion(in, path);
    return parseConfigJson(readString(in), path);
}

BlackBitTrainingState loadCheckpoint(const std::string& path, BlackBitModel& model,
                                      LowRankProjectedOptimizer* optimizer) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("checkpoint BlackBit: impossibile leggere '" + path + "'");
    }

    readHeaderMagicAndVersion(in, path);

    // La configurazione salvata deve coincidere con quella del modello
    // in memoria: caricare i pesi di un modello a 28 layer dentro uno a
    // 12 produrrebbe forme incompatibili a meta' del caricamento, con
    // il modello gia' parzialmente sovrascritto.
    const std::string savedJson = readString(in);
    if (savedJson != toJson(model.config())) {
        throw std::runtime_error("checkpoint BlackBit: '" + path +
                                  "' e' stato salvato con una configurazione diversa da quella del modello "
                                  "corrente.\nNel file:\n" +
                                  savedJson + "\nIn memoria:\n" + toJson(model.config()));
    }

    BlackBitTrainingState state;
    state.step = readScalar<std::uint64_t>(in);
    state.tokensSeen = readScalar<std::uint64_t>(in);
    state.learningRate = readScalar<float>(in);
    state.rngSeed = readScalar<std::uint64_t>(in);
    state.optimizerStep = readScalar<std::uint64_t>(in);
    const bool hasOptimizer = readScalar<std::uint8_t>(in) != 0;

    ParameterCollector collector = collect(model);
    std::unordered_map<std::string, TernaryTensor*> ternaryByName;
    std::unordered_map<std::string, std::vector<float>*> denseByName;
    for (const auto& entry : collector.ternary) {
        ternaryByName[entry.first] = entry.second;
    }
    for (const auto& entry : collector.dense) {
        denseByName[entry.first] = entry.second;
    }

    const auto parameterCount = readScalar<std::uint32_t>(in);
    for (std::uint32_t i = 0; i < parameterCount; ++i) {
        const std::string name = readString(in);
        const auto kind = static_cast<ParameterKind>(readScalar<std::uint8_t>(in));

        if (kind == ParameterKind::Ternary) {
            TernaryTensor loaded = TernaryTensor::deserialize(in);
            auto it = ternaryByName.find(name);
            if (it == ternaryByName.end()) {
                throw std::runtime_error("checkpoint BlackBit: il file contiene il parametro ternario '" + name +
                                          "', che il modello corrente non ha");
            }
            if (loaded.shape() != it->second->shape() || loaded.groupSize() != it->second->groupSize()) {
                throw std::runtime_error("checkpoint BlackBit: il parametro '" + name +
                                          "' ha forma o raggruppamento diversi da quelli del modello");
            }
            *it->second = std::move(loaded);
        } else {
            const auto count = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
            auto it = denseByName.find(name);
            if (it == denseByName.end()) {
                throw std::runtime_error("checkpoint BlackBit: il file contiene il parametro denso '" + name +
                                          "', che il modello corrente non ha");
            }
            if (it->second->size() != count) {
                throw std::runtime_error("checkpoint BlackBit: il parametro denso '" + name + "' ha " +
                                          std::to_string(count) + " valori nel file e " +
                                          std::to_string(it->second->size()) + " nel modello");
            }
            in.read(reinterpret_cast<char*>(it->second->data()),
                    static_cast<std::streamsize>(count * sizeof(float)));
            if (!in) {
                throw std::runtime_error("checkpoint BlackBit: file terminato prima del previsto");
            }
        }
    }

    if (hasOptimizer) {
        if (optimizer != nullptr) {
            optimizer->deserializeState(in);
        }
        // Se il chiamante non ha passato un ottimizzatore, lo stato
        // resta nel file e semplicemente non viene letto: caricare solo
        // i pesi per fare inferenza e' un uso legittimo.
    } else if (optimizer != nullptr) {
        throw std::runtime_error("checkpoint BlackBit: '" + path +
                                  "' non contiene lo stato dell'ottimizzatore, ma ne e' stato richiesto il "
                                  "caricamento (riprendere con momenti azzerati e' una scelta, non un default "
                                  "silenzioso: passa nullptr se e' quello che vuoi)");
    }

    return state;
}

}  // namespace blackforge::blackbit
