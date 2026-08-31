#include "blackforge/blackbit/safetensors.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace blackforge::blackbit {

namespace {

// --------------------------------------------------------------- JSON
//
// Analizzatore del solo sottoinsieme che l'intestazione safetensors usa:
// oggetti, stringhe, array di interi, numeri. Nessun array annidato di
// oggetti, nessun escape unicode oltre quelli semplici. Volutamente
// separato dal parser di configurazione (config_json.cpp), che e' piatto
// per scelta e deve restare tale.

class JsonParser {
public:
    JsonParser(const std::string& text, std::string origin) : text_(text), origin_(std::move(origin)) {}

    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error("safetensors: " + origin_ + " (offset " + std::to_string(pos_) +
                                  "): " + message);
    }

    void skipSpace() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' ||
                                        text_[pos_] == '\r')) {
            ++pos_;
        }
    }

    char peek() {
        skipSpace();
        if (pos_ >= text_.size()) {
            fail("intestazione terminata prima del previsto");
        }
        return text_[pos_];
    }

    void expect(char c) {
        if (peek() != c) {
            fail(std::string("atteso '") + c + "', trovato '" + text_[pos_] + "'");
        }
        ++pos_;
    }

    bool consumeIf(char c) {
        if (pos_ < text_.size() && peek() == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    std::string parseString() {
        expect('"');
        std::string out;
        while (pos_ < text_.size() && text_[pos_] != '"') {
            if (text_[pos_] == '\\' && pos_ + 1 < text_.size()) {
                ++pos_;
                switch (text_[pos_]) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    default: out += text_[pos_]; break;
                }
                ++pos_;
                continue;
            }
            out += text_[pos_++];
        }
        if (pos_ >= text_.size()) {
            fail("stringa non terminata");
        }
        ++pos_;
        return out;
    }

    std::uint64_t parseUnsigned() {
        skipSpace();
        const std::size_t start = pos_;
        while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
            ++pos_;
        }
        if (pos_ == start) {
            fail("atteso un intero non negativo");
        }
        return std::stoull(text_.substr(start, pos_ - start));
    }

    std::vector<std::size_t> parseUnsignedArray() {
        std::vector<std::size_t> values;
        expect('[');
        if (consumeIf(']')) {
            return values;  // tensore a rango 0 (scalare)
        }
        while (true) {
            values.push_back(static_cast<std::size_t>(parseUnsigned()));
            if (consumeIf(',')) {
                continue;
            }
            expect(']');
            return values;
        }
    }

    // Salta un valore qualunque: serve per le chiavi che non ci
    // interessano, senza doverle modellare.
    void skipValue() {
        const char c = peek();
        if (c == '"') {
            (void)parseString();
            return;
        }
        if (c == '{' || c == '[') {
            const char close = c == '{' ? '}' : ']';
            int depth = 0;
            while (pos_ < text_.size()) {
                const char current = text_[pos_];
                if (current == '"') {
                    (void)parseString();
                    continue;
                }
                if (current == c) {
                    ++depth;
                } else if (current == close) {
                    --depth;
                    if (depth == 0) {
                        ++pos_;
                        return;
                    }
                }
                ++pos_;
            }
            fail("valore non terminato");
        }
        // Numero, true, false o null: si consuma fino al separatore.
        while (pos_ < text_.size() && text_[pos_] != ',' && text_[pos_] != '}' && text_[pos_] != ']') {
            ++pos_;
        }
    }

    [[nodiscard]] std::size_t position() const { return pos_; }

private:
    const std::string& text_;
    std::string origin_;
    std::size_t pos_ = 0;
};

SafetensorsDType parseDType(const std::string& name, const JsonParser& parser) {
    if (name == "F64") return SafetensorsDType::F64;
    if (name == "F32") return SafetensorsDType::F32;
    if (name == "F16") return SafetensorsDType::F16;
    if (name == "BF16") return SafetensorsDType::BF16;
    if (name == "I64") return SafetensorsDType::I64;
    if (name == "I32") return SafetensorsDType::I32;
    if (name == "I16") return SafetensorsDType::I16;
    if (name == "I8") return SafetensorsDType::I8;
    if (name == "U8") return SafetensorsDType::U8;
    if (name == "BOOL") return SafetensorsDType::BOOL;
    parser.fail("dtype '" + name + "' non riconosciuto");
}

// Converte 'count' elementi del tipo dato in float32.
void convertToFloat(SafetensorsDType dtype, const unsigned char* source, std::size_t count, float* out) {
    switch (dtype) {
        case SafetensorsDType::F32: {
            std::memcpy(out, source, count * sizeof(float));
            return;
        }
        case SafetensorsDType::BF16: {
            for (std::size_t i = 0; i < count; ++i) {
                std::uint16_t bits = 0;
                std::memcpy(&bits, source + i * 2, 2);
                out[i] = bf16ToFloat(bits);
            }
            return;
        }
        case SafetensorsDType::F16: {
            for (std::size_t i = 0; i < count; ++i) {
                std::uint16_t bits = 0;
                std::memcpy(&bits, source + i * 2, 2);
                out[i] = fp16ToFloat(bits);
            }
            return;
        }
        case SafetensorsDType::F64: {
            for (std::size_t i = 0; i < count; ++i) {
                double value = 0.0;
                std::memcpy(&value, source + i * 8, 8);
                out[i] = static_cast<float>(value);
            }
            return;
        }
        case SafetensorsDType::I64: {
            for (std::size_t i = 0; i < count; ++i) {
                std::int64_t value = 0;
                std::memcpy(&value, source + i * 8, 8);
                out[i] = static_cast<float>(value);
            }
            return;
        }
        case SafetensorsDType::I32: {
            for (std::size_t i = 0; i < count; ++i) {
                std::int32_t value = 0;
                std::memcpy(&value, source + i * 4, 4);
                out[i] = static_cast<float>(value);
            }
            return;
        }
        case SafetensorsDType::I16: {
            for (std::size_t i = 0; i < count; ++i) {
                std::int16_t value = 0;
                std::memcpy(&value, source + i * 2, 2);
                out[i] = static_cast<float>(value);
            }
            return;
        }
        case SafetensorsDType::I8: {
            for (std::size_t i = 0; i < count; ++i) {
                std::int8_t value = 0;
                std::memcpy(&value, source + i, 1);
                out[i] = static_cast<float>(value);
            }
            return;
        }
        case SafetensorsDType::U8:
        case SafetensorsDType::BOOL: {
            for (std::size_t i = 0; i < count; ++i) {
                out[i] = static_cast<float>(source[i]);
            }
            return;
        }
    }
    throw std::runtime_error("safetensors: dtype non gestito nella conversione");
}

}  // namespace

float bf16ToFloat(std::uint16_t bits) {
    // BF16 sono i 16 bit ALTI di un float32: la conversione e' esatta.
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float value = 0.0F;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

float fp16ToFloat(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16;
    std::uint32_t exponent = (bits >> 10) & 0x1FU;
    std::uint32_t mantissa = bits & 0x3FFU;
    std::uint32_t wide = 0;

    if (exponent == 0) {
        if (mantissa != 0) {
            // Subnormale in FP16: normalizzabile in FP32, che ha un
            // esponente molto piu' ampio. Si sposta la mantissa finche'
            // il bit implicito non emerge.
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400U) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3FFU;
            wide = sign | (exponent << 23) | (mantissa << 13);
        } else {
            wide = sign;  // zero (con segno)
        }
    } else if (exponent == 0x1FU) {
        // Infinito o NaN: l'esponente massimo di FP32 e' 0xFF.
        wide = sign | (0xFFU << 23) | (mantissa << 13);
    } else {
        wide = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }

    float value = 0.0F;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

std::size_t safetensorsDTypeSize(SafetensorsDType dtype) {
    switch (dtype) {
        case SafetensorsDType::F64:
        case SafetensorsDType::I64: return 8;
        case SafetensorsDType::F32:
        case SafetensorsDType::I32: return 4;
        case SafetensorsDType::F16:
        case SafetensorsDType::BF16:
        case SafetensorsDType::I16: return 2;
        case SafetensorsDType::I8:
        case SafetensorsDType::U8:
        case SafetensorsDType::BOOL: return 1;
    }
    return 0;
}

const char* safetensorsDTypeName(SafetensorsDType dtype) {
    switch (dtype) {
        case SafetensorsDType::F64: return "F64";
        case SafetensorsDType::F32: return "F32";
        case SafetensorsDType::F16: return "F16";
        case SafetensorsDType::BF16: return "BF16";
        case SafetensorsDType::I64: return "I64";
        case SafetensorsDType::I32: return "I32";
        case SafetensorsDType::I16: return "I16";
        case SafetensorsDType::I8: return "I8";
        case SafetensorsDType::U8: return "U8";
        case SafetensorsDType::BOOL: return "BOOL";
    }
    return "?";
}

std::size_t SafetensorsEntry::elementCount() const {
    std::size_t count = 1;
    for (std::size_t dim : shape) {
        count *= dim;
    }
    return count;
}

std::size_t SafetensorsEntry::rows() const {
    if (shape.empty()) {
        return 1;
    }
    std::size_t count = 1;
    for (std::size_t i = 0; i + 1 < shape.size(); ++i) {
        count *= shape[i];
    }
    return count;
}

std::size_t SafetensorsEntry::rowLength() const { return shape.empty() ? 1 : shape.back(); }

std::string SafetensorsEntry::shapeToString() const {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        out += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
    }
    return out + "]";
}

// --------------------------------------------------------- file singolo

SafetensorsFile::SafetensorsFile(std::string path) : path_(std::move(path)) {
    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error("safetensors: impossibile aprire '" + path_ + "'");
    }

    in.seekg(0, std::ios::end);
    const auto fileSize = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (fileSize < 8) {
        throw std::runtime_error("safetensors: '" + path_ + "' e' troppo corto per contenere un'intestazione");
    }

    std::uint64_t headerLength = 0;
    in.read(reinterpret_cast<char*>(&headerLength), sizeof(headerLength));
    if (!in || headerLength == 0 || headerLength > fileSize - 8) {
        throw std::runtime_error("safetensors: '" + path_ + "' dichiara un'intestazione di " +
                                  std::to_string(headerLength) + " byte, incompatibile con un file di " +
                                  std::to_string(fileSize) + " byte");
    }

    std::string header(static_cast<std::size_t>(headerLength), '\0');
    in.read(header.data(), static_cast<std::streamsize>(headerLength));
    if (!in) {
        throw std::runtime_error("safetensors: intestazione di '" + path_ + "' troncata");
    }
    dataOffset_ = 8 + static_cast<std::size_t>(headerLength);
    const std::size_t dataSize = fileSize - dataOffset_;

    JsonParser parser(header, path_);
    parser.expect('{');
    if (!parser.consumeIf('}')) {
        while (true) {
            const std::string key = parser.parseString();
            parser.expect(':');

            if (key == "__metadata__") {
                // Mappa stringa->stringa, opzionale.
                parser.expect('{');
                if (!parser.consumeIf('}')) {
                    while (true) {
                        const std::string metaKey = parser.parseString();
                        parser.expect(':');
                        metadata_[metaKey] = parser.parseString();
                        if (parser.consumeIf(',')) {
                            continue;
                        }
                        parser.expect('}');
                        break;
                    }
                }
            } else {
                SafetensorsEntry entry;
                entry.name = key;
                bool sawDType = false;
                bool sawShape = false;
                bool sawOffsets = false;

                parser.expect('{');
                while (true) {
                    const std::string field = parser.parseString();
                    parser.expect(':');
                    if (field == "dtype") {
                        entry.dtype = parseDType(parser.parseString(), parser);
                        sawDType = true;
                    } else if (field == "shape") {
                        entry.shape = parser.parseUnsignedArray();
                        sawShape = true;
                    } else if (field == "data_offsets") {
                        const auto offsets = parser.parseUnsignedArray();
                        if (offsets.size() != 2) {
                            parser.fail("'data_offsets' di '" + key + "' non ha due elementi");
                        }
                        entry.byteBegin = offsets[0];
                        entry.byteEnd = offsets[1];
                        sawOffsets = true;
                    } else {
                        parser.skipValue();
                    }
                    if (parser.consumeIf(',')) {
                        continue;
                    }
                    parser.expect('}');
                    break;
                }

                if (!sawDType || !sawShape || !sawOffsets) {
                    parser.fail("il tensore '" + key + "' non dichiara dtype, shape e data_offsets");
                }

                // Coerenza: gli offset devono corrispondere alla forma e
                // stare dentro il file. Un file troncato o un'intestazione
                // sbagliata devono fallire qui, non dopo mezz'ora di
                // import.
                const std::size_t expected = entry.elementCount() * safetensorsDTypeSize(entry.dtype);
                if (entry.byteEnd < entry.byteBegin || entry.byteEnd - entry.byteBegin != expected) {
                    parser.fail("il tensore '" + key + "' " + entry.shapeToString() + " in " +
                                 safetensorsDTypeName(entry.dtype) + " occuperebbe " + std::to_string(expected) +
                                 " byte, ma l'intestazione ne dichiara " +
                                 std::to_string(entry.byteEnd - entry.byteBegin));
                }
                if (entry.byteEnd > dataSize) {
                    parser.fail("il tensore '" + key + "' finisce a " + std::to_string(entry.byteEnd) +
                                 " ma il file ne contiene solo " + std::to_string(dataSize) +
                                 " (file troncato?)");
                }

                byName_[entry.name] = tensors_.size();
                tensors_.push_back(std::move(entry));
            }

            if (parser.consumeIf(',')) {
                continue;
            }
            parser.expect('}');
            break;
        }
    }
}

const SafetensorsEntry* SafetensorsFile::find(const std::string& name) const {
    auto it = byName_.find(name);
    return it == byName_.end() ? nullptr : &tensors_[it->second];
}

void SafetensorsFile::readFloatRows(const std::string& name, std::size_t firstRow, std::size_t rowCount,
                                     float* out) const {
    const SafetensorsEntry* entry = find(name);
    if (entry == nullptr) {
        throw std::runtime_error("safetensors: '" + path_ + "' non contiene il tensore '" + name + "'");
    }
    if (firstRow + rowCount > entry->rows()) {
        throw std::out_of_range("safetensors: righe [" + std::to_string(firstRow) + ", " +
                                 std::to_string(firstRow + rowCount) + ") fuori dal tensore '" + name + "' " +
                                 entry->shapeToString());
    }
    if (rowCount == 0) {
        return;
    }
    if (out == nullptr) {
        throw std::invalid_argument("safetensors: buffer di uscita nullo");
    }

    const std::size_t elementSize = safetensorsDTypeSize(entry->dtype);
    const std::size_t rowLength = entry->rowLength();
    const std::size_t byteCount = rowCount * rowLength * elementSize;

    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error("safetensors: impossibile riaprire '" + path_ + "'");
    }
    in.seekg(static_cast<std::streamoff>(dataOffset_ + entry->byteBegin + firstRow * rowLength * elementSize));

    // Il buffer grezzo e' grande quanto il blocco richiesto, non quanto
    // il tensore: e' il punto dell'API.
    std::vector<unsigned char> raw(byteCount);
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(byteCount));
    if (!in) {
        throw std::runtime_error("safetensors: lettura di '" + name + "' da '" + path_ + "' fallita");
    }

    convertToFloat(entry->dtype, raw.data(), rowCount * rowLength, out);
}

runtime::Tensor SafetensorsFile::readFloat(const std::string& name) const {
    const SafetensorsEntry* entry = find(name);
    if (entry == nullptr) {
        throw std::runtime_error("safetensors: '" + path_ + "' non contiene il tensore '" + name + "'");
    }
    std::vector<float> data(entry->elementCount());
    readFloatRows(name, 0, entry->rows(), data.data());

    std::vector<std::size_t> shape = entry->shape;
    if (shape.empty()) {
        shape.push_back(1);
    }
    return runtime::Tensor(std::move(shape), std::move(data));
}

// ------------------------------------------------------------- modello

SafetensorsModel::SafetensorsModel(const std::string& directory) {
    namespace fs = std::filesystem;

    const fs::path base(directory);
    const fs::path indexPath = base / "model.safetensors.index.json";
    const fs::path singlePath = base / "model.safetensors";

    if (fs::exists(indexPath)) {
        std::ifstream in(indexPath, std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (text.empty()) {
            throw std::runtime_error("safetensors: '" + indexPath.string() + "' e' vuoto");
        }

        // weight_map: nome del tensore -> nome del file.
        JsonParser parser(text, indexPath.string());
        std::unordered_map<std::string, std::string> weightMap;
        parser.expect('{');
        while (true) {
            const std::string key = parser.parseString();
            parser.expect(':');
            if (key == "weight_map") {
                parser.expect('{');
                if (!parser.consumeIf('}')) {
                    while (true) {
                        const std::string tensor = parser.parseString();
                        parser.expect(':');
                        weightMap[tensor] = parser.parseString();
                        if (parser.consumeIf(',')) {
                            continue;
                        }
                        parser.expect('}');
                        break;
                    }
                }
            } else {
                parser.skipValue();
            }
            if (parser.consumeIf(',')) {
                continue;
            }
            parser.expect('}');
            break;
        }

        if (weightMap.empty()) {
            throw std::runtime_error("safetensors: '" + indexPath.string() + "' non contiene 'weight_map'");
        }

        // Un file per shard, aperto una volta sola.
        std::unordered_map<std::string, std::size_t> fileIndex;
        for (const auto& entry : weightMap) {
            auto it = fileIndex.find(entry.second);
            if (it == fileIndex.end()) {
                const fs::path shard = base / entry.second;
                if (!fs::exists(shard)) {
                    throw std::runtime_error("safetensors: l'indice cita '" + entry.second +
                                              "', che non esiste in '" + directory + "'");
                }
                fileIndex[entry.second] = files_.size();
                files_.emplace_back(shard.string());
                it = fileIndex.find(entry.second);
            }
            tensorToFile_[entry.first] = it->second;
        }
    } else if (fs::exists(singlePath)) {
        files_.emplace_back(singlePath.string());
        for (const SafetensorsEntry& entry : files_.front().tensors()) {
            tensorToFile_[entry.name] = 0;
        }
    } else {
        throw std::runtime_error("safetensors: in '" + directory +
                                  "' non ci sono ne' 'model.safetensors.index.json' ne' 'model.safetensors'");
    }
}

const SafetensorsFile& SafetensorsModel::fileFor(const std::string& name) const {
    auto it = tensorToFile_.find(name);
    if (it == tensorToFile_.end()) {
        throw std::runtime_error("safetensors: il modello non contiene il tensore '" + name + "'");
    }
    return files_[it->second];
}

std::vector<std::string> SafetensorsModel::names() const {
    std::vector<std::string> out;
    out.reserve(tensorToFile_.size());
    for (const auto& entry : tensorToFile_) {
        out.push_back(entry.first);
    }
    return out;
}

bool SafetensorsModel::has(const std::string& name) const { return tensorToFile_.count(name) != 0; }

const SafetensorsEntry& SafetensorsModel::entry(const std::string& name) const {
    const SafetensorsEntry* found = fileFor(name).find(name);
    if (found == nullptr) {
        throw std::runtime_error("safetensors: l'indice cita '" + name +
                                  "' ma il file corrispondente non lo contiene");
    }
    return *found;
}

void SafetensorsModel::readFloatRows(const std::string& name, std::size_t firstRow, std::size_t rowCount,
                                      float* out) const {
    fileFor(name).readFloatRows(name, firstRow, rowCount, out);
}

runtime::Tensor SafetensorsModel::readFloat(const std::string& name) const { return fileFor(name).readFloat(name); }

}  // namespace blackforge::blackbit
