#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "blackforge/blackbit/config.hpp"

// Lettura/scrittura della configurazione BlackBit in JSON.
//
// Il parser copre volutamente solo il sottoinsieme che serve: un
// oggetto PIATTO le cui chiavi hanno valori numerici, booleani o
// stringa. Non e' un parser JSON generale (niente array, niente
// annidamento, niente escape unicode): BlackForge non ha dipendenze
// esterne e un parser completo sarebbe codice non usato da nessuno.
// Ogni deviazione dal formato atteso e' un errore con riga e motivo —
// mai un valore di default applicato in silenzio, che nasconderebbe un
// refuso in un nome di campo dietro un modello di dimensioni sbagliate.

namespace blackforge::blackbit {

namespace {

struct Parser {
    const std::string& text;
    const std::string& origin;
    std::size_t pos = 0;
    std::size_t line = 1;

    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error(origin + ":" + std::to_string(line) + ": configurazione BlackBit: " + message);
    }

    void skipSpace() {
        while (pos < text.size()) {
            const char c = text[pos];
            if (c == '\n') {
                ++line;
                ++pos;
            } else if (std::isspace(static_cast<unsigned char>(c)) != 0) {
                ++pos;
            } else if (c == '/' && pos + 1 < text.size() && text[pos + 1] == '/') {
                // Commenti a fine riga: non sono JSON standard, ma i
                // file di configurazione di questo progetto li usano
                // per annotare le scelte dimensionali.
                while (pos < text.size() && text[pos] != '\n') {
                    ++pos;
                }
            } else {
                return;
            }
        }
    }

    char peek() {
        skipSpace();
        return pos < text.size() ? text[pos] : '\0';
    }

    void expect(char c) {
        if (peek() != c) {
            fail(std::string("atteso '") + c + "'");
        }
        ++pos;
    }

    std::string parseString() {
        expect('"');
        std::string out;
        while (pos < text.size() && text[pos] != '"') {
            if (text[pos] == '\\' && pos + 1 < text.size()) {
                ++pos;
            }
            if (text[pos] == '\n') {
                fail("stringa non terminata");
            }
            out += text[pos++];
        }
        if (pos >= text.size()) {
            fail("stringa non terminata");
        }
        ++pos;  // chiude la stringa
        return out;
    }
};

// Un valore JSON del sottoinsieme supportato.
struct Value {
    enum class Kind { Number, Boolean, String } kind = Kind::Number;
    double number = 0.0;
    bool boolean = false;
    std::string string;
};

std::size_t asSize(const Value& value, const std::string& key, const Parser& parser) {
    if (value.kind != Value::Kind::Number) {
        parser.fail("'" + key + "' richiede un numero");
    }
    if (value.number < 0.0 || value.number != std::floor(value.number)) {
        parser.fail("'" + key + "' richiede un intero non negativo");
    }
    return static_cast<std::size_t>(value.number);
}

float asFloat(const Value& value, const std::string& key, const Parser& parser) {
    if (value.kind != Value::Kind::Number) {
        parser.fail("'" + key + "' richiede un numero");
    }
    return static_cast<float>(value.number);
}

bool asBool(const Value& value, const std::string& key, const Parser& parser) {
    if (value.kind != Value::Kind::Boolean) {
        parser.fail("'" + key + "' richiede true o false");
    }
    return value.boolean;
}

sema::DType asWeightDType(const Value& value, const std::string& key, const Parser& parser) {
    if (value.kind != Value::Kind::String) {
        parser.fail("'" + key + "' richiede il nome di un formato tra virgolette");
    }
    if (value.string == "ternary1p58") {
        return sema::DType::TERNARY_1P58;
    }
    if (value.string == "bf16") {
        return sema::DType::BF16;
    }
    if (value.string == "fp16") {
        return sema::DType::FP16;
    }
    if (value.string == "fp32") {
        return sema::DType::FP32;
    }
    parser.fail("'" + key + "': formato '" + value.string + "' non riconosciuto");
}

}  // namespace

BlackBitConfig parseConfigJson(const std::string& text, const std::string& origin) {
    Parser parser{text, origin};
    BlackBitConfig config;

    parser.expect('{');
    if (parser.peek() == '}') {
        ++parser.pos;
        config.validate();
        return config;
    }

    while (true) {
        const std::string key = parser.parseString();
        parser.expect(':');

        Value value;
        const char c = parser.peek();
        if (c == '"') {
            value.kind = Value::Kind::String;
            value.string = parser.parseString();
        } else if (c == 't' || c == 'f') {
            const bool isTrue = text.compare(parser.pos, 4, "true") == 0;
            const bool isFalse = text.compare(parser.pos, 5, "false") == 0;
            if (!isTrue && !isFalse) {
                parser.fail("'" + key + "': atteso true o false");
            }
            value.kind = Value::Kind::Boolean;
            value.boolean = isTrue;
            parser.pos += isTrue ? 4 : 5;
        } else {
            std::size_t consumed = 0;
            try {
                value.number = std::stod(text.substr(parser.pos), &consumed);
            } catch (const std::exception&) {
                parser.fail("'" + key + "': valore numerico non valido");
            }
            if (consumed == 0) {
                parser.fail("'" + key + "': valore numerico non valido");
            }
            parser.pos += consumed;
        }

        if (key == "name") {
            if (value.kind != Value::Kind::String) {
                parser.fail("'name' richiede una stringa");
            }
            config.name = value.string;
        } else if (key == "vocab_size") {
            config.vocabSize = asSize(value, key, parser);
        } else if (key == "hidden_size") {
            config.hiddenSize = asSize(value, key, parser);
        } else if (key == "num_layers") {
            config.numLayers = asSize(value, key, parser);
        } else if (key == "num_heads") {
            config.numHeads = asSize(value, key, parser);
        } else if (key == "num_kv_heads") {
            config.numKvHeads = asSize(value, key, parser);
        } else if (key == "head_dim") {
            config.headDim = asSize(value, key, parser);
        } else if (key == "num_experts") {
            config.numExperts = asSize(value, key, parser);
        } else if (key == "experts_per_tok") {
            config.expertsPerToken = asSize(value, key, parser);
        } else if (key == "expert_hidden") {
            config.expertHidden = asSize(value, key, parser);
        } else if (key == "max_seq_len") {
            config.maxSeqLen = asSize(value, key, parser);
        } else if (key == "ternary_group_size") {
            config.ternaryGroupSize = asSize(value, key, parser);
        } else if (key == "weight_dtype") {
            config.weightDtype = asWeightDType(value, key, parser);
        } else if (key == "norm_dtype") {
            config.normDtype = asWeightDType(value, key, parser);
        } else if (key == "router_dtype") {
            config.routerDtype = asWeightDType(value, key, parser);
        } else if (key == "tie_embeddings") {
            config.tieEmbeddings = asBool(value, key, parser);
        } else if (key == "router_aux_loss_weight") {
            config.routerAuxLossWeight = asFloat(value, key, parser);
        } else if (key == "expert_capacity_factor") {
            config.expertCapacityFactor = asFloat(value, key, parser);
        } else {
            parser.fail("chiave sconosciuta '" + key + "'");
        }

        const char next = parser.peek();
        if (next == ',') {
            ++parser.pos;
            continue;
        }
        if (next == '}') {
            ++parser.pos;
            break;
        }
        parser.fail("attesa ',' o '}' dopo il valore di '" + key + "'");
    }

    config.validate();
    return config;
}

BlackBitConfig loadConfigFromJson(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("impossibile aprire la configurazione BlackBit '" + path + "'");
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return parseConfigJson(buffer.str(), path);
}

std::string toJson(const BlackBitConfig& config) {
    std::ostringstream out;
    out << "{\n";
    out << "    \"name\": \"" << config.name << "\",\n";
    out << "    \"vocab_size\": " << config.vocabSize << ",\n";
    out << "    \"hidden_size\": " << config.hiddenSize << ",\n";
    out << "    \"num_layers\": " << config.numLayers << ",\n";
    out << "    \"num_heads\": " << config.numHeads << ",\n";
    out << "    \"num_kv_heads\": " << config.numKvHeads << ",\n";
    out << "    \"head_dim\": " << config.headDim << ",\n";
    out << "    \"num_experts\": " << config.numExperts << ",\n";
    out << "    \"experts_per_tok\": " << config.expertsPerToken << ",\n";
    out << "    \"expert_hidden\": " << config.expertHidden << ",\n";
    out << "    \"max_seq_len\": " << config.maxSeqLen << ",\n";
    out << "    \"ternary_group_size\": " << config.ternaryGroupSize << ",\n";
    out << "    \"weight_dtype\": \"" << sema::dtypeName(config.weightDtype) << "\",\n";
    out << "    \"norm_dtype\": \"" << sema::dtypeName(config.normDtype) << "\",\n";
    out << "    \"router_dtype\": \"" << sema::dtypeName(config.routerDtype) << "\",\n";
    out << "    \"tie_embeddings\": " << (config.tieEmbeddings ? "true" : "false") << ",\n";
    out << "    \"router_aux_loss_weight\": " << config.routerAuxLossWeight << ",\n";
    out << "    \"expert_capacity_factor\": " << config.expertCapacityFactor << "\n";
    out << "}\n";
    return out.str();
}

}  // namespace blackforge::blackbit
