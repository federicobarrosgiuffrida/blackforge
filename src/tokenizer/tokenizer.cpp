#include "blackforge/tokenizer/tokenizer.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace blackforge::tokenizer {

namespace {

std::uint64_t pairKey(std::uint32_t left, std::uint32_t right) {
    return (static_cast<std::uint64_t>(left) << 32) | static_cast<std::uint64_t>(right);
}

// Classificazione di byte per la pre-tokenizzazione (vedi il commento
// su pretokenize()): lettera ASCII/cifra, oppure un qualunque byte >=
// 0x80 (le sequenze UTF-8 multi-byte per caratteri accentati/CJK/ecc.
// restano cosi' raggruppate in un'unica "parola", invece di essere
// spezzate byte per byte), sono trattati come "word"; whitespace ASCII
// comune come "space"; tutto il resto ("other", tipicamente
// punteggiatura ASCII) diventa un chunk di un solo byte.
enum class ByteClass { Word, Space, Other };

ByteClass classify(std::uint8_t b) {
    if ((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9') || b >= 0x80) {
        return ByteClass::Word;
    }
    if (b == ' ' || b == '\t' || b == '\n' || b == '\r') {
        return ByteClass::Space;
    }
    return ByteClass::Other;
}

// Partiziona 'text' in chunk contigui SENZA perdere alcun byte
// (round-trip garantito dalla semplice concatenazione): run massimali
// di byte "word", con un eventuale run di spazi immediatamente
// precedente incluso come prefisso del chunk (convenzione stile GPT-2:
// " world" e "world" restano chunk distinti, cosi' il tokenizer impara
// che lo spazio iniziale fa parte del token successivo); un run di
// spazi non seguito da una parola (fine stringa) e' il proprio chunk;
// ogni byte "other" e' un chunk a se'. I merge BPE non attraversano mai
// il confine tra due chunk: evita token che uniscono la fine di una
// parola con l'inizio della successiva.
std::vector<std::string> pretokenize(const std::string& text) {
    std::vector<std::string> chunks;
    std::size_t i = 0;
    std::size_t n = text.size();
    while (i < n) {
        std::size_t start = i;
        ByteClass firstClass = classify(static_cast<std::uint8_t>(text[i]));

        if (firstClass == ByteClass::Other) {
            chunks.push_back(text.substr(i, 1));
            ++i;
            continue;
        }

        if (firstClass == ByteClass::Space) {
            std::size_t spaceEnd = i;
            while (spaceEnd < n && classify(static_cast<std::uint8_t>(text[spaceEnd])) == ByteClass::Space) {
                ++spaceEnd;
            }
            if (spaceEnd < n && classify(static_cast<std::uint8_t>(text[spaceEnd])) == ByteClass::Word) {
                std::size_t wordEnd = spaceEnd;
                while (wordEnd < n && classify(static_cast<std::uint8_t>(text[wordEnd])) == ByteClass::Word) {
                    ++wordEnd;
                }
                chunks.push_back(text.substr(start, wordEnd - start));
                i = wordEnd;
            } else {
                chunks.push_back(text.substr(start, spaceEnd - start));
                i = spaceEnd;
            }
            continue;
        }

        // firstClass == Word, nessuno spazio davanti.
        std::size_t wordEnd = i;
        while (wordEnd < n && classify(static_cast<std::uint8_t>(text[wordEnd])) == ByteClass::Word) {
            ++wordEnd;
        }
        chunks.push_back(text.substr(start, wordEnd - start));
        i = wordEnd;
    }
    return chunks;
}

}  // namespace

Tokenizer::Tokenizer() { rebuildFromMerges(); }

Tokenizer::Tokenizer(std::vector<Merge> merges) : merges_(std::move(merges)) { rebuildFromMerges(); }

void Tokenizer::rebuildFromMerges() {
    idToBytes_.clear();
    pairToId_.clear();
    mergeRank_.clear();

    idToBytes_.resize(kFirstMergeId + merges_.size());
    for (std::uint32_t b = 0; b < kBaseVocabSize; ++b) {
        idToBytes_[b] = {static_cast<std::uint8_t>(b)};
    }
    // kPadId/kBosId/kEosId restano con una voce vuota (nessun byte
    // rappresentato): gia' garantito da std::vector::resize().

    for (std::size_t rank = 0; rank < merges_.size(); ++rank) {
        const Merge& m = merges_[rank];
        if (m.resultId != kFirstMergeId + rank) {
            throw std::invalid_argument("Tokenizer: lista di merge non contigua o fuori ordine "
                                         "(atteso resultId=" +
                                         std::to_string(kFirstMergeId + rank) + ", trovato " +
                                         std::to_string(m.resultId) + ")");
        }
        if (m.left >= idToBytes_.size() || m.right >= idToBytes_.size()) {
            throw std::invalid_argument("Tokenizer: merge riferisce un id non ancora definito");
        }

        std::vector<std::uint8_t> bytes = idToBytes_[m.left];
        bytes.insert(bytes.end(), idToBytes_[m.right].begin(), idToBytes_[m.right].end());
        idToBytes_[m.resultId] = std::move(bytes);

        pairToId_[pairKey(m.left, m.right)] = m.resultId;
        mergeRank_[m.resultId] = rank;
    }
}

void Tokenizer::train(const std::string& corpus, std::size_t targetVocabSize) {
    if (targetVocabSize <= kFirstMergeId) {
        throw std::invalid_argument("Tokenizer::train: targetVocabSize deve essere maggiore di " +
                                     std::to_string(kFirstMergeId) +
                                     " (256 byte di base + 3 token speciali)");
    }
    std::size_t maxMerges = targetVocabSize - kFirstMergeId;

    // Frequenza di ogni chunk unico prodotto dalla pre-tokenizzazione
    // (contare i chunk invece di riscannerizzare l'intero corpus ad
    // ogni iterazione e' quello che rende il training trattabile).
    std::unordered_map<std::string, std::uint64_t> chunkFreq;
    for (const std::string& chunk : pretokenize(corpus)) {
        ++chunkFreq[chunk];
    }

    // Rappresentazione corrente di ogni chunk unico come sequenza di id
    // (inizialmente un id per byte), aggiornata ad ogni merge appreso.
    std::vector<std::vector<std::uint32_t>> chunkIds;
    std::vector<std::uint64_t> chunkCounts;
    chunkIds.reserve(chunkFreq.size());
    chunkCounts.reserve(chunkFreq.size());
    for (const auto& [chunk, count] : chunkFreq) {
        std::vector<std::uint32_t> ids;
        ids.reserve(chunk.size());
        for (unsigned char c : chunk) {
            ids.push_back(static_cast<std::uint32_t>(c));
        }
        chunkIds.push_back(std::move(ids));
        chunkCounts.push_back(count);
    }

    merges_.clear();
    std::uint32_t nextId = kFirstMergeId;

    for (std::size_t iteration = 0; iteration < maxMerges; ++iteration) {
        std::unordered_map<std::uint64_t, std::uint64_t> pairCounts;
        for (std::size_t c = 0; c < chunkIds.size(); ++c) {
            const std::vector<std::uint32_t>& ids = chunkIds[c];
            for (std::size_t p = 0; p + 1 < ids.size(); ++p) {
                pairCounts[pairKey(ids[p], ids[p + 1])] += chunkCounts[c];
            }
        }
        if (pairCounts.empty()) {
            break;  // Nessuna coppia adiacente rimasta: il corpus e' esaurito.
        }

        // Coppia piu' frequente; a parita' di frequenza, la chiave
        // numericamente piu' piccola vince (spareggio deterministico,
        // riproducibile a parita' di corpus).
        std::uint64_t bestKey = 0;
        std::uint64_t bestCount = 0;
        for (const auto& [key, count] : pairCounts) {
            if (count > bestCount || (count == bestCount && key < bestKey)) {
                bestKey = key;
                bestCount = count;
            }
        }
        auto bestLeft = static_cast<std::uint32_t>(bestKey >> 32);
        auto bestRight = static_cast<std::uint32_t>(bestKey & 0xFFFFFFFFU);

        std::uint32_t resultId = nextId++;
        merges_.push_back(Merge{bestLeft, bestRight, resultId});

        for (std::vector<std::uint32_t>& ids : chunkIds) {
            std::vector<std::uint32_t> merged;
            merged.reserve(ids.size());
            for (std::size_t p = 0; p < ids.size();) {
                if (p + 1 < ids.size() && ids[p] == bestLeft && ids[p + 1] == bestRight) {
                    merged.push_back(resultId);
                    p += 2;
                } else {
                    merged.push_back(ids[p]);
                    ++p;
                }
            }
            ids = std::move(merged);
        }
    }

    rebuildFromMerges();
}

std::vector<std::uint32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<std::uint32_t> result;

    for (const std::string& chunk : pretokenize(text)) {
        std::vector<std::uint32_t> ids;
        ids.reserve(chunk.size());
        for (unsigned char c : chunk) {
            ids.push_back(static_cast<std::uint32_t>(c));
        }

        // Applica ripetutamente, tra le coppie adiacenti presenti in
        // 'ids', quella con il rank di apprendimento piu' basso (il
        // greedy standard di BPE), finche' ne resta almeno una
        // fondibile.
        while (ids.size() > 1) {
            std::size_t bestPos = ids.size();
            std::size_t bestRank = merges_.size();
            std::uint32_t bestResultId = 0;

            for (std::size_t p = 0; p + 1 < ids.size(); ++p) {
                auto it = pairToId_.find(pairKey(ids[p], ids[p + 1]));
                if (it == pairToId_.end()) {
                    continue;
                }
                std::size_t rank = mergeRank_.at(it->second);
                if (rank < bestRank) {
                    bestRank = rank;
                    bestPos = p;
                    bestResultId = it->second;
                }
            }

            if (bestPos == ids.size()) {
                break;  // Nessuna coppia adiacente e' un merge noto.
            }

            std::vector<std::uint32_t> merged;
            merged.reserve(ids.size() - 1);
            merged.insert(merged.end(), ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(bestPos));
            merged.push_back(bestResultId);
            merged.insert(merged.end(), ids.begin() + static_cast<std::ptrdiff_t>(bestPos) + 2, ids.end());
            ids = std::move(merged);
        }

        result.insert(result.end(), ids.begin(), ids.end());
    }

    return result;
}

std::string Tokenizer::decode(const std::vector<std::uint32_t>& ids) const {
    std::string result;
    for (std::uint32_t id : ids) {
        if (id >= idToBytes_.size()) {
            continue;  // Id sconosciuto: saltato silenziosamente.
        }
        const std::vector<std::uint8_t>& bytes = idToBytes_[id];
        result.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    return result;
}

namespace {

constexpr char kMagic[8] = {'B', 'F', 'T', 'O', 'K', 'N', '1', '\0'};

void writeU32(std::ostream& out, std::uint32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::uint32_t readU32(std::istream& in) {
    std::uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

}  // namespace

void saveTokenizer(const Tokenizer& tok, const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("saveTokenizer: impossibile scrivere il file '" + path + "'");
    }

    out.write(kMagic, sizeof(kMagic));
    const std::vector<Tokenizer::Merge>& merges = tok.merges();
    writeU32(out, static_cast<std::uint32_t>(merges.size()));
    for (const Tokenizer::Merge& m : merges) {
        writeU32(out, m.left);
        writeU32(out, m.right);
        writeU32(out, m.resultId);
    }

    if (!out) {
        throw std::runtime_error("saveTokenizer: scrittura fallita su '" + path + "'");
    }
}

Tokenizer loadTokenizer(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("loadTokenizer: impossibile aprire il file '" + path + "'");
    }

    char magic[8];
    in.read(magic, sizeof(magic));
    if (!in || std::string(magic, sizeof(magic)) != std::string(kMagic, sizeof(kMagic))) {
        throw std::runtime_error("loadTokenizer: '" + path + "' non e' un tokenizer BlackForge valido");
    }

    std::uint32_t count = readU32(in);
    std::vector<Tokenizer::Merge> merges(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        merges[i].left = readU32(in);
        merges[i].right = readU32(in);
        merges[i].resultId = readU32(in);
    }

    if (!in) {
        throw std::runtime_error("loadTokenizer: file '" + path + "' troncato o corrotto");
    }

    return Tokenizer(std::move(merges));
}

}  // namespace blackforge::tokenizer
