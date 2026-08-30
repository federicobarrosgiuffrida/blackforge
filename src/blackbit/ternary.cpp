#include "blackforge/blackbit/ternary.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace blackforge::blackbit {

namespace {

// Lettura/scrittura binaria degli scalari, come nel formato di
// checkpoint esistente: little-endian nativo (x86/ARM comuni).
template <typename T>
void writeScalar(std::ostream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T readScalar(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("TernaryTensor::deserialize: stream terminato prima del previsto");
    }
    return value;
}

}  // namespace

void TernaryTensor::allocate(std::vector<std::size_t> shape, std::size_t groupSize) {
    if (shape.empty()) {
        throw std::invalid_argument("TernaryTensor: la forma non puo' essere vuota");
    }
    if (groupSize == 0 || groupSize % kTritsPerWord != 0) {
        throw std::invalid_argument("TernaryTensor: groupSize deve essere un multiplo positivo di " +
                                     std::to_string(kTritsPerWord) +
                                     " (i gruppi di scala devono cadere su un confine di parola)");
    }

    shape_ = std::move(shape);
    groupSize_ = groupSize;
    rowLength_ = shape_.back();
    rows_ = 1;
    for (std::size_t i = 0; i + 1 < shape_.size(); ++i) {
        rows_ *= shape_[i];
    }
    if (rowLength_ == 0 || rows_ == 0) {
        throw std::invalid_argument("TernaryTensor: nessuna dimensione della forma puo' essere zero");
    }

    // Attenzione: il byte "tutto zero" NON e' 0x00 — il codice
    // posizionale in base 3 rappresenta il trit 0 con la cifra 1,
    // quindi cinque zeri sono 1+3+9+27+81 = 121. Riempire il buffer
    // con questo valore (invece di memset a 0, che significherebbe
    // "cinque trit -1") rende il tensore appena costruito davvero
    // nullo, padding di fine riga compreso: due costruzioni identiche
    // producono lo stesso buffer byte per byte, cosa da cui dipendono
    // la serializzazione e i test di round-trip.
    const std::uint8_t zeroByte = encodeTritByte(0, 0, 0, 0, 0);
    std::uint32_t zeroWord = 0;
    for (int i = 0; i < 4; ++i) {
        zeroWord = setWordByte(zeroWord, i, zeroByte);
    }

    packedWords_.assign(rows_ * wordsPerRow(), zeroWord);
    scales_.assign(rows_ * groupsPerRow(), 1.0F);
}

TernaryTensor::TernaryTensor(std::vector<std::size_t> shape, std::size_t groupSize) {
    allocate(std::move(shape), groupSize);
}

int TernaryTensor::tritAt(std::size_t flatIndex) const {
    if (flatIndex >= elementCount()) {
        throw std::out_of_range("TernaryTensor::tritAt: indice fuori dal tensore");
    }
    const std::size_t row = flatIndex / rowLength_;
    const std::size_t col = flatIndex % rowLength_;
    const std::uint32_t word = packedWords_[row * wordsPerRow() + col / kTritsPerWord];
    const std::size_t inWord = col % kTritsPerWord;
    return decodeTritAt(wordByte(word, static_cast<int>(inWord / kTritsPerByte)),
                        static_cast<int>(inWord % kTritsPerByte));
}

void TernaryTensor::setTritAt(std::size_t flatIndex, int trit) {
    if (flatIndex >= elementCount()) {
        throw std::out_of_range("TernaryTensor::setTritAt: indice fuori dal tensore");
    }
    if (trit < -1 || trit > 1) {
        throw std::invalid_argument("TernaryTensor::setTritAt: un trit vale -1, 0 o +1");
    }
    const std::size_t row = flatIndex / rowLength_;
    const std::size_t col = flatIndex % rowLength_;
    std::uint32_t& word = packedWords_[row * wordsPerRow() + col / kTritsPerWord];
    const std::size_t inWord = col % kTritsPerWord;
    const int byteIndex = static_cast<int>(inWord / kTritsPerByte);
    const int slot = static_cast<int>(inWord % kTritsPerByte);

    int trits[kTritsPerByte];
    decodeTritByte(wordByte(word, byteIndex), trits);
    trits[slot] = trit;
    word = setWordByte(word, byteIndex, encodeTritByte(trits[0], trits[1], trits[2], trits[3], trits[4]));
}

float TernaryTensor::scaleAt(std::size_t flatIndex) const {
    if (flatIndex >= elementCount()) {
        throw std::out_of_range("TernaryTensor::scaleAt: indice fuori dal tensore");
    }
    const std::size_t row = flatIndex / rowLength_;
    const std::size_t col = flatIndex % rowLength_;
    return scales_[row * groupsPerRow() + col / groupSize_];
}

double TernaryTensor::bitsPerWeight() const {
    if (elementCount() == 0) {
        return 0.0;
    }
    return static_cast<double>(totalByteCount()) * 8.0 / static_cast<double>(elementCount());
}

void TernaryTensor::quantizeRowsFrom(std::size_t firstRow, std::size_t rowCount, const float* dense) {
    if (firstRow + rowCount > rows_) {
        throw std::out_of_range("TernaryTensor::quantizeRowsFrom: intervallo di righe fuori dal tensore");
    }
    if (dense == nullptr && rowCount != 0) {
        throw std::invalid_argument("TernaryTensor::quantizeRowsFrom: buffer di ingresso nullo");
    }

    const std::size_t groups = groupsPerRow();
    const std::size_t words = wordsPerRow();
    for (std::size_t r = 0; r < rowCount; ++r) {
        const float* row = dense + r * rowLength_;
        const std::size_t target = firstRow + r;

        for (std::size_t group = 0; group < groups; ++group) {
            const std::size_t first = group * groupSize_;
            const std::size_t last = std::min(first + groupSize_, rowLength_);

            // absmean (BitNet b1.58): la scala e' la media dei valori
            // assoluti del gruppo, calcolata sui soli elementi NON
            // NULLI.
            //
            // Escludere gli zeri e' una deviazione deliberata dalla
            // formulazione originale, e serve a garantire
            // l'IDEMPOTENZA: se un gruppo e' gia' sulla griglia
            // ternaria (valori in {-s, 0, +s}), la media sui non nulli
            // vale esattamente s, quindi quantizzare di nuovo
            // restituisce gli stessi trit e la stessa scala. Con la
            // media su TUTTI gli elementi la scala varrebbe
            // s * (frazione di non nulli) e i pesi si contrarrebbero
            // verso lo zero ad ogni riquantizzazione — un modello
            // salvato e ricaricato piu' volte perderebbe ampiezza senza
            // che nulla lo segnali. Su un gruppo denso (nessuno zero
            // esatto, il caso di qualunque inizializzazione continua)
            // le due formule coincidono.
            //
            // Un gruppo interamente nullo prende scala 1 invece di 0,
            // cosi' la dequantizzazione resta definita e un eventuale
            // aggiornamento successivo ha una griglia su cui muoversi.
            double sum = 0.0;
            std::size_t nonZero = 0;
            for (std::size_t i = first; i < last; ++i) {
                const double magnitude = std::fabs(static_cast<double>(row[i]));
                if (magnitude > 0.0) {
                    sum += magnitude;
                    ++nonZero;
                }
            }
            const double mean = nonZero > 0 ? sum / static_cast<double>(nonZero) : 0.0;
            const float scale = mean > 0.0 ? static_cast<float>(mean) : 1.0F;
            scales_[target * groups + group] = scale;

            // Impacchettamento diretto, cinque trit per volta: passare
            // da setTritAt() elemento per elemento costringerebbe a
            // decodificare e ricodificare lo stesso byte cinque volte
            // (lettura-modifica-scrittura su ogni trit). Su una matrice
            // di embedding da 201 M elementi la differenza fra le due
            // versioni e' un ordine di grandezza sul tempo di
            // inizializzazione.
            const float inverseScale = 1.0F / scale;
            std::size_t i = first;
            while (i < last) {
                const std::size_t wordIndex = target * words + i / kTritsPerWord;
                const int byteIndex = static_cast<int>((i % kTritsPerWord) / kTritsPerByte);
                const std::size_t slotBase = i - (i % kTritsPerByte);

                int trits[kTritsPerByte];
                // Il byte puo' essere a cavallo del confine del gruppo o
                // della riga: le posizioni non coperte da questa
                // iterazione vanno lette, non azzerate.
                decodeTritByte(wordByte(packedWords_[wordIndex], byteIndex), trits);

                const std::size_t slotEnd = std::min(slotBase + kTritsPerByte, last);
                for (std::size_t s = std::max(slotBase, first); s < slotEnd; ++s) {
                    int trit = static_cast<int>(std::lround(row[s] * inverseScale));
                    trit = trit < -1 ? -1 : (trit > 1 ? 1 : trit);
                    trits[s - slotBase] = trit;
                }

                packedWords_[wordIndex] = setWordByte(
                    packedWords_[wordIndex], byteIndex,
                    encodeTritByte(trits[0], trits[1], trits[2], trits[3], trits[4]));
                i = slotEnd;
            }
        }
    }
}

TernaryTensor TernaryTensor::quantizeFrom(const runtime::Tensor& dense, std::size_t groupSize) {
    TernaryTensor result(dense.shape(), groupSize);
    result.quantizeRowsFrom(0, result.rows_, dense.data().data());
    return result;
}

TernaryTensor TernaryTensor::fromTrits(std::vector<std::size_t> shape, const std::vector<std::int8_t>& trits,
                                        const std::vector<float>& scales, std::size_t groupSize) {
    TernaryTensor result(std::move(shape), groupSize);
    if (trits.size() != result.elementCount()) {
        throw std::invalid_argument("TernaryTensor::fromTrits: numero di trit incoerente con la forma");
    }
    if (scales.size() != result.scales_.size()) {
        throw std::invalid_argument("TernaryTensor::fromTrits: numero di scale incoerente con la forma");
    }
    for (std::size_t i = 0; i < trits.size(); ++i) {
        result.setTritAt(i, trits[i]);
    }
    result.scales_ = scales;
    return result;
}

void TernaryTensor::dequantizeRows(std::size_t firstRow, std::size_t rowCount, float* out) const {
    if (firstRow + rowCount > rows_) {
        throw std::out_of_range("TernaryTensor::dequantizeRows: intervallo di righe fuori dal tensore");
    }
    if (out == nullptr && rowCount != 0) {
        throw std::invalid_argument("TernaryTensor::dequantizeRows: buffer di uscita nullo");
    }

    const std::size_t words = wordsPerRow();
    const std::size_t groups = groupsPerRow();

    for (std::size_t r = 0; r < rowCount; ++r) {
        const std::uint32_t* rowWords = packedWords_.data() + (firstRow + r) * words;
        const float* rowScales = scales_.data() + (firstRow + r) * groups;
        float* rowOut = out + r * rowLength_;

        // Scorre le parole in sequenza (accesso lineare, l'ordine che i
        // kernel CUDA useranno) e decodifica 20 pesi per parola con
        // quattro decodeTritByte(): mai una divisione per un divisore
        // variabile, mai una lettura fuori dalla riga corrente.
        std::size_t col = 0;
        for (std::size_t w = 0; w < words && col < rowLength_; ++w) {
            const std::uint32_t word = rowWords[w];
            for (int b = 0; b < 4 && col < rowLength_; ++b) {
                int trits[kTritsPerByte];
                decodeTritByte(wordByte(word, b), trits);
                for (std::size_t s = 0; s < kTritsPerByte && col < rowLength_; ++s, ++col) {
                    rowOut[col] = static_cast<float>(trits[s]) * rowScales[col / groupSize_];
                }
            }
        }
    }
}

runtime::Tensor TernaryTensor::dequantize() const {
    std::vector<float> data(elementCount());
    dequantizeRows(0, rows_, data.data());
    return runtime::Tensor(shape_, std::move(data));
}

void TernaryTensor::serialize(std::ostream& out) const {
    writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(shape_.size()));
    for (std::size_t dim : shape_) {
        writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(dim));
    }
    writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(groupSize_));

    writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(packedWords_.size()));
    out.write(reinterpret_cast<const char*>(packedWords_.data()),
              static_cast<std::streamsize>(packedWords_.size() * sizeof(std::uint32_t)));

    writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(scales_.size()));
    out.write(reinterpret_cast<const char*>(scales_.data()),
              static_cast<std::streamsize>(scales_.size() * sizeof(float)));
}

TernaryTensor TernaryTensor::deserialize(std::istream& in) {
    const auto rank = readScalar<std::uint32_t>(in);
    if (rank == 0) {
        throw std::runtime_error("TernaryTensor::deserialize: rango zero non valido");
    }
    std::vector<std::size_t> shape(rank);
    for (std::uint32_t i = 0; i < rank; ++i) {
        shape[i] = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
    }
    const auto groupSize = static_cast<std::size_t>(readScalar<std::uint64_t>(in));

    TernaryTensor result;
    result.allocate(std::move(shape), groupSize);

    const auto wordCount = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
    if (wordCount != result.packedWords_.size()) {
        throw std::runtime_error("TernaryTensor::deserialize: numero di parole incoerente con la forma dichiarata");
    }
    in.read(reinterpret_cast<char*>(result.packedWords_.data()),
            static_cast<std::streamsize>(wordCount * sizeof(std::uint32_t)));

    const auto scaleCount = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
    if (scaleCount != result.scales_.size()) {
        throw std::runtime_error("TernaryTensor::deserialize: numero di scale incoerente con la forma dichiarata");
    }
    in.read(reinterpret_cast<char*>(result.scales_.data()),
            static_cast<std::streamsize>(scaleCount * sizeof(float)));

    if (!in) {
        throw std::runtime_error("TernaryTensor::deserialize: stream terminato prima del previsto");
    }
    return result;
}

}  // namespace blackforge::blackbit
