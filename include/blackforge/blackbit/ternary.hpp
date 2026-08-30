#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

#include "blackforge/blackbit/device_shared.hpp"
#include "blackforge/runtime/tensor.hpp"

// Rappresentazione ternaria impacchettata {-1, 0, +1} usata da BlackBit
// per TUTTE le matrici grandi (proiezioni di attention, esperti MoE,
// embedding condivisa). Vedi docs/blackbit.md §3 per il ragionamento
// completo dietro il formato; qui il contratto esatto.
//
// FORMATO
//
// Un "trit" ha tre stati. 3^5 = 243 <= 256, quindi 5 pesi ternari
// entrano ESATTAMENTE in un byte con il codice posizionale in base 3
//
//     code = c0 + 3*c1 + 9*c2 + 27*c3 + 81*c4,   c_i in {0,1,2}
//
// dove c_i = trit_i + 1 (cioe' -1 -> 0, 0 -> 1, +1 -> 2). Sono
// 8 bit / 5 pesi = 1,6 bit per peso, contro l'ottimo teorico
// log2(3) = 1,585: efficienza 99,1 %. Nessun trit attraversa mai il
// confine di un byte, quindi decodificare un byte non richiede mai di
// leggerne un altro.
//
// I byte sono raggruppati in parole da 32 bit (4 byte = 20 pesi), che
// e' l'unita' di accesso: un warp che legge una parola per thread
// preleva 128 byte contigui (una transazione di memoria) e ne ricava
// 640 pesi. Ogni RIGA logica (l'ultima dimensione della forma) e'
// impacchettata indipendentemente e paddata a un numero intero di
// parole: l'inizio di ogni riga e' quindi allineato a 4 byte e un tile
// di righe e' un intervallo contiguo del buffer.
//
// La decodifica non usa tabelle ne' divisioni variabili: cinque
// '% 3' / '/= 3' in sequenza su una costante, che ogni compilatore
// (host e nvcc) trasforma in moltiplicazione + shift.
//
// SCALE
//
// Il valore reale di un peso e' 'trit * scala del suo gruppo'. Un
// gruppo e' un blocco di 'groupSize' pesi CONTIGUI lungo l'ultima
// dimensione (default 160 = 8 parole = 32 byte, cosi' un gruppo cade
// sempre su un confine di parola). La quantizzazione usa il criterio
// absmean di BitNet b1.58: scala = media(|w|) sul gruppo, trit =
// round(w / scala) saturato a [-1, +1].

namespace blackforge::blackbit {

// Pesi ternari per byte impacchettato.
inline constexpr std::size_t kTritsPerByte = 5;

// Pesi ternari per parola da 32 bit (l'unita' di accesso in memoria).
inline constexpr std::size_t kTritsPerWord = kTritsPerByte * 4;

// Numero di pesi che condividono una scala, se non specificato
// diversamente. Multiplo di kTritsPerWord per costruzione.
inline constexpr std::size_t kDefaultGroupSize = 160;

// Codifica cinque trit (ciascuno in {-1, 0, +1}) in un byte.
BLACKFORGE_HOST_DEVICE inline std::uint8_t encodeTritByte(int t0, int t1, int t2, int t3, int t4) {
    return static_cast<std::uint8_t>((t0 + 1) + 3 * (t1 + 1) + 9 * (t2 + 1) + 27 * (t3 + 1) + 81 * (t4 + 1));
}

// Decodifica il byte nei suoi cinque trit, in ordine (out[0] e' il piu'
// "basso" nel codice posizionale).
BLACKFORGE_HOST_DEVICE inline void decodeTritByte(std::uint8_t code, int out[kTritsPerByte]) {
    int value = static_cast<int>(code);
    for (int i = 0; i < static_cast<int>(kTritsPerByte); ++i) {
        out[i] = (value % 3) - 1;
        value /= 3;
    }
}

// Estrae un singolo trit da un byte impacchettato ('slot' in [0, 5)).
// Piu' lenta di decodeTritByte() a parita' di pesi estratti: pensata
// per accessi sparsi (test, ispezione), non per la dequantizzazione a
// blocchi.
BLACKFORGE_HOST_DEVICE inline int decodeTritAt(std::uint8_t code, int slot) {
    int value = static_cast<int>(code);
    for (int i = 0; i < slot; ++i) {
        value /= 3;
    }
    return (value % 3) - 1;
}

// Byte all'interno di una parola da 32 bit: il byte j occupa i bit
// [8j, 8j+8) ed e' quello che contiene i trit [20w + 5j, ... + 5).
// Definito con shift espliciti, non con un cast di puntatore: la
// codifica non dipende dall'endianness del processore.
BLACKFORGE_HOST_DEVICE inline std::uint8_t wordByte(std::uint32_t word, int index) {
    return static_cast<std::uint8_t>((word >> (8 * index)) & 0xFFU);
}

BLACKFORGE_HOST_DEVICE inline std::uint32_t setWordByte(std::uint32_t word, int index, std::uint8_t value) {
    const std::uint32_t mask = 0xFFU << (8 * index);
    return (word & ~mask) | (static_cast<std::uint32_t>(value) << (8 * index));
}

// Parole da 32 bit necessarie per impacchettare 'elements' pesi
// ternari contigui (arrotondamento per eccesso: i pesi di padding
// valgono 0).
constexpr std::size_t ternaryWordsPerRow(std::size_t elements) {
    return (elements + kTritsPerWord - 1) / kTritsPerWord;
}

// Byte occupati dalla sola parte impacchettata (senza scale) di una
// matrice [rows, rowLength].
constexpr std::size_t ternaryPackedBytes(std::size_t rows, std::size_t rowLength) {
    return rows * ternaryWordsPerRow(rowLength) * sizeof(std::uint32_t);
}

// Matrice di pesi ternari impacchettati con scale per gruppo.
//
// La forma logica e' quella di un runtime::Tensor: le dimensioni
// iniziali sono appiattite in "righe", l'ultima dimensione e' la
// lunghezza di riga (l'asse lungo cui corrono l'impacchettamento e i
// gruppi di scala). Un tensore a rango 1 e' una sola riga.
//
// Questa classe possiede il buffer HOST. Il caricamento su device e la
// dequantizzazione a tile su GPU leggono lo stesso layout byte per byte
// (vedi packedWords()/scales()): non esiste una seconda codifica per la
// GPU da tenere in sincrono.
class TernaryTensor {
public:
    TernaryTensor() = default;

    // Costruisce un tensore di tutti zeri (trit 0, scale 1). Lancia
    // std::invalid_argument se la forma e' vuota o se groupSize non e'
    // un multiplo positivo di kTritsPerWord (vincolo che tiene i
    // gruppi allineati alle parole).
    explicit TernaryTensor(std::vector<std::size_t> shape, std::size_t groupSize = kDefaultGroupSize);

    // Quantizza un tensore denso con il criterio absmean di BitNet
    // b1.58, gruppo per gruppo: scala = media(|w|), trit =
    // round(w / scala) saturato. Un gruppo interamente nullo riceve
    // scala 1 (e trit tutti 0), evitando una divisione per zero.
    static TernaryTensor quantizeFrom(const runtime::Tensor& dense, std::size_t groupSize = kDefaultGroupSize);

    // Costruisce direttamente da trit e scale gia' noti: 'trits' ha
    // elementCount() valori in {-1, 0, +1} in ordine riga-maggiore,
    // 'scales' ha rows() * groupsPerRow() valori. Usata dai test di
    // round-trip e dalla deserializzazione.
    static TernaryTensor fromTrits(std::vector<std::size_t> shape, const std::vector<std::int8_t>& trits,
                                    const std::vector<float>& scales, std::size_t groupSize = kDefaultGroupSize);

    [[nodiscard]] const std::vector<std::size_t>& shape() const { return shape_; }
    [[nodiscard]] std::size_t elementCount() const { return rows_ * rowLength_; }
    [[nodiscard]] std::size_t rows() const { return rows_; }
    [[nodiscard]] std::size_t rowLength() const { return rowLength_; }
    [[nodiscard]] std::size_t groupSize() const { return groupSize_; }
    [[nodiscard]] std::size_t wordsPerRow() const { return ternaryWordsPerRow(rowLength_); }
    [[nodiscard]] std::size_t groupsPerRow() const { return (rowLength_ + groupSize_ - 1) / groupSize_; }

    // Byte effettivamente occupati dai pesi impacchettati (senza le
    // scale) e byte totali della rappresentazione. Sono i numeri che
    // la telemetria di memoria deve riportare: nessun conto teorico,
    // la dimensione reale dei buffer posseduti.
    [[nodiscard]] std::size_t packedByteCount() const { return packedWords_.size() * sizeof(std::uint32_t); }
    [[nodiscard]] std::size_t scaleByteCount() const { return scales_.size() * sizeof(float); }
    [[nodiscard]] std::size_t totalByteCount() const { return packedByteCount() + scaleByteCount(); }

    // Bit medi realmente occupati per peso logico, padding e scale
    // inclusi: e' il numero onesto da riportare, non il 1,6 teorico.
    [[nodiscard]] double bitsPerWeight() const;

    [[nodiscard]] const std::vector<std::uint32_t>& packedWords() const { return packedWords_; }
    [[nodiscard]] std::vector<std::uint32_t>& packedWords() { return packedWords_; }
    [[nodiscard]] const std::vector<float>& scales() const { return scales_; }
    [[nodiscard]] std::vector<float>& scales() { return scales_; }

    // Accesso a singoli elementi logici (indice riga-maggiore
    // sull'intero tensore). Comodi per test e ispezione, non per i
    // percorsi caldi: usare dequantizeRows() per lavorare a blocchi.
    [[nodiscard]] int tritAt(std::size_t flatIndex) const;
    void setTritAt(std::size_t flatIndex, int trit);
    [[nodiscard]] float scaleAt(std::size_t flatIndex) const;
    [[nodiscard]] float at(std::size_t flatIndex) const { return static_cast<float>(tritAt(flatIndex)) * scaleAt(flatIndex); }

    // Dequantizza UN INTERVALLO di righe in un buffer fornito dal
    // chiamante, di dimensione rowCount * rowLength(). E' il primitivo
    // usato dal percorso a tile di TernaryLinear: il picco di memoria
    // e' quello del tile, non dell'intera matrice.
    void dequantizeRows(std::size_t firstRow, std::size_t rowCount, float* out) const;

    // Quantizza UN INTERVALLO di righe da un buffer denso fornito dal
    // chiamante (rowCount * rowLength() valori), con lo stesso criterio
    // di quantizeFrom(). E' il primitivo che permette di costruire un
    // peso di miliardi di elementi senza mai materializzarne piu' di
    // un blocco in forma densa: l'inizializzazione di BlackBit-9B
    // altrimenti chiederebbe 36 GB di float32 temporanei.
    void quantizeRowsFrom(std::size_t firstRow, std::size_t rowCount, const float* dense);

    // Dequantizza l'INTERA matrice in un tensore denso. Alloca
    // elementCount() float: su una matrice BlackBit reale sono
    // centinaia di MB, quindi e' pensata per i test e per i modelli
    // piccoli, non per il percorso di addestramento (che usa
    // dequantizeRows() a tile).
    [[nodiscard]] runtime::Tensor dequantize() const;

    // Serializzazione binaria (little-endian nativo, stessa convenzione
    // del formato di checkpoint esistente BFCKPT1):
    //   rank:        uint32
    //   shape:       rank * uint64
    //   groupSize:   uint64
    //   wordCount:   uint64,  poi wordCount * uint32
    //   scaleCount:  uint64,  poi scaleCount * float32
    // Non scrive alcun magic ne' versione: e' un blocco componibile,
    // versionato dal contenitore che lo usa (vedi il checkpoint
    // BlackBit).
    void serialize(std::ostream& out) const;

    // Lancia std::runtime_error se lo stream finisce prima del previsto
    // o se i conteggi letti non sono coerenti con la forma.
    static TernaryTensor deserialize(std::istream& in);

private:
    void allocate(std::vector<std::size_t> shape, std::size_t groupSize);

    std::vector<std::size_t> shape_;
    std::size_t rows_ = 0;
    std::size_t rowLength_ = 0;
    std::size_t groupSize_ = kDefaultGroupSize;
    std::vector<std::uint32_t> packedWords_;
    std::vector<float> scales_;
};

}  // namespace blackforge::blackbit
