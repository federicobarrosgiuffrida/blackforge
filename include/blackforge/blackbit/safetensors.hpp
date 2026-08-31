#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "blackforge/runtime/tensor.hpp"

// Lettore del formato safetensors, usato per importare in BlackBit i
// pesi di un modello gia' addestrato (OLMoE, Mixtral, Qwen, ...).
//
// IL FORMATO
//
//   [0, 8)          uint64 little-endian: lunghezza N dell'intestazione
//   [8, 8+N)        intestazione JSON: nome -> {dtype, shape, data_offsets}
//   [8+N, fine)     i tensori, contigui, agli offset dichiarati
//
// L'intestazione sta in testa al file, quindi il manifest completo di un
// modello da 14 GB si legge con 140 KB di I/O. E' anche il motivo per cui
// questa classe apre il file e legge SOLO l'intestazione: i tensori
// vengono letti su richiesta, uno per volta, e a blocchi di righe.
//
// PERCHE' A BLOCCHI DI RIGHE
//
// La tabella di embedding di OLMoE e' [50304, 2048] in BF16: 206 MB sul
// disco, 412 MB una volta convertita in float32. Su BlackBit-9B sarebbero
// 805 MB. Leggerla intera per poi quantizzarla vanificherebbe tutto il
// lavoro fatto per non materializzare mai una matrice completa, e in
// import il picco di memoria conta esattamente come in addestramento.
// readFloatRows() legge, converte e consegna un blocco per volta, ed e'
// la forma che TernaryTensor::quantizeRowsFrom() consuma direttamente.
//
// NESSUNA DIPENDENZA ESTERNA: il sottoinsieme di JSON che l'intestazione
// usa (oggetti, stringhe, array di interi, numeri) e' abbastanza piccolo
// da essere analizzato in un centinaio di righe, e BlackForge non ha
// librerie di terze parti.

namespace blackforge::blackbit {

enum class SafetensorsDType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    BOOL,
};

// Byte occupati da un elemento del tipo dato.
std::size_t safetensorsDTypeSize(SafetensorsDType dtype);

const char* safetensorsDTypeName(SafetensorsDType dtype);

struct SafetensorsEntry {
    std::string name;
    SafetensorsDType dtype = SafetensorsDType::F32;
    std::vector<std::size_t> shape;
    std::size_t byteBegin = 0;  // relativi all'inizio della sezione dati
    std::size_t byteEnd = 0;

    [[nodiscard]] std::size_t elementCount() const;
    [[nodiscard]] std::size_t rows() const;       // tutte le dimensioni tranne l'ultima
    [[nodiscard]] std::size_t rowLength() const;  // l'ultima dimensione (1 se rango 0)
    [[nodiscard]] std::string shapeToString() const;
};

// Un singolo file .safetensors.
class SafetensorsFile {
public:
    // Legge SOLO l'intestazione. Lancia std::runtime_error se il file non
    // si apre, se l'intestazione e' malformata o se dichiara offset
    // incoerenti con la dimensione del file (un file troncato deve
    // fallire subito, non a meta' di un import da mezz'ora).
    explicit SafetensorsFile(std::string path);

    [[nodiscard]] const std::vector<SafetensorsEntry>& tensors() const { return tensors_; }
    [[nodiscard]] const SafetensorsEntry* find(const std::string& name) const;
    [[nodiscard]] const std::string& path() const { return path_; }

    // Metadati opzionali del file (la chiave "__metadata__").
    [[nodiscard]] const std::unordered_map<std::string, std::string>& metadata() const { return metadata_; }

    // Legge le righe [firstRow, firstRow + rowCount) convertendole in
    // float32 nel buffer del chiamante (rowCount * rowLength valori).
    void readFloatRows(const std::string& name, std::size_t firstRow, std::size_t rowCount, float* out) const;

    // Legge il tensore INTERO. Comoda per i tensori piccoli (norm,
    // router); su una matrice grande alloca l'intera matrice in float32,
    // quindi per quelle si usa readFloatRows().
    [[nodiscard]] runtime::Tensor readFloat(const std::string& name) const;

private:
    std::string path_;
    std::size_t dataOffset_ = 0;  // 8 + lunghezza dell'intestazione
    std::vector<SafetensorsEntry> tensors_;
    std::unordered_map<std::string, std::size_t> byName_;
    std::unordered_map<std::string, std::string> metadata_;
};

// Un modello, eventualmente diviso in piu' file .safetensors secondo
// 'model.safetensors.index.json'.
class SafetensorsModel {
public:
    // 'directory' e' la cartella che contiene o
    // 'model.safetensors.index.json' (modello diviso in shard) o
    // 'model.safetensors' (file unico). Lancia std::runtime_error se non
    // trova ne' l'uno ne' l'altro.
    explicit SafetensorsModel(const std::string& directory);

    [[nodiscard]] std::vector<std::string> names() const;
    [[nodiscard]] bool has(const std::string& name) const;
    [[nodiscard]] const SafetensorsEntry& entry(const std::string& name) const;

    void readFloatRows(const std::string& name, std::size_t firstRow, std::size_t rowCount, float* out) const;
    [[nodiscard]] runtime::Tensor readFloat(const std::string& name) const;

    [[nodiscard]] std::size_t fileCount() const { return files_.size(); }

private:
    [[nodiscard]] const SafetensorsFile& fileFor(const std::string& name) const;

    std::vector<SafetensorsFile> files_;
    std::unordered_map<std::string, std::size_t> tensorToFile_;
};

// Conversioni verso float32, esposte perche' i test le verificano
// direttamente.
//
// BF16 sono ESATTAMENTE i 16 bit alti di un float32: la conversione non
// perde nulla ed e' uno shift.
float bf16ToFloat(std::uint16_t bits);

// FP16 (1 segno, 5 esponente, 10 mantissa) richiede invece una vera
// conversione, subnormali e infiniti compresi.
float fp16ToFloat(std::uint16_t bits);

}  // namespace blackforge::blackbit
