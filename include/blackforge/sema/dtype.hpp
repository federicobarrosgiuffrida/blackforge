#pragma once

#include <optional>
#include <string>

#include "blackforge/ast/ast.hpp"

namespace blackforge::sema {

// Formati numerici riconosciuti da BlackForge. TF32 non e' un formato di
// memorizzazione (occupa comunque 4 byte come FP32): e' una modalita' di
// calcolo per operazioni compatibili, ma viene comunque rappresentato
// qui perche' puo' comparire nei campi 'compute' di un blocco precision.
enum class DType {
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    BF16,
    TF32,
    FP32,
};

// Prova a risolvere un nome puntato (es. "bf16", "fp8.e4m3") in un DType
// conosciuto. Restituisce nullopt se il nome non corrisponde a nessun
// formato supportato.
std::optional<DType> parseDType(const ast::DottedName& name);

// Numero di byte occupati in memoria da un valore di questo formato.
// Per TF32 e' convenzionalmente la dimensione di storage equivalente
// (4 byte, come FP32), dato che TF32 e' solo una modalita' di calcolo.
int dtypeSizeInBytes(DType dtype);

// TF32 non e' un formato di memorizzazione: e' una modalita' di calcolo
// per operazioni compatibili (occupa comunque 4 byte come FP32, ma non
// ha senso "salvare i pesi in TF32"). Restituisce false solo per TF32:
// tutti gli altri formati sono validi sia per storage/parameters sia
// per compute/forward/backward.
bool isValidForStorage(DType dtype);

std::string dtypeName(DType dtype);

}  // namespace blackforge::sema
