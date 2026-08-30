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

    // Ternario impacchettato {-1, 0, +1} con una scala per gruppo di
    // pesi contigui, ~1,6 bit/peso reali (vedi
    // blackforge/blackbit/ternary.hpp per il formato esatto). A
    // differenza di ogni altro valore di questo enum NON e' un formato
    // a byte interi: dtypeSizeInBytes() restituisce 0 e la dimensione
    // va calcolata con dtypeBitsPerElement() o, meglio, con
    // blackbit::ternaryPackedBytes().
    //
    // NON e' (ancora) riconosciuto da parseDType(): non esiste alcun
    // percorso che implementi 'precision { storage ternary1p58 }' per i
    // modelli descritti nel linguaggio, e accettarlo nella sintassi
    // significherebbe dichiarare un supporto che non c'e'. E' il
    // formato di memorizzazione del sottosistema BlackBit, che
    // costruisce i propri modelli nativamente.
    TERNARY_1P58,
};

// Prova a risolvere un nome puntato (es. "bf16", "fp8.e4m3") in un DType
// conosciuto. Restituisce nullopt se il nome non corrisponde a nessun
// formato supportato.
std::optional<DType> parseDType(const ast::DottedName& name);

// Numero di byte occupati in memoria da un valore di questo formato.
// Per TF32 e' convenzionalmente la dimensione di storage equivalente
// (4 byte, come FP32), dato che TF32 e' solo una modalita' di calcolo.
// Restituisce 0 per i formati sub-byte (TERNARY_1P58), dove "byte per
// elemento" non e' un numero intero: usare dtypeBitsPerElement().
int dtypeSizeInBytes(DType dtype);

// Bit occupati in media da un elemento, come numero reale: 8/16/32 per
// i formati a byte interi, 1,6 per TERNARY_1P58 (5 pesi ternari per
// byte, vedi blackbit/ternary.hpp). Serve alla contabilita' di memoria
// dei modelli a bassa precisione, dove arrotondare a un byte intero
// falserebbe il conto di un fattore 5.
double dtypeBitsPerElement(DType dtype);

// true per i formati che non occupano un numero intero di byte per
// elemento e richiedono quindi un impacchettamento dedicato.
bool isSubByte(DType dtype);

// TF32 non e' un formato di memorizzazione: e' una modalita' di calcolo
// per operazioni compatibili (occupa comunque 4 byte come FP32, ma non
// ha senso "salvare i pesi in TF32"). Restituisce false solo per TF32:
// tutti gli altri formati sono validi sia per storage/parameters sia
// per compute/forward/backward.
bool isValidForStorage(DType dtype);

std::string dtypeName(DType dtype);

}  // namespace blackforge::sema
