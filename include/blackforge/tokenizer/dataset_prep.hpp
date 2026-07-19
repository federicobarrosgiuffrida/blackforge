#pragma once

#include <cstddef>
#include <string>

#include "blackforge/tokenizer/tokenizer.hpp"

namespace blackforge::tokenizer {

// Costruisce un dataset .bfdata pronto per l'addestramento di un
// modello linguistico (next-token-prediction) a partire da un corpus
// di testo grezzo: tokenizza con 'tok', poi affetta lo stream di id
// risultante in finestre non sovrapposte di 'seqLen' token (stride ==
// seqLen; l'ultima finestra, se il corpus tokenizzato non basta a
// riempirla, viene scartata — non e' un errore, a meno che NESSUNA
// finestra sia completa). Ogni esempio prodotto ha input `[seqLen]`
// (id di token) e target `[seqLen, tok.vocabSize()]` (one-hot del
// token successivo per ogni posizione, shift-by-one causale — la
// stessa forma attesa da `loss cross_entropy` generalizzata a rango 3,
// vedi backend::{cpu,cuda}::softmaxCrossEntropy).
//
// Ritorna il numero di esempi scritti. Lancia std::invalid_argument
// se il corpus tokenizzato non produce nemmeno una finestra completa
// (serve almeno seqLen + 1 token, l'ultimo serve come target
// dell'ultima posizione della finestra).
//
// NOTA: come data::Dataset in generale, l'intero dataset risultante
// viene costruito in memoria prima di essere scritto su disco — per
// corpora molto grandi e' un limite esplicito, che la milestone di
// caricamento dati efficiente (streaming/shard) rimuovera'; questa
// funzione non e' pensata per corpora da gigabyte cosi' come e' oggi.
std::size_t buildLanguageModelDataset(const Tokenizer& tok, const std::string& corpus, std::size_t seqLen,
                                       const std::string& outputPath);

}  // namespace blackforge::tokenizer
