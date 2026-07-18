#pragma once

#include <string>

#include "blackforge/diagnostics/diagnostic.hpp"

namespace blackforge {

// Tipi di token riconosciuti dal lexer di BlackForge.
//
// Il linguaggio e' volutamente verticale: le parole chiave riguardano
// solo i concetti del dominio ML (target, precision, model, dataset, ...).
// I nomi dei formati numerici (bf16, fp8, e4m3, ...) e degli operatori
// tensoriali (linear, silu, ...) NON sono parole chiave: sono identificatori
// ordinari risolti in fase di analisi semantica.
enum class TokenKind {
    // Letterali
    Identifier,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,

    // Parole chiave della configurazione hardware/precisione
    KwTarget,
    KwPrecision,
    KwStorage,
    KwCompute,
    KwAccumulate,
    KwParameters,
    KwForward,
    KwBackward,

    // Parole chiave di dominio ML
    KwModel,
    KwInput,
    KwOutput,
    KwDataset,
    KwLoss,
    KwOptimizer,
    KwTrain,
    KwPretrain,
    KwFinetune,
    KwLora,
    KwForecast,
    KwBenchmark,

    // Parole chiave di 'dataset { ... }' e 'train { ... }'
    KwPath,
    KwLabels,
    KwEpochs,
    KwBatchSize,
    KwLearningRate,
    KwLrSchedule,

    // Parole chiave di 'lora { ... }' (dentro 'train') e 'forecast { ... }'
    KwRank,
    KwAlpha,
    KwHorizon,

    // Punteggiatura
    LBrace,     // {
    RBrace,     // }
    LParen,     // (
    RParen,     // )
    LBracket,   // [
    RBracket,   // ]
    Comma,      // ,
    Colon,      // :
    Semicolon,  // ;
    Dot,        // .
    Equal,      // =

    // Operatori
    Pipeline,  // |>

    // Fine flusso token
    EndOfFile,
};

// Rappresenta un singolo token estratto dal sorgente BlackForge, con
// posizione per la diagnostica (file, riga, colonna).
struct Token {
    TokenKind kind;
    std::string lexeme;
    SourceLocation location;
};

// Nome leggibile di un TokenKind, usato nei messaggi diagnostici e nei test.
std::string tokenKindName(TokenKind kind);

}  // namespace blackforge
