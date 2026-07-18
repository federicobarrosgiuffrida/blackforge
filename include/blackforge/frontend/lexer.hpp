#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "blackforge/diagnostics/diagnostic.hpp"
#include "blackforge/frontend/token.hpp"

namespace blackforge {

// Analizzatore lessicale di BlackForge.
//
// Trasforma il testo sorgente di un file .bf in una sequenza di token.
// In presenza di caratteri non validi non si interrompe: registra una
// diagnostica e prosegue, cosi' un singolo comando puo' riportare tutti
// gli errori lessicali di un file in un'unica esecuzione.
class Lexer {
public:
    Lexer(std::string source, std::string fileName);

    // Esegue l'intera tokenizzazione e restituisce la sequenza di token,
    // terminata da un token EndOfFile. Gli eventuali errori sono
    // disponibili tramite diagnostics().
    std::vector<Token> tokenize();

    [[nodiscard]] const DiagnosticList& diagnostics() const { return diagnostics_; }

private:
    [[nodiscard]] bool isAtEnd() const;
    char peek(std::size_t offset = 0) const;
    char advance();
    bool match(char expected);

    void skipWhitespaceAndComments();

    // Restituisce nullopt quando il carattere corrente non produce un
    // token valido (es. carattere sconosciuto): l'errore e' gia' stato
    // registrato e il chiamante deve semplicemente riprovare.
    std::optional<Token> scanToken();
    Token makeToken(TokenKind kind, std::string lexeme, SourceLocation start) const;
    Token scanIdentifierOrKeyword(SourceLocation start);
    Token scanNumber(SourceLocation start);
    Token scanString(SourceLocation start);

    [[nodiscard]] SourceLocation currentLocation() const;

    std::string source_;
    std::string fileName_;
    std::size_t position_ = 0;
    std::size_t line_ = 1;
    std::size_t column_ = 1;

    DiagnosticList diagnostics_;
};

}  // namespace blackforge
