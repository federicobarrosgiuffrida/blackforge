#include "blackforge/frontend/lexer.hpp"

#include <cctype>
#include <unordered_map>

namespace blackforge {

namespace {

const std::unordered_map<std::string, TokenKind>& keywordTable() {
    static const std::unordered_map<std::string, TokenKind> table = {
        {"target", TokenKind::KwTarget},         {"precision", TokenKind::KwPrecision},
        {"storage", TokenKind::KwStorage},        {"compute", TokenKind::KwCompute},
        {"accumulate", TokenKind::KwAccumulate},  {"parameters", TokenKind::KwParameters},
        {"forward", TokenKind::KwForward},        {"backward", TokenKind::KwBackward},
        {"model", TokenKind::KwModel},            {"input", TokenKind::KwInput},
        {"output", TokenKind::KwOutput},          {"dataset", TokenKind::KwDataset},
        {"loss", TokenKind::KwLoss},              {"optimizer", TokenKind::KwOptimizer},
        {"train", TokenKind::KwTrain},             {"pretrain", TokenKind::KwPretrain},
        {"finetune", TokenKind::KwFinetune},       {"lora", TokenKind::KwLora},
        {"forecast", TokenKind::KwForecast},       {"benchmark", TokenKind::KwBenchmark},
    };
    return table;
}

bool isIdentifierStart(char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_'; }
bool isIdentifierPart(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

}  // namespace

Lexer::Lexer(std::string source, std::string fileName) : source_(std::move(source)), fileName_(std::move(fileName)) {}

bool Lexer::isAtEnd() const { return position_ >= source_.size(); }

char Lexer::peek(std::size_t offset) const {
    std::size_t index = position_ + offset;
    if (index >= source_.size()) {
        return '\0';
    }
    return source_[index];
}

char Lexer::advance() {
    char c = source_[position_++];
    if (c == '\n') {
        ++line_;
        column_ = 1;
    } else {
        ++column_;
    }
    return c;
}

bool Lexer::match(char expected) {
    if (isAtEnd() || peek() != expected) {
        return false;
    }
    advance();
    return true;
}

SourceLocation Lexer::currentLocation() const { return SourceLocation{fileName_, line_, column_}; }

void Lexer::skipWhitespaceAndComments() {
    while (!isAtEnd()) {
        char c = peek();

        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            advance();
            continue;
        }

        if (c == '/' && peek(1) == '/') {
            while (!isAtEnd() && peek() != '\n') {
                advance();
            }
            continue;
        }

        if (c == '/' && peek(1) == '*') {
            SourceLocation start = currentLocation();
            advance();
            advance();
            bool closed = false;
            while (!isAtEnd()) {
                if (peek() == '*' && peek(1) == '/') {
                    advance();
                    advance();
                    closed = true;
                    break;
                }
                advance();
            }
            if (!closed) {
                diagnostics_.addError(start, "commento a blocco non terminato (manca '*/')");
            }
            continue;
        }

        break;
    }
}

Token Lexer::makeToken(TokenKind kind, std::string lexeme, SourceLocation start) const {
    return Token{kind, std::move(lexeme), start};
}

Token Lexer::scanIdentifierOrKeyword(SourceLocation start) {
    std::size_t begin = position_ - 1;
    while (!isAtEnd() && isIdentifierPart(peek())) {
        advance();
    }
    std::string lexeme = source_.substr(begin, position_ - begin);

    const auto& keywords = keywordTable();
    auto it = keywords.find(lexeme);
    TokenKind kind = (it != keywords.end()) ? it->second : TokenKind::Identifier;
    return makeToken(kind, std::move(lexeme), start);
}

Token Lexer::scanNumber(SourceLocation start) {
    std::size_t begin = position_ - 1;
    bool isFloat = false;

    while (!isAtEnd() && std::isdigit(static_cast<unsigned char>(peek()))) {
        advance();
    }

    if (!isAtEnd() && peek() == '.' && std::isdigit(static_cast<unsigned char>(peek(1)))) {
        isFloat = true;
        advance();  // consuma '.'
        while (!isAtEnd() && std::isdigit(static_cast<unsigned char>(peek()))) {
            advance();
        }
    }

    if (!isAtEnd() && (peek() == 'e' || peek() == 'E')) {
        std::size_t lookahead = 1;
        if (peek(lookahead) == '+' || peek(lookahead) == '-') {
            ++lookahead;
        }
        if (std::isdigit(static_cast<unsigned char>(peek(lookahead)))) {
            isFloat = true;
            advance();  // consuma 'e'/'E'
            if (peek() == '+' || peek() == '-') {
                advance();
            }
            while (!isAtEnd() && std::isdigit(static_cast<unsigned char>(peek()))) {
                advance();
            }
        }
    }

    std::string lexeme = source_.substr(begin, position_ - begin);
    return makeToken(isFloat ? TokenKind::FloatLiteral : TokenKind::IntegerLiteral, std::move(lexeme), start);
}

Token Lexer::scanString(SourceLocation start) {
    std::size_t begin = position_;  // dopo la '"' di apertura
    std::string value;

    while (!isAtEnd() && peek() != '"') {
        char c = advance();
        if (c == '\\' && !isAtEnd()) {
            char escaped = advance();
            switch (escaped) {
                case 'n': value += '\n'; break;
                case 't': value += '\t'; break;
                case '"': value += '"'; break;
                case '\\': value += '\\'; break;
                default:
                    diagnostics_.addError(currentLocation(), "sequenza di escape sconosciuta '\\" + std::string(1, escaped) + "'");
                    value += escaped;
                    break;
            }
        } else if (c == '\n') {
            diagnostics_.addError(start, "stringa non terminata prima della fine della riga");
            return makeToken(TokenKind::StringLiteral, value, start);
        } else {
            value += c;
        }
    }

    if (isAtEnd()) {
        diagnostics_.addError(start, "stringa non terminata prima della fine del file");
        return makeToken(TokenKind::StringLiteral, value, start);
    }

    advance();  // consuma '"' di chiusura
    (void)begin;
    return makeToken(TokenKind::StringLiteral, value, start);
}

std::optional<Token> Lexer::scanToken() {
    SourceLocation start = currentLocation();
    char c = advance();

    if (isIdentifierStart(c)) {
        return scanIdentifierOrKeyword(start);
    }
    if (std::isdigit(static_cast<unsigned char>(c))) {
        return scanNumber(start);
    }
    if (c == '"') {
        return scanString(start);
    }

    switch (c) {
        case '{': return makeToken(TokenKind::LBrace, "{", start);
        case '}': return makeToken(TokenKind::RBrace, "}", start);
        case '(': return makeToken(TokenKind::LParen, "(", start);
        case ')': return makeToken(TokenKind::RParen, ")", start);
        case '[': return makeToken(TokenKind::LBracket, "[", start);
        case ']': return makeToken(TokenKind::RBracket, "]", start);
        case ',': return makeToken(TokenKind::Comma, ",", start);
        case ':': return makeToken(TokenKind::Colon, ":", start);
        case ';': return makeToken(TokenKind::Semicolon, ";", start);
        case '.': return makeToken(TokenKind::Dot, ".", start);
        case '=': return makeToken(TokenKind::Equal, "=", start);
        case '|':
            if (match('>')) {
                return makeToken(TokenKind::Pipeline, "|>", start);
            }
            diagnostics_.addError(start, "carattere '|' inatteso: forse intendevi l'operatore di pipeline '|>'?");
            return std::nullopt;
        default:
            diagnostics_.addError(start, "carattere non valido '" + std::string(1, c) + "'");
            return std::nullopt;
    }
}

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;

    while (true) {
        skipWhitespaceAndComments();
        if (isAtEnd()) {
            tokens.push_back(makeToken(TokenKind::EndOfFile, "", currentLocation()));
            break;
        }

        std::optional<Token> token = scanToken();
        if (token.has_value()) {
            tokens.push_back(std::move(*token));
        }
        // Se non e' stato prodotto un token (carattere non valido), il
        // ciclo riprende semplicemente dalla posizione successiva.
    }

    return tokens;
}

}  // namespace blackforge
