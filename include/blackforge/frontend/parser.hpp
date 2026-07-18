#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "blackforge/ast/ast.hpp"
#include "blackforge/diagnostics/diagnostic.hpp"
#include "blackforge/frontend/token.hpp"

namespace blackforge {

// Analizzatore sintattico di BlackForge: trasforma la sequenza di token
// prodotta dal Lexer in un Program (AST).
//
// Il parser recupera dagli errori sintattici: quando una dichiarazione
// top-level non e' valida, la salta (sincronizzandosi sulla prossima
// parola chiave 'target'/'precision'/'model') e continua, cosi' una
// singola esecuzione puo' riportare piu' errori sintattici.
class Parser {
public:
    explicit Parser(std::vector<Token> tokens);

    ast::Program parseProgram();

    [[nodiscard]] const DiagnosticList& diagnostics() const { return diagnostics_; }

private:
    // Eccezione interna usata solo per risalire fino al punto di
    // sincronizzazione quando un errore sintattico non e' recuperabile
    // localmente. Non attraversa mai i confini pubblici della classe.
    struct ParseError {};

    [[nodiscard]] const Token& peek(std::size_t offset = 0) const;
    [[nodiscard]] const Token& previous() const;
    const Token& advance();
    [[nodiscard]] bool check(TokenKind kind) const;
    bool match(TokenKind kind);
    const Token& expect(TokenKind kind, const std::string& errorMessage);
    [[nodiscard]] bool isAtEnd() const;

    [[nodiscard]] ParseError error(const SourceLocation& location, const std::string& message);

    void synchronizeToDeclaration();
    void synchronizeToModelStatement();

    std::optional<ast::Decl> parseDeclaration();
    ast::TargetDecl parseTargetDecl();
    ast::PrecisionDecl parsePrecisionDecl();
    ast::ModelDecl parseModelDecl();
    std::optional<ast::ModelStatement> parseModelStatement();
    ast::InputDecl parseInputDecl(SourceLocation start);
    ast::PipelineStmt parsePipelineStmt();
    ast::DottedName parseDottedName();
    ast::TensorType parseTensorType();
    ast::ShapeDim parseShapeDim();
    ast::Expr parseArgExpr();
    ast::PipelineStage parsePipelineStage();

    std::vector<Token> tokens_;
    std::size_t current_ = 0;
    DiagnosticList diagnostics_;
};

}  // namespace blackforge
