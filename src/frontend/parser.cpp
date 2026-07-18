#include "blackforge/frontend/parser.hpp"

#include <stdexcept>

namespace blackforge {

Parser::Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

const Token& Parser::peek(std::size_t offset) const {
    std::size_t index = current_ + offset;
    if (index >= tokens_.size()) {
        return tokens_.back();  // EndOfFile
    }
    return tokens_[index];
}

const Token& Parser::previous() const { return tokens_[current_ - 1]; }

bool Parser::isAtEnd() const { return peek().kind == TokenKind::EndOfFile; }

const Token& Parser::advance() {
    if (!isAtEnd()) {
        ++current_;
    }
    return previous();
}

bool Parser::check(TokenKind kind) const { return peek().kind == kind; }

bool Parser::match(TokenKind kind) {
    if (!check(kind)) {
        return false;
    }
    advance();
    return true;
}

const Token& Parser::expect(TokenKind kind, const std::string& errorMessage) {
    if (check(kind)) {
        return advance();
    }
    throw error(peek().location, errorMessage + " (trovato " + tokenKindName(peek().kind) + ")");
}

Parser::ParseError Parser::error(const SourceLocation& location, const std::string& message) {
    diagnostics_.addError(location, message);
    return ParseError{};
}

void Parser::synchronizeToDeclaration() {
    while (!isAtEnd()) {
        if (check(TokenKind::KwTarget) || check(TokenKind::KwPrecision) || check(TokenKind::KwModel)) {
            return;
        }
        advance();
    }
}

void Parser::synchronizeToModelStatement() {
    while (!isAtEnd()) {
        if (check(TokenKind::RBrace) || check(TokenKind::KwInput) ||
            (check(TokenKind::Identifier) && peek(1).kind == TokenKind::Pipeline)) {
            return;
        }
        advance();
    }
}

ast::Program Parser::parseProgram() {
    ast::Program program;

    while (!isAtEnd()) {
        try {
            std::optional<ast::Decl> decl = parseDeclaration();
            if (decl.has_value()) {
                program.declarations.push_back(std::move(*decl));
            }
        } catch (const ParseError&) {
            synchronizeToDeclaration();
        }
    }

    return program;
}

std::optional<ast::Decl> Parser::parseDeclaration() {
    if (check(TokenKind::KwTarget)) {
        return ast::Decl{parseTargetDecl()};
    }
    if (check(TokenKind::KwPrecision)) {
        return ast::Decl{parsePrecisionDecl()};
    }
    if (check(TokenKind::KwModel)) {
        return ast::Decl{parseModelDecl()};
    }

    throw error(peek().location,
                "attesa una dichiarazione top-level ('target', 'precision' o 'model'), trovato " +
                    tokenKindName(peek().kind));
}

ast::DottedName Parser::parseDottedName() {
    SourceLocation start = peek().location;
    std::vector<std::string> segments;

    const Token& first = expect(TokenKind::Identifier, "atteso un identificatore");
    segments.push_back(first.lexeme);

    while (match(TokenKind::Dot)) {
        const Token& segment = expect(TokenKind::Identifier, "atteso un identificatore dopo '.'");
        segments.push_back(segment.lexeme);
    }

    return ast::DottedName{std::move(segments), start};
}

ast::TargetDecl Parser::parseTargetDecl() {
    SourceLocation start = peek().location;
    expect(TokenKind::KwTarget, "atteso 'target'");
    ast::DottedName target = parseDottedName();
    return ast::TargetDecl{std::move(target), start};
}

ast::PrecisionDecl Parser::parsePrecisionDecl() {
    SourceLocation start = peek().location;
    expect(TokenKind::KwPrecision, "atteso 'precision'");
    expect(TokenKind::LBrace, "atteso '{' dopo 'precision'");

    std::vector<ast::PrecisionField> fields;
    while (!check(TokenKind::RBrace) && !isAtEnd()) {
        SourceLocation fieldStart = peek().location;
        ast::PrecisionFieldKind kind;

        switch (peek().kind) {
            case TokenKind::KwStorage: kind = ast::PrecisionFieldKind::Storage; break;
            case TokenKind::KwCompute: kind = ast::PrecisionFieldKind::Compute; break;
            case TokenKind::KwAccumulate: kind = ast::PrecisionFieldKind::Accumulate; break;
            case TokenKind::KwParameters: kind = ast::PrecisionFieldKind::Parameters; break;
            case TokenKind::KwForward: kind = ast::PrecisionFieldKind::Forward; break;
            case TokenKind::KwBackward: kind = ast::PrecisionFieldKind::Backward; break;
            default:
                throw error(peek().location,
                            "atteso un campo di precisione ('storage', 'compute', 'accumulate', "
                            "'parameters', 'forward' o 'backward'), trovato " +
                                tokenKindName(peek().kind));
        }

        advance();
        ast::DottedName value = parseDottedName();
        fields.push_back(ast::PrecisionField{kind, std::move(value), fieldStart});
    }

    expect(TokenKind::RBrace, "atteso '}' per chiudere il blocco 'precision'");
    return ast::PrecisionDecl{std::move(fields), start};
}

ast::ShapeDim Parser::parseShapeDim() {
    SourceLocation start = peek().location;

    if (check(TokenKind::IntegerLiteral)) {
        const Token& token = advance();
        try {
            return ast::ShapeDim{false, "", std::stoll(token.lexeme), start};
        } catch (const std::out_of_range&) {
            throw error(start, "valore intero troppo grande per una dimensione: '" + token.lexeme + "'");
        }
    }
    if (check(TokenKind::Identifier)) {
        const Token& token = advance();
        return ast::ShapeDim{true, token.lexeme, 0, start};
    }

    throw error(start, "attesa una dimensione del tensore (intero o nome simbolico), trovato " +
                            tokenKindName(peek().kind));
}

ast::TensorType Parser::parseTensorType() {
    SourceLocation start = peek().location;
    ast::DottedName dtype = parseDottedName();
    expect(TokenKind::LBracket, "atteso '[' per la forma del tensore");

    std::vector<ast::ShapeDim> shape;
    if (!check(TokenKind::RBracket)) {
        shape.push_back(parseShapeDim());
        while (match(TokenKind::Comma)) {
            shape.push_back(parseShapeDim());
        }
    }

    expect(TokenKind::RBracket, "atteso ']' per chiudere la forma del tensore");
    return ast::TensorType{std::move(dtype), std::move(shape), start};
}

ast::InputDecl Parser::parseInputDecl(SourceLocation start) {
    // La parola chiave 'input' e' gia' stata consumata dal chiamante.
    ast::TensorType type = parseTensorType();
    return ast::InputDecl{std::move(type), start};
}

ast::Expr Parser::parseArgExpr() {
    SourceLocation start = peek().location;

    if (check(TokenKind::IntegerLiteral)) {
        const Token& token = advance();
        return ast::Expr{ast::ExprKind::IntegerLiteral, token.lexeme, start};
    }
    if (check(TokenKind::FloatLiteral)) {
        const Token& token = advance();
        return ast::Expr{ast::ExprKind::FloatLiteral, token.lexeme, start};
    }
    if (check(TokenKind::StringLiteral)) {
        const Token& token = advance();
        return ast::Expr{ast::ExprKind::StringLiteral, token.lexeme, start};
    }
    if (check(TokenKind::Identifier)) {
        const Token& token = advance();
        return ast::Expr{ast::ExprKind::Identifier, token.lexeme, start};
    }

    throw error(start, "atteso un argomento (letterale o identificatore), trovato " + tokenKindName(peek().kind));
}

ast::PipelineStage Parser::parsePipelineStage() {
    SourceLocation start = peek().location;
    const Token& nameToken = expect(TokenKind::Identifier, "atteso il nome di una fase della pipeline");

    std::vector<ast::Expr> args;
    if (match(TokenKind::LParen)) {
        if (!check(TokenKind::RParen)) {
            args.push_back(parseArgExpr());
            while (match(TokenKind::Comma)) {
                args.push_back(parseArgExpr());
            }
        }
        expect(TokenKind::RParen, "atteso ')' per chiudere gli argomenti di '" + nameToken.lexeme + "'");
    }

    return ast::PipelineStage{nameToken.lexeme, std::move(args), start};
}

ast::PipelineStmt Parser::parsePipelineStmt() {
    SourceLocation start = peek().location;
    ast::PipelineSource source;
    source.location = start;

    if (check(TokenKind::KwInput)) {
        advance();
        source.kind = ast::PipelineSourceKind::Input;
    } else {
        const Token& id =
            expect(TokenKind::Identifier, "attesa 'input' o un identificatore come sorgente della pipeline");
        source.kind = ast::PipelineSourceKind::Identifier;
        source.identifierName = id.lexeme;
    }

    std::vector<ast::PipelineStage> stages;
    while (match(TokenKind::Pipeline)) {
        stages.push_back(parsePipelineStage());
    }

    if (stages.empty()) {
        throw error(peek().location, "attesa almeno una fase dopo l'operatore di pipeline '|>'");
    }

    return ast::PipelineStmt{source, std::move(stages), start};
}

std::optional<ast::ModelStatement> Parser::parseModelStatement() {
    SourceLocation start = peek().location;

    // 'input' seguito da un tipo tensoriale e' una dichiarazione di input;
    // 'input' seguito da '|>' e' l'inizio di una pipeline che usa l'input
    // come sorgente.
    if (check(TokenKind::KwInput) && peek(1).kind != TokenKind::Pipeline) {
        advance();
        return ast::ModelStatement{parseInputDecl(start)};
    }

    if (check(TokenKind::KwInput) || (check(TokenKind::Identifier) && peek(1).kind == TokenKind::Pipeline)) {
        return ast::ModelStatement{parsePipelineStmt()};
    }

    throw error(start, "attesa una dichiarazione 'input' o l'inizio di una pipeline, trovato " +
                            tokenKindName(peek().kind));
}

ast::ModelDecl Parser::parseModelDecl() {
    SourceLocation start = peek().location;
    expect(TokenKind::KwModel, "atteso 'model'");
    const Token& nameToken = expect(TokenKind::Identifier, "atteso il nome del modello dopo 'model'");
    expect(TokenKind::LBrace, "atteso '{' dopo il nome del modello");

    std::vector<ast::ModelStatement> body;
    while (!check(TokenKind::RBrace) && !isAtEnd()) {
        try {
            std::optional<ast::ModelStatement> statement = parseModelStatement();
            if (statement.has_value()) {
                body.push_back(std::move(*statement));
            }
        } catch (const ParseError&) {
            synchronizeToModelStatement();
        }
    }

    expect(TokenKind::RBrace, "atteso '}' per chiudere il blocco 'model'");
    return ast::ModelDecl{nameToken.lexeme, std::move(body), start};
}

}  // namespace blackforge
