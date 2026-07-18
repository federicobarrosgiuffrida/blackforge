#include "blackforge/frontend/lexer.hpp"

#include <gtest/gtest.h>

#include "blackforge/frontend/token.hpp"

using blackforge::Lexer;
using blackforge::Token;
using blackforge::TokenKind;

namespace {

std::vector<TokenKind> kindsOf(const std::vector<Token>& tokens) {
    std::vector<TokenKind> kinds;
    kinds.reserve(tokens.size());
    for (const auto& token : tokens) {
        kinds.push_back(token.kind);
    }
    return kinds;
}

}  // namespace

TEST(LexerTest, TokenizzaFileVuoto) {
    Lexer lexer("", "test.bf");
    auto tokens = lexer.tokenize();

    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].kind, TokenKind::EndOfFile);
    EXPECT_FALSE(lexer.diagnostics().hasErrors());
}

TEST(LexerTest, RiconosceParoleChiave) {
    Lexer lexer("target precision model input dataset loss optimizer train", "test.bf");
    auto tokens = lexer.tokenize();

    std::vector<TokenKind> expected = {
        TokenKind::KwTarget,   TokenKind::KwPrecision, TokenKind::KwModel,    TokenKind::KwInput,
        TokenKind::KwDataset,  TokenKind::KwLoss,       TokenKind::KwOptimizer, TokenKind::KwTrain,
        TokenKind::EndOfFile,
    };
    EXPECT_EQ(kindsOf(tokens), expected);
}

TEST(LexerTest, RiconosceIdentificatoriNonParoleChiave) {
    Lexer lexer("TinyModel bf16 linear silu hidden_dim", "test.bf");
    auto tokens = lexer.tokenize();

    ASSERT_EQ(tokens.size(), 6u);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(tokens[i].kind, TokenKind::Identifier);
    }
    EXPECT_EQ(tokens[0].lexeme, "TinyModel");
    EXPECT_EQ(tokens[1].lexeme, "bf16");
}

TEST(LexerTest, RiconosceInteriEFloat) {
    Lexer lexer("4096 batch 0.001 1e-4 3.14", "test.bf");
    auto tokens = lexer.tokenize();

    EXPECT_EQ(tokens[0].kind, TokenKind::IntegerLiteral);
    EXPECT_EQ(tokens[0].lexeme, "4096");
    EXPECT_EQ(tokens[1].kind, TokenKind::Identifier);
    EXPECT_EQ(tokens[2].kind, TokenKind::FloatLiteral);
    EXPECT_EQ(tokens[2].lexeme, "0.001");
    EXPECT_EQ(tokens[3].kind, TokenKind::FloatLiteral);
    EXPECT_EQ(tokens[3].lexeme, "1e-4");
    EXPECT_EQ(tokens[4].kind, TokenKind::FloatLiteral);
    EXPECT_EQ(tokens[4].lexeme, "3.14");
}

TEST(LexerTest, RiconoscePipelineEPunteggiatura) {
    Lexer lexer("input |> linear(4096) |> silu", "test.bf");
    auto tokens = lexer.tokenize();

    std::vector<TokenKind> expected = {
        TokenKind::KwInput,  TokenKind::Pipeline, TokenKind::Identifier, TokenKind::LParen,
        TokenKind::IntegerLiteral, TokenKind::RParen, TokenKind::Pipeline, TokenKind::Identifier,
        TokenKind::EndOfFile,
    };
    EXPECT_EQ(kindsOf(tokens), expected);
}

TEST(LexerTest, RiconosceNomiPuntati) {
    Lexer lexer("nvidia.blackwell fp8.e4m3", "test.bf");
    auto tokens = lexer.tokenize();

    std::vector<TokenKind> expected = {
        TokenKind::Identifier, TokenKind::Dot, TokenKind::Identifier,
        TokenKind::Identifier, TokenKind::Dot, TokenKind::Identifier,
        TokenKind::EndOfFile,
    };
    EXPECT_EQ(kindsOf(tokens), expected);
}

TEST(LexerTest, RiconosceStringhe) {
    Lexer lexer(R"("dataset/train.bin" "riga\ncon escape")", "test.bf");
    auto tokens = lexer.tokenize();

    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0].kind, TokenKind::StringLiteral);
    EXPECT_EQ(tokens[0].lexeme, "dataset/train.bin");
    EXPECT_EQ(tokens[1].lexeme, "riga\ncon escape");
}

TEST(LexerTest, IgnoraCommentiRigaEBlocco) {
    Lexer lexer("target // commento a fine riga\n/* commento\nsu piu' righe */ model", "test.bf");
    auto tokens = lexer.tokenize();

    std::vector<TokenKind> expected = {TokenKind::KwTarget, TokenKind::KwModel, TokenKind::EndOfFile};
    EXPECT_EQ(kindsOf(tokens), expected);
}

TEST(LexerTest, TracciaRigaEColonna) {
    Lexer lexer("target\n  model", "test.bf");
    auto tokens = lexer.tokenize();

    EXPECT_EQ(tokens[0].location.line, 1u);
    EXPECT_EQ(tokens[0].location.column, 1u);

    EXPECT_EQ(tokens[1].location.line, 2u);
    EXPECT_EQ(tokens[1].location.column, 3u);
}

TEST(LexerTest, SegnalaCarattereNonValido) {
    Lexer lexer("target $ model", "test.bf");
    auto tokens = lexer.tokenize();

    EXPECT_TRUE(lexer.diagnostics().hasErrors());
    // Il carattere non valido viene saltato e il lexer continua.
    std::vector<TokenKind> expected = {TokenKind::KwTarget, TokenKind::KwModel, TokenKind::EndOfFile};
    EXPECT_EQ(kindsOf(tokens), expected);

    const auto& diags = lexer.diagnostics().all();
    ASSERT_EQ(diags.size(), 1u);
    EXPECT_EQ(diags[0].location.line, 1u);
    EXPECT_EQ(diags[0].location.column, 8u);
}

TEST(LexerTest, SegnalaStringaNonTerminata) {
    Lexer lexer("\"dataset senza chiusura", "test.bf");
    lexer.tokenize();

    EXPECT_TRUE(lexer.diagnostics().hasErrors());
}

TEST(LexerTest, SegnalaCommentoBloccoNonTerminato) {
    Lexer lexer("/* commento mai chiuso", "test.bf");
    lexer.tokenize();

    EXPECT_TRUE(lexer.diagnostics().hasErrors());
}

TEST(LexerTest, TokenizzaEsempioModelloCompleto) {
    Lexer lexer(
        "target nvidia.blackwell\n"
        "\n"
        "precision {\n"
        "    storage bf16\n"
        "    compute fp8.e4m3\n"
        "    accumulate fp32\n"
        "}\n"
        "\n"
        "model TinyModel {\n"
        "    input bf16[batch, 4096]\n"
        "\n"
        "    input\n"
        "        |> linear(4096)\n"
        "        |> silu\n"
        "        |> linear(4096)\n"
        "}\n",
        "tiny_model.bf");

    auto tokens = lexer.tokenize();

    EXPECT_FALSE(lexer.diagnostics().hasErrors());
    EXPECT_GT(tokens.size(), 30u);
    EXPECT_EQ(tokens.back().kind, TokenKind::EndOfFile);
}
