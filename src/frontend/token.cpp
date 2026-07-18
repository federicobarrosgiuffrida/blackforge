#include "blackforge/frontend/token.hpp"

namespace blackforge {

std::string tokenKindName(TokenKind kind) {
    switch (kind) {
        case TokenKind::Identifier: return "identificatore";
        case TokenKind::IntegerLiteral: return "letterale intero";
        case TokenKind::FloatLiteral: return "letterale in virgola mobile";
        case TokenKind::StringLiteral: return "letterale stringa";
        case TokenKind::KwTarget: return "'target'";
        case TokenKind::KwPrecision: return "'precision'";
        case TokenKind::KwStorage: return "'storage'";
        case TokenKind::KwCompute: return "'compute'";
        case TokenKind::KwAccumulate: return "'accumulate'";
        case TokenKind::KwParameters: return "'parameters'";
        case TokenKind::KwForward: return "'forward'";
        case TokenKind::KwBackward: return "'backward'";
        case TokenKind::KwModel: return "'model'";
        case TokenKind::KwInput: return "'input'";
        case TokenKind::KwOutput: return "'output'";
        case TokenKind::KwDataset: return "'dataset'";
        case TokenKind::KwLoss: return "'loss'";
        case TokenKind::KwOptimizer: return "'optimizer'";
        case TokenKind::KwTrain: return "'train'";
        case TokenKind::KwPretrain: return "'pretrain'";
        case TokenKind::KwFinetune: return "'finetune'";
        case TokenKind::KwLora: return "'lora'";
        case TokenKind::KwForecast: return "'forecast'";
        case TokenKind::KwBenchmark: return "'benchmark'";
        case TokenKind::LBrace: return "'{'";
        case TokenKind::RBrace: return "'}'";
        case TokenKind::LParen: return "'('";
        case TokenKind::RParen: return "')'";
        case TokenKind::LBracket: return "'['";
        case TokenKind::RBracket: return "']'";
        case TokenKind::Comma: return "','";
        case TokenKind::Colon: return "':'";
        case TokenKind::Semicolon: return "';'";
        case TokenKind::Dot: return "'.'";
        case TokenKind::Equal: return "'='";
        case TokenKind::Pipeline: return "'|>'";
        case TokenKind::EndOfFile: return "fine file";
    }
    return "token sconosciuto";
}

}  // namespace blackforge
