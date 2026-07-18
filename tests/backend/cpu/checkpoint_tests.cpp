#include "blackforge/backend/cpu/checkpoint.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "blackforge/backend/cpu/model.hpp"
#include "blackforge/frontend/lexer.hpp"
#include "blackforge/frontend/parser.hpp"
#include "blackforge/ir/ir_builder.hpp"

using namespace blackforge;

namespace {

ir::Module buildModule(const std::string& source) {
    Lexer lexer(source, "test.bf");
    auto tokens = lexer.tokenize();
    EXPECT_FALSE(lexer.diagnostics().hasErrors());

    Parser parser(std::move(tokens));
    ast::Program program = parser.parseProgram();
    EXPECT_FALSE(parser.diagnostics().hasErrors());

    ir::IRBuilder builder;
    ir::Module module = builder.build(program);
    EXPECT_FALSE(builder.diagnostics().hasErrors());
    return module;
}

// Wrapper RAII per un file temporaneo nella cartella TEMP di sistema
// (non nella directory del progetto: su Windows, funzionalita' come
// Controlled Folder Access possono impedire a un eseguibile non
// riconosciuto di creare file dentro 'Documents'). Rimuove il file
// anche se un'asserzione fallisce a meta' test.
struct TempFile {
    std::string path;

    explicit TempFile(const std::string& name)
        : path((std::filesystem::temp_directory_path() / name).string()) {}

    ~TempFile() { std::remove(path.c_str()); }
};

}  // namespace

TEST(CheckpointTest, SalvaERicaricaIPesiCorrettamente) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[batch, 4]\n"
        "    input |> linear(3) |> silu |> linear(2)\n"
        "}\n");

    TempFile file("blackforge_test_checkpoint_roundtrip.bfckpt");

    backend::cpu::Model original(module.models.front(), /*seed=*/123);
    backend::cpu::saveCheckpoint(original, file.path);

    backend::cpu::Model reloaded(module.models.front(), /*seed=*/999);  // pesi iniziali diversi
    backend::cpu::loadCheckpoint(reloaded, file.path);

    auto originalParams = original.parameters();
    auto reloadedParams = reloaded.parameters();
    ASSERT_EQ(originalParams.size(), reloadedParams.size());

    for (std::size_t i = 0; i < originalParams.size(); ++i) {
        EXPECT_EQ(originalParams[i]->name, reloadedParams[i]->name);
        ASSERT_EQ(originalParams[i]->value.elementCount(), reloadedParams[i]->value.elementCount());
        for (std::size_t j = 0; j < originalParams[i]->value.elementCount(); ++j) {
            EXPECT_FLOAT_EQ(originalParams[i]->value.at(j), reloadedParams[i]->value.at(j));
        }
    }
}

TEST(CheckpointTest, LoadLanciaSeIlFileNonEsiste) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[4]\n"
        "    input |> linear(2)\n"
        "}\n");
    backend::cpu::Model model(module.models.front());

    TempFile file("blackforge_test_file_inesistente.bfckpt");
    std::remove(file.path.c_str());  // garantisce che non esista davvero

    EXPECT_THROW(backend::cpu::loadCheckpoint(model, file.path), std::runtime_error);
}

TEST(CheckpointTest, LoadLanciaSeIlMagicNonECorretto) {
    ir::Module module = buildModule(
        "model M {\n"
        "    input bf16[4]\n"
        "    input |> linear(2)\n"
        "}\n");
    backend::cpu::Model model(module.models.front());

    TempFile file("blackforge_test_checkpoint_invalido.bfckpt");
    {
        std::ofstream out(file.path, std::ios::binary);
        ASSERT_TRUE(out) << "impossibile creare il file di test '" << file.path << "'";
        out << "questo non e' un checkpoint BlackForge";
    }

    EXPECT_THROW(backend::cpu::loadCheckpoint(model, file.path), std::runtime_error);
}
