#include "blackforge/tokenizer/dataset_prep.hpp"

#include <cstdio>
#include <filesystem>

#include <gtest/gtest.h>

#include "blackforge/data/dataset.hpp"

using namespace blackforge::tokenizer;
using blackforge::data::Dataset;
using blackforge::data::loadDataset;

namespace {

struct TempFile {
    std::string path;

    explicit TempFile(const std::string& name) : path((std::filesystem::temp_directory_path() / name).string()) {}

    ~TempFile() { std::remove(path.c_str()); }
};

}  // namespace

TEST(DatasetPrepTest, CostruisceFinestreNonSovrapposteConTargetOneHotShiftByOne) {
    // Tokenizer senza merge: un id per byte, cosi' il corpus tokenizzato
    // e' prevedibile byte per byte, facile da verificare a mano.
    Tokenizer tok;
    std::string corpus = "ABCDEFGHIJ";  // 10 byte -> 10 id: 65,66,...,74

    TempFile file("blackforge_test_lm_dataset.bfdata");
    std::size_t seqLen = 3;
    std::size_t numExamples = buildLanguageModelDataset(tok, corpus, seqLen, file.path);

    // (10 - 1) / 3 = 3 finestre complete (l'ultimo byte 'J' resta
    // scartato, non basta per completare una quarta finestra).
    EXPECT_EQ(numExamples, 3u);

    Dataset dataset = loadDataset(file.path);
    ASSERT_EQ(dataset.numExamples(), 3u);
    EXPECT_EQ(dataset.inputExampleShape(), (std::vector<std::size_t>{seqLen}));
    EXPECT_EQ(dataset.targetExampleShape(), (std::vector<std::size_t>{seqLen, tok.vocabSize()}));

    Dataset::Batch batch = dataset.batch(0, 3);

    // Finestra 0: input = "ABC" (id 65,66,67), target shift-by-one =
    // "BCD" (id 66,67,68) one-hot.
    EXPECT_FLOAT_EQ(batch.input.at(0), 65.0F);
    EXPECT_FLOAT_EQ(batch.input.at(1), 66.0F);
    EXPECT_FLOAT_EQ(batch.input.at(2), 67.0F);

    std::size_t vocabSize = tok.vocabSize();
    // target[posizione 0] deve avere massa 1 esattamente sull'id 66 ('B').
    for (std::size_t c = 0; c < vocabSize; ++c) {
        float expected = (c == 66) ? 1.0F : 0.0F;
        EXPECT_FLOAT_EQ(batch.target.at(c), expected) << "colonna " << c;
    }
    // target[posizione 2] deve avere massa 1 esattamente sull'id 68 ('D').
    for (std::size_t c = 0; c < vocabSize; ++c) {
        float expected = (c == 68) ? 1.0F : 0.0F;
        EXPECT_FLOAT_EQ(batch.target.at(2 * vocabSize + c), expected) << "colonna " << c;
    }

    // Finestra 1 (stride == seqLen): input = "DEF" (id 68,69,70).
    EXPECT_FLOAT_EQ(batch.input.at(seqLen + 0), 68.0F);
    EXPECT_FLOAT_EQ(batch.input.at(seqLen + 1), 69.0F);
    EXPECT_FLOAT_EQ(batch.input.at(seqLen + 2), 70.0F);
}

TEST(DatasetPrepTest, LanciaSeIlCorpusENonAbbastanzaLungoPerUnaFinestra) {
    Tokenizer tok;
    TempFile file("blackforge_test_lm_dataset_too_short.bfdata");
    EXPECT_THROW(buildLanguageModelDataset(tok, "AB", /*seqLen=*/5, file.path), std::invalid_argument);
}

TEST(DatasetPrepTest, FunzionaConUnTokenizerAddestratoConMerge) {
    Tokenizer tok;
    tok.train("the quick brown fox jumps over the lazy dog many many times over and over",
               /*targetVocabSize=*/Tokenizer::kFirstMergeId + 20);

    TempFile file("blackforge_test_lm_dataset_trained.bfdata");
    std::size_t seqLen = 4;
    std::size_t numExamples =
        buildLanguageModelDataset(tok, "the quick brown fox jumps over the lazy dog", seqLen, file.path);
    EXPECT_GT(numExamples, 0u);

    Dataset dataset = loadDataset(file.path);
    EXPECT_EQ(dataset.targetExampleShape(), (std::vector<std::size_t>{seqLen, tok.vocabSize()}));

    // Ogni riga di target deve essere un one-hot valido (somma esattamente 1).
    Dataset::Batch batch = dataset.batch(0, dataset.numExamples());
    std::size_t vocabSize = tok.vocabSize();
    for (std::size_t row = 0; row < dataset.numExamples() * seqLen; ++row) {
        float sum = 0.0F;
        for (std::size_t c = 0; c < vocabSize; ++c) {
            sum += batch.target.at(row * vocabSize + c);
        }
        EXPECT_FLOAT_EQ(sum, 1.0F) << "riga " << row;
    }
}
