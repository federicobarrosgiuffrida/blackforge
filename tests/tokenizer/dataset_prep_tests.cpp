#include "blackforge/tokenizer/dataset_prep.hpp"

#include <cstdio>
#include <filesystem>

#include <gtest/gtest.h>

#include "blackforge/backend/cpu/loss.hpp"
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

TEST(DatasetPrepTest, CostruisceFinestreNonSovrapposteConTargetSparsoShiftByOne) {
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
    // Target sparso: un indice di classe per posizione, NON un vettore
    // one-hot [seqLen, vocabSize] — l'intero punto di questo formato.
    EXPECT_EQ(dataset.targetExampleShape(), (std::vector<std::size_t>{seqLen}));

    Dataset::Batch batch = dataset.batch(0, 3);

    // Finestra 0: input = "ABC" (id 65,66,67), target shift-by-one =
    // "BCD" (id 66,67,68), come indici sparsi.
    EXPECT_FLOAT_EQ(batch.input.at(0), 65.0F);
    EXPECT_FLOAT_EQ(batch.input.at(1), 66.0F);
    EXPECT_FLOAT_EQ(batch.input.at(2), 67.0F);
    EXPECT_FLOAT_EQ(batch.target.at(0), 66.0F);
    EXPECT_FLOAT_EQ(batch.target.at(1), 67.0F);
    EXPECT_FLOAT_EQ(batch.target.at(2), 68.0F);

    // Finestra 1 (stride == seqLen): input = "DEF" (id 68,69,70), target
    // = "EFG" (id 69,70,71).
    EXPECT_FLOAT_EQ(batch.input.at(seqLen + 0), 68.0F);
    EXPECT_FLOAT_EQ(batch.input.at(seqLen + 1), 69.0F);
    EXPECT_FLOAT_EQ(batch.input.at(seqLen + 2), 70.0F);
    EXPECT_FLOAT_EQ(batch.target.at(seqLen + 0), 69.0F);
    EXPECT_FLOAT_EQ(batch.target.at(seqLen + 1), 70.0F);
    EXPECT_FLOAT_EQ(batch.target.at(seqLen + 2), 71.0F);
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
    EXPECT_EQ(dataset.targetExampleShape(), (std::vector<std::size_t>{seqLen}));

    // Ogni indice di target deve cadere dentro il vocabolario.
    Dataset::Batch batch = dataset.batch(0, dataset.numExamples());
    std::size_t vocabSize = tok.vocabSize();
    for (std::size_t i = 0; i < dataset.numExamples() * seqLen; ++i) {
        float idx = batch.target.at(i);
        EXPECT_GE(idx, 0.0F) << "indice " << i;
        EXPECT_LT(idx, static_cast<float>(vocabSize)) << "indice " << i;
    }
}

TEST(DatasetPrepTest, IlDatasetProdottoSiAllenaConCrossEntropySparse) {
    // Verifica l'integrazione completa: il target sparso prodotto da
    // buildLanguageModelDataset() e' effettivamente utilizzabile da
    // backend::cpu::softmaxCrossEntropySparse (stessa forma, stessa
    // semantica di indice), non solo un dettaglio di formato isolato.
    Tokenizer tok;
    tok.train("aabbccdd aabbccdd aabbccdd aabbccdd", /*targetVocabSize=*/Tokenizer::kFirstMergeId + 5);

    TempFile file("blackforge_test_lm_dataset_ce_sparse.bfdata");
    std::size_t seqLen = 4;
    buildLanguageModelDataset(tok, "aabbccdd aabbccdd aabbccdd", seqLen, file.path);

    Dataset dataset = loadDataset(file.path);
    Dataset::Batch batch = dataset.batch(0, 1);

    // Logit fittizi [1, seqLen, vocabSize]: softmaxCrossEntropySparse
    // non deve lanciare eccezioni e deve produrre una loss finita.
    std::vector<std::size_t> logitsShape = {1, seqLen, tok.vocabSize()};
    std::size_t n = 1;
    for (std::size_t d : logitsShape) {
        n *= d;
    }
    blackforge::runtime::Tensor logits(logitsShape, std::vector<float>(n, 0.0F));

    EXPECT_NO_THROW({
        auto loss = blackforge::backend::cpu::softmaxCrossEntropySparse(logits, batch.target);
        EXPECT_GT(loss.value, 0.0F);
    });
}

TEST(DatasetPrepTest, MlmMascheraLePosizioniCoerentementeConLInputEIlTarget) {
    // Tokenizer senza merge: un id per byte, corpus prevedibile.
    Tokenizer tok;
    std::string corpus = "ABCDEFGHIJKLMNOP";  // 16 byte -> 16 id

    TempFile file("blackforge_test_mlm_dataset.bfdata");
    std::size_t seqLen = 4;
    // maskProb=1.0: OGNI posizione e' mascherata, deterministico e
    // facile da verificare esaustivamente.
    std::size_t numExamples =
        buildMaskedLanguageModelDataset(tok, corpus, seqLen, /*maskProb=*/1.0F, /*seed=*/1, file.path);
    EXPECT_EQ(numExamples, 4u);  // 16 / 4

    Dataset dataset = loadDataset(file.path);
    EXPECT_EQ(dataset.inputExampleShape(), (std::vector<std::size_t>{seqLen}));
    EXPECT_EQ(dataset.targetExampleShape(), (std::vector<std::size_t>{seqLen}));

    Dataset::Batch batch = dataset.batch(0, numExamples);
    for (std::size_t i = 0; i < numExamples * seqLen; ++i) {
        // Ogni posizione e' mascherata: l'input deve essere kPadId, e il
        // target deve essere il token ORIGINALE (id 65..80, 'A'..'P').
        EXPECT_FLOAT_EQ(batch.input.at(i), static_cast<float>(Tokenizer::kPadId)) << "posizione " << i;
        EXPECT_FLOAT_EQ(batch.target.at(i), 65.0F + static_cast<float>(i)) << "posizione " << i;
    }
}

TEST(DatasetPrepTest, MlmNonMascheraNienteConMaskProbVicinoAZero) {
    Tokenizer tok;
    std::string corpus = "ABCDEFGHIJKLMNOP";

    TempFile file("blackforge_test_mlm_dataset_nomask.bfdata");
    std::size_t seqLen = 4;
    // maskProb minuscolo (non zero, il validatore lo rifiuterebbe):
    // con un seme fissato e' comunque possibile (anche se improbabile)
    // che qualche posizione venga mascherata per caso, quindi il test
    // verifica solo la proprieta' strutturale (input == token originale
    // OPPURE input == kPadId con target coerente), non "zero maschere".
    buildMaskedLanguageModelDataset(tok, corpus, seqLen, /*maskProb=*/1e-6F, /*seed=*/7, file.path);

    Dataset dataset = loadDataset(file.path);
    Dataset::Batch batch = dataset.batch(0, dataset.numExamples());
    for (std::size_t i = 0; i < dataset.numExamples() * seqLen; ++i) {
        float input = batch.input.at(i);
        float target = batch.target.at(i);
        if (input == static_cast<float>(Tokenizer::kPadId)) {
            EXPECT_NE(target, -1.0F) << "posizione mascherata " << i << " ha target ignorato";
        } else {
            EXPECT_FLOAT_EQ(target, -1.0F) << "posizione non mascherata " << i << " ha un target != -1";
        }
    }
}

TEST(DatasetPrepTest, MlmLanciaSeMaskProbFuoriRange) {
    Tokenizer tok;
    TempFile file("blackforge_test_mlm_dataset_badprob.bfdata");
    EXPECT_THROW(buildMaskedLanguageModelDataset(tok, "ABCDEFGH", 4, /*maskProb=*/0.0F, 1, file.path),
                 std::invalid_argument);
    EXPECT_THROW(buildMaskedLanguageModelDataset(tok, "ABCDEFGH", 4, /*maskProb=*/1.5F, 1, file.path),
                 std::invalid_argument);
}

TEST(DatasetPrepTest, MlmLanciaSeIlCorpusENonAbbastanzaLungoPerUnaFinestra) {
    Tokenizer tok;
    TempFile file("blackforge_test_mlm_dataset_short.bfdata");
    EXPECT_THROW(buildMaskedLanguageModelDataset(tok, "AB", /*seqLen=*/5, /*maskProb=*/0.5F, 1, file.path),
                 std::invalid_argument);
}

TEST(DatasetPrepTest, IlDatasetMlmProdottoSiAllenaConCrossEntropyMasked) {
    Tokenizer tok;
    tok.train("aabbccdd aabbccdd aabbccdd aabbccdd", /*targetVocabSize=*/Tokenizer::kFirstMergeId + 5);

    TempFile file("blackforge_test_mlm_dataset_ce_masked.bfdata");
    std::size_t seqLen = 4;
    buildMaskedLanguageModelDataset(tok, "aabbccdd aabbccdd aabbccdd", seqLen, /*maskProb=*/0.3F, /*seed=*/3,
                                     file.path);

    Dataset dataset = loadDataset(file.path);
    Dataset::Batch batch = dataset.batch(0, 1);

    std::vector<std::size_t> logitsShape = {1, seqLen, tok.vocabSize()};
    std::size_t n = 1;
    for (std::size_t d : logitsShape) {
        n *= d;
    }
    blackforge::runtime::Tensor logits(logitsShape, std::vector<float>(n, 0.0F));

    EXPECT_NO_THROW({ (void)blackforge::backend::cpu::softmaxCrossEntropyMasked(logits, batch.target); });
}
