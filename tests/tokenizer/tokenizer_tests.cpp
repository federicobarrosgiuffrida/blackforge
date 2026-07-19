#include "blackforge/tokenizer/tokenizer.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

using namespace blackforge::tokenizer;

namespace {

struct TempFile {
    std::string path;

    explicit TempFile(const std::string& name) : path((std::filesystem::temp_directory_path() / name).string()) {}

    ~TempFile() { std::remove(path.c_str()); }
};

}  // namespace

TEST(TokenizerTest, TokenizerVuotoHaSoloIlVocabolarioDiBase) {
    Tokenizer tok;
    EXPECT_EQ(tok.vocabSize(), Tokenizer::kFirstMergeId);
    EXPECT_TRUE(tok.merges().empty());
}

TEST(TokenizerTest, SenzaMergeEncodeProduceUnIdPerByte) {
    Tokenizer tok;
    std::vector<std::uint32_t> ids = tok.encode("AB");
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], static_cast<std::uint32_t>('A'));
    EXPECT_EQ(ids[1], static_cast<std::uint32_t>('B'));
}

TEST(TokenizerTest, DecodeDiIdDiByteRicostruisceIlTesto) {
    Tokenizer tok;
    std::string text = "Hello, World! 123";
    std::vector<std::uint32_t> ids = tok.encode(text);
    EXPECT_EQ(tok.decode(ids), text);
}

TEST(TokenizerTest, TrainApprendeIMergePiuFrequenti) {
    // "ab" e' la coppia adiacente piu' frequente nel corpus (compare in
    // ogni occorrenza di "ababab..."): il primo merge appreso deve
    // essere esattamente 'a'+'b'.
    Tokenizer tok;
    tok.train("ababababab ababababab", /*targetVocabSize=*/Tokenizer::kFirstMergeId + 1);

    ASSERT_EQ(tok.merges().size(), 1u);
    EXPECT_EQ(tok.merges().front().left, static_cast<std::uint32_t>('a'));
    EXPECT_EQ(tok.merges().front().right, static_cast<std::uint32_t>('b'));
}

TEST(TokenizerTest, TrainSiFermaSeIlCorpusSiEsaurisce) {
    // Un corpus di 4 byte distinti tutti diversi non ha alcuna coppia
    // adiacente ripetuta: nessun merge e' possibile, targetVocabSize
    // resta irraggiungibile ma train() non deve lanciare eccezioni.
    Tokenizer tok;
    tok.train("wxyz", /*targetVocabSize=*/Tokenizer::kFirstMergeId + 100);
    EXPECT_LT(tok.merges().size(), 100u);
}

TEST(TokenizerTest, TrainLanciaSeTargetVocabSizeENonRaggiungibile) {
    Tokenizer tok;
    EXPECT_THROW(tok.train("qualunque cosa", /*targetVocabSize=*/Tokenizer::kFirstMergeId), std::invalid_argument);
    EXPECT_THROW(tok.train("qualunque cosa", /*targetVocabSize=*/10), std::invalid_argument);
}

TEST(TokenizerTest, EncodeDecodeRoundTripDopoIlTrainingSuTestoVario) {
    // Un corpus con ripetizione sufficiente da produrre merge reali,
    // poi verifica che qualunque testo (anche fuori dal corpus di
    // training, incluso testo non visto) sopravviva a un round-trip
    // encode->decode esatto: e' la proprieta' fondamentale di un
    // tokenizer byte-level (nessuna normalizzazione, nessun <unk>).
    Tokenizer tok;
    std::string corpus =
        "the quick brown fox jumps over the lazy dog. "
        "the dog barks at the fox. the fox runs away quickly. "
        "quick thinking saved the day, the quick fox thought.";
    tok.train(corpus, /*targetVocabSize=*/Tokenizer::kFirstMergeId + 40);

    EXPECT_GT(tok.merges().size(), 0u);

    for (const std::string& text : {std::string("the quick fox"), std::string("completely unseen text!"),
                                     std::string(""), std::string("a"), corpus}) {
        std::vector<std::uint32_t> ids = tok.encode(text);
        EXPECT_EQ(tok.decode(ids), text) << "round-trip fallito per: " << text;
    }
}

TEST(TokenizerTest, EncodeDecodeRoundTripSuTestoUtf8MultiByte) {
    // Caratteri accentati/non-ASCII (UTF-8 multi-byte): verifica che la
    // pre-tokenizzazione (che tratta ogni byte >= 0x80 come "word") non
    // spezzi le sequenze multi-byte in modo da romperne il round-trip.
    Tokenizer tok;
    tok.train("caffè caffè caffè città città naïve naïve naïve", /*targetVocabSize=*/Tokenizer::kFirstMergeId + 10);

    std::string text = "una città con un caffè naïve";
    std::vector<std::uint32_t> ids = tok.encode(text);
    EXPECT_EQ(tok.decode(ids), text);
}

TEST(TokenizerTest, MergeAppresoRiduceLaLunghezzaDellaSequenza) {
    // Con un merge appreso su una coppia ripetuta, encode() deve
    // produrre MENO id di una tokenizzazione byte-per-byte pura: e'
    // l'intero punto di BPE (compressione della sequenza).
    Tokenizer trained;
    trained.train("aaaaaaaaaaaaaaaaaaaa", /*targetVocabSize=*/Tokenizer::kFirstMergeId + 5);

    Tokenizer untrained;
    std::string text = "aaaaaaaaaaaaaaaaaaaa";
    EXPECT_LT(trained.encode(text).size(), untrained.encode(text).size());
}

TEST(TokenizerTest, TokenSpecialiRestanoFuoriDalRangeDeiByteEHannoIdFissi) {
    EXPECT_EQ(Tokenizer::kPadId, 256u);
    EXPECT_EQ(Tokenizer::kBosId, 257u);
    EXPECT_EQ(Tokenizer::kEosId, 258u);
    EXPECT_EQ(Tokenizer::kFirstMergeId, 259u);

    // decode() su un id di token speciale non produce byte (nessuna
    // rappresentazione testuale), ma non deve lanciare eccezioni.
    Tokenizer tok;
    std::string decoded = tok.decode({Tokenizer::kBosId, static_cast<std::uint32_t>('x'), Tokenizer::kEosId});
    EXPECT_EQ(decoded, "x");
}

TEST(TokenizerTest, SaveELoadRicostruisconoUnTokenizerEquivalente) {
    Tokenizer tok;
    tok.train(
        "the quick brown fox jumps over the lazy dog repeatedly, the fox and the dog become good friends "
        "eventually, quick foxes and lazy dogs coexist peacefully in this small testing corpus",
        /*targetVocabSize=*/Tokenizer::kFirstMergeId + 30);

    TempFile file("blackforge_test_tokenizer.bftok");
    saveTokenizer(tok, file.path);
    Tokenizer loaded = loadTokenizer(file.path);

    ASSERT_EQ(loaded.merges().size(), tok.merges().size());
    for (std::size_t i = 0; i < tok.merges().size(); ++i) {
        EXPECT_EQ(loaded.merges()[i].left, tok.merges()[i].left) << "merge " << i;
        EXPECT_EQ(loaded.merges()[i].right, tok.merges()[i].right) << "merge " << i;
        EXPECT_EQ(loaded.merges()[i].resultId, tok.merges()[i].resultId) << "merge " << i;
    }

    std::string text = "the quick fox and the lazy dog";
    EXPECT_EQ(loaded.encode(text), tok.encode(text));
    EXPECT_EQ(loaded.decode(tok.encode(text)), text);
}

TEST(TokenizerTest, LoadLanciaSeIlFileNonEUnTokenizerValido) {
    TempFile file("blackforge_test_not_a_tokenizer.bftok");
    {
        std::ofstream out(file.path, std::ios::binary);
        out << "not a real tokenizer file";
    }
    EXPECT_THROW(loadTokenizer(file.path), std::runtime_error);
}

TEST(TokenizerTest, LoadLanciaSeIlFileNonEsiste) {
    EXPECT_THROW(loadTokenizer("blackforge_percorso_inesistente.bftok"), std::runtime_error);
}
