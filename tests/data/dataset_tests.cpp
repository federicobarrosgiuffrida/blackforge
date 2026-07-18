#include "blackforge/data/dataset.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

using namespace blackforge;

namespace {

// Wrapper RAII per un file temporaneo nella cartella TEMP di sistema
// (non nella directory del progetto: vedi la nota su Controlled Folder
// Access nei test di checkpoint).
struct TempFile {
    std::string path;

    explicit TempFile(const std::string& name)
        : path((std::filesystem::temp_directory_path() / name).string()) {}

    ~TempFile() { std::remove(path.c_str()); }
};

}  // namespace

TEST(DatasetTest, CostruttoreValidaLaQuantitaDiDati) {
    EXPECT_THROW(data::Dataset({4}, {2}, std::vector<float>(3), std::vector<float>(2 * 1), 1),
                 std::invalid_argument);
}

TEST(DatasetTest, BatchEstraeGliEsempiCorretti) {
    // 3 esempi, input a 2 feature, target a 1 feature.
    data::Dataset dataset({2}, {1}, {1.0F, 2.0F, 10.0F, 20.0F, 100.0F, 200.0F}, {1.0F, 2.0F, 3.0F}, 3);

    data::Dataset::Batch batch = dataset.batch(0, 2);
    EXPECT_EQ(batch.input.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ(batch.input.at(0), 1.0F);
    EXPECT_FLOAT_EQ(batch.input.at(1), 2.0F);
    EXPECT_FLOAT_EQ(batch.input.at(2), 10.0F);
    EXPECT_FLOAT_EQ(batch.input.at(3), 20.0F);

    EXPECT_EQ(batch.target.shape(), (std::vector<std::size_t>{2, 1}));
    EXPECT_FLOAT_EQ(batch.target.at(0), 1.0F);
    EXPECT_FLOAT_EQ(batch.target.at(1), 2.0F);
}

TEST(DatasetTest, BatchAvvolgeQuandoSuperaNumExamples) {
    data::Dataset dataset({1}, {1}, {1.0F, 2.0F, 3.0F}, {10.0F, 20.0F, 30.0F}, 3);

    // Partendo dall'esempio 2 (l'ultimo) con batchSize=2, il secondo
    // elemento del batch deve avvolgere all'esempio 0.
    data::Dataset::Batch batch = dataset.batch(2, 2);
    EXPECT_FLOAT_EQ(batch.input.at(0), 3.0F);  // esempio 2
    EXPECT_FLOAT_EQ(batch.input.at(1), 1.0F);  // esempio 0 (avvolto)
}

TEST(DatasetTest, ShuffleRiordinaGliEsempiPreservandoInputETarget) {
    // 5 esempi, ogni input[i] == target[i] (a meno di scala): dopo lo
    // shuffle, la corrispondenza deve restare intatta (l'esempio che
    // finisce in una data posizione deve avere sia l'input sia il
    // target dello STESSO esempio originale, non mescolati tra loro).
    data::Dataset dataset({1}, {1}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F}, {10.0F, 20.0F, 30.0F, 40.0F, 50.0F}, 5);
    dataset.shuffle(/*seed=*/42);

    data::Dataset::Batch batch = dataset.batch(0, 5);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(batch.target.at(i), batch.input.at(i) * 10.0F) << "indice " << i;
    }
}

TEST(DatasetTest, ShuffleCambiaDavveroLOrdine) {
    std::vector<float> original{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    data::Dataset dataset({1}, {1}, original, original, 8);
    dataset.shuffle(/*seed=*/7);

    data::Dataset::Batch batch = dataset.batch(0, 8);
    bool anyDifferent = false;
    for (std::size_t i = 0; i < 8; ++i) {
        if (batch.input.at(i) != original[i]) {
            anyDifferent = true;
            break;
        }
    }
    EXPECT_TRUE(anyDifferent) << "un rimescolamento di 8 esempi con seme fisso non dovrebbe restituire "
                                 "l'ordine originale (probabilita' trascurabile se il codice e' corretto)";
}

TEST(DatasetTest, ShuffleEDeterministaAParitaDiSeme) {
    std::vector<float> original{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    data::Dataset a({1}, {1}, original, original, 5);
    data::Dataset b({1}, {1}, original, original, 5);

    a.shuffle(99);
    b.shuffle(99);

    data::Dataset::Batch batchA = a.batch(0, 5);
    data::Dataset::Batch batchB = b.batch(0, 5);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(batchA.input.at(i), batchB.input.at(i)) << "indice " << i;
    }
}

TEST(DatasetTest, ShuffleNonVaInCrashConZeroOUnEsempio) {
    data::Dataset empty({1}, {1}, {}, {}, 0);
    EXPECT_NO_THROW(empty.shuffle(1));

    data::Dataset single({1}, {1}, {1.0F}, {10.0F}, 1);
    EXPECT_NO_THROW(single.shuffle(1));
}

TEST(DatasetTest, SalvaERicaricaCorrettamente) {
    TempFile file("blackforge_test_dataset_roundtrip.bfdata");

    std::vector<std::size_t> inputShape{2};
    std::vector<std::size_t> targetShape{1};
    std::vector<float> inputs{1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> targets{10.0F, 20.0F};

    data::saveDataset(file.path, inputShape, targetShape, inputs, targets, 2);
    data::Dataset loaded = data::loadDataset(file.path);

    EXPECT_EQ(loaded.numExamples(), 2u);
    EXPECT_EQ(loaded.inputExampleShape(), inputShape);
    EXPECT_EQ(loaded.targetExampleShape(), targetShape);

    data::Dataset::Batch batch = loaded.batch(0, 2);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        EXPECT_FLOAT_EQ(batch.input.at(i), inputs[i]);
    }
    for (std::size_t i = 0; i < targets.size(); ++i) {
        EXPECT_FLOAT_EQ(batch.target.at(i), targets[i]);
    }
}

TEST(DatasetTest, LoadLanciaSeIlFileNonEsiste) {
    EXPECT_THROW(data::loadDataset("blackforge_test_dataset_inesistente.bfdata"), std::runtime_error);
}

TEST(DatasetTest, LoadLanciaSeIlMagicNonECorretto) {
    TempFile file("blackforge_test_dataset_invalido.bfdata");
    {
        std::ofstream out(file.path, std::ios::binary);
        ASSERT_TRUE(out);
        out << "non e' un dataset BlackForge";
    }
    EXPECT_THROW(data::loadDataset(file.path), std::runtime_error);
}
