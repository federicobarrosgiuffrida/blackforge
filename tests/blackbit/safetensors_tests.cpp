#include "blackforge/blackbit/safetensors.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

using namespace blackforge;
using blackforge::blackbit::SafetensorsFile;
using blackforge::blackbit::SafetensorsModel;

namespace {

std::uint16_t floatToBf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);  // troncamento, non arrotondamento
}

// Cartella temporanea che si cancella da sola.
class TempDir {
public:
    explicit TempDir(const std::string& name)
        : path_(std::filesystem::temp_directory_path() / ("bfst_" + name)) {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~TempDir() { std::filesystem::remove_all(path_); }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] std::string file(const std::string& name) const { return (path_ / name).string(); }
    [[nodiscard]] std::string path() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

// Scrive un file safetensors valido a partire da un'intestazione JSON e
// dai byte dei dati.
void writeSafetensors(const std::string& path, const std::string& header, const std::vector<unsigned char>& data) {
    std::ofstream out(path, std::ios::binary);
    const std::uint64_t length = header.size();
    out.write(reinterpret_cast<const char*>(&length), sizeof(length));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

// Un file con un solo tensore BF16 [rows, cols] di valori noti.
void writeBf16Tensor(const std::string& path, const std::string& name, std::size_t rows, std::size_t cols,
                      const std::vector<float>& values) {
    std::vector<unsigned char> data(rows * cols * 2);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::uint16_t bits = floatToBf16(values[i]);
        std::memcpy(data.data() + i * 2, &bits, 2);
    }
    const std::string header = "{\"" + name + "\":{\"dtype\":\"BF16\",\"shape\":[" + std::to_string(rows) + "," +
                                std::to_string(cols) + "],\"data_offsets\":[0," + std::to_string(data.size()) +
                                "]}}";
    writeSafetensors(path, header, data);
}

}  // namespace

TEST(SafetensorsTest, Bf16SonoISediciBitAltiDiUnFloat32) {
    // La conversione BF16 -> FP32 non perde NULLA: BF16 e' letteralmente
    // la meta' alta di un float32. Quello che si perde e' nel senso
    // opposto (FP32 -> BF16), ed e' esattamente la mantissa bassa.
    //
    // Quindi il round-trip e' esatto solo per i valori che stanno gia'
    // negli 8 bit di mantissa di BF16.
    for (float value : {0.0F, 1.0F, -1.0F, 0.5F, -0.25F, 3.5F, 256.0F, -0.001953125F}) {
        EXPECT_FLOAT_EQ(blackbit::bf16ToFloat(floatToBf16(value)), value) << "valore " << value;
    }

    // Per gli altri il troncamento costa al massimo 2^-7 relativo:
    // BF16 ha SETTE bit espliciti di mantissa (1 segno + 8 esponente +
    // 7 mantissa), quindi troncare azzera tutto sotto 2^-7 rispetto al
    // bit implicito. Il limite vale per qualunque ordine di grandezza,
    // che e' il punto: BF16 ha lo stesso esponente di FP32, quindi la
    // conversione non degrada agli estremi come farebbe FP16.
    for (float value : {1.2345678F, 1e-10F, 1e10F, -7.77e-30F, 3.3e30F}) {
        const float back = blackbit::bf16ToFloat(floatToBf16(value));
        EXPECT_NEAR(back / value, 1.0F, 1.0F / 128.0F) << "valore " << value;
    }

    EXPECT_TRUE(std::isnan(blackbit::bf16ToFloat(0x7FC0)));
    EXPECT_TRUE(std::isinf(blackbit::bf16ToFloat(0x7F80)));
    EXPECT_LT(blackbit::bf16ToFloat(0xFF80), 0.0F);
}

TEST(SafetensorsTest, Fp16GestisceNormaliSubnormaliEInfiniti) {
    EXPECT_FLOAT_EQ(blackbit::fp16ToFloat(0x0000), 0.0F);
    EXPECT_FLOAT_EQ(blackbit::fp16ToFloat(0x3C00), 1.0F);
    EXPECT_FLOAT_EQ(blackbit::fp16ToFloat(0xBC00), -1.0F);
    EXPECT_FLOAT_EQ(blackbit::fp16ToFloat(0x4000), 2.0F);
    EXPECT_FLOAT_EQ(blackbit::fp16ToFloat(0x3800), 0.5F);
    // Massimo normale di FP16.
    EXPECT_FLOAT_EQ(blackbit::fp16ToFloat(0x7BFF), 65504.0F);
    // Il subnormale piu' piccolo: 2^-24. E' il caso che una conversione
    // ingenua sbaglia in silenzio.
    EXPECT_FLOAT_EQ(blackbit::fp16ToFloat(0x0001), std::ldexp(1.0F, -24));
    // Il subnormale piu' grande.
    EXPECT_FLOAT_EQ(blackbit::fp16ToFloat(0x03FF), std::ldexp(1023.0F, -24));
    EXPECT_TRUE(std::isinf(blackbit::fp16ToFloat(0x7C00)));
    EXPECT_TRUE(std::isnan(blackbit::fp16ToFloat(0x7E00)));
}

TEST(SafetensorsTest, LeggeUnFileConUnSoloTensore) {
    TempDir dir("single");
    const std::string path = dir.file("model.safetensors");

    std::vector<float> values(3 * 4);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i) * 0.5F;  // rappresentabili esattamente in BF16
    }
    writeBf16Tensor(path, "peso", 3, 4, values);

    SafetensorsFile file(path);
    ASSERT_EQ(file.tensors().size(), 1U);
    const auto* entry = file.find("peso");
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->dtype, blackbit::SafetensorsDType::BF16);
    EXPECT_EQ(entry->shape, (std::vector<std::size_t>{3, 4}));
    EXPECT_EQ(entry->rows(), 3U);
    EXPECT_EQ(entry->rowLength(), 4U);

    const runtime::Tensor loaded = file.readFloat("peso");
    ASSERT_EQ(loaded.elementCount(), values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_FLOAT_EQ(loaded.at(i), values[i]) << "elemento " << i;
    }
}

TEST(SafetensorsTest, LaLetturaABlocchiDiRigheCoincideConQuellaCompleta) {
    // E' la garanzia su cui poggia l'import: leggere a blocchi non deve
    // dare risultati diversi dal leggere tutto, e non deve allocare
    // l'intero tensore.
    TempDir dir("rows");
    const std::string path = dir.file("model.safetensors");

    constexpr std::size_t kRows = 17;
    constexpr std::size_t kCols = 5;
    std::vector<float> values(kRows * kCols);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(static_cast<int>(i) - 40) * 0.25F;
    }
    writeBf16Tensor(path, "grande", kRows, kCols, values);

    SafetensorsFile file(path);
    const runtime::Tensor full = file.readFloat("grande");

    std::vector<float> block(4 * kCols);
    for (std::size_t first = 0; first < kRows; first += 4) {
        const std::size_t count = std::min<std::size_t>(4, kRows - first);
        file.readFloatRows("grande", first, count, block.data());
        for (std::size_t r = 0; r < count; ++r) {
            for (std::size_t c = 0; c < kCols; ++c) {
                ASSERT_FLOAT_EQ(block[r * kCols + c], full.at((first + r) * kCols + c))
                    << "riga " << (first + r) << " colonna " << c;
            }
        }
    }
}

TEST(SafetensorsTest, RifiutaFileMalformati) {
    TempDir dir("bad");

    // File troppo corto.
    {
        const std::string path = dir.file("corto.safetensors");
        std::ofstream out(path, std::ios::binary);
        out << "abc";
    }
    EXPECT_THROW(SafetensorsFile(dir.file("corto.safetensors")), std::runtime_error);

    // Lunghezza dell'intestazione assurda.
    {
        const std::string path = dir.file("lungo.safetensors");
        std::ofstream out(path, std::ios::binary);
        const std::uint64_t length = 1ULL << 40;
        out.write(reinterpret_cast<const char*>(&length), sizeof(length));
        out << "{}";
    }
    EXPECT_THROW(SafetensorsFile(dir.file("lungo.safetensors")), std::runtime_error);

    // Offset incoerenti con la forma dichiarata: 2x3 BF16 sono 12 byte,
    // non 8. E' il controllo che intercetta un file troncato PRIMA di
    // un import da mezz'ora.
    {
        const std::string header = R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,8]}})";
        writeSafetensors(dir.file("offset.safetensors"), header, std::vector<unsigned char>(8));
    }
    EXPECT_THROW(SafetensorsFile(dir.file("offset.safetensors")), std::runtime_error);

    // Tensore che finisce oltre la fine del file.
    {
        const std::string header = R"({"w":{"dtype":"F32","shape":[100],"data_offsets":[0,400]}})";
        writeSafetensors(dir.file("troncato.safetensors"), header, std::vector<unsigned char>(40));
    }
    EXPECT_THROW(SafetensorsFile(dir.file("troncato.safetensors")), std::runtime_error);

    // dtype sconosciuto.
    {
        const std::string header = R"({"w":{"dtype":"FP8","shape":[4],"data_offsets":[0,4]}})";
        writeSafetensors(dir.file("dtype.safetensors"), header, std::vector<unsigned char>(4));
    }
    EXPECT_THROW(SafetensorsFile(dir.file("dtype.safetensors")), std::runtime_error);

    EXPECT_THROW(SafetensorsFile(dir.file("inesistente.safetensors")), std::runtime_error);
}

TEST(SafetensorsTest, LeggeIMetadatiEIgnoraICampiSconosciuti) {
    TempDir dir("meta");
    const std::string path = dir.file("model.safetensors");

    // '__metadata__' e un campo extra che il lettore non modella: deve
    // saltarli senza rompersi, perche' i produttori ne aggiungono di
    // nuovi nel tempo.
    const std::string header =
        R"({"__metadata__":{"format":"pt","autore":"prova"},)"
        R"("w":{"dtype":"F32","shape":[2],"data_offsets":[0,8],"extra":{"annidato":[1,2,3]}}})";
    std::vector<unsigned char> data(8);
    const float values[2] = {1.5F, -2.5F};
    std::memcpy(data.data(), values, 8);
    writeSafetensors(path, header, data);

    SafetensorsFile file(path);
    EXPECT_EQ(file.metadata().at("format"), "pt");
    EXPECT_EQ(file.metadata().at("autore"), "prova");
    const runtime::Tensor loaded = file.readFloat("w");
    EXPECT_FLOAT_EQ(loaded.at(0), 1.5F);
    EXPECT_FLOAT_EQ(loaded.at(1), -2.5F);
}

TEST(SafetensorsTest, ModelloDivisoInPiuShard) {
    TempDir dir("shard");
    writeBf16Tensor(dir.file("model-00001-of-00002.safetensors"), "a", 2, 2, {1.0F, 2.0F, 3.0F, 4.0F});
    writeBf16Tensor(dir.file("model-00002-of-00002.safetensors"), "b", 1, 3, {5.0F, 6.0F, 7.0F});
    {
        std::ofstream out(dir.file("model.safetensors.index.json"));
        out << R"({"metadata":{"total_size":22},"weight_map":{)"
            << R"("a":"model-00001-of-00002.safetensors",)"
            << R"("b":"model-00002-of-00002.safetensors"}})";
    }

    SafetensorsModel model(dir.path());
    EXPECT_EQ(model.fileCount(), 2U);
    EXPECT_EQ(model.names().size(), 2U);
    EXPECT_TRUE(model.has("a"));
    EXPECT_TRUE(model.has("b"));
    EXPECT_FALSE(model.has("c"));

    EXPECT_FLOAT_EQ(model.readFloat("a").at(3), 4.0F);
    EXPECT_FLOAT_EQ(model.readFloat("b").at(2), 7.0F);
    EXPECT_EQ(model.entry("b").shape, (std::vector<std::size_t>{1, 3}));
    EXPECT_THROW((void)model.readFloat("c"), std::runtime_error);
}

TEST(SafetensorsTest, ModelloAFileUnicoSenzaIndice) {
    TempDir dir("nolist");
    writeBf16Tensor(dir.file("model.safetensors"), "solo", 2, 2, {1.0F, 2.0F, 3.0F, 4.0F});

    SafetensorsModel model(dir.path());
    EXPECT_EQ(model.fileCount(), 1U);
    EXPECT_TRUE(model.has("solo"));
}

TEST(SafetensorsTest, UnaCartellaSenzaModelloEUnErroreChiaro) {
    TempDir dir("vuota");
    try {
        SafetensorsModel model(dir.path());
        FAIL() << "avrebbe dovuto rifiutare una cartella senza modello";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("model.safetensors"), std::string::npos);
    }
}

TEST(SafetensorsTest, UnIndiceCheCitaUnoShardMancanteFallisceSubito) {
    TempDir dir("mancante");
    {
        std::ofstream out(dir.file("model.safetensors.index.json"));
        out << R"({"weight_map":{"a":"non-esiste.safetensors"}})";
    }
    EXPECT_THROW(SafetensorsModel(dir.path()), std::runtime_error);
}
