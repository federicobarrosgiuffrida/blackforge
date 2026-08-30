#include "blackforge/blackbit/ternary.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <sstream>

using namespace blackforge;
using blackforge::blackbit::TernaryTensor;

namespace {

// Trit deterministici e riproducibili in {-1, 0, +1}.
std::vector<std::int8_t> pseudoTrits(std::size_t count, unsigned int seed) {
    std::mt19937 rng(seed);
    std::vector<std::int8_t> trits(count);
    for (std::size_t i = 0; i < count; ++i) {
        trits[i] = static_cast<std::int8_t>(static_cast<int>(rng() % 3) - 1);
    }
    return trits;
}

std::vector<float> positiveScales(std::size_t count, unsigned int seed) {
    std::mt19937 rng(seed);
    std::vector<float> scales(count);
    for (std::size_t i = 0; i < count; ++i) {
        scales[i] = 0.01F + static_cast<float>(rng() % 1000) / 1000.0F;
    }
    return scales;
}

}  // namespace

TEST(TernaryCodecTest, CodificaEDecodificaTutteLe243Combinazioni) {
    // Il codice posizionale in base 3 deve essere una biiezione fra le
    // 3^5 = 243 cinquine di trit e i byte [0, 243): se due cinquine
    // collidessero, l'impacchettamento perderebbe informazione in modo
    // silenzioso.
    std::vector<bool> seen(256, false);
    int produced = 0;

    for (int a = -1; a <= 1; ++a) {
        for (int b = -1; b <= 1; ++b) {
            for (int c = -1; c <= 1; ++c) {
                for (int d = -1; d <= 1; ++d) {
                    for (int e = -1; e <= 1; ++e) {
                        const std::uint8_t code = blackbit::encodeTritByte(a, b, c, d, e);
                        ASSERT_LT(code, 243) << "codice fuori dall'intervallo previsto";
                        ASSERT_FALSE(seen[code]) << "due cinquine diverse producono lo stesso byte";
                        seen[code] = true;
                        ++produced;

                        int out[blackbit::kTritsPerByte];
                        blackbit::decodeTritByte(code, out);
                        EXPECT_EQ(out[0], a);
                        EXPECT_EQ(out[1], b);
                        EXPECT_EQ(out[2], c);
                        EXPECT_EQ(out[3], d);
                        EXPECT_EQ(out[4], e);

                        EXPECT_EQ(blackbit::decodeTritAt(code, 0), a);
                        EXPECT_EQ(blackbit::decodeTritAt(code, 1), b);
                        EXPECT_EQ(blackbit::decodeTritAt(code, 2), c);
                        EXPECT_EQ(blackbit::decodeTritAt(code, 3), d);
                        EXPECT_EQ(blackbit::decodeTritAt(code, 4), e);
                    }
                }
            }
        }
    }

    EXPECT_EQ(produced, 243);
}

TEST(TernaryCodecTest, ByteDiUnaParolaSonoIndipendenti) {
    std::uint32_t word = 0;
    word = blackbit::setWordByte(word, 0, 11);
    word = blackbit::setWordByte(word, 1, 200);
    word = blackbit::setWordByte(word, 2, 3);
    word = blackbit::setWordByte(word, 3, 242);

    EXPECT_EQ(blackbit::wordByte(word, 0), 11);
    EXPECT_EQ(blackbit::wordByte(word, 1), 200);
    EXPECT_EQ(blackbit::wordByte(word, 2), 3);
    EXPECT_EQ(blackbit::wordByte(word, 3), 242);

    // Riscrivere un byte non tocca gli altri.
    word = blackbit::setWordByte(word, 2, 77);
    EXPECT_EQ(blackbit::wordByte(word, 0), 11);
    EXPECT_EQ(blackbit::wordByte(word, 1), 200);
    EXPECT_EQ(blackbit::wordByte(word, 2), 77);
    EXPECT_EQ(blackbit::wordByte(word, 3), 242);
}

TEST(TernaryTensorTest, UnTensoreAppenaCostruitoETuttoZero) {
    TernaryTensor tensor({4, 37});
    for (std::size_t i = 0; i < tensor.elementCount(); ++i) {
        EXPECT_EQ(tensor.tritAt(i), 0) << "elemento " << i;
        EXPECT_FLOAT_EQ(tensor.at(i), 0.0F);
    }
}

TEST(TernaryTensorTest, RoundTripEsattoSuFormeNonAllineate) {
    // 37 non e' multiplo di 5 ne' di 20: l'ultima parola di ogni riga
    // e' parzialmente padding, il caso in cui un impacchettamento
    // sbagliato tipicamente rompe.
    for (std::vector<std::size_t> shape : std::vector<std::vector<std::size_t>>{
             {37}, {3, 37}, {2, 3, 41}, {1, 20}, {5, 160}, {7, 161}}) {
        std::size_t elements = 1;
        for (std::size_t dim : shape) {
            elements *= dim;
        }

        const auto trits = pseudoTrits(elements, 7);
        TernaryTensor tensor(shape);
        for (std::size_t i = 0; i < elements; ++i) {
            tensor.setTritAt(i, trits[i]);
        }

        for (std::size_t i = 0; i < elements; ++i) {
            ASSERT_EQ(tensor.tritAt(i), trits[i]) << "forma " << shape.size() << "d, elemento " << i;
        }
    }
}

TEST(TernaryTensorTest, DequantizzazioneCompletaCoincideConTritPerScala) {
    const std::vector<std::size_t> shape{6, 250};
    TernaryTensor tensor(shape, 20);
    const auto trits = pseudoTrits(tensor.elementCount(), 11);
    const auto scales = positiveScales(tensor.scales().size(), 13);

    TernaryTensor built = TernaryTensor::fromTrits(shape, trits, scales, 20);
    const runtime::Tensor dense = built.dequantize();

    ASSERT_EQ(dense.elementCount(), built.elementCount());
    for (std::size_t i = 0; i < built.elementCount(); ++i) {
        const std::size_t row = i / built.rowLength();
        const std::size_t col = i % built.rowLength();
        const float expected =
            static_cast<float>(trits[i]) * scales[row * built.groupsPerRow() + col / built.groupSize()];
        ASSERT_FLOAT_EQ(dense.at(i), expected) << "elemento " << i;
    }
}

TEST(TernaryTensorTest, DequantizzazioneATileCoincideConQuellaCompleta) {
    // E' la garanzia su cui si regge TernaryLinear: lavorare a blocchi
    // di righe deve dare esattamente lo stesso risultato che
    // materializzare l'intera matrice, che invece per BlackBit-9B non
    // entrerebbe in memoria.
    const std::vector<std::size_t> shape{17, 133};
    const auto trits = pseudoTrits(17 * 133, 21);
    TernaryTensor tensor = TernaryTensor::fromTrits(shape, trits, positiveScales(17, 23), 160);
    ASSERT_EQ(tensor.groupsPerRow(), 1U);

    const runtime::Tensor full = tensor.dequantize();

    std::vector<float> tile(5 * tensor.rowLength());
    for (std::size_t first = 0; first < tensor.rows(); first += 5) {
        const std::size_t count = std::min<std::size_t>(5, tensor.rows() - first);
        tensor.dequantizeRows(first, count, tile.data());
        for (std::size_t r = 0; r < count; ++r) {
            for (std::size_t c = 0; c < tensor.rowLength(); ++c) {
                ASSERT_FLOAT_EQ(tile[r * tensor.rowLength() + c], full.at((first + r) * tensor.rowLength() + c))
                    << "riga " << (first + r) << " colonna " << c;
            }
        }
    }
}

TEST(TernaryTensorTest, QuantizzareUnTensoreGiaSullaGrigliaLoRecuperaEsattamente) {
    // Se i valori densi sono gia' multipli esatti della media dei
    // valori assoluti del loro gruppo, quantizzare e dequantizzare deve
    // essere l'identita': altrimenti la quantizzazione starebbe
    // introducendo un errore anche dove non ce n'e' nessuno da
    // introdurre.
    const std::vector<std::size_t> shape{4, 40};
    std::vector<float> dense(4 * 40);
    for (std::size_t row = 0; row < 4; ++row) {
        const float scale = 0.25F * static_cast<float>(row + 1);
        for (std::size_t col = 0; col < 40; ++col) {
            const int trit = static_cast<int>((row + col) % 3) - 1;
            dense[row * 40 + col] = static_cast<float>(trit) * scale;
        }
    }
    // Nota: il gruppo contiene anche degli zeri, ed e' proprio il caso
    // in cui la absmean "classica" (media su TUTTI gli elementi)
    // contrarrebbe la scala di un fattore 2/3. Vedi quantizeFrom().

    const runtime::Tensor source(shape, dense);
    const TernaryTensor packed = TernaryTensor::quantizeFrom(source, 20);
    const runtime::Tensor recovered = packed.dequantize();

    for (std::size_t i = 0; i < source.elementCount(); ++i) {
        ASSERT_NEAR(recovered.at(i), source.at(i), 1e-6F) << "elemento " << i;
    }
}

TEST(TernaryTensorTest, QuantizzazioneAbsmeanProduceSoloValoriTernari) {
    std::mt19937 rng(99);
    std::normal_distribution<float> normal(0.0F, 0.5F);
    std::vector<float> dense(3 * 200);
    for (float& value : dense) {
        value = normal(rng);
    }

    const TernaryTensor packed = TernaryTensor::quantizeFrom(runtime::Tensor({3, 200}, dense), 20);
    for (std::size_t i = 0; i < packed.elementCount(); ++i) {
        const int trit = packed.tritAt(i);
        ASSERT_GE(trit, -1);
        ASSERT_LE(trit, 1);
        ASSERT_GT(packed.scaleAt(i), 0.0F) << "una scala nulla renderebbe il peso non aggiornabile";
    }

    // Con dati gaussiani centrati la quantizzazione non deve collassare
    // tutto a zero ne' saturare tutto: e' il segnale minimo che la
    // soglia absmean sia sensata.
    std::size_t nonZero = 0;
    for (std::size_t i = 0; i < packed.elementCount(); ++i) {
        nonZero += packed.tritAt(i) != 0 ? 1 : 0;
    }
    EXPECT_GT(nonZero, packed.elementCount() / 10);
    EXPECT_LT(nonZero, packed.elementCount());
}

TEST(TernaryTensorTest, UnGruppoInteramenteNulloRestaAggiornabile) {
    const runtime::Tensor zeros = runtime::Tensor::zeros({1, 40});
    const TernaryTensor packed = TernaryTensor::quantizeFrom(zeros, 20);
    for (std::size_t i = 0; i < packed.elementCount(); ++i) {
        EXPECT_EQ(packed.tritAt(i), 0);
        EXPECT_FLOAT_EQ(packed.scaleAt(i), 1.0F);
    }
}

TEST(TernaryTensorTest, SerializzazioneERiletturaSonoIdentiche) {
    const std::vector<std::size_t> shape{9, 77};
    TernaryTensor original = TernaryTensor::fromTrits(
        shape, pseudoTrits(9 * 77, 31), positiveScales(9 * ((77 + 19) / 20), 37), 20);

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    original.serialize(stream);

    const TernaryTensor reloaded = TernaryTensor::deserialize(stream);

    EXPECT_EQ(reloaded.shape(), original.shape());
    EXPECT_EQ(reloaded.groupSize(), original.groupSize());
    EXPECT_EQ(reloaded.packedWords(), original.packedWords());
    EXPECT_EQ(reloaded.scales(), original.scales());
    for (std::size_t i = 0; i < original.elementCount(); ++i) {
        ASSERT_EQ(reloaded.tritAt(i), original.tritAt(i));
    }
}

TEST(TernaryTensorTest, LaDensitaDichiarataEQuellaReale) {
    // 1,6 bit/peso e' il numero teorico; questo test verifica quello
    // REALE (buffer posseduti / pesi logici), padding e scale inclusi.
    TernaryTensor tensor({1024, 3072}, 160);

    // 3072 non e' multiplo di 20: l'ultima parola di ogni riga e'
    // parzialmente padding (154 parole invece di 153,6). E' lo 0,26 %
    // di spreco, ed e' incluso nel conto reale sotto.
    EXPECT_EQ(tensor.wordsPerRow(), 154U);
    EXPECT_EQ(tensor.packedByteCount(), 1024U * 154U * 4U);
    EXPECT_EQ(tensor.groupsPerRow(), 20U);  // ceil(3072 / 160)
    EXPECT_EQ(tensor.scaleByteCount(), 1024U * 20U * sizeof(float));

    // 1,6 bit di pesi (+ padding) + 4 byte ogni 160 pesi = ~0,21 bit di
    // scale.
    EXPECT_NEAR(tensor.bitsPerWeight(), 1.81, 0.01);
    EXPECT_LT(tensor.bitsPerWeight(), 2.0) << "non deve superare un formato a 2 bit puro";

    // I soli pesi, senza le scale, devono stare sotto il limite
    // dichiarato di 1,6 bit + padding.
    const double packedBits =
        static_cast<double>(tensor.packedByteCount()) * 8.0 / static_cast<double>(tensor.elementCount());
    EXPECT_NEAR(packedBits, 1.6, 0.01);
}

TEST(TernaryTensorTest, RifiutaConfigurazioniNonValide) {
    EXPECT_THROW(TernaryTensor({}, 160), std::invalid_argument);
    EXPECT_THROW(TernaryTensor({4, 0}, 160), std::invalid_argument);
    EXPECT_THROW(TernaryTensor({4, 40}, 0), std::invalid_argument);
    EXPECT_THROW(TernaryTensor({4, 40}, 30), std::invalid_argument);  // non multiplo di 20

    TernaryTensor tensor({2, 40});
    EXPECT_THROW((void)tensor.tritAt(80), std::out_of_range);
    EXPECT_THROW(tensor.setTritAt(0, 2), std::invalid_argument);
    EXPECT_THROW(tensor.dequantizeRows(1, 5, nullptr), std::out_of_range);
}
