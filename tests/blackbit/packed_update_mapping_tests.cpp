#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "blackforge/blackbit/stochastic_round.hpp"
#include "blackforge/blackbit/ternary.hpp"

namespace {

using blackforge::blackbit::counterRandom;
using blackforge::blackbit::decodeTritByte;
using blackforge::blackbit::encodeTritByte;
using blackforge::blackbit::kTritsPerByte;
using blackforge::blackbit::kTritsPerWord;
using blackforge::blackbit::setWordByte;
using blackforge::blackbit::splitMix64;
using blackforge::blackbit::stochasticRoundToTrit;
using blackforge::blackbit::ternaryWordsPerRow;
using blackforge::blackbit::wordByte;

constexpr std::size_t kBlockSize = 256;
constexpr std::size_t kComponentTile = 32;

using Trits = std::array<int, kTritsPerWord>;
using Updates = std::array<float, kTritsPerWord>;

float projection(std::uint64_t seed, std::uint64_t epoch, std::size_t row,
                 std::size_t component, std::size_t rank) {
    const std::uint64_t mixed =
        counterRandom(seed ^ (epoch * 0x9E3779B97F4A7C15ULL),
                      static_cast<std::uint64_t>(row) * rank + component);
    return ((mixed & 1ULL) != 0 ? 1.0F : -1.0F) / std::sqrt(static_cast<float>(rank));
}

Trits decodeWord(std::uint32_t word) {
    Trits trits{};
    for (int byte = 0; byte < 4; ++byte) {
        decodeTritByte(wordByte(word, byte),
                       trits.data() + static_cast<std::size_t>(byte) * kTritsPerByte);
    }
    return trits;
}

std::uint32_t encodeWord(std::uint32_t word, const Trits& trits) {
    for (int byte = 0; byte < 4; ++byte) {
        const int* byteTrits = trits.data() + static_cast<std::size_t>(byte) * kTritsPerByte;
        word = setWordByte(word, byte,
                          encodeTritByte(byteTrits[0], byteTrits[1], byteTrits[2],
                                        byteTrits[3], byteTrits[4]));
    }
    return word;
}

std::uint32_t applyUpdates(std::uint32_t word, const Updates& updates, std::size_t row,
                           std::size_t firstColumn, std::size_t cols, std::uint64_t seed,
                           std::uint64_t step) {
    Trits trits = decodeWord(word);
    const std::uint64_t stepSeed = splitMix64(seed ^ (step * 0xD1B54A32D192ED03ULL));
    for (std::size_t slot = 0; slot < kTritsPerWord; ++slot) {
        const std::size_t col = firstColumn + slot;
        if (col >= cols || !std::isfinite(updates[slot])) continue;
        const float target = static_cast<float>(trits[slot]) + updates[slot];
        trits[slot] = stochasticRoundToTrit(target, stepSeed, row * cols + col);
    }
    return encodeWord(word, trits);
}

// Modello host del launch precedente: indice lineare, un thread per word.
void updateWithLinearMapping(std::vector<std::uint32_t>& packed, std::size_t rows,
                             std::size_t cols, const std::vector<float>& direction,
                             std::size_t rank, float learningRate, std::uint64_t seed,
                             std::uint64_t epoch, std::uint64_t step) {
    const std::size_t wordsPerRow = ternaryWordsPerRow(cols);
    const std::size_t wordCount = rows * wordsPerRow;
    const std::size_t blockCount = (wordCount + kBlockSize - 1) / kBlockSize;
    for (std::size_t block = 0; block < blockCount; ++block) {
        for (std::size_t thread = 0; thread < kBlockSize; ++thread) {
            const std::size_t wordIndex = block * kBlockSize + thread;
            if (wordIndex >= wordCount) continue;
            const std::size_t row = wordIndex / wordsPerRow;
            const std::size_t firstColumn = (wordIndex % wordsPerRow) * kTritsPerWord;
            Updates updates{};
            for (std::size_t component = 0; component < rank; ++component) {
                const float scaled =
                    -learningRate * projection(seed, epoch, row, component, rank);
                const float* directionRow = direction.data() + component * cols;
                for (std::size_t slot = 0; slot < kTritsPerWord; ++slot) {
                    const std::size_t col = firstColumn + slot;
                    if (col < cols) updates[slot] += scaled * directionRow[col];
                }
            }
            packed[wordIndex] = applyUpdates(packed[wordIndex], updates, row, firstColumn,
                                             cols, seed, step);
        }
    }
}

// Modello host del nuovo launch: un blocco per colonna di word e 256
// righe. Il caricamento della tile riproduce anche la cooperazione dei
// thread inattivi nell'ultimo blocco di righe.
void updateWithTiledMapping(std::vector<std::uint32_t>& packed, std::size_t rows,
                            std::size_t cols, const std::vector<float>& direction,
                            std::size_t rank, float learningRate, std::uint64_t seed,
                            std::uint64_t epoch, std::uint64_t step) {
    const std::size_t wordsPerRow = ternaryWordsPerRow(cols);
    const std::size_t rowBlockCount = (rows + kBlockSize - 1) / kBlockSize;
    for (std::size_t wordColumn = 0; wordColumn < wordsPerRow; ++wordColumn) {
        const std::size_t firstColumn = wordColumn * kTritsPerWord;
        for (std::size_t rowBlock = 0; rowBlock < rowBlockCount; ++rowBlock) {
            std::array<Updates, kBlockSize> updates{};
            std::array<float, kComponentTile * kTritsPerWord> sharedDirection{};

            for (std::size_t componentBase = 0; componentBase < rank;
                 componentBase += kComponentTile) {
                const std::size_t chunk = std::min(kComponentTile, rank - componentBase);
                for (std::size_t thread = 0; thread < kBlockSize; ++thread) {
                    for (std::size_t index = thread; index < chunk * kTritsPerWord;
                         index += kBlockSize) {
                        const std::size_t component =
                            componentBase + index / kTritsPerWord;
                        const std::size_t column = firstColumn + index % kTritsPerWord;
                        sharedDirection[index] =
                            column < cols ? direction[component * cols + column] : 0.0F;
                    }
                }

                for (std::size_t thread = 0; thread < kBlockSize; ++thread) {
                    const std::size_t row = rowBlock * kBlockSize + thread;
                    if (row >= rows) continue;
                    for (std::size_t offset = 0; offset < chunk; ++offset) {
                        const float scaled = -learningRate *
                                             projection(seed, epoch, row,
                                                        componentBase + offset, rank);
                        const float* tile =
                            sharedDirection.data() + offset * kTritsPerWord;
                        for (std::size_t slot = 0; slot < kTritsPerWord; ++slot) {
                            if (firstColumn + slot < cols) {
                                updates[thread][slot] += scaled * tile[slot];
                            }
                        }
                    }
                }
            }

            for (std::size_t thread = 0; thread < kBlockSize; ++thread) {
                const std::size_t row = rowBlock * kBlockSize + thread;
                if (row >= rows) continue;
                const std::size_t wordIndex = row * wordsPerRow + wordColumn;
                packed[wordIndex] = applyUpdates(packed[wordIndex], updates[thread], row,
                                                 firstColumn, cols, seed, step);
            }
        }
    }
}

std::vector<std::uint32_t> randomPacked(std::size_t rows, std::size_t cols,
                                        std::mt19937& generator) {
    std::uniform_int_distribution<int> trit(-1, 1);
    std::vector<std::uint32_t> packed(rows * ternaryWordsPerRow(cols));
    for (std::uint32_t& word : packed) {
        Trits values{};
        for (int& value : values) value = trit(generator);
        word = encodeWord(0, values);
    }
    return packed;
}

struct MappingCase {
    std::size_t rows;
    std::size_t cols;
    std::size_t rank;
};

}  // namespace

TEST(PackedUpdateMappingTest, LaMappaturaATileProduceWordBitIdentiche) {
    // Ogni caso attraversa un blocco di righe parziale e un'ultima word
    // con padding. I ranghi 8 e 40 coprono rispettivamente una tile
    // parziale e una tile piena seguita da una parziale; 3072 e' la
    // larghezza reale di BlackBit-9B (154 word, 8 slot di padding).
    constexpr std::array<MappingCase, 3> kCases{{
        {257, 47, 8},
        {259, 67, 40},
        {3, 3072, 8},
    }};
    constexpr float kLearningRate = 0.19F;
    constexpr std::uint64_t kSeed = 0x123456789ABCDEF0ULL;
    constexpr std::uint64_t kEpoch = 7;
    constexpr std::uint64_t kStep = 23;
    std::mt19937 generator(0xB1ACB17U);
    std::uniform_real_distribution<float> directionValue(-1.25F, 1.25F);

    for (const MappingCase& testCase : kCases) {
        SCOPED_TRACE("rows=" + std::to_string(testCase.rows) +
                     " cols=" + std::to_string(testCase.cols) +
                     " rank=" + std::to_string(testCase.rank));
        std::vector<std::uint32_t> linear =
            randomPacked(testCase.rows, testCase.cols, generator);
        std::vector<std::uint32_t> tiled = linear;
        std::vector<float> direction(testCase.rank * testCase.cols);
        for (float& value : direction) value = directionValue(generator);

        updateWithLinearMapping(linear, testCase.rows, testCase.cols, direction,
                                testCase.rank, kLearningRate, kSeed, kEpoch, kStep);
        updateWithTiledMapping(tiled, testCase.rows, testCase.cols, direction,
                               testCase.rank, kLearningRate, kSeed, kEpoch, kStep);

        ASSERT_EQ(linear.size(), tiled.size());
        for (std::size_t word = 0; word < linear.size(); ++word) {
            ASSERT_EQ(linear[word], tiled[word]) << "word " << word;
        }
    }
}
