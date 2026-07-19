#include "blackforge/tokenizer/dataset_prep.hpp"

#include <stdexcept>

#include "blackforge/data/dataset.hpp"

namespace blackforge::tokenizer {

std::size_t buildLanguageModelDataset(const Tokenizer& tok, const std::string& corpus, std::size_t seqLen,
                                       const std::string& outputPath) {
    std::vector<std::uint32_t> ids = tok.encode(corpus);
    if (ids.size() < seqLen + 1) {
        throw std::invalid_argument("buildLanguageModelDataset: il corpus tokenizzato ha solo " +
                                     std::to_string(ids.size()) + " token, ne servono almeno " +
                                     std::to_string(seqLen + 1) + " (seqLen + 1) per una singola finestra");
    }

    std::size_t vocabSize = tok.vocabSize();
    std::size_t numExamples = (ids.size() - 1) / seqLen;

    std::vector<float> inputs;
    std::vector<float> targets;
    inputs.reserve(numExamples * seqLen);
    targets.reserve(numExamples * seqLen * vocabSize);

    for (std::size_t example = 0; example < numExamples; ++example) {
        std::size_t windowStart = example * seqLen;
        for (std::size_t s = 0; s < seqLen; ++s) {
            inputs.push_back(static_cast<float>(ids[windowStart + s]));

            std::size_t nextToken = ids[windowStart + s + 1];
            for (std::size_t c = 0; c < vocabSize; ++c) {
                targets.push_back(c == nextToken ? 1.0F : 0.0F);
            }
        }
    }

    data::saveDataset(outputPath, {seqLen}, {seqLen, vocabSize}, inputs, targets, numExamples);
    return numExamples;
}

}  // namespace blackforge::tokenizer
