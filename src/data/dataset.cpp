#include "blackforge/data/dataset.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace blackforge::data {

namespace {

constexpr char kMagic[8] = {'B', 'F', 'D', 'A', 'T', 'A', '1', '\0'};

std::size_t product(const std::vector<std::size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies<>());
}

void writeU32(std::ostream& out, std::uint32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void writeU64(std::ostream& out, std::uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::uint32_t readU32(std::istream& in) {
    std::uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

std::uint64_t readU64(std::istream& in) {
    std::uint64_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

void writeShape(std::ostream& out, const std::vector<std::size_t>& shape) {
    writeU32(out, static_cast<std::uint32_t>(shape.size()));
    for (std::size_t dim : shape) {
        writeU64(out, static_cast<std::uint64_t>(dim));
    }
}

std::vector<std::size_t> readShape(std::istream& in) {
    std::uint32_t rank = readU32(in);
    std::vector<std::size_t> shape(rank);
    for (std::uint32_t i = 0; i < rank; ++i) {
        shape[i] = static_cast<std::size_t>(readU64(in));
    }
    return shape;
}

}  // namespace

Dataset::Dataset(std::vector<std::size_t> inputExampleShape, std::vector<std::size_t> targetExampleShape,
                  std::vector<float> inputs, std::vector<float> targets, std::size_t numExamples)
    : inputShape_(std::move(inputExampleShape)),
      targetShape_(std::move(targetExampleShape)),
      inputs_(std::move(inputs)),
      targets_(std::move(targets)),
      numExamples_(numExamples) {
    if (inputs_.size() != numExamples_ * product(inputShape_)) {
        throw std::invalid_argument("Dataset: la quantita' di dati di input non corrisponde a numExamples * forma");
    }
    if (targets_.size() != numExamples_ * product(targetShape_)) {
        throw std::invalid_argument(
            "Dataset: la quantita' di dati di target non corrisponde a numExamples * forma");
    }
}

Dataset::Batch Dataset::batch(std::size_t startIndex, std::size_t batchSize) const {
    if (numExamples_ == 0) {
        throw std::invalid_argument("Dataset::batch: il dataset non contiene esempi");
    }

    std::size_t inputExampleSize = product(inputShape_);
    std::size_t targetExampleSize = product(targetShape_);

    std::vector<float> inputData(batchSize * inputExampleSize);
    std::vector<float> targetData(batchSize * targetExampleSize);

    for (std::size_t i = 0; i < batchSize; ++i) {
        std::size_t exampleIndex = (startIndex + i) % numExamples_;
        std::copy_n(inputs_.begin() + static_cast<std::ptrdiff_t>(exampleIndex * inputExampleSize), inputExampleSize,
                    inputData.begin() + static_cast<std::ptrdiff_t>(i * inputExampleSize));
        std::copy_n(targets_.begin() + static_cast<std::ptrdiff_t>(exampleIndex * targetExampleSize),
                    targetExampleSize, targetData.begin() + static_cast<std::ptrdiff_t>(i * targetExampleSize));
    }

    std::vector<std::size_t> inputBatchShape{batchSize};
    inputBatchShape.insert(inputBatchShape.end(), inputShape_.begin(), inputShape_.end());
    std::vector<std::size_t> targetBatchShape{batchSize};
    targetBatchShape.insert(targetBatchShape.end(), targetShape_.begin(), targetShape_.end());

    return Batch{runtime::Tensor(std::move(inputBatchShape), std::move(inputData)),
                 runtime::Tensor(std::move(targetBatchShape), std::move(targetData))};
}

void saveDataset(const std::string& path, const std::vector<std::size_t>& inputExampleShape,
                  const std::vector<std::size_t>& targetExampleShape, const std::vector<float>& inputs,
                  const std::vector<float>& targets, std::size_t numExamples) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("saveDataset: impossibile scrivere il file '" + path + "'");
    }

    out.write(kMagic, sizeof(kMagic));
    writeU32(out, static_cast<std::uint32_t>(numExamples));
    writeShape(out, inputExampleShape);
    writeShape(out, targetExampleShape);
    out.write(reinterpret_cast<const char*>(inputs.data()), static_cast<std::streamsize>(inputs.size() * sizeof(float)));
    out.write(reinterpret_cast<const char*>(targets.data()),
              static_cast<std::streamsize>(targets.size() * sizeof(float)));

    if (!out) {
        throw std::runtime_error("saveDataset: scrittura fallita su '" + path + "'");
    }
}

Dataset loadDataset(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("loadDataset: impossibile aprire il file '" + path + "'");
    }

    char magic[8];
    in.read(magic, sizeof(magic));
    if (!in || std::string(magic, sizeof(magic)) != std::string(kMagic, sizeof(kMagic))) {
        throw std::runtime_error("loadDataset: '" + path + "' non e' un dataset BlackForge valido");
    }

    std::uint32_t numExamples = readU32(in);
    std::vector<std::size_t> inputShape = readShape(in);
    std::vector<std::size_t> targetShape = readShape(in);

    std::size_t inputCount = static_cast<std::size_t>(numExamples) * product(inputShape);
    std::size_t targetCount = static_cast<std::size_t>(numExamples) * product(targetShape);

    std::vector<float> inputs(inputCount);
    std::vector<float> targets(targetCount);
    in.read(reinterpret_cast<char*>(inputs.data()), static_cast<std::streamsize>(inputCount * sizeof(float)));
    in.read(reinterpret_cast<char*>(targets.data()), static_cast<std::streamsize>(targetCount * sizeof(float)));

    if (!in) {
        throw std::runtime_error("loadDataset: file '" + path + "' troncato o corrotto");
    }

    return Dataset(std::move(inputShape), std::move(targetShape), std::move(inputs), std::move(targets),
                    numExamples);
}

}  // namespace blackforge::data
