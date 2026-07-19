#include "blackforge/backend/cpu/loss.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace blackforge::backend::cpu {

LossResult meanSquaredError(const runtime::Tensor& prediction, const runtime::Tensor& target) {
    if (prediction.shape() != target.shape()) {
        throw std::invalid_argument("meanSquaredError: forme incompatibili " + prediction.shapeToString() + " e " +
                                     target.shapeToString());
    }

    std::size_t n = prediction.elementCount();
    double sumSquares = 0.0;
    std::vector<float> grad(n);

    for (std::size_t i = 0; i < n; ++i) {
        float diff = prediction.at(i) - target.at(i);
        sumSquares += static_cast<double>(diff) * static_cast<double>(diff);
        // d/dprediction[i] mean((pred-target)^2) = 2*(pred-target)/n
        grad[i] = 2.0F * diff / static_cast<float>(n);
    }

    float value = static_cast<float>(sumSquares / static_cast<double>(n));
    return LossResult{value, runtime::Tensor(prediction.shape(), std::move(grad))};
}

LossResult softmaxCrossEntropy(const runtime::Tensor& logits, const runtime::Tensor& target) {
    if (logits.shape() != target.shape()) {
        throw std::invalid_argument("softmaxCrossEntropy: forme incompatibili " + logits.shapeToString() + " e " +
                                     target.shapeToString());
    }
    if (logits.rank() < 2) {
        throw std::invalid_argument("softmaxCrossEntropy: richiede un tensore a rango >= 2 [..., classi], trovato " +
                                     logits.shapeToString());
    }

    // Generalizzato a rango >= 2 (stessa idea di rmsnorm/softmax in
    // ops.cpp): per un tensore [batch, seq, classi] tratta ogni "riga"
    // (ogni posizione di ogni esempio del batch) come un esempio
    // indipendente di classificazione, e la loss e' la media su tutte
    // le righe (non solo sul batch) — la convenzione standard per la
    // loss di next-token-prediction di un modello linguistico.
    std::size_t numClasses = logits.shape().back();
    std::size_t batch = logits.elementCount() / numClasses;

    std::vector<float> grad(logits.elementCount());
    double totalLoss = 0.0;
    std::vector<double> probs(numClasses);

    for (std::size_t b = 0; b < batch; ++b) {
        std::size_t rowOffset = b * numClasses;

        float maxLogit = logits.at(rowOffset);
        for (std::size_t c = 1; c < numClasses; ++c) {
            maxLogit = std::max(maxLogit, logits.at(rowOffset + c));
        }

        double sumExp = 0.0;
        for (std::size_t c = 0; c < numClasses; ++c) {
            probs[c] = std::exp(static_cast<double>(logits.at(rowOffset + c) - maxLogit));
            sumExp += probs[c];
        }

        for (std::size_t c = 0; c < numClasses; ++c) {
            probs[c] /= sumExp;
            float t = target.at(rowOffset + c);
            if (t != 0.0F) {
                // 1e-12 evita log(0) se una probabilita' softmax collassa
                // a zero in virgola mobile (target con massa nulla non
                // contribuiscono comunque alla somma).
                totalLoss += -static_cast<double>(t) * std::log(std::max(probs[c], 1e-12));
            }
            grad[rowOffset + c] = static_cast<float>((probs[c] - static_cast<double>(t)) / static_cast<double>(batch));
        }
    }

    float value = static_cast<float>(totalLoss / static_cast<double>(batch));
    return LossResult{value, runtime::Tensor(logits.shape(), std::move(grad))};
}

LossResult softmaxCrossEntropySparse(const runtime::Tensor& logits, const runtime::Tensor& targetIndices) {
    if (logits.rank() < 2) {
        throw std::invalid_argument(
            "softmaxCrossEntropySparse: richiede logits a rango >= 2 [..., classi], trovato " +
            logits.shapeToString());
    }
    std::size_t numClasses = logits.shape().back();
    std::size_t batch = logits.elementCount() / numClasses;

    if (targetIndices.elementCount() != batch) {
        throw std::invalid_argument("softmaxCrossEntropySparse: targetIndices ha " +
                                     std::to_string(targetIndices.elementCount()) +
                                     " elementi, ne servono " + std::to_string(batch) +
                                     " (uno per riga di logits, cioe' " + logits.shapeToString() +
                                     " senza l'ultima dimensione)");
    }

    // Matematicamente identico a softmaxCrossEntropy() con un target
    // one-hot (stessa formula, stesso gradiente in forma chiusa), ma
    // 'targetIndices' e' l'indice della classe corretta per riga
    // (arrotondato con std::lround, la stessa convenzione degli id di
    // token altrove in BlackForge — vedi embeddingLookup), non un
    // vettore one-hot denso: evita di materializzare mai un target di
    // dimensione [..., classi], fondamentale quando 'classi' e' un
    // vocabolario da decine di migliaia di token (next-token-prediction
    // di un modello linguistico), dove il target denso sprecherebbe
    // 'classi' volte piu' memoria del necessario.
    std::vector<float> grad(logits.elementCount());
    double totalLoss = 0.0;
    std::vector<double> probs(numClasses);

    for (std::size_t b = 0; b < batch; ++b) {
        std::size_t rowOffset = b * numClasses;

        auto targetClassSigned = static_cast<long long>(std::lround(targetIndices.at(b)));
        if (targetClassSigned < 0 || static_cast<std::size_t>(targetClassSigned) >= numClasses) {
            throw std::invalid_argument("softmaxCrossEntropySparse: indice di classe " +
                                         std::to_string(targetClassSigned) + " fuori da [0, " +
                                         std::to_string(numClasses) + ")");
        }
        auto targetClass = static_cast<std::size_t>(targetClassSigned);

        float maxLogit = logits.at(rowOffset);
        for (std::size_t c = 1; c < numClasses; ++c) {
            maxLogit = std::max(maxLogit, logits.at(rowOffset + c));
        }

        double sumExp = 0.0;
        for (std::size_t c = 0; c < numClasses; ++c) {
            probs[c] = std::exp(static_cast<double>(logits.at(rowOffset + c) - maxLogit));
            sumExp += probs[c];
        }

        for (std::size_t c = 0; c < numClasses; ++c) {
            probs[c] /= sumExp;
            float t = (c == targetClass) ? 1.0F : 0.0F;
            grad[rowOffset + c] = static_cast<float>((probs[c] - static_cast<double>(t)) / static_cast<double>(batch));
        }
        totalLoss += -std::log(std::max(probs[targetClass], 1e-12));
    }

    float value = static_cast<float>(totalLoss / static_cast<double>(batch));
    return LossResult{value, runtime::Tensor(logits.shape(), std::move(grad))};
}

LossResult softmaxCrossEntropyMasked(const runtime::Tensor& logits, const runtime::Tensor& targetIndices) {
    if (logits.rank() < 2) {
        throw std::invalid_argument(
            "softmaxCrossEntropyMasked: richiede logits a rango >= 2 [..., classi], trovato " +
            logits.shapeToString());
    }
    std::size_t numClasses = logits.shape().back();
    std::size_t batch = logits.elementCount() / numClasses;

    if (targetIndices.elementCount() != batch) {
        throw std::invalid_argument("softmaxCrossEntropyMasked: targetIndices ha " +
                                     std::to_string(targetIndices.elementCount()) + " elementi, ne servono " +
                                     std::to_string(batch));
    }

    std::vector<float> grad(logits.elementCount(), 0.0F);
    double totalLoss = 0.0;
    std::size_t numMaskedRows = 0;
    std::vector<double> probs(numClasses);

    for (std::size_t b = 0; b < batch; ++b) {
        auto rawIndex = static_cast<long long>(std::lround(targetIndices.at(b)));
        if (rawIndex == -1) {
            continue;  // Riga non mascherata: nessun contributo, gradiente resta a zero.
        }
        if (rawIndex < 0 || static_cast<std::size_t>(rawIndex) >= numClasses) {
            throw std::invalid_argument("softmaxCrossEntropyMasked: indice di classe " + std::to_string(rawIndex) +
                                         " fuori da [0, " + std::to_string(numClasses) + ") (o -1 per ignorare)");
        }
        auto targetClass = static_cast<std::size_t>(rawIndex);
        ++numMaskedRows;

        std::size_t rowOffset = b * numClasses;
        float maxLogit = logits.at(rowOffset);
        for (std::size_t c = 1; c < numClasses; ++c) {
            maxLogit = std::max(maxLogit, logits.at(rowOffset + c));
        }

        double sumExp = 0.0;
        for (std::size_t c = 0; c < numClasses; ++c) {
            probs[c] = std::exp(static_cast<double>(logits.at(rowOffset + c) - maxLogit));
            sumExp += probs[c];
        }
        for (std::size_t c = 0; c < numClasses; ++c) {
            probs[c] /= sumExp;
        }
        totalLoss += -std::log(std::max(probs[targetClass], 1e-12));

        // Il gradiente per riga viene diviso per numMaskedRows, non per
        // 'batch': ogni riga mascherata pesa 1/numMaskedRows sulla loss
        // media, esattamente come nella versione non mascherata ogni
        // riga pesa 1/batch. numMaskedRows non e' ancora noto per
        // intero a questo punto del ciclo (si scopre solo alla fine),
        // quindi si scrive qui il gradiente NON normalizzato (prob -
        // target) e si normalizza in un secondo passaggio sotto.
        for (std::size_t c = 0; c < numClasses; ++c) {
            float t = (c == targetClass) ? 1.0F : 0.0F;
            grad[rowOffset + c] = static_cast<float>(probs[c]) - t;
        }
    }

    float value = 0.0F;
    if (numMaskedRows > 0) {
        value = static_cast<float>(totalLoss / static_cast<double>(numMaskedRows));
        float invMaskedRows = 1.0F / static_cast<float>(numMaskedRows);
        for (float& g : grad) {
            g *= invMaskedRows;
        }
    }
    // Se numMaskedRows == 0, 'grad' e' gia' tutto zero (mai scritto) e
    // 'value' resta 0.0F: un batch senza righe mascherate non ha nulla
    // da imparare, non e' un errore.

    return LossResult{value, runtime::Tensor(logits.shape(), std::move(grad))};
}

}  // namespace blackforge::backend::cpu
