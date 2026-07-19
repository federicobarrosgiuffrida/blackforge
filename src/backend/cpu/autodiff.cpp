#include "blackforge/backend/cpu/autodiff.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "blackforge/backend/cpu/ops.hpp"

namespace blackforge::backend::cpu {

namespace {

constexpr float kGeluCoeff = 0.044715F;
constexpr float kGeluCoeffDeriv = 3.0F * kGeluCoeff;  // 0.134145
constexpr float kSqrt2OverPi = 0.7978845608F;         // sqrt(2/pi)
constexpr float kRmsNormEps = 1e-6F;                  // deve coincidere con ops.cpp

// Duplicato deliberatamente da ops.cpp (dove sono anch'esse in un
// namespace anonimo, quindi non condivisibili tra le due unita' di
// traduzione senza una dichiarazione in un header): sono helper interni
// piccoli, non fa parte dell'API pubblica di nessuno dei due file.
Tensor extractHead(const Tensor& full, std::size_t b, std::size_t h, std::size_t seq, std::size_t dim,
                    std::size_t headDim) {
    std::vector<float> result(seq * headDim);
    for (std::size_t s = 0; s < seq; ++s) {
        std::size_t srcOffset = (b * seq + s) * dim + h * headDim;
        std::copy_n(full.data().begin() + static_cast<std::ptrdiff_t>(srcOffset), headDim,
                    result.begin() + static_cast<std::ptrdiff_t>(s * headDim));
    }
    return Tensor({seq, headDim}, std::move(result));
}

void scatterHead(std::vector<float>& full, const Tensor& headSlice, std::size_t b, std::size_t h, std::size_t seq,
                  std::size_t dim, std::size_t headDim) {
    for (std::size_t s = 0; s < seq; ++s) {
        std::size_t dstOffset = (b * seq + s) * dim + h * headDim;
        std::copy_n(headSlice.data().begin() + static_cast<std::ptrdiff_t>(s * headDim), headDim,
                    full.begin() + static_cast<std::ptrdiff_t>(dstOffset));
    }
}

// Appiattisce un tensore a rango >= 2 a [rows, features] (stesso
// layout flat riga-maggiore, vedi il commento in ops.cpp/addBias):
// serve per poter chiamare matmulBackward() (primitivo puramente 2D)
// su input a rango > 2 come [batch, seq, features].
Tensor flatten2D(const Tensor& t) {
    std::size_t features = t.shape().back();
    std::size_t rows = t.elementCount() / features;
    return Tensor({rows, features}, t.data());
}

Tensor applyCausalMask(const Tensor& scores) {
    std::size_t seq = scores.dim(0);
    std::vector<float> result(scores.data());
    for (std::size_t i = 0; i < seq; ++i) {
        for (std::size_t j = i + 1; j < seq; ++j) {
            result[i * seq + j] = -std::numeric_limits<float>::infinity();
        }
    }
    return Tensor({seq, seq}, std::move(result));
}

Tensor elementwiseGrad(const Tensor& input, const Tensor& gradOutput, float (*derivative)(float)) {
    if (input.shape() != gradOutput.shape()) {
        throw std::invalid_argument("backward elementwise: forme incompatibili " + input.shapeToString() + " e " +
                                     gradOutput.shapeToString());
    }
    std::vector<float> result(input.elementCount());
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = gradOutput.at(i) * derivative(input.at(i));
    }
    return Tensor(input.shape(), std::move(result));
}

}  // namespace

MatmulGrad matmulBackward(const Tensor& a, const Tensor& b, const Tensor& gradOutput) {
    if (a.rank() != 2 || b.rank() != 2 || gradOutput.rank() != 2 || a.dim(1) != b.dim(0) ||
        gradOutput.dim(0) != a.dim(0) || gradOutput.dim(1) != b.dim(1)) {
        throw std::invalid_argument("matmulBackward: forme incompatibili A=" + a.shapeToString() +
                                     " B=" + b.shapeToString() + " gradOutput=" + gradOutput.shapeToString());
    }

    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(1);

    std::vector<float> dA(m * k, 0.0F);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            float sum = 0.0F;
            for (std::size_t j = 0; j < n; ++j) {
                sum += gradOutput.at(i * n + j) * b.at(p * n + j);
            }
            dA[i * k + p] = sum;
        }
    }

    std::vector<float> dB(k * n, 0.0F);
    for (std::size_t p = 0; p < k; ++p) {
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0.0F;
            for (std::size_t i = 0; i < m; ++i) {
                sum += a.at(i * k + p) * gradOutput.at(i * n + j);
            }
            dB[p * n + j] = sum;
        }
    }

    return MatmulGrad{Tensor({m, k}, std::move(dA)), Tensor({k, n}, std::move(dB))};
}

AddBiasGrad addBiasBackward(const Tensor& gradOutput) {
    // Generalizzato a rango >= 2 (vedi il commento in ops.cpp/addBias):
    // dBias somma il gradiente su tutte le "righe" (tutte le dimensioni
    // tranne l'ultima), non solo sul batch.
    if (gradOutput.rank() < 2) {
        throw std::invalid_argument("addBiasBackward: atteso un gradiente a rango >= 2, trovato " +
                                     gradOutput.shapeToString());
    }

    std::size_t features = gradOutput.shape().back();
    std::size_t rows = gradOutput.elementCount() / features;

    std::vector<float> dBias(features, 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < features; ++col) {
            dBias[col] += gradOutput.at(row * features + col);
        }
    }

    return AddBiasGrad{gradOutput, Tensor({features}, std::move(dBias))};
}

MatmulTransposeBGrad matmulTransposeBBackward(const Tensor& a, const Tensor& b, const Tensor& gradOutput) {
    // a:[m,k], b:[n,k], C = a @ b^T:[m,n]. Con D = b^T:[k,n], C = a @ D:
    // dA = dC @ D^T = dC @ b (stessa forma del matmulBackward standard
    // per il primo operando); dD = a^T @ dC, quindi dB = dD^T = dC^T @ a.
    if (a.rank() != 2 || b.rank() != 2 || gradOutput.rank() != 2 || a.dim(1) != b.dim(1) ||
        gradOutput.dim(0) != a.dim(0) || gradOutput.dim(1) != b.dim(0)) {
        throw std::invalid_argument("matmulTransposeBBackward: forme incompatibili A=" + a.shapeToString() +
                                     " B=" + b.shapeToString() + " gradOutput=" + gradOutput.shapeToString());
    }

    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(0);

    std::vector<float> dA(m * k, 0.0F);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            float sum = 0.0F;
            for (std::size_t j = 0; j < n; ++j) {
                sum += gradOutput.at(i * n + j) * b.at(j * k + p);
            }
            dA[i * k + p] = sum;
        }
    }

    std::vector<float> dB(n * k, 0.0F);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t p = 0; p < k; ++p) {
            float sum = 0.0F;
            for (std::size_t i = 0; i < m; ++i) {
                sum += gradOutput.at(i * n + j) * a.at(i * k + p);
            }
            dB[j * k + p] = sum;
        }
    }

    return MatmulTransposeBGrad{Tensor({m, k}, std::move(dA)), Tensor({n, k}, std::move(dB))};
}

Tensor siluBackward(const Tensor& input, const Tensor& gradOutput) {
    return elementwiseGrad(input, gradOutput, [](float x) {
        float sigmoid = 1.0F / (1.0F + std::exp(-x));
        return sigmoid * (1.0F + x * (1.0F - sigmoid));
    });
}

Tensor reluBackward(const Tensor& input, const Tensor& gradOutput) {
    return elementwiseGrad(input, gradOutput, [](float x) { return x > 0.0F ? 1.0F : 0.0F; });
}

Tensor geluBackward(const Tensor& input, const Tensor& gradOutput) {
    return elementwiseGrad(input, gradOutput, [](float x) {
        float inner = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
        float t = std::tanh(inner);
        float innerDerivative = kSqrt2OverPi * (1.0F + kGeluCoeffDeriv * x * x);
        return 0.5F * (1.0F + t) + 0.5F * x * (1.0F - t * t) * innerDerivative;
    });
}

Tensor rmsnormBackward(const Tensor& input, const Tensor& gradOutput) {
    if (input.shape() != gradOutput.shape()) {
        throw std::invalid_argument("rmsnormBackward: forme incompatibili " + input.shapeToString() + " e " +
                                     gradOutput.shapeToString());
    }
    if (input.rank() < 2) {
        throw std::invalid_argument("rmsnormBackward: richiede un tensore a rango >= 2, trovato " +
                                     input.shapeToString());
    }

    std::size_t features = input.shape().back();
    std::size_t rows = input.elementCount() / features;
    std::vector<float> result(input.elementCount());

    // Derivazione: r = sqrt(mean(x^2) + eps), y_i = x_i / r.
    // dr/dx_i = x_i / (features * r), quindi, con S = sum_j(gOut_j * x_j):
    // dL/dx_i = gOut_i / r  -  x_i * S / (features * r^3).
    for (std::size_t row = 0; row < rows; ++row) {
        std::size_t rowOffset = row * features;

        double sumSquares = 0.0;
        double weightedSum = 0.0;  // S = sum_j(gOut_j * x_j)
        for (std::size_t col = 0; col < features; ++col) {
            double x = static_cast<double>(input.at(rowOffset + col));
            sumSquares += x * x;
            weightedSum += static_cast<double>(gradOutput.at(rowOffset + col)) * x;
        }
        double meanSquare = sumSquares / static_cast<double>(features);
        double r = std::sqrt(meanSquare + static_cast<double>(kRmsNormEps));

        for (std::size_t col = 0; col < features; ++col) {
            double x = static_cast<double>(input.at(rowOffset + col));
            double gOut = static_cast<double>(gradOutput.at(rowOffset + col));
            double grad = gOut / r - x * weightedSum / (static_cast<double>(features) * r * r * r);
            result[rowOffset + col] = static_cast<float>(grad);
        }
    }

    return Tensor(input.shape(), std::move(result));
}

Tensor softmaxBackward(const Tensor& input, const Tensor& gradOutput) {
    if (input.shape() != gradOutput.shape()) {
        throw std::invalid_argument("softmaxBackward: forme incompatibili " + input.shapeToString() + " e " +
                                     gradOutput.shapeToString());
    }
    if (input.rank() < 2) {
        throw std::invalid_argument("softmaxBackward: richiede un tensore a rango >= 2, trovato " +
                                     input.shapeToString());
    }

    Tensor y = softmax(input);
    std::size_t features = input.shape().back();
    std::size_t rows = input.elementCount() / features;
    std::vector<float> result(input.elementCount());

    // dx_j = y_j * (gOut_j - S), S = sum_i(gOut_i * y_i): la forma
    // chiusa standard dello Jacobiano di softmax.
    for (std::size_t row = 0; row < rows; ++row) {
        std::size_t rowOffset = row * features;

        double weightedSum = 0.0;
        for (std::size_t col = 0; col < features; ++col) {
            weightedSum += static_cast<double>(gradOutput.at(rowOffset + col)) * static_cast<double>(y.at(rowOffset + col));
        }

        for (std::size_t col = 0; col < features; ++col) {
            double yVal = static_cast<double>(y.at(rowOffset + col));
            double gOut = static_cast<double>(gradOutput.at(rowOffset + col));
            result[rowOffset + col] = static_cast<float>(yVal * (gOut - weightedSum));
        }
    }

    return Tensor(input.shape(), std::move(result));
}

Tensor embeddingLookupBackward(const Tensor& tokenIds, const Tensor& gradOutput, std::size_t vocabSize) {
    // Nessun gradiente rispetto ai token id (indici, non differenziabili):
    // solo dTable, via scatter-add (piu' token uguali nella stessa
    // sequenza/batch accumulano il gradiente sulla stessa riga della
    // tabella).
    if (tokenIds.rank() != 2 || gradOutput.rank() != 3 || tokenIds.dim(0) != gradOutput.dim(0) ||
        tokenIds.dim(1) != gradOutput.dim(1)) {
        throw std::invalid_argument("embeddingLookupBackward: forme incompatibili tokenIds=" +
                                     tokenIds.shapeToString() + " gradOutput=" + gradOutput.shapeToString());
    }

    std::size_t dim = gradOutput.dim(2);
    std::vector<float> dTable(vocabSize * dim, 0.0F);

    std::size_t totalTokens = tokenIds.elementCount();
    for (std::size_t i = 0; i < totalTokens; ++i) {
        auto tokenId = static_cast<long long>(std::lround(tokenIds.at(i)));
        if (tokenId < 0 || static_cast<std::size_t>(tokenId) >= vocabSize) {
            throw std::invalid_argument("embeddingLookupBackward: token id " + std::to_string(tokenId) +
                                         " fuori dal vocabolario [0, " + std::to_string(vocabSize) + ")");
        }
        std::size_t tableOffset = static_cast<std::size_t>(tokenId) * dim;
        std::size_t gradOffset = i * dim;
        for (std::size_t d = 0; d < dim; ++d) {
            dTable[tableOffset + d] += gradOutput.at(gradOffset + d);
        }
    }

    return Tensor({vocabSize, dim}, std::move(dTable));
}

PositionalEmbeddingGrad addPositionalEmbeddingBackward(const Tensor& gradOutput, std::size_t maxSeqLen) {
    // L'addizione e' un'identita' nel gradiente per l'ingresso; dTable
    // somma il gradiente su tutto il batch, posizione per posizione (le
    // righe della tabella oltre 'seq', se maxSeqLen > seq, non sono
    // state usate in questo forward e ricevono gradiente zero).
    if (gradOutput.rank() != 3) {
        throw std::invalid_argument("addPositionalEmbeddingBackward: atteso un gradiente a rango 3 "
                                     "[batch, seq, dim], trovato " +
                                     gradOutput.shapeToString());
    }

    std::size_t batch = gradOutput.dim(0);
    std::size_t seq = gradOutput.dim(1);
    std::size_t dim = gradOutput.dim(2);
    if (seq > maxSeqLen) {
        throw std::invalid_argument("addPositionalEmbeddingBackward: la sequenza (" + std::to_string(seq) +
                                     ") supera maxSeqLen (" + std::to_string(maxSeqLen) + ")");
    }

    std::vector<float> dTable(maxSeqLen * dim, 0.0F);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t s = 0; s < seq; ++s) {
            std::size_t gradOffset = (b * seq + s) * dim;
            std::size_t tableOffset = s * dim;
            for (std::size_t d = 0; d < dim; ++d) {
                dTable[tableOffset + d] += gradOutput.at(gradOffset + d);
            }
        }
    }

    return PositionalEmbeddingGrad{gradOutput, Tensor({maxSeqLen, dim}, std::move(dTable))};
}

FeedForwardGrad feedForwardBackward(const Tensor& input, const Tensor& w1, const Tensor& b1, const Tensor& w2,
                                     const Tensor& b2, const Tensor& gradOutput) {
    // b2 non serve al calcolo (il gradiente di un bias non dipende dal
    // suo valore, solo dalla somma di gradOutput): resta nella firma
    // solo per simmetria con feedForward(), piu' facile da richiamare a
    // specchio dal codice chiamante.
    (void)b2;

    // Ricalcola gli stessi passaggi del forward (vedi ops.cpp/feedForward)
    Tensor normed = rmsnorm(input);
    Tensor preActivation = linear(normed, w1, b1);
    Tensor hidden = silu(preActivation);

    // y = input + linear2(hidden): il residual manda gradOutput sia
    // direttamente a dInput sia, tramite linear2/silu/linear1/rmsnorm,
    // indietro fino a dInput di nuovo (i due contributi si sommano).
    // matmulBackward() e' un primitivo puramente 2D: input/hidden/normed
    // possono essere a rango 3 ([batch, seq, features]), quindi vanno
    // appiattiti prima di ogni chiamata (stessa idea di linear() nel
    // forward) e la loro uscita ri-alzata di rango dopo.
    AddBiasGrad addGrad2 = addBiasBackward(gradOutput);
    MatmulGrad matGrad2 = matmulBackward(flatten2D(hidden), w2, flatten2D(addGrad2.dInput));
    Tensor dHidden(hidden.shape(), std::move(matGrad2.dA.data()));

    Tensor dPreActivation = siluBackward(preActivation, dHidden);

    AddBiasGrad addGrad1 = addBiasBackward(dPreActivation);
    MatmulGrad matGrad1 = matmulBackward(flatten2D(normed), w1, flatten2D(addGrad1.dInput));
    Tensor dNormed(normed.shape(), std::move(matGrad1.dA.data()));

    Tensor dInputFromBranch = rmsnormBackward(input, dNormed);
    Tensor dInput = add(gradOutput, dInputFromBranch);

    return FeedForwardGrad{dInput, matGrad1.dB, addGrad1.dBias, matGrad2.dB, addGrad2.dBias};
}

namespace {

// Nucleo condiviso di selfAttentionBackward()/
// bidirectionalSelfAttentionBackward(): ricalcola lo stesso forward
// (causale o bidirezionale a seconda di 'causal', deve corrispondere
// esattamente a quale delle due e' stata usata nel forward originale)
// per ricostruire gli stati intermedi necessari, poi applica la stessa
// formula analitica in entrambi i casi (la maschera causale, quando
// presente, non richiede un passaggio di backward dedicato: le
// posizioni mascherate hanno probs == 0, quindi softmaxBackward da'
// loro gradiente esattamente zero).
SelfAttentionGrad selfAttentionBackwardImpl(const Tensor& input, const Tensor& wq, const Tensor& wk,
                                             const Tensor& wv, const Tensor& wout, std::size_t numHeads,
                                             const Tensor& gradOutput, bool causal) {
    std::size_t batch = input.dim(0);
    std::size_t seq = input.dim(1);
    std::size_t dim = input.dim(2);
    std::size_t headDim = dim / numHeads;
    float scaleFactor = 1.0F / std::sqrt(static_cast<float>(headDim));

    Tensor normed = rmsnorm(input);
    Tensor normedFlat({batch * seq, dim}, normed.data());
    Tensor qFlat = matmul(normedFlat, wq);
    Tensor kFlat = matmul(normedFlat, wk);
    Tensor vFlat = matmul(normedFlat, wv);
    Tensor q({batch, seq, dim}, qFlat.data());
    Tensor k({batch, seq, dim}, kFlat.data());
    Tensor v({batch, seq, dim}, vFlat.data());

    std::vector<float> concatenatedData(batch * seq * dim);
    // Cachati per il backward: per ogni (b,h), gli score mascherati
    // (pre-softmax) e Q/K/V della testa.
    std::vector<Tensor> maskedScoresPerHead(batch * numHeads);
    std::vector<Tensor> qHeadPerHead(batch * numHeads);
    std::vector<Tensor> kHeadPerHead(batch * numHeads);
    std::vector<Tensor> vHeadPerHead(batch * numHeads);

    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < numHeads; ++h) {
            std::size_t idx = b * numHeads + h;
            Tensor qHead = extractHead(q, b, h, seq, dim, headDim);
            Tensor kHead = extractHead(k, b, h, seq, dim, headDim);
            Tensor vHead = extractHead(v, b, h, seq, dim, headDim);

            Tensor scores = scale(matmulTransposeB(qHead, kHead), scaleFactor);
            Tensor maskedScores = causal ? applyCausalMask(scores) : scores;
            Tensor probs = softmax(maskedScores);
            Tensor headOut = matmul(probs, vHead);

            scatterHead(concatenatedData, headOut, b, h, seq, dim, headDim);

            maskedScoresPerHead[idx] = std::move(maskedScores);
            qHeadPerHead[idx] = std::move(qHead);
            kHeadPerHead[idx] = std::move(kHead);
            vHeadPerHead[idx] = std::move(vHead);
        }
    }
    Tensor concatenated({batch, seq, dim}, std::move(concatenatedData));

    // --- backward attraverso la proiezione Wout ---
    Tensor concatenatedFlat({batch * seq, dim}, concatenated.data());
    Tensor gradOutputFlat({batch * seq, dim}, gradOutput.data());
    MatmulGrad woutGrad = matmulBackward(concatenatedFlat, wout, gradOutputFlat);
    Tensor dConcatenated({batch, seq, dim}, std::move(woutGrad.dA.data()));

    // --- backward per ogni testa, accumulando dQ/dK/dV su tutto il tensore ---
    std::vector<float> dQData(batch * seq * dim, 0.0F);
    std::vector<float> dKData(batch * seq * dim, 0.0F);
    std::vector<float> dVData(batch * seq * dim, 0.0F);

    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < numHeads; ++h) {
            std::size_t idx = b * numHeads + h;
            Tensor dHeadOut = extractHead(dConcatenated, b, h, seq, dim, headDim);

            Tensor probs = softmax(maskedScoresPerHead[idx]);
            MatmulGrad pvGrad = matmulBackward(probs, vHeadPerHead[idx], dHeadOut);
            // pvGrad.dA = dProbs, pvGrad.dB = dVHead

            Tensor dMaskedScores = softmaxBackward(maskedScoresPerHead[idx], pvGrad.dA);
            // La maschera causale non richiede un passaggio di backward
            // dedicato: le posizioni mascherate hanno probs == 0, quindi
            // la formula di softmaxBackward da' loro gradiente esattamente
            // zero (coerente con il non aver partecipato al forward).
            Tensor dScores = scale(dMaskedScores, scaleFactor);

            MatmulTransposeBGrad qkGrad =
                matmulTransposeBBackward(qHeadPerHead[idx], kHeadPerHead[idx], dScores);

            scatterHead(dQData, qkGrad.dA, b, h, seq, dim, headDim);
            scatterHead(dKData, qkGrad.dB, b, h, seq, dim, headDim);
            scatterHead(dVData, pvGrad.dB, b, h, seq, dim, headDim);
        }
    }

    Tensor dQFlat({batch * seq, dim}, std::move(dQData));
    Tensor dKFlat({batch * seq, dim}, std::move(dKData));
    Tensor dVFlat({batch * seq, dim}, std::move(dVData));

    // --- backward attraverso le proiezioni Q/K/V ---
    MatmulGrad qBackward = matmulBackward(normedFlat, wq, dQFlat);
    MatmulGrad kBackward = matmulBackward(normedFlat, wk, dKFlat);
    MatmulGrad vBackward = matmulBackward(normedFlat, wv, dVFlat);

    Tensor dNormedFlat = add(add(qBackward.dA, kBackward.dA), vBackward.dA);
    Tensor dNormed({batch, seq, dim}, std::move(dNormedFlat.data()));

    // --- backward attraverso rmsnorm, poi residual ---
    Tensor dInputFromBranch = rmsnormBackward(input, dNormed);
    Tensor dInput = add(gradOutput, dInputFromBranch);

    return SelfAttentionGrad{dInput, qBackward.dB, kBackward.dB, vBackward.dB, woutGrad.dB};
}

}  // namespace

SelfAttentionGrad selfAttentionBackward(const Tensor& input, const Tensor& wq, const Tensor& wk, const Tensor& wv,
                                         const Tensor& wout, std::size_t numHeads, const Tensor& gradOutput) {
    return selfAttentionBackwardImpl(input, wq, wk, wv, wout, numHeads, gradOutput, /*causal=*/true);
}

SelfAttentionGrad bidirectionalSelfAttentionBackward(const Tensor& input, const Tensor& wq, const Tensor& wk,
                                                       const Tensor& wv, const Tensor& wout, std::size_t numHeads,
                                                       const Tensor& gradOutput) {
    return selfAttentionBackwardImpl(input, wq, wk, wv, wout, numHeads, gradOutput, /*causal=*/false);
}

}  // namespace blackforge::backend::cpu
