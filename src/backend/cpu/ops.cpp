#include "blackforge/backend/cpu/ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace blackforge::backend::cpu {

namespace {

constexpr float kGeluCoeff = 0.044715F;
constexpr float kSqrt2OverPi = 0.7978845608F;  // sqrt(2/pi)
constexpr float kRmsNormEps = 1e-6F;

void requireSameShape(const Tensor& a, const Tensor& b, const char* opName) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(std::string(opName) + ": forme incompatibili " + a.shapeToString() + " e " +
                                     b.shapeToString());
    }
}

// Estrae la "testa" h dell'esempio b da un tensore [batch, seq, dim]
// (concettualmente [batch, seq, numHeads, headDim], con le teste
// contigue nell'ultima dimensione): restituisce [seq, headDim].
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

// Scrive la "testa" h dell'esempio b (forma [seq, headDim]) dentro un
// buffer piatto [batch, seq, dim] gia' allocato: le teste partizionano
// esattamente 'dim', quindi e' una scrittura diretta, non serve
// accumulare (nessuna sovrapposizione tra teste diverse).
void scatterHead(std::vector<float>& full, const Tensor& headSlice, std::size_t b, std::size_t h, std::size_t seq,
                  std::size_t dim, std::size_t headDim) {
    for (std::size_t s = 0; s < seq; ++s) {
        std::size_t dstOffset = (b * seq + s) * dim + h * headDim;
        std::copy_n(headSlice.data().begin() + static_cast<std::ptrdiff_t>(s * headDim), headDim,
                    full.begin() + static_cast<std::ptrdiff_t>(dstOffset));
    }
}

// Applica la maschera causale a una matrice di score [seq, seq]:
// posizione (i,j) con j > i (future rispetto a i) viene impostata a
// -infinito, cosi' softmax() le azzera esattamente (necessario per un
// modello linguistico autoregressivo: la posizione i non deve poter
// "vedere" token futuri).
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

Tensor elementwise(const Tensor& input, float (*fn)(float)) {
    std::vector<float> result(input.elementCount());
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = fn(input.at(i));
    }
    return Tensor(input.shape(), std::move(result));
}

}  // namespace

Tensor add(const Tensor& a, const Tensor& b) {
    requireSameShape(a, b, "add");
    std::vector<float> result(a.elementCount());
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = a.at(i) + b.at(i);
    }
    return Tensor(a.shape(), std::move(result));
}

Tensor scale(const Tensor& input, float factor) {
    std::vector<float> result(input.elementCount());
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = input.at(i) * factor;
    }
    return Tensor(input.shape(), std::move(result));
}

Tensor addBias(const Tensor& input, const Tensor& bias) {
    // Generalizzato a rango >= 2: 'features' e' sempre l'ultima
    // dimensione, tutte le altre (batch, e per i tensori di sequenza
    // anche la posizione) sono trattate come righe indipendenti. Per un
    // input a rango 2 [batch, features] il comportamento e' identico a
    // prima; per un input a rango 3 [batch, seq, features] (uscita di
    // 'embedding') il bias si applica a ogni token di ogni sequenza. Il
    // layout flat riga-maggiore di [batch, seq, features] e' identico,
    // byte per byte, a quello di [batch*seq, features]: non serve una
    // vera riorganizzazione dei dati, solo una diversa interpretazione
    // della stessa forma.
    if (input.rank() < 2 || bias.rank() != 1 || input.shape().back() != bias.dim(0)) {
        throw std::invalid_argument("addBias: attesi input [..., features] e bias [features] con features "
                                     "corrispondenti, trovati " +
                                     input.shapeToString() + " e " + bias.shapeToString());
    }

    std::size_t features = input.shape().back();
    std::size_t rows = input.elementCount() / features;
    std::vector<float> result(input.elementCount());

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < features; ++col) {
            std::size_t idx = row * features + col;
            result[idx] = input.at(idx) + bias.at(col);
        }
    }

    return Tensor(input.shape(), std::move(result));
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.rank() != 2 || b.rank() != 2 || a.dim(1) != b.dim(0)) {
        throw std::invalid_argument("matmul: forme incompatibili " + a.shapeToString() + " x " + b.shapeToString());
    }

    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(1);

    std::vector<float> result(m * n, 0.0F);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            float aVal = a.at(i * k + p);
            for (std::size_t j = 0; j < n; ++j) {
                result[i * n + j] += aVal * b.at(p * n + j);
            }
        }
    }

    return Tensor({m, n}, std::move(result));
}

Tensor matmulTransposeB(const Tensor& a, const Tensor& b) {
    // a: [m,k], b: [n,k] -> a @ b^T: [m,n]. Serve per gli score di
    // attention (Q @ K^T): un primitivo dedicato invece di trasporre
    // esplicitamente b e chiamare matmul() evita una copia e un
    // secondo passaggio sui dati.
    if (a.rank() != 2 || b.rank() != 2 || a.dim(1) != b.dim(1)) {
        throw std::invalid_argument("matmulTransposeB: forme incompatibili " + a.shapeToString() + " e " +
                                     b.shapeToString());
    }

    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(0);

    std::vector<float> result(m * n, 0.0F);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0.0F;
            for (std::size_t p = 0; p < k; ++p) {
                sum += a.at(i * k + p) * b.at(j * k + p);
            }
            result[i * n + j] = sum;
        }
    }

    return Tensor({m, n}, std::move(result));
}

Tensor silu(const Tensor& input) {
    return elementwise(input, [](float x) { return x / (1.0F + std::exp(-x)); });
}

Tensor relu(const Tensor& input) {
    return elementwise(input, [](float x) { return x > 0.0F ? x : 0.0F; });
}

Tensor gelu(const Tensor& input) {
    return elementwise(input, [](float x) {
        float inner = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
        return 0.5F * x * (1.0F + std::tanh(inner));
    });
}

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    if (input.rank() == 2) {
        return addBias(matmul(input, weight), bias);
    }
    if (input.rank() < 2) {
        throw std::invalid_argument("linear: richiede un tensore a rango >= 2, trovato " + input.shapeToString());
    }

    // Rango > 2 (es. [batch, seq, features], uscita di 'embedding'):
    // matmul() e' un primitivo puramente 2D, quindi si appiattisce
    // temporaneamente a [batch*seq, features] (stesso layout flat
    // riga-maggiore, nessuna vera riorganizzazione dei dati) prima del
    // prodotto matriciale, poi si ripristina la forma originale
    // sostituendo solo l'ultima dimensione con le feature in uscita.
    std::size_t inFeatures = input.shape().back();
    std::size_t rows = input.elementCount() / inFeatures;
    Tensor flatInput({rows, inFeatures}, input.data());
    Tensor flatOutput = addBias(matmul(flatInput, weight), bias);

    std::vector<std::size_t> outputShape = input.shape();
    outputShape.back() = weight.dim(1);
    return Tensor(std::move(outputShape), std::move(flatOutput.data()));
}

Tensor rmsnorm(const Tensor& input) {
    // Generalizzato a rango >= 2 (vedi il commento in addBias): normalizza
    // ogni "riga" (tutte le dimensioni tranne l'ultima, che e' 'features').
    if (input.rank() < 2) {
        throw std::invalid_argument("rmsnorm: richiede un tensore a rango >= 2, trovato " + input.shapeToString());
    }

    std::size_t features = input.shape().back();
    std::size_t rows = input.elementCount() / features;
    std::vector<float> result(input.elementCount());

    for (std::size_t row = 0; row < rows; ++row) {
        std::size_t rowOffset = row * features;

        double sumSquares = 0.0;
        for (std::size_t col = 0; col < features; ++col) {
            float x = input.at(rowOffset + col);
            sumSquares += static_cast<double>(x) * static_cast<double>(x);
        }
        double meanSquare = sumSquares / static_cast<double>(features);
        float rms = static_cast<float>(std::sqrt(meanSquare + static_cast<double>(kRmsNormEps)));

        for (std::size_t col = 0; col < features; ++col) {
            result[rowOffset + col] = input.at(rowOffset + col) / rms;
        }
    }

    return Tensor(input.shape(), std::move(result));
}

Tensor softmax(const Tensor& input) {
    // Generalizzato a rango >= 2, stessa logica di rmsnorm sopra:
    // softmax lungo l'ultima dimensione, per ogni "riga" indipendente
    // (utile anche per gli score di attention, forma [batch, heads,
    // seq, seq] o piu' semplicemente applicato riga per riga a matrici
    // [seq, seq] individuali).
    if (input.rank() < 2) {
        throw std::invalid_argument("softmax: richiede un tensore a rango >= 2, trovato " + input.shapeToString());
    }

    std::size_t features = input.shape().back();
    std::size_t rows = input.elementCount() / features;
    std::vector<float> result(input.elementCount());

    for (std::size_t row = 0; row < rows; ++row) {
        std::size_t rowOffset = row * features;

        float maxVal = input.at(rowOffset);
        for (std::size_t col = 1; col < features; ++col) {
            maxVal = std::max(maxVal, input.at(rowOffset + col));
        }

        double sumExp = 0.0;
        for (std::size_t col = 0; col < features; ++col) {
            double e = std::exp(static_cast<double>(input.at(rowOffset + col) - maxVal));
            result[rowOffset + col] = static_cast<float>(e);
            sumExp += e;
        }

        for (std::size_t col = 0; col < features; ++col) {
            result[rowOffset + col] = static_cast<float>(static_cast<double>(result[rowOffset + col]) / sumExp);
        }
    }

    return Tensor(input.shape(), std::move(result));
}

Tensor embeddingLookup(const Tensor& tokenIds, const Tensor& table) {
    // tokenIds: [batch, seq], valori non negativi che rappresentano
    // indici interi memorizzati come float (BlackForge non ha ancora un
    // dtype intero dedicato: e' una semplificazione pragmatica,
    // documentata, non un dtype separato). table: [vocabSize, dim].
    // Uscita: [batch, seq, dim].
    if (tokenIds.rank() != 2) {
        throw std::invalid_argument("embeddingLookup: richiede token id a rango 2 [batch, seq], trovato " +
                                     tokenIds.shapeToString());
    }
    if (table.rank() != 2) {
        throw std::invalid_argument("embeddingLookup: richiede una tabella a rango 2 [vocabSize, dim], trovato " +
                                     table.shapeToString());
    }

    std::size_t batch = tokenIds.dim(0);
    std::size_t seq = tokenIds.dim(1);
    std::size_t vocabSize = table.dim(0);
    std::size_t dim = table.dim(1);

    std::vector<float> result(batch * seq * dim);
    for (std::size_t i = 0; i < batch * seq; ++i) {
        auto tokenId = static_cast<long long>(std::lround(tokenIds.at(i)));
        if (tokenId < 0 || static_cast<std::size_t>(tokenId) >= vocabSize) {
            throw std::invalid_argument("embeddingLookup: token id " + std::to_string(tokenId) +
                                         " fuori dal vocabolario [0, " + std::to_string(vocabSize) + ")");
        }
        std::size_t tableOffset = static_cast<std::size_t>(tokenId) * dim;
        std::copy_n(table.data().begin() + static_cast<std::ptrdiff_t>(tableOffset), dim,
                    result.begin() + static_cast<std::ptrdiff_t>(i * dim));
    }

    return Tensor({batch, seq, dim}, std::move(result));
}

Tensor addPositionalEmbedding(const Tensor& input, const Tensor& table) {
    // input: [batch, seq, dim]. table: [maxSeqLen, dim] (un vettore
    // allenabile per ogni posizione assoluta). output[b,s,d] =
    // input[b,s,d] + table[s,d], trasmesso (broadcast) su ogni esempio
    // del batch.
    if (input.rank() != 3) {
        throw std::invalid_argument("addPositionalEmbedding: richiede un input a rango 3 [batch, seq, dim], "
                                     "trovato " +
                                     input.shapeToString());
    }
    if (table.rank() != 2 || table.dim(1) != input.dim(2)) {
        throw std::invalid_argument("addPositionalEmbedding: richiede una tabella [maxSeqLen, dim] con dim "
                                     "coerente con l'input, trovato " +
                                     table.shapeToString() + " per input " + input.shapeToString());
    }

    std::size_t batch = input.dim(0);
    std::size_t seq = input.dim(1);
    std::size_t dim = input.dim(2);
    std::size_t maxSeqLen = table.dim(0);
    if (seq > maxSeqLen) {
        throw std::invalid_argument("addPositionalEmbedding: la sequenza (" + std::to_string(seq) +
                                     ") supera maxSeqLen (" + std::to_string(maxSeqLen) + ") della tabella");
    }

    std::vector<float> result(input.elementCount());
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t s = 0; s < seq; ++s) {
            std::size_t inOffset = (b * seq + s) * dim;
            std::size_t tableOffset = s * dim;
            for (std::size_t d = 0; d < dim; ++d) {
                result[inOffset + d] = input.at(inOffset + d) + table.at(tableOffset + d);
            }
        }
    }

    return Tensor(input.shape(), std::move(result));
}

Tensor feedForward(const Tensor& input, const Tensor& w1, const Tensor& b1, const Tensor& w2, const Tensor& b2) {
    Tensor normed = rmsnorm(input);
    Tensor hidden = silu(linear(normed, w1, b1));
    Tensor out = linear(hidden, w2, b2);
    return add(input, out);
}

Tensor selfAttention(const Tensor& input, const Tensor& wq, const Tensor& wk, const Tensor& wv, const Tensor& wout,
                      std::size_t numHeads) {
    if (input.rank() != 3) {
        throw std::invalid_argument("selfAttention: richiede un input a rango 3 [batch, seq, dim], trovato " +
                                     input.shapeToString());
    }
    std::size_t batch = input.dim(0);
    std::size_t seq = input.dim(1);
    std::size_t dim = input.dim(2);
    if (numHeads == 0 || dim % numHeads != 0) {
        throw std::invalid_argument("selfAttention: numHeads (" + std::to_string(numHeads) +
                                     ") deve dividere esattamente dim (" + std::to_string(dim) + ")");
    }
    std::size_t headDim = dim / numHeads;
    float scaleFactor = 1.0F / std::sqrt(static_cast<float>(headDim));

    Tensor normed = rmsnorm(input);

    // Le proiezioni Q/K/V/Out non hanno bias (come in LLaMA): matmul()
    // diretto invece di linear(), su 'normed' appiattito a 2D.
    Tensor normedFlat({batch * seq, dim}, normed.data());
    Tensor qFlat = matmul(normedFlat, wq);
    Tensor kFlat = matmul(normedFlat, wk);
    Tensor vFlat = matmul(normedFlat, wv);
    Tensor q({batch, seq, dim}, qFlat.data());
    Tensor k({batch, seq, dim}, kFlat.data());
    Tensor v({batch, seq, dim}, vFlat.data());

    std::vector<float> concatenatedData(batch * seq * dim);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < numHeads; ++h) {
            Tensor qHead = extractHead(q, b, h, seq, dim, headDim);
            Tensor kHead = extractHead(k, b, h, seq, dim, headDim);
            Tensor vHead = extractHead(v, b, h, seq, dim, headDim);

            Tensor scores = scale(matmulTransposeB(qHead, kHead), scaleFactor);
            Tensor maskedScores = applyCausalMask(scores);
            Tensor probs = softmax(maskedScores);
            Tensor headOut = matmul(probs, vHead);

            scatterHead(concatenatedData, headOut, b, h, seq, dim, headDim);
        }
    }
    Tensor concatenated({batch, seq, dim}, std::move(concatenatedData));

    Tensor concatenatedFlat({batch * seq, dim}, concatenated.data());
    Tensor projectedFlat = matmul(concatenatedFlat, wout);
    Tensor projected({batch, seq, dim}, std::move(projectedFlat.data()));

    return add(input, projected);
}

}  // namespace blackforge::backend::cpu
