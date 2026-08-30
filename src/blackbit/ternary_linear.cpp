#include "blackforge/blackbit/ternary_linear.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "blackforge/backend/cpu/quantize.hpp"
#include "blackforge/backend/cpu/random_init.hpp"
#include "blackforge/blackbit/telemetry.hpp"

namespace blackforge::blackbit {

namespace {

// Righe (token) di un tensore di attivazioni [..., features]: tutte le
// dimensioni tranne l'ultima, appiattite. Stessa convenzione di
// backend::cpu::linear.
std::size_t rowsOf(const runtime::Tensor& tensor) {
    std::size_t rows = 1;
    for (std::size_t i = 0; i + 1 < tensor.rank(); ++i) {
        rows *= tensor.dim(i);
    }
    return rows;
}

void requireFeatureCount(const runtime::Tensor& tensor, std::size_t expected, const char* what,
                          const std::string& name) {
    if (tensor.rank() < 2) {
        throw std::invalid_argument("TernaryLinear '" + name + "': " + what + " deve avere rango >= 2");
    }
    if (tensor.shape().back() != expected) {
        throw std::invalid_argument("TernaryLinear '" + name + "': " + what + " ha " +
                                     std::to_string(tensor.shape().back()) + " feature, attese " +
                                     std::to_string(expected));
    }
}

}  // namespace

TernaryLinear::TernaryLinear(std::string name, std::size_t inFeatures, std::size_t outFeatures,
                              std::size_t groupSize, std::size_t tileRows)
    : name_(std::move(name)),
      inFeatures_(inFeatures),
      outFeatures_(outFeatures),
      weight_({outFeatures, inFeatures}, groupSize) {
    // Le dimensioni nulle sono gia' rifiutate dal costruttore di
    // TernaryTensor, che gira prima di questo corpo.
    setTileRows(tileRows);
    accountedBytes_ = weight_.totalByteCount();
    MemoryTelemetry::instance().recordAllocation(MemoryArena::Parameter, accountedBytes_);
}

TernaryLinear::~TernaryLinear() {
    if (accountedBytes_ != 0) {
        MemoryTelemetry::instance().recordRelease(MemoryArena::Parameter, accountedBytes_);
    }
}

TernaryLinear::TernaryLinear(TernaryLinear&& other) noexcept
    : name_(std::move(other.name_)),
      inFeatures_(other.inFeatures_),
      outFeatures_(other.outFeatures_),
      tileRows_(other.tileRows_),
      compute_(other.compute_),
      weight_(std::move(other.weight_)),
      accountedBytes_(other.accountedBytes_) {
    other.accountedBytes_ = 0;
}

TernaryLinear& TernaryLinear::operator=(TernaryLinear&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (accountedBytes_ != 0) {
        MemoryTelemetry::instance().recordRelease(MemoryArena::Parameter, accountedBytes_);
    }
    name_ = std::move(other.name_);
    inFeatures_ = other.inFeatures_;
    outFeatures_ = other.outFeatures_;
    tileRows_ = other.tileRows_;
    compute_ = other.compute_;
    weight_ = std::move(other.weight_);
    accountedBytes_ = other.accountedBytes_;
    other.accountedBytes_ = 0;
    return *this;
}

void TernaryLinear::setTileRows(std::size_t rows) {
    if (rows == 0) {
        throw std::invalid_argument("TernaryLinear '" + name_ + "': tileRows deve essere positivo");
    }
    tileRows_ = std::min(rows, outFeatures_);
}

void TernaryLinear::initialize(unsigned int seed) {
    // Inizializzazione a blocchi di righe: il tensore denso temporaneo
    // non supera mai il tile, anche su una matrice da 201 M elementi.
    std::vector<float> block(tileRows_ * inFeatures_);
    const ScopedMemory scope(MemoryArena::Workspace, block.size() * sizeof(float));

    // Scala di Xavier/Glorot: sqrt(6 / (fan_in + fan_out)). Il resto
    // del motore usa ancora un uniforme fisso in [-0,1, 0,1] (vedi
    // random_init.hpp), che per hidden 3072 sarebbe troppo grande di un
    // ordine di grandezza; qui serve un'inizializzazione che regga 28
    // layer, quindi si usa quella corretta invece di ereditare il
    // segnaposto.
    const float limit =
        std::sqrt(6.0F / static_cast<float>(inFeatures_ + outFeatures_));

    for (std::size_t first = 0; first < outFeatures_; first += tileRows_) {
        const std::size_t count = std::min(tileRows_, outFeatures_ - first);
        for (std::size_t r = 0; r < count; ++r) {
            // Un seme per riga: il risultato non dipende da tileRows_,
            // quindi cambiare la dimensione del tile non cambia il
            // modello inizializzato.
            const runtime::Tensor row = backend::cpu::randomTensor(
                {1, inFeatures_}, backend::cpu::seedFor(seed, first + r, 0x5B17U));
            for (std::size_t c = 0; c < inFeatures_; ++c) {
                // randomTensor produce uniforme in [-0,1, 0,1]:
                // riscalato all'intervallo di Xavier.
                block[r * inFeatures_ + c] = row.at(c) * (limit / 0.1F);
            }
        }
        weight_.quantizeRowsFrom(first, count, block.data());
    }
}

void TernaryLinear::loadDense(const runtime::Tensor& dense) {
    if (dense.rank() != 2 || dense.dim(0) != outFeatures_ || dense.dim(1) != inFeatures_) {
        throw std::invalid_argument("TernaryLinear '" + name_ + "': la matrice densa deve avere forma [" +
                                     std::to_string(outFeatures_) + ", " + std::to_string(inFeatures_) + "]");
    }
    weight_.quantizeRowsFrom(0, outFeatures_, dense.data().data());
}

void TernaryLinear::dequantizeTile(std::size_t firstRow, std::size_t rowCount, std::vector<float>& buffer) const {
    weight_.dequantizeRows(firstRow, rowCount, buffer.data());

    if (compute_ == ComputeDType::BF16) {
        // Il tile viene arrotondato al formato in cui il GEMM avverra'
        // davvero. Non e' quantizzazione simulata "cosmetica": il
        // backward differenzia esattamente questa operazione, perche'
        // usa lo stesso tile arrotondato (vedi backward()).
        for (std::size_t i = 0; i < rowCount * inFeatures_; ++i) {
            buffer[i] = backend::cpu::quantizeScalar(buffer[i], sema::DType::BF16);
        }
    }
}

runtime::Tensor TernaryLinear::forward(const runtime::Tensor& input) const {
    requireFeatureCount(input, inFeatures_, "l'ingresso", name_);
    const std::size_t rows = rowsOf(input);

    std::vector<std::size_t> outputShape = input.shape();
    outputShape.back() = outFeatures_;
    std::vector<float> output(rows * outFeatures_, 0.0F);

    std::vector<float> tile(tileRows_ * inFeatures_);
    const ScopedMemory scope(MemoryArena::Workspace, tile.size() * sizeof(float));

    const float* x = input.data().data();

    for (std::size_t firstRow = 0; firstRow < outFeatures_; firstRow += tileRows_) {
        const std::size_t count = std::min(tileRows_, outFeatures_ - firstRow);
        dequantizeTile(firstRow, count, tile);

        // Y[m, n] = somma_k X[m, k] * Wtile[n - firstRow, k]. E' il
        // prodotto X @ Wtile^T: la stessa forma di
        // backend::cpu::matmulTransposeB, scritta qui direttamente sul
        // tile perche' passare per un runtime::Tensor costringerebbe a
        // COPIARE il tile ad ogni blocco — proprio la memoria che
        // questo schema esiste per non spendere.
        for (std::size_t m = 0; m < rows; ++m) {
            const float* xRow = x + m * inFeatures_;
            float* outRow = output.data() + m * outFeatures_ + firstRow;
            for (std::size_t n = 0; n < count; ++n) {
                const float* wRow = tile.data() + n * inFeatures_;
                float sum = 0.0F;
                for (std::size_t k = 0; k < inFeatures_; ++k) {
                    sum += xRow[k] * wRow[k];
                }
                outRow[n] = sum;
            }
        }
    }

    return runtime::Tensor(std::move(outputShape), std::move(output));
}

runtime::Tensor TernaryLinear::backward(const runtime::Tensor& input, const runtime::Tensor& gradOutput,
                                         GradientSink* sink) const {
    requireFeatureCount(input, inFeatures_, "l'ingresso", name_);
    requireFeatureCount(gradOutput, outFeatures_, "il gradiente dell'uscita", name_);

    const std::size_t rows = rowsOf(input);
    if (rowsOf(gradOutput) != rows) {
        throw std::invalid_argument("TernaryLinear '" + name_ +
                                     "': ingresso e gradiente dell'uscita hanno un numero di righe diverso");
    }

    std::vector<float> gradInput(rows * inFeatures_, 0.0F);

    std::vector<float> tile(tileRows_ * inFeatures_);
    std::vector<float> weightGradBlock(tileRows_ * inFeatures_);
    const ScopedMemory tileScope(MemoryArena::Workspace, tile.size() * sizeof(float));

    const float* x = input.data().data();
    const float* dy = gradOutput.data().data();

    for (std::size_t firstRow = 0; firstRow < outFeatures_; firstRow += tileRows_) {
        const std::size_t count = std::min(tileRows_, outFeatures_ - firstRow);
        const std::size_t blockValues = count * inFeatures_;

        // Il blocco di gradiente esiste SOLO dentro questa iterazione:
        // viene azzerato, riempito, consegnato al sink e riusato dal
        // blocco successivo. La telemetria lo registra come vivo
        // esattamente per quel tempo — e' cosi' che il benchmark puo'
        // dimostrare, con un numero, che non esiste un buffer di
        // gradienti a dimensione modello.
        const ScopedMemory gradScope(MemoryArena::Gradient, blockValues * sizeof(float));
        GradientLifetimeStats& stats = gradientLifetimeStats();
        stats.liveBytes += blockValues * sizeof(float);
        stats.cumulativeBytes += blockValues * sizeof(float);
        ++stats.blocksProduced;
        stats.peakLiveBytes = std::max(stats.peakLiveBytes, stats.liveBytes);

        dequantizeTile(firstRow, count, tile);
        std::fill(weightGradBlock.begin(), weightGradBlock.begin() + static_cast<std::ptrdiff_t>(blockValues),
                  0.0F);

        for (std::size_t m = 0; m < rows; ++m) {
            const float* xRow = x + m * inFeatures_;
            const float* dyRow = dy + m * outFeatures_ + firstRow;
            float* dxRow = gradInput.data() + m * inFeatures_;

            for (std::size_t n = 0; n < count; ++n) {
                const float g = dyRow[n];
                if (g == 0.0F) {
                    continue;
                }
                const float* wRow = tile.data() + n * inFeatures_;
                float* gwRow = weightGradBlock.data() + n * inFeatures_;
                for (std::size_t k = 0; k < inFeatures_; ++k) {
                    // dX = dY @ W (con W nel layout [out, in]).
                    dxRow[k] += g * wRow[k];
                    // dW[n, k] = somma_m dY[m, n] * X[m, k]. Straight-
                    // through estimator: e' il gradiente rispetto al
                    // peso REALE (trit * scala), non rispetto al trit.
                    gwRow[k] += g * xRow[k];
                }
            }
        }

        if (sink != nullptr) {
            sink->consumeWeightGradientBlock(parameterId(), firstRow, count, weightGradBlock.data());
        }

        stats.liveBytes -= blockValues * sizeof(float);
        ++stats.blocksReleased;
    }

    return runtime::Tensor(input.shape(), std::move(gradInput));
}

}  // namespace blackforge::blackbit
