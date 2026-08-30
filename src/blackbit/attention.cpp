#include "blackforge/blackbit/attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "blackforge/blackbit/rope.hpp"
#include "blackforge/blackbit/telemetry.hpp"

namespace blackforge::blackbit {

namespace {

void applyRopeToTensor(runtime::Tensor& tensor, std::size_t heads, std::size_t headDim) {
    const std::size_t batch = tensor.dim(0);
    const std::size_t seq = tensor.dim(1);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t s = 0; s < seq; ++s) {
            float* row = tensor.data().data() + (b * seq + s) * heads * headDim;
            for (std::size_t h = 0; h < heads; ++h) {
                applyRope(row + h * headDim, headDim, s);
            }
        }
    }
}

}  // namespace

GqaAttention::GqaAttention(const std::string& namePrefix, const BlackBitConfig& config)
    : config_(config),
      q_(namePrefix + ".q", config.hiddenSize, config.queryDim(), config.ternaryGroupSize, 128),
      k_(namePrefix + ".k", config.hiddenSize, config.kvDim(), config.ternaryGroupSize, 128),
      v_(namePrefix + ".v", config.hiddenSize, config.kvDim(), config.ternaryGroupSize, 128),
      o_(namePrefix + ".o", config.queryDim(), config.hiddenSize, config.ternaryGroupSize, 128) {
    config.validate();
    if (config.headDim % 2 != 0) {
        throw std::invalid_argument("GqaAttention: head_dim deve essere pari (RoPE ruota coppie di componenti)");
    }
}

void GqaAttention::initialize(unsigned int seed) {
    q_.initialize(seed);
    k_.initialize(seed + 1);
    v_.initialize(seed + 2);
    o_.initialize(seed + 3);
}

void GqaAttention::setComputeDType(ComputeDType dtype) {
    q_.setComputeDType(dtype);
    k_.setComputeDType(dtype);
    v_.setComputeDType(dtype);
    o_.setComputeDType(dtype);
}

runtime::Tensor GqaAttention::forward(const runtime::Tensor& input, AttentionCache& cache) const {
    if (input.rank() != 3 || input.dim(2) != config_.hiddenSize) {
        throw std::invalid_argument("GqaAttention: atteso un ingresso [batch, seq, hidden]");
    }

    const std::size_t batch = input.dim(0);
    const std::size_t seq = input.dim(1);
    const std::size_t heads = config_.numHeads;
    const std::size_t kvHeads = config_.numKvHeads;
    const std::size_t headDim = config_.headDim;
    const std::size_t group = config_.headsPerKvGroup();
    const float scale = 1.0F / std::sqrt(static_cast<float>(headDim));

    cache.query = q_.forward(input);
    cache.key = k_.forward(input);
    cache.value = v_.forward(input);

    applyRopeToTensor(cache.query, heads, headDim);
    applyRopeToTensor(cache.key, kvHeads, headDim);

    cache.rowMax.assign(batch * heads * seq, 0.0F);
    cache.rowSum.assign(batch * heads * seq, 0.0F);

    std::vector<float> attnOutput(batch * seq * heads * headDim, 0.0F);
    const ScopedMemory scope(MemoryArena::Activation, attnOutput.size() * sizeof(float));

    const float* q = cache.query.data().data();
    const float* k = cache.key.data().data();
    const float* v = cache.value.data().data();

    // Un accumulatore per riga di query: e' l'unico stato O(headDim) del
    // ciclo. La matrice di score non viene mai scritta da nessuna parte.
    std::vector<float> accumulator(headDim);

    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < heads; ++h) {
            // Raggruppamento GQA: la testa di query h legge la testa
            // K/V h / group. Nessuna copia, solo un offset di indice.
            const std::size_t kvHead = h / group;

            for (std::size_t i = 0; i < seq; ++i) {
                const float* qRow = q + ((b * seq + i) * heads + h) * headDim;

                std::fill(accumulator.begin(), accumulator.end(), 0.0F);
                float runningMax = -std::numeric_limits<float>::infinity();
                float runningSum = 0.0F;

                // Solo j <= i: le posizioni oltre il limite causale non
                // vengono nemmeno visitate, invece di essere calcolate e
                // poi mascherate.
                for (std::size_t j = 0; j <= i; ++j) {
                    const float* kRow = k + ((b * seq + j) * kvHeads + kvHead) * headDim;

                    // Accumulo in double: e' il prodotto scalare che
                    // entra in un'esponenziale, dove un errore relativo
                    // si amplifica (requisito 15).
                    double dot = 0.0;
                    for (std::size_t d = 0; d < headDim; ++d) {
                        dot += static_cast<double>(qRow[d]) * static_cast<double>(kRow[d]);
                    }
                    const float score = static_cast<float>(dot) * scale;

                    // Softmax online: quando arriva un massimo nuovo si
                    // riscala cio' che si e' gia' accumulato, invece di
                    // rileggere tutto.
                    const float newMax = std::max(runningMax, score);
                    const float correction = runningMax == -std::numeric_limits<float>::infinity()
                                                  ? 0.0F
                                                  : std::exp(runningMax - newMax);
                    const float weight = std::exp(score - newMax);

                    const float* vRow = v + ((b * seq + j) * kvHeads + kvHead) * headDim;
                    for (std::size_t d = 0; d < headDim; ++d) {
                        accumulator[d] = accumulator[d] * correction + weight * vRow[d];
                    }
                    runningSum = runningSum * correction + weight;
                    runningMax = newMax;
                }

                const std::size_t statIndex = (b * heads + h) * seq + i;
                cache.rowMax[statIndex] = runningMax;
                cache.rowSum[statIndex] = runningSum;

                float* out = attnOutput.data() + ((b * seq + i) * heads + h) * headDim;
                for (std::size_t d = 0; d < headDim; ++d) {
                    out[d] = accumulator[d] / runningSum;
                }
            }
        }
    }

    cache.attnOutput = runtime::Tensor({batch, seq, heads * headDim}, std::move(attnOutput));
    return o_.forward(cache.attnOutput);
}

runtime::Tensor GqaAttention::backward(const runtime::Tensor& input, const runtime::Tensor& gradOutput,
                                        const AttentionCache& cache, GradientSink* sink) const {
    const std::size_t batch = input.dim(0);
    const std::size_t seq = input.dim(1);
    const std::size_t heads = config_.numHeads;
    const std::size_t kvHeads = config_.numKvHeads;
    const std::size_t headDim = config_.headDim;
    const std::size_t group = config_.headsPerKvGroup();
    const float scale = 1.0F / std::sqrt(static_cast<float>(headDim));

    const runtime::Tensor gradAttnOutput = o_.backward(cache.attnOutput, gradOutput, sink);

    std::vector<float> gradQ(batch * seq * heads * headDim, 0.0F);
    std::vector<float> gradK(batch * seq * kvHeads * headDim, 0.0F);
    std::vector<float> gradV(batch * seq * kvHeads * headDim, 0.0F);
    const ScopedMemory scope(MemoryArena::Gradient,
                              (gradQ.size() + gradK.size() + gradV.size()) * sizeof(float));

    const float* q = cache.query.data().data();
    const float* k = cache.key.data().data();
    const float* v = cache.value.data().data();
    const float* out = cache.attnOutput.data().data();
    const float* dOut = gradAttnOutput.data().data();

    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < heads; ++h) {
            const std::size_t kvHead = h / group;

            for (std::size_t i = 0; i < seq; ++i) {
                const std::size_t statIndex = (b * heads + h) * seq + i;
                const float rowMax = cache.rowMax[statIndex];
                const float rowSum = cache.rowSum[statIndex];

                const float* qRow = q + ((b * seq + i) * heads + h) * headDim;
                const float* outRow = out + ((b * seq + i) * heads + h) * headDim;
                const float* dOutRow = dOut + ((b * seq + i) * heads + h) * headDim;
                float* dqRow = gradQ.data() + ((b * seq + i) * heads + h) * headDim;

                // D = <uscita, gradiente dell'uscita>: e' il termine di
                // correzione della softmax-backward, e si ottiene
                // dall'uscita gia' calcolata senza dover materializzare
                // le probabilita' (stessa identita' usata da
                // fusedAttentionBackward nel backend CUDA).
                double correction = 0.0;
                for (std::size_t d = 0; d < headDim; ++d) {
                    correction += static_cast<double>(outRow[d]) * static_cast<double>(dOutRow[d]);
                }

                for (std::size_t j = 0; j <= i; ++j) {
                    const float* kRow = k + ((b * seq + j) * kvHeads + kvHead) * headDim;
                    const float* vRow = v + ((b * seq + j) * kvHeads + kvHead) * headDim;
                    float* dkRow = gradK.data() + ((b * seq + j) * kvHeads + kvHead) * headDim;
                    float* dvRow = gradV.data() + ((b * seq + j) * kvHeads + kvHead) * headDim;

                    double dot = 0.0;
                    double dProbability = 0.0;
                    for (std::size_t d = 0; d < headDim; ++d) {
                        dot += static_cast<double>(qRow[d]) * static_cast<double>(kRow[d]);
                        dProbability += static_cast<double>(dOutRow[d]) * static_cast<double>(vRow[d]);
                    }
                    // Ricalcolo della probabilita' dalle statistiche
                    // salvate: O(1) di memoria per riga invece di
                    // O(seq^2) di matrice conservata.
                    const float probability = std::exp(static_cast<float>(dot) * scale - rowMax) / rowSum;
                    const float gradScore =
                        probability * static_cast<float>(dProbability - correction) * scale;

                    for (std::size_t d = 0; d < headDim; ++d) {
                        dqRow[d] += gradScore * kRow[d];
                        // Piu' teste di query condividono questa testa
                        // K/V: i contributi si SOMMANO qui, che e'
                        // esattamente il gradiente corretto del
                        // raggruppamento GQA — senza aver mai duplicato
                        // K/V nel forward.
                        dkRow[d] += gradScore * qRow[d];
                        dvRow[d] += probability * dOutRow[d];
                    }
                }
            }
        }
    }

    // Backward di RoPE: e' una rotazione, quindi la sua trasposta e' la
    // rotazione opposta.
    runtime::Tensor gradQTensor({batch, seq, heads * headDim}, std::move(gradQ));
    runtime::Tensor gradKTensor({batch, seq, kvHeads * headDim}, std::move(gradK));
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t s = 0; s < seq; ++s) {
            for (std::size_t h = 0; h < heads; ++h) {
                applyRopeTranspose(gradQTensor.data().data() + ((b * seq + s) * heads + h) * headDim, headDim, s);
            }
            for (std::size_t h = 0; h < kvHeads; ++h) {
                applyRopeTranspose(gradKTensor.data().data() + ((b * seq + s) * kvHeads + h) * headDim, headDim,
                                    s);
            }
        }
    }

    const runtime::Tensor fromQ = q_.backward(input, gradQTensor, sink);
    const runtime::Tensor fromK = k_.backward(input, gradKTensor, sink);
    const runtime::Tensor fromV =
        v_.backward(input, runtime::Tensor({batch, seq, kvHeads * headDim}, std::move(gradV)), sink);

    std::vector<float> gradInput(input.elementCount());
    for (std::size_t i = 0; i < gradInput.size(); ++i) {
        gradInput[i] = fromQ.at(i) + fromK.at(i) + fromV.at(i);
    }
    return runtime::Tensor(input.shape(), std::move(gradInput));
}

}  // namespace blackforge::blackbit
