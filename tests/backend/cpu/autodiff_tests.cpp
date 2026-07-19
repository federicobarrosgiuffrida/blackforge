#include "blackforge/backend/cpu/autodiff.hpp"

#include <functional>

#include <gtest/gtest.h>

#include "blackforge/backend/cpu/ops.hpp"

using blackforge::runtime::Tensor;
namespace cpu = blackforge::backend::cpu;

namespace {

// Derivata numerica (differenze centrali) di f rispetto a t.at(index).
// E' il modo standard per verificare che una formula di backward sia
// corretta: si confronta col prodotto vettore-Jacobiano analitico.
float numericalDerivative(const std::function<float(const Tensor&)>& f, Tensor t, std::size_t index,
                           float eps = 1e-3F) {
    float original = t.at(index);
    t.at(index) = original + eps;
    float plus = f(t);
    t.at(index) = original - eps;
    float minus = f(t);
    return (plus - minus) / (2.0F * eps);
}

float dot(const Tensor& a, const Tensor& b) {
    float sum = 0.0F;
    for (std::size_t i = 0; i < a.elementCount(); ++i) {
        sum += a.at(i) * b.at(i);
    }
    return sum;
}

}  // namespace

TEST(AutodiffTest, MatmulBackwardCorrispondeAllaDerivataNumerica) {
    Tensor a({2, 3}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F});
    Tensor b({3, 2}, {0.7F, -0.1F, 0.2F, 0.3F, -0.4F, 0.5F});
    Tensor gradOutput({2, 2}, {1.0F, -1.0F, 0.5F, 2.0F});

    cpu::MatmulGrad analytic = cpu::matmulBackward(a, b, gradOutput);

    auto fA = [&](const Tensor& aVar) { return dot(cpu::matmul(aVar, b), gradOutput); };
    for (std::size_t i = 0; i < a.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dA.at(i), numericalDerivative(fA, a, i), 1e-2F) << "dA indice " << i;
    }

    auto fB = [&](const Tensor& bVar) { return dot(cpu::matmul(a, bVar), gradOutput); };
    for (std::size_t i = 0; i < b.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dB.at(i), numericalDerivative(fB, b, i), 1e-2F) << "dB indice " << i;
    }
}

TEST(AutodiffTest, AddBiasBackwardCorrispondeAllaDerivataNumerica) {
    Tensor input({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    Tensor bias({2}, {0.5F, -0.5F});
    Tensor gradOutput({2, 2}, {1.0F, 0.5F, -1.0F, 2.0F});

    cpu::AddBiasGrad analytic = cpu::addBiasBackward(gradOutput);

    auto fInput = [&](const Tensor& inVar) { return dot(cpu::addBias(inVar, bias), gradOutput); };
    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dInput.at(i), numericalDerivative(fInput, input, i), 1e-2F) << "dInput indice " << i;
    }

    auto fBias = [&](const Tensor& biasVar) { return dot(cpu::addBias(input, biasVar), gradOutput); };
    for (std::size_t i = 0; i < bias.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dBias.at(i), numericalDerivative(fBias, bias, i), 1e-2F) << "dBias indice " << i;
    }
}

TEST(AutodiffTest, SiluBackwardCorrispondeAllaDerivataNumerica) {
    Tensor input({4}, {-1.5F, -0.3F, 0.7F, 2.0F});
    Tensor gradOutput({4}, {1.0F, -0.5F, 2.0F, 0.3F});

    Tensor analytic = cpu::siluBackward(input, gradOutput);
    auto f = [&](const Tensor& x) { return dot(cpu::silu(x), gradOutput); };

    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(analytic.at(i), numericalDerivative(f, input, i), 1e-2F) << "indice " << i;
    }
}

TEST(AutodiffTest, ReluBackwardCorrispondeAllaDerivataNumerica) {
    // Valori lontani da zero: relu non e' differenziabile esattamente in 0.
    Tensor input({4}, {-2.0F, -0.4F, 0.6F, 3.0F});
    Tensor gradOutput({4}, {1.0F, -0.5F, 2.0F, 0.3F});

    Tensor analytic = cpu::reluBackward(input, gradOutput);
    auto f = [&](const Tensor& x) { return dot(cpu::relu(x), gradOutput); };

    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(analytic.at(i), numericalDerivative(f, input, i), 1e-2F) << "indice " << i;
    }
}

TEST(AutodiffTest, GeluBackwardCorrispondeAllaDerivataNumerica) {
    Tensor input({4}, {-1.5F, -0.3F, 0.7F, 2.0F});
    Tensor gradOutput({4}, {1.0F, -0.5F, 2.0F, 0.3F});

    Tensor analytic = cpu::geluBackward(input, gradOutput);
    auto f = [&](const Tensor& x) { return dot(cpu::gelu(x), gradOutput); };

    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(analytic.at(i), numericalDerivative(f, input, i), 1e-2F) << "indice " << i;
    }
}

TEST(AutodiffTest, SoftmaxBackwardCorrispondeAllaDerivataNumerica) {
    Tensor input({2, 3}, {0.5F, -1.2F, 0.3F, 2.0F, 0.1F, -0.5F});
    Tensor gradOutput({2, 3}, {1.0F, -0.5F, 2.0F, 0.3F, -1.0F, 0.7F});

    Tensor analytic = cpu::softmaxBackward(input, gradOutput);
    auto f = [&](const Tensor& x) { return dot(cpu::softmax(x), gradOutput); };

    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(analytic.at(i), numericalDerivative(f, input, i), 1e-2F) << "indice " << i;
    }
}

TEST(AutodiffTest, RmsnormBackwardCorrispondeAllaDerivataNumerica) {
    // Batch di 2 righe, cosi' la formula di backward deve normalizzare
    // correttamente riga per riga (non mescolare le righe tra loro).
    Tensor input({2, 3}, {0.5F, -1.2F, 0.3F, 2.0F, 0.1F, -0.5F});
    Tensor gradOutput({2, 3}, {1.0F, -0.5F, 2.0F, 0.3F, -1.0F, 0.7F});

    Tensor analytic = cpu::rmsnormBackward(input, gradOutput);
    auto f = [&](const Tensor& x) { return dot(cpu::rmsnorm(x), gradOutput); };

    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(analytic.at(i), numericalDerivative(f, input, i), 1e-2F) << "indice " << i;
    }
}

TEST(AutodiffTest, RmsnormBackwardCorrispondeAllaDerivataNumericaARango3) {
    // Conferma che la generalizzazione a rango >= 2 sia corretta anche
    // nel backward, non solo nel forward: [batch=2, seq=2, dim=3].
    Tensor input3({2, 2, 3}, {0.5F, -1.2F, 0.3F, 2.0F, 0.1F, -0.5F, -0.8F, 1.5F, 0.2F, 0.4F, -0.3F, 1.1F});
    Tensor gradOutput3({2, 2, 3}, {1.0F, -0.5F, 2.0F, 0.3F, -1.0F, 0.7F, 0.6F, 0.2F, -0.4F, -0.9F, 1.3F, 0.5F});

    Tensor analytic3 = cpu::rmsnormBackward(input3, gradOutput3);
    auto f3 = [&](const Tensor& x) { return dot(cpu::rmsnorm(x), gradOutput3); };

    for (std::size_t i = 0; i < input3.elementCount(); ++i) {
        EXPECT_NEAR(analytic3.at(i), numericalDerivative(f3, input3, i), 1e-2F) << "indice " << i;
    }
}

TEST(AutodiffTest, SoftmaxBackwardCorrispondeAllaDerivataNumericaARango3) {
    Tensor input({2, 2, 3}, {0.5F, -1.2F, 0.3F, 2.0F, 0.1F, -0.5F, -0.8F, 1.5F, 0.2F, 0.4F, -0.3F, 1.1F});
    Tensor gradOutput({2, 2, 3}, {1.0F, -0.5F, 2.0F, 0.3F, -1.0F, 0.7F, 0.6F, 0.2F, -0.4F, -0.9F, 1.3F, 0.5F});

    Tensor analytic = cpu::softmaxBackward(input, gradOutput);
    auto f = [&](const Tensor& x) { return dot(cpu::softmax(x), gradOutput); };

    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(analytic.at(i), numericalDerivative(f, input, i), 1e-2F) << "indice " << i;
    }
}

TEST(AutodiffTest, MatmulTransposeBBackwardCorrispondeAllaDerivataNumerica) {
    Tensor a({2, 3}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F});
    Tensor b({4, 3}, {0.7F, -0.1F, 0.2F, 0.3F, -0.4F, 0.5F, 0.2F, 0.1F, -0.3F, -0.6F, 0.4F, 0.1F});
    Tensor gradOutput({2, 4}, {1.0F, -1.0F, 0.5F, 2.0F, -0.3F, 0.8F, -1.2F, 0.4F});

    cpu::MatmulTransposeBGrad analytic = cpu::matmulTransposeBBackward(a, b, gradOutput);

    auto fA = [&](const Tensor& aVar) { return dot(cpu::matmulTransposeB(aVar, b), gradOutput); };
    for (std::size_t i = 0; i < a.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dA.at(i), numericalDerivative(fA, a, i), 1e-2F) << "dA indice " << i;
    }

    auto fB = [&](const Tensor& bVar) { return dot(cpu::matmulTransposeB(a, bVar), gradOutput); };
    for (std::size_t i = 0; i < b.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dB.at(i), numericalDerivative(fB, b, i), 1e-2F) << "dB indice " << i;
    }
}

TEST(AutodiffTest, EmbeddingLookupBackwardAccumulaSuTokenRipetuti) {
    // vocabolario di 3, dim=2; token id [0, 1, 0] (il token 0 compare
    // due volte: il suo gradiente deve essere la SOMMA dei contributi
    // di entrambe le occorrenze, non solo l'ultima).
    Tensor tokenIds({1, 3}, {0.0F, 1.0F, 0.0F});
    Tensor gradOutputEmb({1, 3, 2}, {1.0F, 2.0F, 10.0F, 20.0F, 3.0F, 4.0F});

    Tensor dTable = cpu::embeddingLookupBackward(tokenIds, gradOutputEmb, /*vocabSize=*/3);
    ASSERT_EQ(dTable.shape(), (std::vector<std::size_t>{3, 2}));

    // riga 0 (token 0): somma delle occorrenze in posizione 0 e 2.
    EXPECT_FLOAT_EQ(dTable.at(0), 1.0F + 3.0F);
    EXPECT_FLOAT_EQ(dTable.at(1), 2.0F + 4.0F);
    // riga 1 (token 1): solo la posizione 1.
    EXPECT_FLOAT_EQ(dTable.at(2), 10.0F);
    EXPECT_FLOAT_EQ(dTable.at(3), 20.0F);
    // riga 2 (token 2, mai usato): gradiente zero.
    EXPECT_FLOAT_EQ(dTable.at(4), 0.0F);
    EXPECT_FLOAT_EQ(dTable.at(5), 0.0F);
}

TEST(AutodiffTest, EmbeddingLookupBackwardCorrispondeAllaDerivataNumericaDellaTabella) {
    Tensor tokenIds({1, 3}, {0.0F, 1.0F, 0.0F});
    Tensor table({2, 2}, {0.5F, -0.5F, 1.0F, 2.0F});
    Tensor gradOutput({1, 3, 2}, {1.0F, 2.0F, 10.0F, 20.0F, 3.0F, 4.0F});

    Tensor analytic = cpu::embeddingLookupBackward(tokenIds, gradOutput, /*vocabSize=*/2);
    auto f = [&](const Tensor& t) { return dot(cpu::embeddingLookup(tokenIds, t), gradOutput); };

    for (std::size_t i = 0; i < table.elementCount(); ++i) {
        EXPECT_NEAR(analytic.at(i), numericalDerivative(f, table, i), 1e-2F) << "indice " << i;
    }
}

TEST(AutodiffTest, AddPositionalEmbeddingBackwardCorrispondeAllaDerivataNumericaDellaTabella) {
    Tensor input({2, 2, 2}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F, 0.2F, -0.1F});
    Tensor table({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    Tensor gradOutput({2, 2, 2}, {1.0F, -0.5F, 2.0F, 0.3F, -1.0F, 0.7F, 0.6F, 0.2F});

    cpu::PositionalEmbeddingGrad analytic = cpu::addPositionalEmbeddingBackward(gradOutput, /*maxSeqLen=*/2);

    // dInput deve essere esattamente gradOutput (l'addizione e'
    // un'identita' nel gradiente).
    for (std::size_t i = 0; i < gradOutput.elementCount(); ++i) {
        EXPECT_FLOAT_EQ(analytic.dInput.at(i), gradOutput.at(i)) << "indice " << i;
    }

    auto f = [&](const Tensor& t) { return dot(cpu::addPositionalEmbedding(input, t), gradOutput); };
    for (std::size_t i = 0; i < table.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dTable.at(i), numericalDerivative(f, table, i), 1e-2F) << "indice " << i;
    }
}

TEST(AutodiffTest, FeedForwardBackwardCorrispondeAllaDerivataNumerica) {
    Tensor input({1, 2, 3}, {0.3F, -0.6F, 0.2F, -0.4F, 0.5F, 0.1F});
    Tensor w1({3, 4}, {0.2F, -0.1F, 0.3F, 0.1F, -0.2F, 0.4F, 0.1F, -0.3F, 0.3F, 0.2F, -0.1F, 0.2F});
    Tensor b1({4}, {0.1F, -0.1F, 0.05F, 0.0F});
    Tensor w2({4, 3}, {0.1F, -0.2F, 0.3F, 0.2F, 0.1F, -0.1F, -0.3F, 0.2F, 0.1F, 0.1F, -0.1F, 0.2F});
    Tensor b2({3}, {0.05F, -0.05F, 0.1F});
    Tensor gradOutput({1, 2, 3}, {1.0F, -0.5F, 0.3F, -0.2F, 0.7F, -0.4F});

    cpu::FeedForwardGrad analytic = cpu::feedForwardBackward(input, w1, b1, w2, b2, gradOutput);

    auto fInput = [&](const Tensor& x) { return dot(cpu::feedForward(x, w1, b1, w2, b2), gradOutput); };
    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dInput.at(i), numericalDerivative(fInput, input, i), 1e-2F) << "dInput indice " << i;
    }

    auto fW1 = [&](const Tensor& w) { return dot(cpu::feedForward(input, w, b1, w2, b2), gradOutput); };
    for (std::size_t i = 0; i < w1.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dW1.at(i), numericalDerivative(fW1, w1, i), 1e-2F) << "dW1 indice " << i;
    }

    auto fB1 = [&](const Tensor& b) { return dot(cpu::feedForward(input, w1, b, w2, b2), gradOutput); };
    for (std::size_t i = 0; i < b1.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dB1.at(i), numericalDerivative(fB1, b1, i), 1e-2F) << "dB1 indice " << i;
    }

    auto fW2 = [&](const Tensor& w) { return dot(cpu::feedForward(input, w1, b1, w, b2), gradOutput); };
    for (std::size_t i = 0; i < w2.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dW2.at(i), numericalDerivative(fW2, w2, i), 1e-2F) << "dW2 indice " << i;
    }

    auto fB2 = [&](const Tensor& b) { return dot(cpu::feedForward(input, w1, b1, w2, b), gradOutput); };
    for (std::size_t i = 0; i < b2.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dB2.at(i), numericalDerivative(fB2, b2, i), 1e-2F) << "dB2 indice " << i;
    }
}

TEST(AutodiffTest, SelfAttentionBackwardCorrispondeAllaDerivataNumerica) {
    // batch=1, seq=3, dim=4, numHeads=2 (headDim=2): abbastanza piccolo
    // da restare veloce, abbastanza grande da esercitare davvero la
    // maschera causale (seq>1) e la suddivisione multi-testa (numHeads>1).
    Tensor input({1, 3, 4}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F, 0.2F, -0.1F, 0.3F, 0.2F, -0.4F, 0.5F});
    Tensor wq({4, 4}, {0.3F, -0.1F, 0.2F, 0.05F, 0.1F, 0.4F, -0.2F, 0.15F, -0.3F, 0.2F, 0.1F, -0.05F, 0.2F, -0.1F,
                        0.3F, 0.1F});
    Tensor wk({4, 4}, {0.1F, 0.2F, -0.1F, 0.3F, -0.2F, 0.1F, 0.4F, -0.1F, 0.3F, -0.3F, 0.1F, 0.2F, -0.1F, 0.2F,
                        -0.2F, 0.1F});
    Tensor wv({4, 4}, {0.2F, 0.1F, -0.1F, 0.3F, 0.1F, -0.2F, 0.3F, 0.1F, -0.1F, 0.3F, 0.2F, -0.1F, 0.3F, 0.1F, -0.2F,
                        0.2F});
    Tensor wout({4, 4}, {0.1F, -0.1F, 0.2F, 0.1F, 0.2F, 0.1F, -0.1F, 0.2F, -0.1F, 0.2F, 0.1F, -0.1F, 0.1F, 0.2F,
                          -0.1F, 0.2F});
    Tensor gradOutput({1, 3, 4}, {1.0F, -0.5F, 0.3F, -0.2F, 0.7F, -0.4F, 0.2F, -0.1F, -0.3F, 0.6F, 0.1F, -0.5F});

    cpu::SelfAttentionGrad analytic = cpu::selfAttentionBackward(input, wq, wk, wv, wout, /*numHeads=*/2, gradOutput);

    auto fInput = [&](const Tensor& x) {
        return dot(cpu::selfAttention(x, wq, wk, wv, wout, 2), gradOutput);
    };
    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dInput.at(i), numericalDerivative(fInput, input, i), 3e-2F) << "dInput indice " << i;
    }

    auto fWq = [&](const Tensor& w) { return dot(cpu::selfAttention(input, w, wk, wv, wout, 2), gradOutput); };
    for (std::size_t i = 0; i < wq.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dWq.at(i), numericalDerivative(fWq, wq, i), 3e-2F) << "dWq indice " << i;
    }

    auto fWk = [&](const Tensor& w) { return dot(cpu::selfAttention(input, wq, w, wv, wout, 2), gradOutput); };
    for (std::size_t i = 0; i < wk.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dWk.at(i), numericalDerivative(fWk, wk, i), 3e-2F) << "dWk indice " << i;
    }

    auto fWv = [&](const Tensor& w) { return dot(cpu::selfAttention(input, wq, wk, w, wout, 2), gradOutput); };
    for (std::size_t i = 0; i < wv.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dWv.at(i), numericalDerivative(fWv, wv, i), 3e-2F) << "dWv indice " << i;
    }

    auto fWout = [&](const Tensor& w) { return dot(cpu::selfAttention(input, wq, wk, wv, w, 2), gradOutput); };
    for (std::size_t i = 0; i < wout.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dWout.at(i), numericalDerivative(fWout, wout, i), 3e-2F) << "dWout indice " << i;
    }
}

TEST(AutodiffTest, SelfAttentionBackwardFunzionaAncheConUnaSolaTesta) {
    // numHeads=1: caso limite (nessuna suddivisione multi-testa).
    Tensor input({1, 2, 2}, {0.2F, -0.3F, 0.4F, 0.1F});
    Tensor wq({2, 2}, {0.3F, -0.1F, 0.2F, 0.15F});
    Tensor wk({2, 2}, {0.1F, 0.2F, -0.2F, 0.1F});
    Tensor wv({2, 2}, {0.2F, 0.1F, -0.1F, 0.3F});
    Tensor wout({2, 2}, {0.1F, -0.1F, 0.2F, 0.1F});
    Tensor gradOutput({1, 2, 2}, {1.0F, -0.5F, 0.3F, -0.2F});

    cpu::SelfAttentionGrad analytic = cpu::selfAttentionBackward(input, wq, wk, wv, wout, /*numHeads=*/1, gradOutput);
    auto f = [&](const Tensor& x) { return dot(cpu::selfAttention(x, wq, wk, wv, wout, 1), gradOutput); };

    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_NEAR(analytic.dInput.at(i), numericalDerivative(f, input, i), 3e-2F) << "indice " << i;
    }
}
