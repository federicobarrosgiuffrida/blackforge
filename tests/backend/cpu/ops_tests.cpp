#include "blackforge/backend/cpu/ops.hpp"

#include <cmath>

#include <gtest/gtest.h>

using blackforge::runtime::Tensor;
namespace cpu = blackforge::backend::cpu;

TEST(CpuOpsTest, AddSommaElementwise) {
    Tensor a({2}, {1.0F, 2.0F});
    Tensor b({2}, {10.0F, 20.0F});
    Tensor result = cpu::add(a, b);

    EXPECT_FLOAT_EQ(result.at(0), 11.0F);
    EXPECT_FLOAT_EQ(result.at(1), 22.0F);
}

TEST(CpuOpsTest, AddLanciaSuFormeIncompatibili) {
    Tensor a({2}, {1.0F, 2.0F});
    Tensor b({3}, {1.0F, 2.0F, 3.0F});
    EXPECT_THROW(cpu::add(a, b), std::invalid_argument);
}

TEST(CpuOpsTest, AddBiasTrasmetteSuOgniRiga) {
    Tensor input({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    Tensor bias({2}, {10.0F, 20.0F});
    Tensor result = cpu::addBias(input, bias);

    EXPECT_FLOAT_EQ(result.at(0), 11.0F);
    EXPECT_FLOAT_EQ(result.at(1), 22.0F);
    EXPECT_FLOAT_EQ(result.at(2), 13.0F);
    EXPECT_FLOAT_EQ(result.at(3), 24.0F);
}

TEST(CpuOpsTest, MatmulCalcolaIlProdottoCorretto) {
    Tensor a({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    Tensor b({2, 2}, {5.0F, 6.0F, 7.0F, 8.0F});
    Tensor result = cpu::matmul(a, b);

    ASSERT_EQ(result.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ(result.at(0), 19.0F);  // 1*5 + 2*7
    EXPECT_FLOAT_EQ(result.at(1), 22.0F);  // 1*6 + 2*8
    EXPECT_FLOAT_EQ(result.at(2), 43.0F);  // 3*5 + 4*7
    EXPECT_FLOAT_EQ(result.at(3), 50.0F);  // 3*6 + 4*8
}

TEST(CpuOpsTest, MatmulLanciaSuDimensioniIncompatibili) {
    Tensor a({2, 3}, std::vector<float>(6, 1.0F));
    Tensor b({2, 2}, std::vector<float>(4, 1.0F));
    EXPECT_THROW(cpu::matmul(a, b), std::invalid_argument);
}

TEST(CpuOpsTest, LinearCombinaMatmulEBias) {
    // input [1,2] (1x2), weight identita' (2x2), bias [10, 20]
    Tensor input({1, 2}, {3.0F, 4.0F});
    Tensor weight({2, 2}, {1.0F, 0.0F, 0.0F, 1.0F});
    Tensor bias({2}, {10.0F, 20.0F});

    Tensor result = cpu::linear(input, weight, bias);
    EXPECT_FLOAT_EQ(result.at(0), 13.0F);
    EXPECT_FLOAT_EQ(result.at(1), 24.0F);
}

TEST(CpuOpsTest, ReluAzzeraIValoriNegativi) {
    Tensor input({3}, {-2.0F, 0.0F, 3.0F});
    Tensor result = cpu::relu(input);

    EXPECT_FLOAT_EQ(result.at(0), 0.0F);
    EXPECT_FLOAT_EQ(result.at(1), 0.0F);
    EXPECT_FLOAT_EQ(result.at(2), 3.0F);
}

TEST(CpuOpsTest, SiluValeZeroInZeroEConvergeAXPerValoriGrandi) {
    Tensor input({3}, {0.0F, 20.0F, -20.0F});
    Tensor result = cpu::silu(input);

    EXPECT_FLOAT_EQ(result.at(0), 0.0F);
    EXPECT_NEAR(result.at(1), 20.0F, 0.01F);  // sigmoid(20) ~= 1
    EXPECT_NEAR(result.at(2), 0.0F, 0.01F);   // sigmoid(-20) ~= 0
}

TEST(CpuOpsTest, GeluValeZeroInZeroEConvergeAXPerValoriGrandi) {
    Tensor input({3}, {0.0F, 20.0F, -20.0F});
    Tensor result = cpu::gelu(input);

    EXPECT_FLOAT_EQ(result.at(0), 0.0F);
    EXPECT_NEAR(result.at(1), 20.0F, 0.01F);
    EXPECT_NEAR(result.at(2), 0.0F, 0.01F);
}

TEST(CpuOpsTest, RmsnormNormalizzaOgniRigaARadiceMediaQuadraticaUnitaria) {
    Tensor input({2, 4}, {1.0F, 2.0F, 3.0F, 4.0F, -1.0F, -1.0F, -1.0F, -1.0F});
    Tensor result = cpu::rmsnorm(input);

    ASSERT_EQ(result.shape(), (std::vector<std::size_t>{2, 4}));
    for (std::size_t row = 0; row < 2; ++row) {
        double sumSquares = 0.0;
        for (std::size_t col = 0; col < 4; ++col) {
            float v = result.at(row * 4 + col);
            sumSquares += static_cast<double>(v) * static_cast<double>(v);
        }
        // Con eps trascurabile rispetto ai valori usati, la RMS
        // dell'uscita normalizzata deve essere vicina a 1.
        double rms = std::sqrt(sumSquares / 4.0);
        EXPECT_NEAR(rms, 1.0, 1e-3) << "riga " << row;
    }
}

TEST(CpuOpsTest, RmsnormPreservaIlSegnoEIlRapportoTraElementi) {
    Tensor input({1, 2}, {2.0F, -4.0F});
    Tensor result = cpu::rmsnorm(input);

    EXPECT_GT(result.at(0), 0.0F);
    EXPECT_LT(result.at(1), 0.0F);
    // Il rapporto tra i due elementi (entrambi divisi per la stessa
    // costante rms) deve restare invariato: -2.
    EXPECT_NEAR(result.at(1) / result.at(0), -2.0F, 1e-4F);
}

TEST(CpuOpsTest, RmsnormLanciaSeNonERango2) {
    Tensor input({4}, {1.0F, 2.0F, 3.0F, 4.0F});
    EXPECT_THROW((void)cpu::rmsnorm(input), std::invalid_argument);
}

TEST(CpuOpsTest, SoftmaxOgniRigaSommaAUno) {
    Tensor input({2, 3}, {1.0F, 2.0F, 3.0F, -1.0F, 0.0F, 1.0F});
    Tensor result = cpu::softmax(input);

    for (std::size_t row = 0; row < 2; ++row) {
        float sum = result.at(row * 3) + result.at(row * 3 + 1) + result.at(row * 3 + 2);
        EXPECT_NEAR(sum, 1.0F, 1e-5F) << "riga " << row;
    }
}

TEST(CpuOpsTest, SoftmaxTuttiIValoriSonoPositivi) {
    // Valori abbastanza vicini da non far sottostimare exp() a zero per
    // underflow (softmax(-100 rispetto a un massimo di 50) sarebbe
    // legittimamente 0.0f in float32, non un bug: qui si verifica la
    // proprieta' "tutti positivi" in un caso dove ha senso attenderla).
    Tensor input({1, 4}, {-10.0F, 0.0F, 5.0F, 3.0F});
    Tensor result = cpu::softmax(input);
    for (std::size_t i = 0; i < result.elementCount(); ++i) {
        EXPECT_GT(result.at(i), 0.0F) << "indice " << i;
    }
}

TEST(CpuOpsTest, SoftmaxSuLogitUniformiDaProbabilitaUniformi) {
    Tensor input({1, 4}, {2.0F, 2.0F, 2.0F, 2.0F});
    Tensor result = cpu::softmax(input);
    for (std::size_t i = 0; i < result.elementCount(); ++i) {
        EXPECT_NEAR(result.at(i), 0.25F, 1e-5F) << "indice " << i;
    }
}

TEST(CpuOpsTest, SoftmaxERobustoALogitGrandi) {
    // Senza sottrazione del massimo, exp(1000) andrebbe in overflow
    // (inf) e il risultato sarebbe NaN: verifica la stabilita' numerica.
    Tensor input({1, 3}, {1000.0F, 1000.0F, 1000.0F});
    Tensor result = cpu::softmax(input);
    for (std::size_t i = 0; i < result.elementCount(); ++i) {
        EXPECT_NEAR(result.at(i), 1.0F / 3.0F, 1e-4F) << "indice " << i;
    }
}

TEST(CpuOpsTest, SoftmaxLanciaSeNonERango2) {
    Tensor input({4}, {1.0F, 2.0F, 3.0F, 4.0F});
    EXPECT_THROW((void)cpu::softmax(input), std::invalid_argument);
}

// --- Generalizzazione a rango 3 [batch, seq, features] ---

TEST(CpuOpsTest, LinearFunzionaSuTensoriARango3) {
    // [batch=2, seq=3, inFeatures=2] |> linear(2 -> 1)
    Tensor input({2, 3, 2}, {1.0F, 1.0F, 2.0F, 2.0F, 3.0F, 3.0F, 4.0F, 4.0F, 5.0F, 5.0F, 6.0F, 6.0F});
    Tensor weight({2, 1}, {1.0F, 1.0F});  // somma le due feature
    Tensor bias({1}, {0.0F});

    Tensor result = cpu::linear(input, weight, bias);
    ASSERT_EQ(result.shape(), (std::vector<std::size_t>{2, 3, 1}));
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(result.at(i), input.at(i * 2) + input.at(i * 2 + 1)) << "token " << i;
    }
}

TEST(CpuOpsTest, RmsnormFunzionaSuTensoriARango3EOgniTokenESeparato) {
    Tensor input({1, 2, 4}, {1.0F, 2.0F, 3.0F, 4.0F, -1.0F, -1.0F, -1.0F, -1.0F});
    Tensor result = cpu::rmsnorm(input);
    ASSERT_EQ(result.shape(), (std::vector<std::size_t>{1, 2, 4}));

    for (std::size_t token = 0; token < 2; ++token) {
        double sumSquares = 0.0;
        for (std::size_t d = 0; d < 4; ++d) {
            float v = result.at(token * 4 + d);
            sumSquares += static_cast<double>(v) * static_cast<double>(v);
        }
        EXPECT_NEAR(std::sqrt(sumSquares / 4.0), 1.0, 1e-3) << "token " << token;
    }
}

TEST(CpuOpsTest, MatmulTransposeBCalcolaAPerBTrasposta) {
    // a: [2,3], b: [2,3] -> a @ b^T: [2,2]
    Tensor a({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    Tensor b({2, 3}, {1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F});

    Tensor result = cpu::matmulTransposeB(a, b);
    ASSERT_EQ(result.shape(), (std::vector<std::size_t>{2, 2}));
    // riga 0 di a . riga 0 di b = 1*1+2*0+3*0 = 1; riga 0 di a . riga 1 di b = 1*0+2*1+3*0 = 2
    EXPECT_FLOAT_EQ(result.at(0), 1.0F);
    EXPECT_FLOAT_EQ(result.at(1), 2.0F);
    // riga 1 di a . riga 0 di b = 4; riga 1 di a . riga 1 di b = 5
    EXPECT_FLOAT_EQ(result.at(2), 4.0F);
    EXPECT_FLOAT_EQ(result.at(3), 5.0F);
}

TEST(CpuOpsTest, MatmulTransposeBCorrispondeATrasporreEChiamareMatmul) {
    Tensor a({3, 2}, {0.5F, -1.0F, 2.0F, 0.3F, -0.7F, 1.2F});
    Tensor b({4, 2}, {1.0F, 2.0F, -1.0F, 0.5F, 0.3F, -0.2F, 2.0F, 1.0F});

    // Trasposizione esplicita di b: [2,4]
    std::vector<float> bTransposedData(8);
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            bTransposedData[j * 4 + i] = b.at(i * 2 + j);
        }
    }
    Tensor bTransposed({2, 4}, bTransposedData);

    Tensor expected = cpu::matmul(a, bTransposed);
    Tensor actual = cpu::matmulTransposeB(a, b);

    ASSERT_EQ(actual.shape(), expected.shape());
    for (std::size_t i = 0; i < expected.elementCount(); ++i) {
        EXPECT_NEAR(actual.at(i), expected.at(i), 1e-5F) << "indice " << i;
    }
}

TEST(CpuOpsTest, EmbeddingLookupEstraeLeRigheCorretteDallaTabella) {
    // vocabolario di 3 token, dim=2
    Tensor table({3, 2}, {10.0F, 11.0F, 20.0F, 21.0F, 30.0F, 31.0F});
    // batch=1, seq=3: token id [0, 2, 1]
    Tensor tokenIds({1, 3}, {0.0F, 2.0F, 1.0F});

    Tensor result = cpu::embeddingLookup(tokenIds, table);
    ASSERT_EQ(result.shape(), (std::vector<std::size_t>{1, 3, 2}));
    EXPECT_FLOAT_EQ(result.at(0), 10.0F);
    EXPECT_FLOAT_EQ(result.at(1), 11.0F);
    EXPECT_FLOAT_EQ(result.at(2), 30.0F);
    EXPECT_FLOAT_EQ(result.at(3), 31.0F);
    EXPECT_FLOAT_EQ(result.at(4), 20.0F);
    EXPECT_FLOAT_EQ(result.at(5), 21.0F);
}

TEST(CpuOpsTest, EmbeddingLookupLanciaSeIlTokenIdENegativoOFuoriVocabolario) {
    Tensor table({3, 2}, {10.0F, 11.0F, 20.0F, 21.0F, 30.0F, 31.0F});

    Tensor tooLarge({1, 1}, {3.0F});
    EXPECT_THROW((void)cpu::embeddingLookup(tooLarge, table), std::invalid_argument);

    Tensor negative({1, 1}, {-1.0F});
    EXPECT_THROW((void)cpu::embeddingLookup(negative, table), std::invalid_argument);
}

TEST(CpuOpsTest, AddPositionalEmbeddingSommaPerPosizioneETrasmetteSulBatch) {
    // batch=2, seq=2, dim=2
    Tensor input({2, 2, 2}, {0.0F, 0.0F, 0.0F, 0.0F, 100.0F, 100.0F, 100.0F, 100.0F});
    Tensor table({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});  // posizione 0 -> [1,2], posizione 1 -> [3,4]

    Tensor result = cpu::addPositionalEmbedding(input, table);
    ASSERT_EQ(result.shape(), (std::vector<std::size_t>{2, 2, 2}));
    // esempio 0, posizione 0
    EXPECT_FLOAT_EQ(result.at(0), 1.0F);
    EXPECT_FLOAT_EQ(result.at(1), 2.0F);
    // esempio 0, posizione 1
    EXPECT_FLOAT_EQ(result.at(2), 3.0F);
    EXPECT_FLOAT_EQ(result.at(3), 4.0F);
    // esempio 1, posizione 0 (stesso vettore posizionale dell'esempio 0)
    EXPECT_FLOAT_EQ(result.at(4), 101.0F);
    EXPECT_FLOAT_EQ(result.at(5), 102.0F);
}

TEST(CpuOpsTest, AddPositionalEmbeddingLanciaSeLaSequenzaSuperaMaxSeqLen) {
    Tensor input({1, 3, 2}, std::vector<float>(6, 0.0F));
    Tensor table({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});  // maxSeqLen=2 < seq=3
    EXPECT_THROW((void)cpu::addPositionalEmbedding(input, table), std::invalid_argument);
}

// --- feedForward / selfAttention (blocchi transformer con residual e pre-norm) ---

TEST(CpuOpsTest, FeedForwardConPesiNulliRestituisceLInputInvariato) {
    // Con w1/b1/w2/b2 tutti a zero, il ramo FFN contribuisce 0: il
    // residual lascia passare l'input esattamente invariato.
    Tensor input({1, 2, 3}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F});
    Tensor w1 = Tensor::zeros({3, 4});
    Tensor b1 = Tensor::zeros({4});
    Tensor w2 = Tensor::zeros({4, 3});
    Tensor b2 = Tensor::zeros({3});

    Tensor result = cpu::feedForward(input, w1, b1, w2, b2);
    ASSERT_EQ(result.shape(), input.shape());
    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_FLOAT_EQ(result.at(i), input.at(i)) << "indice " << i;
    }
}

TEST(CpuOpsTest, SelfAttentionConPesiNulliRestituisceLInputInvariato) {
    // Con Wq/Wk/Wv/Wout a zero, Q=K=V=0 -> gli score sono 0 sulle
    // posizioni ammesse dalla maschera causale (softmax uniforme su di
    // esse) ma V=0 rende comunque nullo l'output di ogni testa: il
    // residual lascia passare l'input invariato, come per feedForward.
    Tensor input({1, 3, 4}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F, 0.2F, -0.1F, 0.3F, 0.2F, -0.4F, 0.5F});
    Tensor wq = Tensor::zeros({4, 4});
    Tensor wk = Tensor::zeros({4, 4});
    Tensor wv = Tensor::zeros({4, 4});
    Tensor wout = Tensor::zeros({4, 4});

    Tensor result = cpu::selfAttention(input, wq, wk, wv, wout, /*numHeads=*/2);
    ASSERT_EQ(result.shape(), input.shape());
    for (std::size_t i = 0; i < input.elementCount(); ++i) {
        EXPECT_FLOAT_EQ(result.at(i), input.at(i)) << "indice " << i;
    }
}

TEST(CpuOpsTest, SelfAttentionRispettaLaMascheraCausale) {
    // Se si cambia il token FUTURO (posizione 2) senza toccare gli
    // altri, l'uscita alle posizioni 0 e 1 non deve cambiare: la
    // maschera causale impedisce loro di "vedere" il futuro.
    Tensor wq({4, 4}, {0.3F, -0.1F, 0.2F, 0.05F, 0.1F, 0.4F, -0.2F, 0.15F, -0.3F, 0.2F, 0.1F, -0.05F, 0.2F, -0.1F,
                        0.3F, 0.1F});
    Tensor wk({4, 4}, {0.1F, 0.2F, -0.1F, 0.3F, -0.2F, 0.1F, 0.4F, -0.1F, 0.3F, -0.3F, 0.1F, 0.2F, -0.1F, 0.2F,
                        -0.2F, 0.1F});
    Tensor wv({4, 4}, {0.2F, 0.1F, -0.1F, 0.3F, 0.1F, -0.2F, 0.3F, 0.1F, -0.1F, 0.3F, 0.2F, -0.1F, 0.3F, 0.1F, -0.2F,
                        0.2F});
    Tensor wout({4, 4}, {0.1F, -0.1F, 0.2F, 0.1F, 0.2F, 0.1F, -0.1F, 0.2F, -0.1F, 0.2F, 0.1F, -0.1F, 0.1F, 0.2F,
                          -0.1F, 0.2F});

    Tensor inputA({1, 3, 4}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F, 0.2F, -0.1F, 0.3F, 0.2F, -0.4F, 0.5F});
    Tensor inputB = inputA;
    inputB.at(8) = 99.0F;  // token 2 (posizione futura per 0 e 1), primo elemento

    Tensor resultA = cpu::selfAttention(inputA, wq, wk, wv, wout, /*numHeads=*/2);
    Tensor resultB = cpu::selfAttention(inputB, wq, wk, wv, wout, /*numHeads=*/2);

    // posizioni 0 e 1 (indici 0-3 e 4-7 del tensore [1,3,4]): invariate
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(resultA.at(i), resultB.at(i)) << "indice " << i;
    }
    // posizione 2 (indici 8-11): PUO' essere diversa (dipende anche
    // dal proprio input, che e' cambiato) — non asseriamo nulla qui,
    // la proprieta' interessante e' solo quella sopra.
}

TEST(CpuOpsTest, SelfAttentionLanciaSeNumHeadsNonDivideDim) {
    Tensor input({1, 2, 3}, std::vector<float>(6, 0.0F));
    Tensor w = Tensor::zeros({3, 3});
    EXPECT_THROW((void)cpu::selfAttention(input, w, w, w, w, /*numHeads=*/2), std::invalid_argument);
}

TEST(CpuOpsTest, BidirectionalSelfAttentionNonRispettaLaMascheraCausale) {
    // L'esatto opposto di SelfAttentionRispettaLaMascheraCausale: qui
    // cambiare il token FUTURO (posizione 2) DEVE cambiare l'uscita
    // alle posizioni precedenti (0 e 1), perche' l'attention
    // bidirezionale le fa dipendere anche dal contesto a destra — se
    // non cambiasse, la maschera causale sarebbe ancora attiva per
    // errore.
    Tensor wq({4, 4}, {0.3F, -0.1F, 0.2F, 0.05F, 0.1F, 0.4F, -0.2F, 0.15F, -0.3F, 0.2F, 0.1F, -0.05F, 0.2F, -0.1F,
                        0.3F, 0.1F});
    Tensor wk({4, 4}, {0.1F, 0.2F, -0.1F, 0.3F, -0.2F, 0.1F, 0.4F, -0.1F, 0.3F, -0.3F, 0.1F, 0.2F, -0.1F, 0.2F,
                        -0.2F, 0.1F});
    Tensor wv({4, 4}, {0.2F, 0.1F, -0.1F, 0.3F, 0.1F, -0.2F, 0.3F, 0.1F, -0.1F, 0.3F, 0.2F, -0.1F, 0.3F, 0.1F, -0.2F,
                        0.2F});
    Tensor wout({4, 4}, {0.1F, -0.1F, 0.2F, 0.1F, 0.2F, 0.1F, -0.1F, 0.2F, -0.1F, 0.2F, 0.1F, -0.1F, 0.1F, 0.2F,
                          -0.1F, 0.2F});

    Tensor inputA({1, 3, 4}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F, 0.2F, -0.1F, 0.3F, 0.2F, -0.4F, 0.5F});
    Tensor inputB = inputA;
    inputB.at(8) = 99.0F;  // token 2 (futuro per 0 e 1), primo elemento

    Tensor resultA = cpu::bidirectionalSelfAttention(inputA, wq, wk, wv, wout, /*numHeads=*/2);
    Tensor resultB = cpu::bidirectionalSelfAttention(inputB, wq, wk, wv, wout, /*numHeads=*/2);

    bool anyDifferenceInPastPositions = false;
    for (std::size_t i = 0; i < 8; ++i) {
        if (std::abs(resultA.at(i) - resultB.at(i)) > 1e-4F) {
            anyDifferenceInPastPositions = true;
            break;
        }
    }
    EXPECT_TRUE(anyDifferenceInPastPositions)
        << "l'attention bidirezionale dovrebbe far dipendere le posizioni passate anche dal futuro";
}

TEST(CpuOpsTest, SelfAttentionIncrementaleCorrispondeASelfAttentionSuTutteLePosizioni) {
    // Stessi pesi di SelfAttentionRispettaLaMascheraCausale: alimenta
    // l'intera sequenza [1,3,4] a selfAttention() in un colpo solo, poi
    // le stesse 3 posizioni una alla volta a selfAttentionIncremental()
    // (usando la STESSA cache K/V via riferimento, che cresce ad ogni
    // chiamata) — le due devono produrre esattamente lo stesso output,
    // posizione per posizione: il KV-cache e' una riorganizzazione del
    // calcolo, non un'approssimazione.
    Tensor wq({4, 4}, {0.3F, -0.1F, 0.2F, 0.05F, 0.1F, 0.4F, -0.2F, 0.15F, -0.3F, 0.2F, 0.1F, -0.05F, 0.2F, -0.1F,
                        0.3F, 0.1F});
    Tensor wk({4, 4}, {0.1F, 0.2F, -0.1F, 0.3F, -0.2F, 0.1F, 0.4F, -0.1F, 0.3F, -0.3F, 0.1F, 0.2F, -0.1F, 0.2F,
                        -0.2F, 0.1F});
    Tensor wv({4, 4}, {0.2F, 0.1F, -0.1F, 0.3F, 0.1F, -0.2F, 0.3F, 0.1F, -0.1F, 0.3F, 0.2F, -0.1F, 0.3F, 0.1F, -0.2F,
                        0.2F});
    Tensor wout({4, 4}, {0.1F, -0.1F, 0.2F, 0.1F, 0.2F, 0.1F, -0.1F, 0.2F, -0.1F, 0.2F, 0.1F, -0.1F, 0.1F, 0.2F,
                          -0.1F, 0.2F});

    Tensor input({1, 3, 4}, {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F, 0.2F, -0.1F, 0.3F, 0.2F, -0.4F, 0.5F});
    Tensor fullResult = cpu::selfAttention(input, wq, wk, wv, wout, /*numHeads=*/2);

    cpu::KVCache cache;
    for (std::size_t s = 0; s < 3; ++s) {
        Tensor oneToken({1, 1, 4},
                         {input.at(s * 4 + 0), input.at(s * 4 + 1), input.at(s * 4 + 2), input.at(s * 4 + 3)});
        Tensor incResult = cpu::selfAttentionIncremental(oneToken, wq, wk, wv, wout, /*numHeads=*/2, cache);

        ASSERT_EQ(incResult.shape(), (std::vector<std::size_t>{1, 1, 4}));
        for (std::size_t d = 0; d < 4; ++d) {
            EXPECT_NEAR(incResult.at(d), fullResult.at(s * 4 + d), 1e-4F) << "posizione " << s << " indice " << d;
        }
    }
    EXPECT_EQ(cache.length, 3u);
}

TEST(CpuOpsTest, SelfAttentionIncrementaleLanciaSeNumHeadsNonDivideDim) {
    Tensor input({1, 1, 3}, std::vector<float>(3, 0.0F));
    Tensor w = Tensor::zeros({3, 3});
    cpu::KVCache cache;
    EXPECT_THROW((void)cpu::selfAttentionIncremental(input, w, w, w, w, /*numHeads=*/2, cache),
                 std::invalid_argument);
}

TEST(CpuOpsTest, AddPositionalEmbeddingAtCorrispondeAllaVersioneSenzaOffsetQuandoOffsetEZero) {
    Tensor input({1, 2, 2}, {0.1F, -0.2F, 0.3F, 0.4F});
    Tensor table({4, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});

    Tensor withoutOffset = cpu::addPositionalEmbedding(input, table);
    Tensor withZeroOffset = cpu::addPositionalEmbeddingAt(input, table, /*offset=*/0);

    for (std::size_t i = 0; i < withoutOffset.elementCount(); ++i) {
        EXPECT_FLOAT_EQ(withZeroOffset.at(i), withoutOffset.at(i)) << "indice " << i;
    }
}

TEST(CpuOpsTest, AddPositionalEmbeddingAtUsaLeRigheDellaTabellaSpostateDiOffset) {
    Tensor input({1, 1, 2}, {0.0F, 0.0F});
    Tensor table({4, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});

    Tensor result = cpu::addPositionalEmbeddingAt(input, table, /*offset=*/2);
    // offset=2, s=0 -> riga 2 della tabella: {5.0, 6.0}.
    EXPECT_FLOAT_EQ(result.at(0), 5.0F);
    EXPECT_FLOAT_EQ(result.at(1), 6.0F);
}

TEST(CpuOpsTest, AddPositionalEmbeddingAtLanciaSeOffsetPiuSeqSuperaMaxSeqLen) {
    Tensor input({1, 2, 2}, {0.0F, 0.0F, 0.0F, 0.0F});
    Tensor table({3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    EXPECT_THROW((void)cpu::addPositionalEmbeddingAt(input, table, /*offset=*/2), std::invalid_argument);
}
