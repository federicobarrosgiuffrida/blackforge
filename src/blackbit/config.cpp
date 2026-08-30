#include "blackforge/blackbit/config.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "blackforge/blackbit/ternary.hpp"

namespace blackforge::blackbit {

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::invalid_argument("BlackBitConfig: " + message);
    }
}

// Byte occupati dalle sole scale della forma [rows, cols]: usa la
// stessa formula di raggruppamento del tensore reale, cosi' la stima
// non puo' divergere dal formato.
std::size_t ternaryScaleBytesFor(std::size_t rows, std::size_t cols, std::size_t groupSize) {
    const std::size_t groups = (cols + groupSize - 1) / groupSize;
    return rows * groups * sizeof(float);
}

}  // namespace

void BlackBitConfig::validate() const {
    require(vocabSize > 0, "vocab_size deve essere positivo");
    require(hiddenSize > 0, "hidden_size deve essere positivo");
    require(numLayers > 0, "num_layers deve essere positivo");
    require(numHeads > 0, "num_heads deve essere positivo");
    require(numKvHeads > 0, "num_kv_heads deve essere positivo");
    require(headDim > 0, "head_dim deve essere positivo");
    require(numHeads % numKvHeads == 0,
            "num_heads deve essere un multiplo di num_kv_heads (ogni gruppo GQA condivide una testa K/V)");
    require(numExperts > 0, "num_experts deve essere positivo");
    require(expertsPerToken > 0 && expertsPerToken <= numExperts,
            "experts_per_tok deve essere in [1, num_experts]");
    require(expertHidden > 0, "expert_hidden deve essere positivo");
    require(maxSeqLen > 0, "max_seq_len deve essere positivo");
    require(ternaryGroupSize > 0 && ternaryGroupSize % kTritsPerWord == 0,
            "ternary_group_size deve essere un multiplo positivo di 20 (allineamento alle parole impacchettate)");
    require(expertCapacityFactor > 0.0F, "expert_capacity_factor deve essere positivo");
    require(routerAuxLossWeight >= 0.0F, "router_aux_loss_weight non puo' essere negativo");
    require(weightDtype == sema::DType::TERNARY_1P58 || weightDtype == sema::DType::FP32,
            "weight_dtype supporta 'ternary1p58' (percorso reale) o 'fp32' (modalita' di riferimento)");
}

ParameterCount countParameters(const BlackBitConfig& config) {
    ParameterCount count;

    const std::size_t qDim = config.queryDim();
    const std::size_t kvDim = config.kvDim();

    // q + o proiettano su/da hiddenSize; k e v proiettano sulla
    // dimensione ridotta delle teste K/V: e' l'intero risparmio di GQA
    // sui parametri (oltre a quello sulla cache K/V).
    const std::size_t attentionPerLayer =
        config.hiddenSize * qDim + 2 * config.hiddenSize * kvDim + qDim * config.hiddenSize;

    // SwiGLU: gate e up proiettano hidden -> expertHidden, down torna
    // indietro.
    const std::size_t expertsPerLayer = config.numExperts * 3 * config.hiddenSize * config.expertHidden;

    count.attention = attentionPerLayer * config.numLayers;
    count.experts = expertsPerLayer * config.numLayers;
    count.router = config.hiddenSize * config.numExperts * config.numLayers;
    // Due RMSNorm per layer (pre-attention, pre-MoE) piu' la norm
    // finale prima della proiezione di uscita.
    count.norms = 2 * config.hiddenSize * config.numLayers + config.hiddenSize;
    count.embedding = config.vocabSize * config.hiddenSize * (config.tieEmbeddings ? 1 : 2);

    return count;
}

std::size_t countActiveParameters(const BlackBitConfig& config) {
    const std::size_t qDim = config.queryDim();
    const std::size_t kvDim = config.kvDim();

    const std::size_t attentionPerLayer =
        config.hiddenSize * qDim + 2 * config.hiddenSize * kvDim + qDim * config.hiddenSize;
    const std::size_t expertsPerLayer = config.expertsPerToken * 3 * config.hiddenSize * config.expertHidden;
    const std::size_t routerPerLayer = config.hiddenSize * config.numExperts;
    const std::size_t normsPerLayer = 2 * config.hiddenSize;

    // La proiezione di uscita moltiplica per l'INTERA tabella (serve un
    // logit per ogni token del vocabolario); il lookup di ingresso
    // legge una riga sola e non e' contato.
    const std::size_t outputProjection = config.vocabSize * config.hiddenSize;

    return (attentionPerLayer + expertsPerLayer + routerPerLayer + normsPerLayer) * config.numLayers +
           outputProjection + config.hiddenSize;
}

MemoryEstimate estimateTrainingMemory(const BlackBitConfig& config, const TrainingShape& shape,
                                       const LowMemoryOptions& options) {
    config.validate();

    MemoryEstimate estimate;
    const std::size_t group = config.ternaryGroupSize;
    const std::size_t h = config.hiddenSize;
    const std::size_t qDim = config.queryDim();
    const std::size_t kvDim = config.kvDim();
    const std::size_t eh = config.expertHidden;

    // --- pesi ternari, matrice per matrice (layout [out, in]) ---
    auto addTernary = [&](std::size_t rows, std::size_t cols, std::size_t repeat) {
        estimate.packedWeightBytes += repeat * ternaryPackedBytes(rows, cols);
        estimate.scaleBytes += repeat * ternaryScaleBytesFor(rows, cols, group);
    };

    addTernary(qDim, h, config.numLayers);              // q_proj
    addTernary(kvDim, h, 2 * config.numLayers);         // k_proj, v_proj
    addTernary(h, qDim, config.numLayers);              // o_proj
    addTernary(eh, h, 2 * config.numExperts * config.numLayers);  // gate, up
    addTernary(h, eh, config.numExperts * config.numLayers);      // down
    addTernary(config.vocabSize, h, config.tieEmbeddings ? 1 : 2);

    // --- parametri densi (router + norm), BF16 ---
    const ParameterCount count = countParameters(config);
    estimate.denseParameterBytes = count.dense() * 2;

    // --- stato optimizer low-rank ---
    //
    // Per una matrice [m, n] la proiezione e' R = P^T G con R di forma
    // [r, n] (vedi low_rank_optimizer.hpp): si tengono TRE buffer di
    // r*n valori (primo momento, secondo momento, accumulatore del
    // passo) invece dei due di m*n di AdamW. Il rango non puo' superare
    // m: proiettare su piu' direzioni di quante la matrice ne abbia
    // costerebbe piu' del parametro stesso.
    const std::size_t sb = options.optimizerStateBytes;
    auto addLowRank = [&](std::size_t rows, std::size_t cols, std::size_t repeat) {
        const std::size_t rank = std::min(options.optimizerRank, rows);
        estimate.optimizerBytes += repeat * 3 * rank * cols * sb;
    };
    addLowRank(qDim, h, config.numLayers);
    addLowRank(kvDim, h, 2 * config.numLayers);
    addLowRank(h, qDim, config.numLayers);
    addLowRank(eh, h, 2 * config.numExperts * config.numLayers);
    addLowRank(h, eh, config.numExperts * config.numLayers);
    addLowRank(config.vocabSize, h, config.tieEmbeddings ? 1 : 2);
    // Router e norm sono cosi' piccoli che uno stato Adam ordinario
    // (2 valori per parametro) e' piu' semplice e piu' stabile di una
    // proiezione: vengono contati come tali.
    estimate.optimizerBytes += count.dense() * 2 * sb;

    // --- attivazioni ---
    const std::size_t tokens = shape.microBatch * shape.seqLen;
    if (options.activationCheckpointing) {
        // Un solo tensore [tokens, hidden] per confine di layer, piu' il
        // materiale che il ricalcolo di UN layer produce (norm, q/k/v,
        // uscita attention, hidden degli esperti selezionati).
        estimate.activationBytes = (config.numLayers + 1) * tokens * h * sizeof(float);
        estimate.activationBytes +=
            tokens * (h + qDim + 2 * kvDim + qDim + config.expertsPerToken * 2 * eh) * sizeof(float);
    } else {
        // Stima grossolana del percorso senza ricalcolo: ogni layer
        // conserva tutto cio' che il suo backward richiede.
        estimate.activationBytes =
            config.numLayers * tokens * (3 * h + qDim + 2 * kvDim + qDim + config.expertsPerToken * 2 * eh) *
            sizeof(float);
    }

    // --- picco di gradiente: un solo BLOCCO DI RIGHE di una sola
    // matrice per volta (requisito 6). Non basta "una matrice per
    // volta": la tabella di embedding di BlackBit-9B da sola sarebbe
    // 65536 * 3072 * 4 B = 805 MB, un decimo del budget totale per un
    // buffer che vive qualche microsecondo. ---
    const std::size_t tileRows = std::max<std::size_t>(options.gradientTileRows, 1);
    auto gradientBlock = [&](std::size_t rows, std::size_t cols) {
        return std::min(rows, tileRows) * cols * sizeof(float);
    };
    std::size_t largestBlock = 0;
    largestBlock = std::max(largestBlock, gradientBlock(qDim, h));
    largestBlock = std::max(largestBlock, gradientBlock(kvDim, h));
    largestBlock = std::max(largestBlock, gradientBlock(h, qDim));
    largestBlock = std::max(largestBlock, gradientBlock(eh, h));
    largestBlock = std::max(largestBlock, gradientBlock(h, eh));
    largestBlock = std::max(largestBlock, gradientBlock(config.vocabSize, h));
    estimate.gradientPeakBytes = largestBlock;

    // --- workspace: tile dequantizzati (doppio buffer) + un blocco di
    // logit per la cross-entropy a blocchi di vocabolario ---
    const std::size_t tileCols = std::max({h, eh, qDim});
    estimate.workspaceBytes = 2 * options.dequantTileRows * tileCols * sizeof(float);
    estimate.workspaceBytes += tokens * std::min<std::size_t>(config.vocabSize, 4096) * sizeof(float);

    // --- confronto con l'approccio ordinario ---
    // master copy BF16 (2 B) + gradiente denso FP32 (4 B) + AdamW
    // completo (2 momenti FP32, 8 B) per OGNI parametro.
    estimate.conventionalBytes = count.total() * (2 + 4 + 8);

    return estimate;
}

BlackBitConfig blackBitTiny() {
    BlackBitConfig config;
    config.name = "blackbit-tiny";
    config.vocabSize = 8192;
    config.hiddenSize = 384;
    config.numLayers = 6;
    config.numHeads = 6;
    config.numKvHeads = 2;
    config.headDim = 64;
    config.numExperts = 4;
    config.expertsPerToken = 2;
    config.expertHidden = 1024;
    config.maxSeqLen = 512;
    return config;
}

BlackBitConfig blackBitSmall() {
    BlackBitConfig config;
    config.name = "blackbit-small";
    config.vocabSize = 16384;
    config.hiddenSize = 512;
    config.numLayers = 8;
    config.numHeads = 8;
    config.numKvHeads = 2;
    config.headDim = 64;
    config.numExperts = 6;
    config.expertsPerToken = 2;
    config.expertHidden = 1408;
    config.maxSeqLen = 1024;
    return config;
}

BlackBitConfig blackBitMedium() {
    BlackBitConfig config;
    config.name = "blackbit-medium";
    config.vocabSize = 32768;
    config.hiddenSize = 768;
    config.numLayers = 12;
    config.numHeads = 12;
    config.numKvHeads = 4;
    config.headDim = 64;
    config.numExperts = 8;
    config.expertsPerToken = 2;
    config.expertHidden = 1792;
    config.maxSeqLen = 1024;
    return config;
}

BlackBitConfig blackBit9bA3b() {
    BlackBitConfig config;
    config.name = "blackbit-9b-a3b";
    config.vocabSize = 65536;
    config.hiddenSize = 3072;
    config.numLayers = 28;
    config.numHeads = 24;
    config.numKvHeads = 6;
    config.headDim = 128;
    config.numExperts = 8;
    config.expertsPerToken = 2;
    config.expertHidden = 3968;
    config.maxSeqLen = 4096;
    return config;
}

BlackBitConfig blackBitPreset(const std::string& name) {
    if (name == "tiny" || name == "blackbit-tiny") {
        return blackBitTiny();
    }
    if (name == "small" || name == "blackbit-small") {
        return blackBitSmall();
    }
    if (name == "medium" || name == "blackbit-medium") {
        return blackBitMedium();
    }
    if (name == "9b" || name == "9b-a3b" || name == "blackbit-9b-a3b") {
        return blackBit9bA3b();
    }
    throw std::invalid_argument("blackBitPreset: configurazione '" + name +
                                 "' sconosciuta (attese: tiny, small, medium, 9b-a3b)");
}

}  // namespace blackforge::blackbit
