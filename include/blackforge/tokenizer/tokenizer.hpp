#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace blackforge::tokenizer {

// Tokenizer BPE (Byte Pair Encoding) byte-level, nello stile di GPT-2:
// il vocabolario di base sono i 256 valori di byte possibili (nessun
// token <unk> necessario, ogni sequenza di byte — quindi qualunque
// testo UTF-8, o dato binario arbitrario — e' rappresentabile),
// esteso con token appresi fondendo iterativamente le coppie adiacenti
// piu' frequenti in un corpus di training (Sennrich et al. 2016,
// "Neural Machine Translation of Rare Words with Subword Units").
//
// Tre id di token speciali sono riservati SUBITO DOPO i 256 byte di
// base (id 256/257/258), prima di qualunque merge appreso: restano
// fissi indipendentemente dalla dimensione del vocabolario allenato.
// I merge appresi partono quindi sempre da id 259 in su.
class Tokenizer {
public:
    static constexpr std::uint32_t kBaseVocabSize = 256;
    static constexpr std::uint32_t kPadId = 256;
    static constexpr std::uint32_t kBosId = 257;
    static constexpr std::uint32_t kEosId = 258;
    static constexpr std::uint32_t kFirstMergeId = 259;

    // Una fusione appresa: la coppia (left, right) di id diventa un
    // nuovo token con id 'resultId'. L'ordine nel vettore merges() E'
    // la priorita' (rank) usata da encode(): tra le coppie fondibili in
    // un dato momento, si applica sempre quella appresa per prima.
    struct Merge {
        std::uint32_t left;
        std::uint32_t right;
        std::uint32_t resultId;
    };

    // Tokenizer con solo il vocabolario di base (256 byte + 3 token
    // speciali), nessun merge appreso: stato di partenza per train(),
    // o utilizzabile cosi' com'e' per una tokenizzazione byte-level
    // pura (un token per byte, nessuna compressione).
    Tokenizer();

    // Ricostruisce un tokenizer da una lista di merge gia' appresa
    // (usato da loadTokenizer()): applica i merge nell'ordine dato,
    // esattamente come farebbe train() se li avesse appresi lei stessa.
    explicit Tokenizer(std::vector<Merge> merges);

    // Addestra i merge su 'corpus' (testo grezzo, trattato come una
    // sequenza di byte) fino a raggiungere 'targetVocabSize' token
    // totali (base + speciali + merge appresi). Sovrascrive qualunque
    // merge precedentemente presente. Lancia std::invalid_argument se
    // 'targetVocabSize' <= kFirstMergeId (non c'e' spazio per nemmeno
    // un merge). Si ferma prima di raggiungere 'targetVocabSize' se il
    // corpus si esaurisce (nessuna coppia adiacente rimasta da
    // fondere): non e' un errore, il corpus era semplicemente troppo
    // piccolo o troppo poco ripetitivo per quel vocabolario.
    //
    // Implementazione di riferimento: O(numMerge * numChunkUnici) per
    // il conteggio delle coppie ad ogni iterazione, non la struttura
    // dati a coda di priorita' + lista concatenata delle implementazioni
    // ottimizzate per corpora da gigabyte — correttezza prima delle
    // prestazioni, coerente con il resto del progetto; adeguata per i
    // corpora di addestramento di piccola/media scala che questo
    // linguaggio puo' oggi caricare in memoria (vedi data/dataset.hpp).
    void train(const std::string& corpus, std::size_t targetVocabSize);

    // Codifica un testo in una sequenza di id di token, applicando i
    // merge appresi (l'algoritmo greedy standard di BPE: ad ogni passo
    // si fonde, tra le coppie adiacenti presenti nella sequenza
    // corrente, quella con il rank di apprendimento piu' basso). Non
    // aggiunge automaticamente bos/eos: e' responsabilita' del
    // chiamante (vedi kBosId/kEosId).
    [[nodiscard]] std::vector<std::uint32_t> encode(const std::string& text) const;

    // Decodifica una sequenza di id nel testo originale, concatenando i
    // byte rappresentati da ogni id. Round-trip esatto con encode()
    // (proprieta' fondamentale di un tokenizer byte-level: nessuna
    // normalizzazione, nessuna perdita). Id di token speciali (pad/bos/
    // eos) o fuori dal vocabolario vengono saltati silenziosamente (non
    // hanno una rappresentazione testuale).
    [[nodiscard]] std::string decode(const std::vector<std::uint32_t>& ids) const;

    [[nodiscard]] std::size_t vocabSize() const { return kFirstMergeId + merges_.size(); }
    [[nodiscard]] const std::vector<Merge>& merges() const { return merges_; }

private:
    std::vector<Merge> merges_;

    // id -> byte che rappresenta (per decode()); indicizzato per id,
    // dimensione == vocabSize() (gli id 256/257/258, pad/bos/eos, hanno
    // una voce vuota: non rappresentano byte).
    std::vector<std::vector<std::uint8_t>> idToBytes_;

    // (left << 32 | right) -> id risultante, per un lookup O(1) delle
    // coppie fondibili durante train()/encode().
    std::unordered_map<std::uint64_t, std::uint32_t> pairToId_;

    // id risultante -> rank (indice in merges_, l'ordine di
    // apprendimento): usato da encode() per scegliere quale coppia
    // fondere per prima quando piu' di una e' disponibile.
    std::unordered_map<std::uint32_t, std::size_t> mergeRank_;

    void rebuildFromMerges();
};

// Salva/carica un tokenizer addestrato in un formato binario
// proprietario di BlackForge (magic "BFTOKN1\0", non compatibile con
// formati esterni come tiktoken/sentencepiece/HuggingFace tokenizers):
// la lista dei merge appresi, nell'ordine di apprendimento (da cui
// l'intero stato del tokenizer — vocabolario, priorita' — e'
// interamente ricostruibile).
void saveTokenizer(const Tokenizer& tok, const std::string& path);
Tokenizer loadTokenizer(const std::string& path);

}  // namespace blackforge::tokenizer
