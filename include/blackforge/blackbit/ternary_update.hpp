#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "blackforge/blackbit/gradient.hpp"
#include "blackforge/blackbit/ternary.hpp"

// Come si aggiorna un peso che vale {-1, 0, +1} — senza tenerne da
// nessuna parte una copia continua (requisiti 3 e 8).
//
// IL PROBLEMA, IN NUMERI
//
// Un peso ternario ha una griglia di passo 'scala' (~0,03 su una
// matrice appena inizializzata). Un aggiornamento tipico vale
// lr * gradiente ~ 1e-4. Arrotondato al valore piu' vicino:
//
//     round((0,03 + 0,0001) / 0,03) = round(1,0033) = 1
//
// cioe' NIENTE. Ripetuto per milioni di passi, sempre niente: il
// modello resterebbe fermo al punto di inizializzazione mentre la loss
// sembra "semplicemente non scendere". E' esattamente il modo in cui un
// motore puo' fingere di allenare pesi a bassa precisione.
//
// La risposta della letteratura e' tenere pesi latenti in fp16 e
// riquantizzare (BitNet): per BlackBit-9B sarebbero 18 GB, cioe' il
// progetto non esiste piu'.
//
// LA RISPOSTA DI BLACKBIT
//
// Arrotondamento STOCASTICO. Lo stesso aggiornamento diventa un cambio
// di trit con probabilita' 0,0033, ed e' esatto in valore atteso:
//
//     E[trit_nuovo * scala] = trit_vecchio * scala + delta
//
// Non serve nessuna informazione continua persistente: l'aggiornamento
// non viene "accumulato finche' non basta", viene applicato subito con
// la probabilita' giusta. Su una matrice di esperto (12,2 M pesi) un
// passo con quei numeri muove ~40 000 trit: il modello si sposta
// davvero, e nessun byte di master copy esiste.
//
// Il prezzo e' la varianza: ogni singolo passo e' rumoroso. E' il
// motivo per cui il residuo low-rank del requisito 9 resta utile (riduce
// la varianza accumulando il sotto-passo prima di consolidarlo), non il
// motivo per cui sarebbe indispensabile.

namespace blackforge::blackbit {

// Quanti trit ha davvero cambiato valore un aggiornamento. E' la
// metrica che dimostra che i pesi ternari NON sono congelati: se
// 'flips' resta zero passo dopo passo, l'addestramento non sta
// toccando le matrici grandi, per quanto la loss possa muoversi grazie
// ai parametri densi.
struct TernaryUpdateStats {
    std::size_t elementsConsidered = 0;
    std::size_t flips = 0;

    [[nodiscard]] double flipFraction() const {
        return elementsConsidered == 0 ? 0.0
                                        : static_cast<double>(flips) / static_cast<double>(elementsConsidered);
    }

    TernaryUpdateStats& operator+=(const TernaryUpdateStats& other) {
        elementsConsidered += other.elementsConsidered;
        flips += other.flips;
        return *this;
    }
};

// Hash a 64 bit di un nome di parametro (FNV-1a), usato per dare a ogni
// matrice una sequenza pseudocasuale propria pur partendo da un solo
// seme globale.
std::uint64_t parameterNameHash(const std::string& name);

// In quali unita' e' espresso un aggiornamento.
enum class TernaryUpdateUnits {
    // Unita' del peso reale: il delta viene diviso per la scala del
    // gruppo prima di essere confrontato con la griglia.
    Weight,
    // Unita' della GRIGLIA: 1,0 significa "un passo ternario intero".
    // E' la forma naturale per un ottimizzatore, perche' su una griglia
    // quantizzata il modulo assoluto del gradiente non dice nulla —
    // conta solo quanto e' grande l'aggiornamento RISPETTO al passo
    // della griglia.
    Grid,
};

// Applica 'delta' alle righe
// [firstRow, firstRow + rowCount) di 'weight', arrotondando
// stocasticamente sulla griglia ternaria del gruppo corrispondente.
//
// 'seed' e 'step' determinano interamente la sequenza casuale: due
// esecuzioni con gli stessi valori producono gli stessi flip, il che
// rende riproducibile un addestramento intero.
TernaryUpdateStats applyTernaryUpdateBlock(TernaryTensor& weight, std::size_t firstRow, std::size_t rowCount,
                                            const float* delta, std::uint64_t seed, std::uint64_t step,
                                            TernaryUpdateUnits units = TernaryUpdateUnits::Weight);

// Sink che consuma il gradiente e aggiorna IMMEDIATAMENTE i pesi:
// discesa del gradiente, con arrotondamento stocastico sui pesi ternari
// e aggiornamento ordinario su quelli densi.
//
// AGGIORNAMENTO NORMALIZZATO (attivo per default)
//
// Il delta applicato non e' 'lr * gradiente' ma
//
//     lr * gradiente / rms(gradiente del blocco)     [unita' di griglia]
//
// cioe' il learning rate misura direttamente QUANTA FRAZIONE DI PASSO
// TERNARIO ci si aspetta di percorrere, indipendentemente dal modulo
// del gradiente e dalla scala di quantizzazione di quella matrice.
//
// Non e' un dettaglio di comodo: senza normalizzazione, con una loss
// mediata su qualche migliaio di elementi il gradiente vale ~1e-4 e il
// passo della griglia ~0,2, quindi la probabilita' di flip e' ~5e-4 —
// misurata su un blocco MoE minuscolo, 300 passi producevano 2 flip in
// tutto e la loss non si muoveva. Il segnale c'era, ma la griglia era
// troppo grossa per riceverlo. La stessa cosa la fa Adam su pesi
// continui, normalizzando il passo per la radice del secondo momento:
// qui e' ancora piu' necessaria, perche' un passo troppo piccolo non
// diventa un aggiornamento piccolo, diventa NESSUN aggiornamento.
//
// Disattivabile (setNormalizeUpdates(false)) per confrontare i due
// comportamenti. Non tiene alcuno stato per parametro: la memoria
// dell'ottimizzatore e' esattamente zero.
class TernarySgdSink : public GradientSink {
public:
    explicit TernarySgdSink(float learningRate, std::uint64_t seed = 0x9E3779B97F4A7C15ULL)
        : learningRate_(learningRate), seed_(seed) {}

    // Registra i tensori che questo sink puo' aggiornare. Un gradiente
    // per un parametro non registrato e' un errore (indica un modello e
    // un ottimizzatore fuori sincrono), non un aggiornamento saltato in
    // silenzio.
    void registerTernary(const std::string& name, TernaryTensor& weight);
    void registerDense(const std::string& name, std::vector<float>& values);

    void consumeWeightGradientBlock(const ParameterId& id, std::size_t firstRow, std::size_t rowCount,
                                     const float* block) override;
    void consumeDenseGradient(const ParameterId& id, const float* values, std::size_t count) override;

    void setLearningRate(float learningRate) { learningRate_ = learningRate; }
    [[nodiscard]] float learningRate() const { return learningRate_; }

    void setNormalizeUpdates(bool normalize) { normalizeUpdates_ = normalize; }
    [[nodiscard]] bool normalizeUpdates() const { return normalizeUpdates_; }

    // Avanza il contatore di passo: va chiamata una volta per ogni
    // aggiornamento completo del modello, cosi' la sequenza casuale non
    // si ripete fra un passo e il successivo.
    void endStep() { ++step_; }
    [[nodiscard]] std::uint64_t step() const { return step_; }

    [[nodiscard]] const TernaryUpdateStats& stats() const { return stats_; }
    void resetStats() { stats_ = TernaryUpdateStats{}; }

    // Byte di stato posseduti dall'ottimizzatore: zero per costruzione.
    // Il benchmark lo riporta accanto a quello dell'ottimizzatore
    // low-rank, come termine di paragone.
    [[nodiscard]] static std::size_t stateBytes() { return 0; }

private:
    float learningRate_;
    std::uint64_t seed_;
    bool normalizeUpdates_ = true;
    std::uint64_t step_ = 0;
    TernaryUpdateStats stats_;
    std::unordered_map<std::string, TernaryTensor*> ternary_;
    std::unordered_map<std::string, std::vector<float>*> dense_;
};

}  // namespace blackforge::blackbit
