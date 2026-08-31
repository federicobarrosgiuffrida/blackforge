#pragma once

#include <cstddef>
#include <string>

// Attribuzione del tempo GPU per fase (requisito: sapere DOVE va il
// tempo prima di ottimizzare).
//
// PERCHE' ESISTE
//
// Il benchmark riporta forward/backward/optimizer per passo, ma dentro
// il forward non distingue niente. E il conto aritmetico dice che il
// tempo non e' dove ci si aspetterebbe:
//
//   BlackBit-9B, seq 512, micro-batch 1, 75,781 token/s
//     FLOP del passo (fwd+bwd, 2,91 G attivi)      8,94 TFLOP
//     tempo del passo                              6,76 s
//     throughput effettivo                         1,32 TFLOP/s
//     i soli GEMM, a 50-100 TFLOP/s di picco       89-179 ms  (1,3-2,7 %)
//     traffico dei pesi impacchettati a 448 GB/s     16 ms  (0,2 %)
//
// Il 97 % del passo non e' ne' nella matematica ne' nella lettura dei
// pesi. Ottimizzare i GEMM degli esperti — che era il piano — attaccherebbe
// al massimo il 2,7 %. Prima di riscrivere il percorso caldo serve
// sapere quale fase consuma il resto.
//
// COME MISURA
//
// Coppie di cudaEvent attorno a ogni regione, accumulate durante il
// passo e risolte con UNA sola sincronizzazione alla fine. Non usa
// l'orologio dell'host: i lanci sono asincroni, e un timer host attorno
// a una chiamata misurerebbe il tempo di accodamento, non quello di
// esecuzione (lo stesso motivo per cui backend/cpu/benchmark.hpp
// documenta di non avere un breakdown per operazione su CUDA).
//
// Gli eventi sono presi da un pool per fase, creati alla prima
// necessita' e riusati per tutti i passi successivi: un passo instrumenta
// qualche centinaio di coppie, non decine di migliaia.
//
// A profiler spento begin()/end() sono un confronto booleano: si puo'
// lasciare l'istrumentazione nel percorso caldo senza pagarla.

namespace blackforge::blackbit::cuda {

enum class GpuPhase {
    Embedding,   // lookup della tabella condivisa
    Norm,        // RMSNorm, forward e backward
    Attention,   // proiezioni Q/K/V/O, RoPE, nucleo GQA
    Router,      // logit, softmax, selezione top-k, gradiente del router
    Experts,     // gather, gate/up/down degli esperti, ricombinazione
    Head,        // testa di uscita a blocchi di vocabolario e loss
    Optimizer,   // proiezione low-rank e aggiornamento dei trit
    Count,
};

inline constexpr std::size_t kGpuPhaseCount = static_cast<std::size_t>(GpuPhase::Count);

const char* gpuPhaseName(GpuPhase phase);

class GpuProfiler {
public:
    // Stato per fase: definito nell'unita' di traduzione CUDA perche'
    // contiene cudaEvent_t, e questo header viene incluso anche da
    // main.cpp, che non vede <cuda_runtime.h>. E' pubblico soltanto
    // perche' l'implementazione lo nomina da funzioni non membro; resta
    // un tipo incompleto per chiunque includa questo header.
    struct PhaseState;

    // Di processo, come MemoryTelemetry: c'e' un solo modello per
    // processo nei percorsi previsti, e passare un contesto attraverso
    // ogni firma renderebbe illeggibile il codice numerico.
    static GpuProfiler& instance();

    void setEnabled(bool enabled) { enabled_ = enabled; }
    [[nodiscard]] bool enabled() const { return enabled_; }

    // Nessuna annidatura della STESSA fase: begin() restituisce false se
    // una regione di quella fase e' gia' aperta, e in quel caso NON apre
    // niente. Chi chiama deve evitare la end() corrispondente, altrimenti
    // chiuderebbe la regione esterna a meta' — e' proprio quello che fa
    // GpuPhaseScope, che ricorda se ha davvero aperto.
    // Fasi diverse invece si sovrappongono di proposito: l'ottimizzatore
    // gira dentro il backward, quindi il suo tempo e' anche tempo di
    // esperti/attenzione/testa, e la somma delle fasi puo' superare il
    // tempo del passo. E' un'informazione, non un errore.
    bool begin(GpuPhase phase);
    void end(GpuPhase phase);

    // Azzera gli accumulatori riusando gli eventi gia' creati.
    void reset();

    // Sincronizza e somma i tempi. Va chiamata prima di leggere i
    // risultati.
    void resolve();

    [[nodiscard]] double milliseconds(GpuPhase phase) const;
    [[nodiscard]] double totalMilliseconds() const;
    [[nodiscard]] std::size_t regionCount(GpuPhase phase) const;

    // Una riga per fase, con la percentuale rispetto al tempo di parete
    // del passo. Le percentuali possono NON sommare a 100: le fasi
    // possono sovrapporsi, e la differenza rispetto a 100 e' proprio
    // il tempo non attribuito, che e' l'informazione che si cerca.
    [[nodiscard]] std::string report(double wallStepMilliseconds) const;

private:
    GpuProfiler() = default;

    bool enabled_ = false;
    bool resolved_ = false;
    PhaseState& state(GpuPhase phase);
    [[nodiscard]] const PhaseState& state(GpuPhase phase) const;
};

// Marca una regione per tutta la durata di uno scope. Non fa nulla se
// il profiler e' spento.
class GpuPhaseScope {
public:
    explicit GpuPhaseScope(GpuPhase phase)
        : phase_(phase), opened_(GpuProfiler::instance().begin(phase)) {}
    ~GpuPhaseScope() {
        if (opened_) GpuProfiler::instance().end(phase_);
    }

    GpuPhaseScope(const GpuPhaseScope&) = delete;
    GpuPhaseScope& operator=(const GpuPhaseScope&) = delete;
    GpuPhaseScope(GpuPhaseScope&&) = delete;
    GpuPhaseScope& operator=(GpuPhaseScope&&) = delete;

private:
    GpuPhase phase_;
    bool opened_;
};

}  // namespace blackforge::blackbit::cuda
