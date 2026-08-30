#pragma once

#include <array>
#include <cstddef>
#include <string>

// Telemetria di memoria di BlackBit (requisito 11).
//
// Non e' un profiler: e' la contabilita' di CHI possiede quanta memoria
// in ogni istante, tenuta dal codice stesso che alloca. Serve a due
// cose che nessuna misura esterna puo' dare:
//
//   1. dimostrare con un numero — non con una dichiarazione — che il
//      picco dei gradienti vivi contemporaneamente e' quello di un
//      blocco, non del modello intero (requisito 6/17);
//   2. rifiutare una configurazione PRIMA di allocare, quando la somma
//      prevista supera il budget dichiarato.
//
// Ogni allocazione BlackBit passa di qui. Un'allocazione che non
// comparisse in questa contabilita' renderebbe falso l'intero rapporto
// del benchmark: e' per questo che le arene (parameter/activation/...)
// sono un enum chiuso e non una stringa libera.
//
// Assunzione esplicita, come nel resto di questo motore: nessun accesso
// concorrente da piu' thread.

namespace blackforge::blackbit {

enum class MemoryArena {
    Parameter,   // pesi ternari impacchettati, scale, norm, router
    Activation,  // attivazioni conservate fra forward e backward
    Gradient,    // gradienti vivi in questo istante
    Optimizer,   // stato low-rank dell'ottimizzatore
    Workspace,   // tile dequantizzati, buffer temporanei di calcolo
};

inline constexpr std::size_t kMemoryArenaCount = 5;

const char* memoryArenaName(MemoryArena arena);

class MemoryTelemetry {
public:
    // Contabilita' di processo: c'e' un solo modello BlackBit per
    // processo nei percorsi previsti (addestramento, benchmark), e
    // passare un contesto attraverso ogni firma renderebbe illeggibile
    // il codice numerico senza aggiungere nulla.
    static MemoryTelemetry& instance();

    // Registra un'allocazione e, se un budget e' attivo, la CONTROLLA:
    // lancia std::runtime_error con l'arena, i byte, l'occupazione
    // corrente e cosa ridurre.
    //
    // Il controllo vive qui, e non nei singoli siti di allocazione,
    // perche' e' l'unico punto che tutti attraversano: un budget che si
    // possa aggirare dimenticando una chiamata non e' un budget. Il
    // prezzo e' che il buffer risulta gia' allocato quando l'eccezione
    // parte (viene liberato dallo svolgimento dello stack) — la
    // diagnostica e' comunque quella giusta, e l'esecuzione si ferma
    // prima dell'allocazione SUCCESSIVA invece di arrivare a esaurire
    // davvero la memoria.
    void recordAllocation(MemoryArena arena, std::size_t bytes);

    // Un rilascio piu' grande dell'occupazione corrente e' un errore di
    // contabilita' (o una reset() chiamata mentre oggetti BlackBit erano
    // ancora vivi). NON lancia: questa funzione viene chiamata dai
    // distruttori, e un'eccezione da li' terminerebbe il processo
    // trasformando un difetto di telemetria in un crash. Il contatore
    // viene saturato a zero e l'anomalia registrata in
    // inconsistencies(), che i test verificano essere zero.
    void recordRelease(MemoryArena arena, std::size_t bytes);

    [[nodiscard]] std::size_t current(MemoryArena arena) const;
    [[nodiscard]] std::size_t peak(MemoryArena arena) const;
    [[nodiscard]] std::size_t allocationCount(MemoryArena arena) const;

    [[nodiscard]] std::size_t currentTotal() const;
    [[nodiscard]] std::size_t peakTotal() const;

    // Numero di rilasci incoerenti visti finora: deve restare zero in
    // un'esecuzione corretta.
    [[nodiscard]] std::size_t inconsistencies() const { return inconsistencies_; }

    // Azzera contatori correnti, picchi e conteggi.
    //
    // ATTENZIONE: va chiamata SOLO quando nessun oggetto BlackBit che
    // ha registrato memoria e' ancora vivo, altrimenti i loro
    // distruttori scaricheranno byte che questo azzeramento ha gia'
    // tolto (vedi inconsistencies()). Per misurare il picco di una fase
    // dentro un'esecuzione gia' avviata si usa resetPeaks().
    void reset();

    // Azzera i SOLI picchi, lasciando invariata l'occupazione corrente:
    // serve a misurare il picco di un singolo passo di addestramento
    // senza contare cio' che era gia' allocato prima (i parametri).
    void resetPeaks();

    // Rapporto testuale, una riga per arena piu' il totale.
    [[nodiscard]] std::string report() const;

private:
    MemoryTelemetry() = default;

    std::array<std::size_t, kMemoryArenaCount> current_{};
    std::array<std::size_t, kMemoryArenaCount> peak_{};
    std::array<std::size_t, kMemoryArenaCount> allocations_{};
    std::size_t inconsistencies_ = 0;
};

// Registra un'allocazione alla costruzione e il rilascio alla
// distruzione: l'unico modo per non dimenticare un rilascio su un
// percorso di eccezione (e ogni percorso di questo motore puo'
// lanciare: le forme sono validate a runtime).
class ScopedMemory {
public:
    ScopedMemory(MemoryArena arena, std::size_t bytes) : arena_(arena), bytes_(bytes) {
        MemoryTelemetry::instance().recordAllocation(arena_, bytes_);
    }

    ~ScopedMemory() { MemoryTelemetry::instance().recordRelease(arena_, bytes_); }

    ScopedMemory(const ScopedMemory&) = delete;
    ScopedMemory& operator=(const ScopedMemory&) = delete;
    ScopedMemory(ScopedMemory&&) = delete;
    ScopedMemory& operator=(ScopedMemory&&) = delete;

private:
    MemoryArena arena_;
    std::size_t bytes_;
};

// Budget massimo di memoria dichiarato dall'utente (requisito 11):
// superarlo e' un errore diagnosticabile, non un cudaErrorMemoryAllocation
// a meta' del secondo step.
class MemoryBudget {
public:
    static MemoryBudget& instance();

    // 0 = nessun limite (default).
    void setLimitBytes(std::size_t bytes) { limitBytes_ = bytes; }
    [[nodiscard]] std::size_t limitBytes() const { return limitBytes_; }

    // Lancia std::runtime_error con un messaggio che nomina l'arena, i
    // byte richiesti, l'occupazione corrente per arena e il budget, se
    // l'allocazione richiesta lo supererebbe. Da chiamare PRIMA di
    // allocare davvero.
    void checkBeforeAllocation(MemoryArena arena, std::size_t bytes) const;

private:
    MemoryBudget() = default;

    std::size_t limitBytes_ = 0;
};

}  // namespace blackforge::blackbit
