#include "blackforge/blackbit/telemetry.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace blackforge::blackbit {

namespace {

std::size_t indexOf(MemoryArena arena) { return static_cast<std::size_t>(arena); }

std::string mebibytes(std::size_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MiB";
    return out.str();
}

}  // namespace

const char* memoryArenaName(MemoryArena arena) {
    switch (arena) {
        case MemoryArena::Parameter: return "parametri";
        case MemoryArena::Activation: return "attivazioni";
        case MemoryArena::Gradient: return "gradienti";
        case MemoryArena::Optimizer: return "optimizer";
        case MemoryArena::Workspace: return "workspace";
    }
    return "?";
}

MemoryTelemetry& MemoryTelemetry::instance() {
    static MemoryTelemetry telemetry;
    return telemetry;
}

void MemoryTelemetry::recordAllocation(MemoryArena arena, std::size_t bytes) {
    const std::size_t i = indexOf(arena);
    current_[i] += bytes;
    ++allocations_[i];
    if (current_[i] > peak_[i]) {
        peak_[i] = current_[i];
    }
}

void MemoryTelemetry::recordRelease(MemoryArena arena, std::size_t bytes) {
    const std::size_t i = indexOf(arena);
    if (bytes > current_[i]) {
        // Un rilascio piu' grande dell'occupazione corrente significa
        // che qualcuno ha sbagliato a contare: e' un errore di
        // programmazione, non una condizione da assorbire in silenzio
        // (un contatore che va in underflow renderebbe inutile tutta la
        // telemetria, e con essa la prova che i gradienti non si
        // accumulano).
        throw std::logic_error(std::string("MemoryTelemetry: rilascio di ") + std::to_string(bytes) +
                               " byte nell'arena '" + memoryArenaName(arena) + "' che ne ha solo " +
                               std::to_string(current_[i]));
    }
    current_[i] -= bytes;
}

std::size_t MemoryTelemetry::current(MemoryArena arena) const { return current_[indexOf(arena)]; }

std::size_t MemoryTelemetry::peak(MemoryArena arena) const { return peak_[indexOf(arena)]; }

std::size_t MemoryTelemetry::allocationCount(MemoryArena arena) const { return allocations_[indexOf(arena)]; }

std::size_t MemoryTelemetry::currentTotal() const {
    std::size_t total = 0;
    for (std::size_t value : current_) {
        total += value;
    }
    return total;
}

std::size_t MemoryTelemetry::peakTotal() const {
    // NOTA: e' la somma dei picchi per arena, non il picco della somma.
    // E' un limite superiore (i picchi possono non coincidere nel
    // tempo), scelto perche' e' il numero prudente da confrontare con
    // un budget: sovrastimare porta a rifiutare una configurazione che
    // forse entrerebbe, sottostimare porta a esaurire la VRAM a meta'
    // di un addestramento.
    std::size_t total = 0;
    for (std::size_t value : peak_) {
        total += value;
    }
    return total;
}

void MemoryTelemetry::reset() {
    current_.fill(0);
    peak_.fill(0);
    allocations_.fill(0);
}

void MemoryTelemetry::resetPeaks() { peak_ = current_; }

std::string MemoryTelemetry::report() const {
    std::ostringstream out;
    for (std::size_t i = 0; i < kMemoryArenaCount; ++i) {
        const auto arena = static_cast<MemoryArena>(i);
        out << "  " << std::left << std::setw(14) << memoryArenaName(arena) << " corrente " << std::right
            << std::setw(12) << mebibytes(current_[i]) << "   picco " << std::setw(12) << mebibytes(peak_[i])
            << "   allocazioni " << allocations_[i] << "\n";
    }
    out << "  " << std::left << std::setw(14) << "TOTALE" << " corrente " << std::right << std::setw(12)
        << mebibytes(currentTotal()) << "   picco " << std::setw(12) << mebibytes(peakTotal()) << "\n";
    return out.str();
}

MemoryBudget& MemoryBudget::instance() {
    static MemoryBudget budget;
    return budget;
}

void MemoryBudget::checkBeforeAllocation(MemoryArena arena, std::size_t bytes) const {
    if (limitBytes_ == 0) {
        return;
    }

    const MemoryTelemetry& telemetry = MemoryTelemetry::instance();
    const std::size_t after = telemetry.currentTotal() + bytes;
    if (after <= limitBytes_) {
        return;
    }

    std::ostringstream out;
    out << "budget di memoria BlackBit superato: allocazione di " << mebibytes(bytes) << " nell'arena '"
        << memoryArenaName(arena) << "' porterebbe il totale a " << mebibytes(after) << ", oltre il limite di "
        << mebibytes(limitBytes_) << ".\nOccupazione corrente:\n"
        << telemetry.report()
        << "Riduci --micro-batch o --seq-len, abbassa il rango dell'optimizer, oppure alza --max-vram-mb se la "
           "scheda ha davvero piu' memoria.";
    throw std::runtime_error(out.str());
}

}  // namespace blackforge::blackbit
