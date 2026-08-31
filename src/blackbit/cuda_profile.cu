#include "blackforge/blackbit/cuda_profile.hpp"

#include <array>
#include <iomanip>
#include <sstream>
#include <vector>

#include "blackforge/backend/cuda/cuda_check.hpp"

namespace blackforge::blackbit::cuda {

// Fuori dal namespace anonimo di proposito: e' un membro di
// GpuProfiler::PhaseState, e un tipo a collegamento interno dentro un
// membro di una classe a collegamento esterno fa scattare
// -Wsubobject-linkage.
struct EventPair {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
};

struct GpuProfiler::PhaseState {
    // Pool di coppie di eventi, creato pigramente e riusato per tutti i
    // passi: creare un cudaEvent costa, e un passo instrumenta qualche
    // centinaio di regioni. Mai distrutti esplicitamente — stessa scelta
    // gia' fatta per gli handle cuBLAS e per i buffer del pool di
    // memoria: il runtime CUDA li libera all'uscita del processo.
    std::vector<EventPair> pool;
    std::size_t used = 0;   // coppie chiuse in questo passo
    std::size_t open = 0;   // indice della coppia aperta da begin()
    bool isOpen = false;
    double milliseconds = 0.0;
};

namespace {

std::array<GpuProfiler::PhaseState, kGpuPhaseCount>& phaseStates() {
    static std::array<GpuProfiler::PhaseState, kGpuPhaseCount> states;
    return states;
}

}  // namespace

const char* gpuPhaseName(GpuPhase phase) {
    switch (phase) {
        case GpuPhase::Embedding: return "embedding";
        case GpuPhase::Norm: return "rmsnorm";
        case GpuPhase::Attention: return "attention";
        case GpuPhase::Router: return "router";
        case GpuPhase::Experts: return "esperti";
        case GpuPhase::Head: return "testa/loss";
        case GpuPhase::Optimizer: return "optimizer";
        case GpuPhase::Count: break;
    }
    return "?";
}

GpuProfiler& GpuProfiler::instance() {
    static GpuProfiler profiler;
    return profiler;
}

GpuProfiler::PhaseState& GpuProfiler::state(GpuPhase phase) {
    return phaseStates()[static_cast<std::size_t>(phase)];
}

const GpuProfiler::PhaseState& GpuProfiler::state(GpuPhase phase) const {
    return phaseStates()[static_cast<std::size_t>(phase)];
}

bool GpuProfiler::begin(GpuPhase phase) {
    if (!enabled_) {
        return false;
    }
    PhaseState& phaseState = state(phase);
    if (phaseState.isOpen) {
        // Annidare la stessa fase falserebbe l'attribuzione senza dirlo:
        // la regione esterna verrebbe chiusa dall'end() interno. Si
        // ignora l'apertura annidata e si dice a chi chiama di non
        // chiudere.
        return false;
    }
    if (phaseState.used >= phaseState.pool.size()) {
        EventPair pair;
        BLACKFORGE_CUDA_CHECK(cudaEventCreate(&pair.start));
        BLACKFORGE_CUDA_CHECK(cudaEventCreate(&pair.stop));
        phaseState.pool.push_back(pair);
    }
    phaseState.open = phaseState.used;
    phaseState.isOpen = true;
    BLACKFORGE_CUDA_CHECK(cudaEventRecord(phaseState.pool[phaseState.open].start));
    return true;
}

void GpuProfiler::end(GpuPhase phase) {
    if (!enabled_) {
        return;
    }
    PhaseState& phaseState = state(phase);
    if (!phaseState.isOpen) {
        return;
    }
    BLACKFORGE_CUDA_CHECK(cudaEventRecord(phaseState.pool[phaseState.open].stop));
    phaseState.isOpen = false;
    ++phaseState.used;
    resolved_ = false;
}

void GpuProfiler::reset() {
    for (std::size_t i = 0; i < kGpuPhaseCount; ++i) {
        PhaseState& phaseState = phaseStates()[i];
        phaseState.used = 0;
        phaseState.isOpen = false;
        phaseState.milliseconds = 0.0;
    }
    resolved_ = false;
}

void GpuProfiler::resolve() {
    if (!enabled_ || resolved_) {
        return;
    }
    // UNA sola sincronizzazione per tutto il passo: gli eventi sono gia'
    // stati registrati nello stream, qui si legge soltanto.
    BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());
    for (std::size_t i = 0; i < kGpuPhaseCount; ++i) {
        PhaseState& phaseState = phaseStates()[i];
        phaseState.milliseconds = 0.0;
        for (std::size_t pair = 0; pair < phaseState.used; ++pair) {
            float elapsed = 0.0F;
            BLACKFORGE_CUDA_CHECK(
                cudaEventElapsedTime(&elapsed, phaseState.pool[pair].start, phaseState.pool[pair].stop));
            phaseState.milliseconds += static_cast<double>(elapsed);
        }
    }
    resolved_ = true;
}

double GpuProfiler::milliseconds(GpuPhase phase) const { return state(phase).milliseconds; }

std::size_t GpuProfiler::regionCount(GpuPhase phase) const { return state(phase).used; }

double GpuProfiler::totalMilliseconds() const {
    double total = 0.0;
    for (std::size_t i = 0; i < kGpuPhaseCount; ++i) {
        total += phaseStates()[i].milliseconds;
    }
    return total;
}

std::string GpuProfiler::report(double wallStepMilliseconds) const {
    std::ostringstream out;
    out << "Attribuzione del tempo GPU per fase\n";
    out << "  " << std::left << std::setw(14) << "fase" << std::right << std::setw(12) << "ms"
        << std::setw(10) << "% passo" << std::setw(12) << "regioni" << "\n";

    for (std::size_t i = 0; i < kGpuPhaseCount; ++i) {
        const auto phase = static_cast<GpuPhase>(i);
        const PhaseState& phaseState = phaseStates()[i];
        if (phaseState.used == 0) {
            continue;
        }
        const double percent =
            wallStepMilliseconds > 0.0 ? 100.0 * phaseState.milliseconds / wallStepMilliseconds : 0.0;
        out << "  " << std::left << std::setw(14) << gpuPhaseName(phase) << std::right << std::fixed
            << std::setprecision(2) << std::setw(12) << phaseState.milliseconds << std::setw(9) << percent
            << " %" << std::setw(12) << phaseState.used << "\n";
    }

    const double attributed = totalMilliseconds();
    const double percent = wallStepMilliseconds > 0.0 ? 100.0 * attributed / wallStepMilliseconds : 0.0;
    out << "  " << std::left << std::setw(14) << "ATTRIBUITO" << std::right << std::fixed
        << std::setprecision(2) << std::setw(12) << attributed << std::setw(9) << percent << " %\n";
    out << "  " << std::left << std::setw(14) << "non attribuito" << std::right << std::setw(12)
        << (wallStepMilliseconds - attributed) << std::setw(9) << (100.0 - percent) << " %\n";
    out << "\n  Le fasi si sovrappongono di proposito: 'optimizer' include la proiezione\n"
           "  low-rank che gira DENTRO il backward, quindi il suo tempo e' contato anche\n"
           "  in esperti/attenzione/testa e la somma puo' superare il 100 %. Per lo stesso\n"
           "  motivo un 'non attribuito' negativo non e' un errore.\n"
           "  Un 'non attribuito' grande e positivo significa invece tempo speso fuori\n"
           "  dalle regioni instrumentate: accodamento dei lanci sull'host, allocazioni,\n"
           "  o sincronizzazioni implicite (cudaMemcpy sincrono, cudaMemset).\n";
    return out.str();
}

}  // namespace blackforge::blackbit::cuda
