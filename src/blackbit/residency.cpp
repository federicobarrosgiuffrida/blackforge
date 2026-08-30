#include "blackforge/blackbit/residency.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "blackforge/blackbit/ternary.hpp"

namespace blackforge::blackbit {

namespace {

std::size_t ternaryBytes(std::size_t rows, std::size_t cols, std::size_t groupSize) {
    const std::size_t groups = (cols + groupSize - 1) / groupSize;
    return ternaryPackedBytes(rows, cols) + rows * groups * sizeof(float);
}

std::string mib(std::size_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MiB";
    return out.str();
}

}  // namespace

const char* residencyStateName(ResidencyState state) {
    switch (state) {
        case ResidencyState::GpuResident: return "gpu";
        case ResidencyState::CpuPinned: return "host-pinned";
        case ResidencyState::Paged: return "paginato";
    }
    return "?";
}

ResidencyPlan planResidency(const BlackBitConfig& config, const ResidencyPolicy& policy) {
    config.validate();

    ResidencyPlan plan;
    const std::size_t group = config.ternaryGroupSize;
    const std::size_t h = config.hiddenSize;

    auto add = [&](ResidencyState state, std::size_t bytes) {
        plan.bytesByState[static_cast<std::size_t>(state)] += bytes;
    };

    // Parte condivisa: attention, router, norm, embedding legata.
    const std::size_t attentionBytes =
        config.numLayers * (ternaryBytes(config.queryDim(), h, group) +
                             2 * ternaryBytes(config.kvDim(), h, group) +
                             ternaryBytes(h, config.queryDim(), group));
    const ParameterCount count = countParameters(config);
    const std::size_t sharedBytes =
        attentionBytes + ternaryBytes(config.vocabSize, h, group) + count.dense() * sizeof(float);

    add(policy.keepSharedResident ? ResidencyState::GpuResident : policy.coldExpertState, sharedBytes);

    // Esperti: gate/up/down per esperto per layer.
    const std::size_t bytesPerExpert = 2 * ternaryBytes(config.expertHidden, h, group) +
                                        ternaryBytes(h, config.expertHidden, group);
    const std::size_t hot =
        policy.hotExpertsPerLayer == 0 ? config.numExperts : std::min(policy.hotExpertsPerLayer, config.numExperts);
    const std::size_t cold = config.numExperts - hot;

    add(ResidencyState::GpuResident, config.numLayers * hot * bytesPerExpert);
    if (cold > 0) {
        add(policy.coldExpertState, config.numLayers * cold * bytesPerExpert);
    }

    // Nel caso peggiore un token seleziona expertsPerToken esperti tutti
    // freddi, in ogni layer: e' il traffico che la politica genera.
    const std::size_t coldSelected = cold == 0 ? 0 : std::min(config.expertsPerToken, cold);
    plan.worstCaseTransferBytesPerToken = config.numLayers * coldSelected * bytesPerExpert;

    return plan;
}

std::string ResidencyPlan::report() const {
    std::ostringstream out;
    for (std::size_t i = 0; i < bytesByState.size(); ++i) {
        const auto state = static_cast<ResidencyState>(i);
        out << "  " << std::left << std::setw(14) << residencyStateName(state) << std::right << std::setw(14)
            << mib(bytesByState[i]) << "\n";
    }
    out << "  " << std::left << std::setw(14) << "TOTALE" << std::right << std::setw(14) << mib(total()) << "\n";
    if (worstCaseTransferBytesPerToken > 0) {
        out << "  trasferimento host->device nel caso peggiore, per token: "
            << mib(worstCaseTransferBytesPerToken) << "\n";
    }
    return out.str();
}

}  // namespace blackforge::blackbit
