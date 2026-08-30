#include "blackforge/blackbit/telemetry.hpp"

#include <gtest/gtest.h>

#include <string>

using namespace blackforge;
using blackforge::blackbit::MemoryArena;
using blackforge::blackbit::MemoryBudget;
using blackforge::blackbit::MemoryTelemetry;

namespace {

// I test di questo file manipolano contatori di processo: si
// ripristinano lo stato pulito prima e dopo, altrimenti sporcherebbero
// gli altri test (e viceversa).
class TelemetryFixture : public ::testing::Test {
protected:
    void SetUp() override {
        MemoryBudget::instance().setLimitBytes(0);
        MemoryTelemetry::instance().reset();
    }
    void TearDown() override {
        MemoryBudget::instance().setLimitBytes(0);
        MemoryTelemetry::instance().reset();
    }
};

}  // namespace

TEST_F(TelemetryFixture, ContaOccupazioneCorrenteEPicco) {
    MemoryTelemetry& telemetry = MemoryTelemetry::instance();

    telemetry.recordAllocation(MemoryArena::Gradient, 1000);
    telemetry.recordAllocation(MemoryArena::Gradient, 500);
    EXPECT_EQ(telemetry.current(MemoryArena::Gradient), 1500U);
    EXPECT_EQ(telemetry.peak(MemoryArena::Gradient), 1500U);

    telemetry.recordRelease(MemoryArena::Gradient, 1200);
    EXPECT_EQ(telemetry.current(MemoryArena::Gradient), 300U);
    EXPECT_EQ(telemetry.peak(MemoryArena::Gradient), 1500U) << "il picco non deve scendere";
    EXPECT_EQ(telemetry.allocationCount(MemoryArena::Gradient), 2U);
    EXPECT_EQ(telemetry.inconsistencies(), 0U);
}

TEST_F(TelemetryFixture, ScopedMemoryRilasciaAncheSuUnPercorsoDiEccezione) {
    MemoryTelemetry& telemetry = MemoryTelemetry::instance();

    try {
        const blackbit::ScopedMemory scope(MemoryArena::Workspace, 4096);
        EXPECT_EQ(telemetry.current(MemoryArena::Workspace), 4096U);
        throw std::runtime_error("qualcosa e' andato storto a meta' del calcolo");
    } catch (const std::runtime_error&) {
        // atteso
    }

    EXPECT_EQ(telemetry.current(MemoryArena::Workspace), 0U);
    EXPECT_EQ(telemetry.peak(MemoryArena::Workspace), 4096U);
}

TEST_F(TelemetryFixture, ResetPeaksNonToccaLOccupazioneCorrente) {
    MemoryTelemetry& telemetry = MemoryTelemetry::instance();
    telemetry.recordAllocation(MemoryArena::Parameter, 8000);
    telemetry.recordAllocation(MemoryArena::Activation, 2000);
    telemetry.recordRelease(MemoryArena::Activation, 2000);

    telemetry.resetPeaks();
    EXPECT_EQ(telemetry.current(MemoryArena::Parameter), 8000U);
    EXPECT_EQ(telemetry.peak(MemoryArena::Parameter), 8000U);
    EXPECT_EQ(telemetry.peak(MemoryArena::Activation), 0U)
        << "il picco di un'arena ora vuota deve ripartire da zero";

    telemetry.recordRelease(MemoryArena::Parameter, 8000);
}

TEST_F(TelemetryFixture, UnRilascioIncoerenteVieneRegistratoInveceDiLanciare) {
    // Questa funzione e' raggiunta dai distruttori: un'eccezione da li'
    // terminerebbe il processo.
    MemoryTelemetry& telemetry = MemoryTelemetry::instance();
    EXPECT_NO_THROW(telemetry.recordRelease(MemoryArena::Optimizer, 999));
    EXPECT_EQ(telemetry.current(MemoryArena::Optimizer), 0U);
    EXPECT_EQ(telemetry.inconsistencies(), 1U);
}

TEST_F(TelemetryFixture, IlBudgetFermaLAllocazionePrimaDiSforare) {
    MemoryBudget::instance().setLimitBytes(10000);
    MemoryTelemetry& telemetry = MemoryTelemetry::instance();

    EXPECT_NO_THROW(telemetry.recordAllocation(MemoryArena::Parameter, 6000));
    EXPECT_NO_THROW(telemetry.recordAllocation(MemoryArena::Optimizer, 3000));

    try {
        telemetry.recordAllocation(MemoryArena::Activation, 2000);
        FAIL() << "l'allocazione avrebbe portato il totale a 11 000 byte, oltre il limite di 10 000";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("attivazioni"), std::string::npos) << "il messaggio deve nominare l'arena";
        EXPECT_NE(message.find("--seq-len"), std::string::npos) << "e dire cosa ridurre";
    }

    // Il rifiuto non deve aver alterato i contatori.
    EXPECT_EQ(telemetry.current(MemoryArena::Activation), 0U);
    EXPECT_EQ(telemetry.currentTotal(), 9000U);

    telemetry.recordRelease(MemoryArena::Parameter, 6000);
    telemetry.recordRelease(MemoryArena::Optimizer, 3000);
}

TEST_F(TelemetryFixture, SenzaBudgetNonCEAlcunLimite) {
    MemoryBudget::instance().setLimitBytes(0);
    EXPECT_NO_THROW(MemoryTelemetry::instance().recordAllocation(MemoryArena::Parameter, 1ULL << 40));
    MemoryTelemetry::instance().recordRelease(MemoryArena::Parameter, 1ULL << 40);
}

TEST_F(TelemetryFixture, IlRapportoNominaOgniArena) {
    const std::string report = MemoryTelemetry::instance().report();
    for (const char* arena : {"parametri", "attivazioni", "gradienti", "optimizer", "workspace", "TOTALE"}) {
        EXPECT_NE(report.find(arena), std::string::npos) << "manca l'arena '" << arena << "'";
    }
}
