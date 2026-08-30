#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// Ciclo di vita dei gradienti nel percorso a bassa memoria di BlackBit
// (requisito 6).
//
// IL PROBLEMA
//
// Il modello ordinario di BlackForge (backend::cpu::Parameter,
// backend::cuda::Parameter) tiene per ogni parametro un tensore 'grad'
// PERSISTENTE, azzerato da zeroGrad() e letto da Optimizer::step().
// Per BlackBit-9B sarebbero 9,05 G float32 = 36,2 GB vivi
// contemporaneamente: quattro volte e mezzo la VRAM della scheda
// bersaglio. Nessuna quantizzazione dei pesi salva un progetto che
// alloca i gradienti in questo modo.
//
// LA SOSTITUZIONE
//
// Il gradiente di un peso non viene mai posseduto da nessuno: viene
// prodotto a blocchi di righe, consegnato a un GradientSink che lo
// consuma immediatamente (proiezione low-rank, aggiornamento dei trit),
// e il buffer viene riusato dal blocco successivo. Il picco e' quello
// di UN blocco.
//
// A blocchi di RIGHE e non di matrici intere perche'
//
//     dW[n, k] = somma_m dY[m, n] * X[m, k]
//
// dipende, per la riga n, solo dalla colonna n di dY: un blocco di
// righe si calcola in isolamento. Senza questo, il gradiente della sola
// tabella di embedding di BlackBit-9B sarebbe 805 MB.
//
// CONVENZIONE SUL SIGNIFICATO DEL GRADIENTE
//
// Il blocco consegnato al sink e' sempre dL/dW_efficace, cioe' il
// gradiente rispetto al peso REALE (trit * scala), nelle stesse unita'
// del peso — non rispetto al trit e non rispetto alla scala. E' la
// convenzione dello straight-through estimator: la quantizzazione e'
// trattata come identita' nel backward. Tradurre questo gradiente in
// un movimento sulla griglia ternaria e' compito dell'ottimizzatore,
// che e' l'unico a sapere con quale scala e con quale politica di
// arrotondamento farlo.

namespace blackforge::blackbit {

// Identifica il parametro a cui appartiene un blocco di gradiente.
struct ParameterId {
    std::string name;
    std::size_t rows = 0;
    std::size_t cols = 0;
};

class GradientSink {
public:
    virtual ~GradientSink() = default;

    // Consuma le righe [firstRow, firstRow + rowCount) del gradiente di
    // 'id'. 'block' punta a rowCount * id.cols valori in ordine
    // riga-maggiore ed e' valido SOLO per la durata della chiamata: il
    // sink deve consumarlo o copiarne cio' che gli serve, mai
    // conservarne il puntatore.
    virtual void consumeWeightGradientBlock(const ParameterId& id, std::size_t firstRow, std::size_t rowCount,
                                             const float* block) = 0;

    // Consuma il gradiente completo di un parametro piccolo e denso
    // (gamma di una RMSNorm, matrice di routing): sono migliaia di
    // valori, non miliardi, e spezzarli non avrebbe senso. Stesse
    // regole di validita' del puntatore.
    virtual void consumeDenseGradient(const ParameterId& id, const float* values, std::size_t count) = 0;
};

// Sink di riferimento che ACCUMULA i gradienti in tensori densi in
// memoria host.
//
// Esiste per i test e per i modelli minuscoli: e' esattamente il
// comportamento che BlackBit deve evitare su scala reale, e serve come
// termine di paragone verificabile ("il gradiente calcolato in
// streaming coincide con quello calcolato tutto insieme?"). Il
// benchmark riporta esplicitamente se un sink di questo tipo e' in uso:
// se lo fosse su BlackBit-9B, la riga "FULL MODEL GRADIENT BUFFER"
// direbbe SI.
class DenseGradientCollector : public GradientSink {
public:
    void consumeWeightGradientBlock(const ParameterId& id, std::size_t firstRow, std::size_t rowCount,
                                     const float* block) override;
    void consumeDenseGradient(const ParameterId& id, const float* values, std::size_t count) override;

    // Gradiente accumulato di un parametro; lancia std::out_of_range se
    // il parametro non ha mai ricevuto blocchi.
    [[nodiscard]] const std::vector<float>& gradient(const std::string& name) const;
    [[nodiscard]] bool has(const std::string& name) const { return gradients_.count(name) != 0; }

    void clear() { gradients_.clear(); }

    // Byte densi che questo collettore sta tenendo in vita: e' il
    // numero che rende visibile il costo che il percorso in streaming
    // evita.
    [[nodiscard]] std::size_t heldBytes() const;

private:
    std::unordered_map<std::string, std::vector<float>> gradients_;
};

// Statistiche sul ciclo di vita dei gradienti, di processo (requisito
// 17): quanti byte di gradiente sono vivi ADESSO e quanti lo sono stati
// al massimo contemporaneamente.
//
// E' la prova numerica richiesta da "FULL MODEL GRADIENT BUFFER:
// NO": se il picco resta nell'ordine di un blocco mentre il modello ha
// miliardi di parametri, nessun buffer a dimensione modello puo' essere
// esistito.
struct GradientLifetimeStats {
    std::size_t liveBytes = 0;
    std::size_t peakLiveBytes = 0;
    std::size_t blocksProduced = 0;
    std::size_t blocksReleased = 0;

    // Byte totali di gradiente prodotti nel corso dell'esecuzione, da
    // confrontare con il picco: il rapporto fra i due dice quante volte
    // lo stesso spazio e' stato riusato.
    std::size_t cumulativeBytes = 0;
};

GradientLifetimeStats& gradientLifetimeStats();
void resetGradientLifetimeStats();

}  // namespace blackforge::blackbit
