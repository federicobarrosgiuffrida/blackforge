#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "blackforge/runtime/tensor.hpp"

namespace blackforge::data {

// Un dataset di addestramento caricato in memoria: coppie
// (input, target) usate da 'blackforge train'.
//
// Il formato su disco (vedi saveDataset/loadDataset) e' un formato
// binario minimale e proprietario di BlackForge (magic "BFDATA1"),
// scelto per semplicita' e non per compatibilita' con formati esterni.
// Non c'e' ancora alcuno strumento per costruire un dataset a partire
// da sorgenti reali (CSV, immagini, testo, ...): serve chiamare
// saveDataset() con i dati gia' pronti in memoria.
class Dataset {
public:
    Dataset(std::vector<std::size_t> inputExampleShape, std::vector<std::size_t> targetExampleShape,
            std::vector<float> inputs, std::vector<float> targets, std::size_t numExamples);

    [[nodiscard]] std::size_t numExamples() const { return numExamples_; }
    [[nodiscard]] const std::vector<std::size_t>& inputExampleShape() const { return inputShape_; }
    [[nodiscard]] const std::vector<std::size_t>& targetExampleShape() const { return targetShape_; }

    struct Batch {
        runtime::Tensor input;
        runtime::Tensor target;
    };

    // Restituisce un batch di 'batchSize' esempi a partire
    // dall'esempio 'startIndex', avvolgendo (wraparound) se supera
    // numExamples(): permette di iterare su piu' epoche senza dover
    // gestire manualmente i confini del dataset.
    [[nodiscard]] Batch batch(std::size_t startIndex, std::size_t batchSize) const;

    // Rimescola l'ordine degli esempi in-place (Fisher-Yates, seminato
    // con 'seed': stesso seme => stesso rimescolamento, riproducibile).
    // Va chiamata all'inizio di ogni epoca con un seme diverso (es.
    // derivato dal numero di epoca) per evitare che ogni epoca veda gli
    // esempi sempre nello stesso ordine — importante per SGD/AdamW, che
    // altrimenti vedrebbero sempre gli stessi batch nello stesso
    // ordine, con lo stesso bias di conseguenza. Non fa nulla se il
    // dataset ha 0 o 1 esempi.
    void shuffle(unsigned int seed);

private:
    std::vector<std::size_t> inputShape_;  // forma di un singolo esempio (senza dimensione di batch)
    std::vector<std::size_t> targetShape_;
    std::vector<float> inputs_;   // numExamples_ esempi concatenati
    std::vector<float> targets_;  // numExamples_ esempi concatenati
    std::size_t numExamples_;
};

// Lancia std::runtime_error se il file non puo' essere scritto.
void saveDataset(const std::string& path, const std::vector<std::size_t>& inputExampleShape,
                  const std::vector<std::size_t>& targetExampleShape, const std::vector<float>& inputs,
                  const std::vector<float>& targets, std::size_t numExamples);

// Lancia std::runtime_error se il file non puo' essere letto, ha un
// magic non valido o e' troncato/corrotto.
Dataset loadDataset(const std::string& path);

}  // namespace blackforge::data
