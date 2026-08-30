#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "blackforge/blackbit/gradient.hpp"
#include "blackforge/blackbit/ternary.hpp"
#include "blackforge/runtime/tensor.hpp"

// Proiezione lineare con pesi memorizzati in ternario impacchettato
// (requisito 2).
//
// MEMORIZZAZIONE E CALCOLO SONO COSE DIVERSE
//
// Il peso vive SEMPRE e SOLO come TernaryTensor: non esiste, in nessun
// momento, una copia densa dell'intera matrice. Il calcolo avviene
// invece nel formato che l'hardware sa moltiplicare — oggi float32 sul
// backend di riferimento CPU (e BF16 su Tensor Core quando il percorso
// CUDA sara' collegato), domani FP4 su Blackwell.
//
// Il ponte fra i due e' la DEQUANTIZZAZIONE A TILE:
//
//     per ogni blocco di 'tileRows' righe del peso:
//         decodifica il blocco nel formato di calcolo   (buffer riusato)
//         moltiplica                                    (GEMM sul blocco)
//
// Il picco di memoria e' 'tileRows * inFeatures' valori, non
// 'outFeatures * inFeatures': per una matrice di esperto di BlackBit-9B
// sono 2 MB invece di 48 MB, e per la tabella di embedding 1,5 MB
// invece di 805 MB. Il buffer del tile e' allocato UNA VOLTA e riusato
// da tutti i blocchi e da tutte le chiamate.
//
// LAYOUT [outFeatures, inFeatures]
//
// Trasposto rispetto al 'linear' del resto del motore ([in, out]). Tre
// motivi concreti:
//   1. i gruppi di scala corrono lungo la dimensione di riduzione, che
//      e' il raggruppamento statisticamente sensato;
//   2. Y = X @ W^T, la forma che il primitivo matmulTransposeB del
//      motore implementa gia' su CPU e su CUDA;
//   3. un tile di righe di W e' un intervallo CONTIGUO del buffer
//      impacchettato: letture perfettamente sequenziali, che e'
//      esattamente cio' che un kernel CUDA vuole.
//
// NESSUN BIAS: come le proiezioni di LLaMA (e come selfAttention nel
// resto di questo motore), che ne fanno a meno senza perdita misurabile.

namespace blackforge::blackbit {

// Formato in cui il tile decodificato viene moltiplicato. Storage e
// compute restano indipendenti: aggiungere FP8/FP4 qui non cambia
// nulla nella firma di forward()/backward() ne' nel modello che le usa.
enum class ComputeDType {
    FP32,  // percorso di riferimento, CPU e CUDA
    BF16,  // Tensor Core (arrotondamento del tile a BF16 prima del GEMM)
};

class TernaryLinear {
public:
    // Costruisce una proiezione con tutti i pesi a zero. 'name' e'
    // l'identificativo usato nei checkpoint e nei blocchi di gradiente.
    TernaryLinear(std::string name, std::size_t inFeatures, std::size_t outFeatures,
                   std::size_t groupSize = kDefaultGroupSize, std::size_t tileRows = 128);

    // Il peso e' registrato nella telemetria alla costruzione e
    // scaricato alla distruzione: un layer che sparisse senza
    // scaricarsi renderebbe il rapporto di memoria una finzione. Non
    // copiabile (una copia raddoppierebbe i byte contabilizzati senza
    // che nessuno se ne accorga), spostabile (i modelli tengono i layer
    // in vettori).
    ~TernaryLinear();
    TernaryLinear(const TernaryLinear&) = delete;
    TernaryLinear& operator=(const TernaryLinear&) = delete;
    TernaryLinear(TernaryLinear&& other) noexcept;
    TernaryLinear& operator=(TernaryLinear&& other) noexcept;

    // Inizializza i pesi con una distribuzione uniforme deterministica
    // e li quantizza subito: il tensore denso intermedio esiste solo
    // per una riga alla volta, mai per l'intera matrice.
    void initialize(unsigned int seed);

    // Sostituisce il peso quantizzando una matrice densa [out, in].
    // Usata dai test e dalla conversione di modelli esistenti; NON e'
    // un percorso del ciclo di addestramento.
    void loadDense(const runtime::Tensor& dense);

    // input [..., inFeatures] -> output [..., outFeatures].
    [[nodiscard]] runtime::Tensor forward(const runtime::Tensor& input) const;

    // Restituisce dL/dInput e consegna dL/dW a blocchi di righe al
    // sink (se non nullo), un blocco per volta, liberandolo subito.
    // 'input' e 'gradOutput' sono quelli dell'ultima forward().
    [[nodiscard]] runtime::Tensor backward(const runtime::Tensor& input, const runtime::Tensor& gradOutput,
                                            GradientSink* sink) const;

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] std::size_t inFeatures() const { return inFeatures_; }
    [[nodiscard]] std::size_t outFeatures() const { return outFeatures_; }
    [[nodiscard]] const TernaryTensor& weight() const { return weight_; }
    [[nodiscard]] TernaryTensor& weight() { return weight_; }
    [[nodiscard]] ParameterId parameterId() const { return ParameterId{name_, outFeatures_, inFeatures_}; }

    [[nodiscard]] std::size_t tileRows() const { return tileRows_; }
    void setTileRows(std::size_t rows);

    [[nodiscard]] ComputeDType computeDType() const { return compute_; }
    void setComputeDType(ComputeDType dtype) { compute_ = dtype; }

    // Byte realmente occupati dal parametro (impacchettato + scale).
    [[nodiscard]] std::size_t parameterBytes() const { return weight_.totalByteCount(); }

    // Byte del buffer di tile riusato: il picco temporaneo del percorso
    // di dequantizzazione, che il benchmark riporta come "unpacked
    // temporary memory".
    [[nodiscard]] std::size_t tileWorkspaceBytes() const { return tileRows_ * inFeatures_ * sizeof(float); }

private:
    // Decodifica un blocco di righe nel buffer di lavoro, applicando
    // l'arrotondamento del formato di calcolo scelto.
    void dequantizeTile(std::size_t firstRow, std::size_t rowCount, std::vector<float>& buffer) const;

    std::string name_;
    std::size_t inFeatures_ = 0;
    std::size_t outFeatures_ = 0;
    std::size_t tileRows_ = 128;
    ComputeDType compute_ = ComputeDType::FP32;
    TernaryTensor weight_;

    // Byte registrati nell'arena dei parametri da QUESTA istanza: zero
    // dopo uno spostamento, cosi' il distruttore dell'oggetto svuotato
    // non scarica byte che ora appartengono a un altro.
    std::size_t accountedBytes_ = 0;
};

}  // namespace blackforge::blackbit
