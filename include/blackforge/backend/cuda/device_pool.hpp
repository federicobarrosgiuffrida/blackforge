#pragma once

#include <cstddef>

namespace blackforge::backend::cuda {

// Pool di memoria device per-processo, indicizzato per DEVICE CUDA e
// dimensione in byte (free list a bucket esatti): quasi ogni operazione
// di questo backend (matmul, addBias, silu, ..., attention) alloca
// tensori intermedi che vivono solo per la durata di una singola
// chiamata, e li libera subito dopo. cudaMalloc/cudaFree sono
// operazioni relativamente costose e sincronizzanti sul driver — farle
// ad ogni singola operazione (potenzialmente migliaia di volte per
// step di training) e' un collo di bottiglia reale, misurato su un
// training end-to-end (batch piccoli, tanti step: l'overhead fisso per
// allocazione domina rispetto al calcolo effettivo).
//
// Un buffer liberato con devicePoolRelease() NON viene restituito al
// driver CUDA: resta nella free list del device corrente, pronto per
// essere riusato dalla prossima richiesta della STESSA dimensione in
// byte su quel device. Le forme dei tensori intermedi di un training
// loop sono determinate dal modello e dal batch_size, quindi si
// ripetono identiche ad ogni step: dopo il primo step (che "scalda" il
// pool allocando dal driver), gli step successivi non chiamano quasi
// mai cudaMalloc.
//
// Indicizzato per device (non un pool globale unico): un puntatore
// cudaMalloc'd e' valido solo sul device attivo al momento
// dell'allocazione — essenziale per il training multi-GPU (vedi
// backend::cuda::runMultiGpuTraining), che alterna il device attivo
// tra repliche dello stesso processo.
//
// Nessuna deallocazione esplicita a fine processo: i buffer restano
// nella free list (mai cudaFree'd esplicitamente) fino all'uscita del
// processo, quando il runtime CUDA libera comunque tutta la memoria di
// ogni contesto — stessa scelta gia' fatta per gli handle cuBLAS/
// cuBLASLt condivisi (vedi ops_gemm.cu/ops_tensorcore.cu).
//
// Assunzione esplicita: nessun accesso concorrente da piu' thread
// (stessa assunzione documentata altrove in questo backend).

// Richiede 'bytes' byte di memoria device sul device CUDA attualmente
// attivo: riusa un buffer libero della stessa dimensione se disponibile
// nel pool, altrimenti alloca con cudaMalloc. Restituisce nullptr se
// 'bytes' e' 0 (nessuna allocazione, coerente con DeviceTensor per
// tensori a 0 elementi).
void* devicePoolAcquire(std::size_t bytes);

// Restituisce al pool del device attualmente attivo un buffer di
// 'bytes' byte precedentemente ottenuto da devicePoolAcquire() con la
// STESSA dimensione (il chiamante e' responsabile di passare la stessa
// 'bytes' usata per l'acquisizione: il pool non traccia le dimensioni
// dei puntatori che gestisce). Nessun effetto se 'ptr' e' nullptr.
void devicePoolRelease(void* ptr, std::size_t bytes);

}  // namespace blackforge::backend::cuda
