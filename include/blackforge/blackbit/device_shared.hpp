#pragma once

// Marcatore per il codice che deve esistere IDENTICO sull'host e nei
// kernel CUDA: codifica/decodifica dei trit, arrotondamento
// stocastico, generatore pseudocasuale a contatore.
//
// Il punto non e' la comodita': e' che i test di questo repository
// girano sulla CPU, e se la GPU usasse una seconda implementazione
// delle stesse formule un test verde non direbbe nulla sul percorso
// che poi addestra davvero il modello. Tenendo un solo corpo di
// funzione compilato per entrambi i target, la copertura si trasferisce
// per costruzione.
//
// Le funzioni marcate cosi' devono restare: senza stato, senza
// allocazioni, senza eccezioni, senza dipendenze dalla libreria
// standard oltre <cstdint>/<cmath>.

#if defined(__CUDACC__)
#define BLACKFORGE_HOST_DEVICE __host__ __device__
#else
#define BLACKFORGE_HOST_DEVICE
#endif
