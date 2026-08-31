# Handoff — BlackBit su BlackForge

Repo: `federicobarrosgiuffrida/blackforge`
Branch: `claude/blackbit-9b-moe-engine-lxqj4x`
Ultimo commit: `6adcef0` — "Attribuzione del tempo GPU per fase e verifica del CUDA senza GPU"

---

## 1. Stato in una riga

Il modello BlackBit-9B-A3B gira davvero su RTX 5060 (8 GB), 9 miliardi di
parametri ternari, picco 4,379 GiB, nessun NaN, i trit cambiano davvero.
Il lavoro aperto **non** e' farlo funzionare: e' che è **lento**, 6,76 s
per passo, e stiamo attaccando il kernel che ne consuma il 24 %.

## 2. Vincolo dell'ambiente — leggere prima di tutto

**In questo container non esiste il toolkit CUDA.** `cmake` compila solo
il ramo CPU. Nessuno dei 29 file CUDA del repository viene compilato qui,
e non c'e' nessuna GPU su cui misurare.

Conseguenze pratiche:

- Ogni modifica ai `.cu` va verificata con `./tools/check_cuda_syntax.sh`
  (aggiunto nel commit `6adcef0`, vedi §5). Cattura sintassi, tipi,
  argomenti dei kernel, name lookup. **Non** cattura errori di `nvcc`
  (funzioni host chiamate dal device, pressione sui registri) ne'
  ovviamente errori di esecuzione.
- Ogni numero di prestazioni in questo documento che non sia marcato
  MISURATO e' un modello aritmetico, non una misura.
- La build CPU (`cmake --build build && ctest`) deve restare verde:
  **387/387 test passano** oggi. È l'unica verifica automatica reale.

## 3. Lavoro NON committato — è dove eravamo

Un solo file modificato: `src/blackbit/cuda_low_rank_optimizer.cu`
(+97 / −30). Copia del diff: vedi `git diff` sul branch, oppure
rigenerarlo. **Passa `check_cuda_syntax.sh`, non ha un test.**

### Cosa fa la modifica

Rimappa `updatePackedKernel` da "un thread per word consecutiva" a
"un blocco = una colonna di word × 256 righe", con le direzioni caricate
in memoria condivisa.

- Griglia: da `<<<gridFor(rows*wordsPerRow), 256>>>` a
  `<<<dim3(wordsPerRow, gridFor(rows)), 256>>>`.
- `firstColumn` diventa uniforme dentro il blocco, quindi le `rank * 20`
  direzioni servono a tutti i thread → si caricano una volta in
  `__shared__ float sharedDirection[kComponentTile * kTritsPerWord]`
  (`kComponentTile = 32`, quindi 2560 byte fissi, indipendenti da `rank`)
  e si rileggono in broadcast.

### Perché — il ragionamento, da rifare se non convince

Il profilo Nsight (già in `docs/blackbit.md` §7.8, MISURATO da una
sessione precedente su hardware vero) dice `updatePacked = 1228 ms/step`,
24 % del passo. La spiegazione **non** è il volume di traffico:

- rank in uso è **8** (non 32, non 64: `--optimizer-rank 8` in tutti i
  run 9B). Traffico direzioni = `rank × 4 B × 9,05e9` = 290 GB/step, che
  a 448 GB/s sarebbe ~0,65 s, e in più `direction` (8 × 3968 × 4 = 127 KB)
  sta comodamente in L2. Il volume da solo non spiega 1228 ms.
- La spiegazione è il **numero di transazioni**. Nella mappatura vecchia
  un warp copre 32 word consecutive della stessa riga, quindi per ogni
  `(componente, slot)` legge `direction[c*cols + 20*t + slot]`: passo 20
  float, 32 indirizzi su ~40 linee di cache. Con rank 8 sono 160 letture
  così per thread → **~6400 transazioni L2 per warp**. Su 452 M word
  (14,1 M warp) sono ~90 G transazioni × 32 B ≈ 2,9 TB da L2. A ~1,9 TB/s
  di banda L2 fa ~1,5 s, che è l'ordine di grandezza dei 1228 ms
  misurati. **Il modello combacia con la misura.**
- I 9,05 miliardi di arrotondamenti stocastici, che il documento
  indicava come sospetto principale, sono aritmetica di registro:
  ~12,7 G `splitMix64` per passo, stimati ~28 ms. Non sono loro.

Costo del cambio, dichiarato apertamente nel commento del kernel:
`packed[row*wordsPerRow + wordColumn]` diventa un accesso a passo
`wordsPerRow`, quindi **non coalescente**. 3,8 GiB/passo (lettura +
scrittura) con amplificazione ~8× ≈ 30 GiB ≈ 68 ms a 448 GB/s. È un
compromesso: ~1 s risparmiato contro ~68 ms pagati. **Va verificato su
GPU, non è un miglioramento su tutti i fronti.**

### Invarianza dichiarata (da verificare, non ancora testata)

I trit prodotti dovrebbero essere **bit-identici**:

- `update` resta accumulato per componente crescente (`componentBase`
  crescente, poi `offset` crescente) → stessa somma, stessi
  raggruppamenti;
- `projection(seed, epoch, row, component, rank)` dipende da riga e
  componente, non dalla forma della griglia;
- `stochasticRoundToTrit(target, stepSeed, row*cols + col)` dipende da
  (riga, colonna), non dalla griglia;
- la guardia `firstColumn + slot < cols` sugli slot di padding
  dell'ultima word è stata **mantenuta** apposta (senza, accumulerebbero
  `scaled * 0`, che è 0 salvo che `scaled` non sia finito).

Cambia **solo** il raggruppamento di `updateSquared`, che alimenta il
solo `updateRms` diagnostico. Verificato con grep: nessun consumatore
fuori dal campo della struct in `cuda_low_rank_optimizer.hpp`. Stessa
categoria di cambiamento già accettata e documentata da una sessione
precedente per `gradientStatsKernel`.

### PROSSIMO PASSO — era esattamente qui che ci si è fermati

Scrivere `tests/blackbit/packed_update_mapping_tests.cpp` e registrarlo
in `tests/CMakeLists.txt` (elenco esplicito, riga ~41, non c'è GLOB).

Il test **non** ha bisogno di CUDA: reimplementa su host le due
mappature (vecchia e nuova, compresa la piastrellatura su
`kComponentTile` e la memoria condivisa simulata come array locale) su
una matrice piccola con `packed` e `direction` casuali, e asserisce che
le word prodotte siano **bit-identiche**. È la parte davvero rischiosa
del cambio: l'aritmetica degli indici.

Casi che devono essere coperti, perché sono i bordi veri:

- `rows` **non** multiplo di 256 (thread di coda inattivi);
- `rank` **non** multiplo di `kComponentTile = 32` (ultimo `chunk`
  parziale) — provare rank 8 (il valore reale) e rank 40;
- `cols` **non** multiplo di `kTritsPerWord = 20` (padding dell'ultima
  word: 3072 non è multiplo di 20, wordsPerRow = 154, ~0,26 % di spreco
  — già documentato in `docs/blackbit.md`).

Le funzioni del codec (`decodeTritByte`, `encodeTritByte`, `wordByte`,
`setWordByte`) e `counterRandom`/`stochasticRoundToTrit` sono
`BLACKFORGE_HOST_DEVICE`, quindi chiamabili dal test. `projection()` è
`__device__` dentro il `.cu` e va replicata nel test (è tre righe:
`counterRandom(seed ^ epoch*0x9E3779B97F4A7C15, row*rank + component)`,
poi `±1/sqrtf(rank)` sul bit basso). Il test verifica l'**equivalenza
fra le due mappature**, quindi replicare `projection` non indebolisce
ciò che si sta testando.

Dopo il test: aggiornare `docs/blackbit.md` §7.8 e committare.

### Una cosa da NON dimenticare

Il commento nel kernel dice "Un blocco copre 5120 trit, quindi i
contatori stanno in 32 bit" — resta vero (256 thread × 20 trit), ma
**va riletto** dopo il cambio di mappatura per essere sicuri che
descriva ancora la realtà.

## 4. Cosa è stato committato in `6adcef0`

Due cose indipendenti.

### a) Profiler per fase (`--profile`)

`include/blackforge/blackbit/cuda_profile.hpp` + `src/blackbit/cuda_profile.cu`.

Motivo: la tabella Nsight in `docs/blackbit.md` somma **5,1 s di kernel
su un passo da 6,76 s**. 1,6 s (24 %) non sta in nessun kernel, e Nsight
non dice a quale parte del modello appartenga. Il profiler misura
dall'interno con coppie di `cudaEvent`, fasi: embedding, rmsnorm,
attenzione, router, esperti, testa/loss, optimizer. Una sola
`cudaDeviceSynchronize` a fine passo, **fuori** dal cronometro.

Due scelte da sapere leggendo il report:

- la fase `optimizer` copre **anche** la proiezione low-rank che gira
  dentro il backward, non solo `endStep()`. Si sovrappone quindi di
  proposito alle fasi che la contengono, e la somma può superare il
  100 %. Senza questa scelta i 622 ms di "proiezione + statistiche" di
  Nsight finirebbero contati come tempo degli esperti;
- `non attribuito` grande e positivo = tempo fuori dai kernel
  (accodamento host, allocazioni, `cudaMemcpy` sincrono dopo
  `routeForwardKernel`, i `cudaMemset` del dispatch).

Spento salvo `--profile`; a profiler spento `begin`/`end` sono un
confronto booleano.

**Due difetti trovati e corretti scrivendo l'istrumentazione** (erano
bug veri, non cosmetici):

1. lo scope `Experts` in `MoELayer::backward` era a livello di funzione
   e si prendeva anche il softmax backward e il GEMM del router. Ora è
   delimitato al solo ciclo, e il backward del router è marcato `Router`.
2. `GpuPhaseScope` chiamava `end()` anche quando `begin()` aveva
   rifiutato di aprire una regione annidata: **chiudeva la regione
   esterna a metà**. `begin()` ora restituisce se ha davvero aperto.

### b) `tools/check_cuda_syntax.sh` + `tools/cuda_syntax_stub/`

Riscrive `kernel<<<griglia, blocchi>>>(args)` in
`(blackforgeLaunchConfiguration(griglia, blocchi), kernel)(args)` — una
espressione con virgola che vale `kernel` e poi lo chiama. Non basta
cancellare la configurazione: le variabili usate solo lì diventerebbero
falsi `-Wunused-variable` (è successo, su `cuda_attention.cu`).

Stub minimali per `cuda_runtime.h`, `cublas_v2.h`, `cublasLt.h`,
`cuda_bf16.h` (con `static_assert` sui 2 byte, così i `sizeof` nelle
allocazioni restano corretti), più le `isfinite`/`min`/`max` che CUDA
mette allo scope globale.

**Tutti e 29 i file CUDA del repository passano, zero warning**, incluso
il ramo `#if BLACKFORGE_HAS_CUDA` di `main.cpp`:

```bash
./tools/check_cuda_syntax.sh                                   # i 29 file
./tools/check_cuda_syntax.sh src/blackbit/cuda_moe.cu          # uno solo
EXTRA_FLAGS='-DBLACKFORGE_HAS_CUDA=1 -DBLACKFORGE_VERSION="x"' \
  ./tools/check_cuda_syntax.sh src/main.cpp
```

## 5. Comandi

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBLACKFORGE_ENABLE_CUDA=OFF
cmake --build build -j$(nproc)
(cd build && ctest --output-on-failure)      # 387/387 devono passare
./tools/check_cuda_syntax.sh                 # dopo OGNI modifica ai .cu
```

Su una macchina con GPU:

```bash
./blackforge benchmark blackbit 9b-a3b --device cuda \
  --seq 512 --micro-batch 1 --steps 3 --optimizer-rank 8 \
  --max-vram-mb 7800 --profile
```

## 6. Dove andare dopo (ranking per tempo misurato, `docs/blackbit.md` §7.8)

| componente | ms/step | % | stato |
|---|---:|---:|---|
| updatePacked (optimizer) | 1 228 | 24 % | **in corso**, §3 |
| decode ternario | 822 | 16 % | aperto |
| proiezione + statistiche | 622 | 12 % | aperto |
| GEMM cuBLASLt (tutti) | ~570 | 11 % | **non toccare** |
| GQA forward | 333 | 6 % | aperto |
| GQA backward | 265 | 5 % | aperto |
| routing forward | 229 | 4 % | `<<<1,1>>>`, vedi sotto |
| adam direction | 104 | 2 % | aperto |

Due avvertenze concrete:

- **I GEMM non sono il collo di bottiglia.** Il throughput effettivo è
  1,32 TFLOP/s su 8,94 TFLOP per passo; a 50–100 TFLOP/s di picco i GEMM
  sarebbero l'1,3–2,7 % del passo (Nsight dice 11 %, stesso verdetto).
  Una sessione precedente ha proposto "GEMM degli esperti in batch" come
  correzione principale ed **era sbagliato**: attaccherebbe ~2 %. Non
  ripetere l'errore.
- `routeForwardKernel` gira **a thread singolo** (`<<<1,1>>>`,
  `src/blackbit/cuda_moe.cu`), 229 ms/passo di pura serializzazione.
  Parallelizzarlo è possibile ma delicato: il riempimento greedy della
  capacità è **sensibile all'ordine** e non deve cambiare quali esperti
  vengono scelti. Esiste già un bug storico su questo percorso (il
  dispatch scartava uno slot quando l'esperto preferito era pieno anche
  se altri avevano capacità: perdeva lavoro **e** faceva sembrare il
  benchmark più veloce). Qualunque modifica va confrontata contro il
  conteggio di `droppedAssignments` e l'entropia di routing.

Altri `<<<N,1>>>` (un thread per blocco) da guardare:
`cuda_model.cu:updateHeadStatsKernel`, `cuda_moe.cu:softmaxSmallKernel`,
`cuda_moe.cu:routingGradientKernel`, `cuda_moe.cu:softmaxBackwardSmallKernel`.

**Tetto previsto**: anche azzerando ogni overhead dell'host, con questa
architettura di calcolo BF16 il limite è ~100 token/s. Oltre quella
soglia non basta togliere overhead: va ridotto il lavoro dei kernel.
≥150 token/s non sembra realistico su questa GPU.

## 7. Filone parallelo, sospeso: importare OLMoE-1B-7B

Task aperti (l'utente aveva chiesto di convertire un MoE preaddestrato
in ternario, poi ha dato priorità alla GPU):

- **fatto e committato** (`9d0a419`): lettore safetensors,
  `include/blackforge/blackbit/safetensors.hpp` + `.cpp`, conversione
  BF16/FP16, letture a blocchi di righe, 10 test passano.
- **da fare**: estensioni architetturali per OLMoE — embedding non
  legate (oggi `throw`), QK-norm su q/k, `norm_topk_prob` configurabile,
  `rms_norm_eps` configurabile (OLMoE usa **1e-5**, BlackBit ha 1e-6
  cablato), `rope_theta` configurabile.
- **da fare**: importatore OLMoE → checkpoint BlackBit ternario (mappa
  dei nomi, quantizzazione a blocchi di righe, comando CLI, report
  dell'errore di quantizzazione per tensore).

Le convenzioni OLMoE sono state **verificate leggendo
`modeling_olmoe.py`**, non indovinate: RMSNorm è `weight * hidden` con
init a 1,0; ordine QK-norm = proj → norm sulla dimensione piena →
reshape → RoPE; `rotate_half` coincide con quello di BlackBit; router =
softmax su tutti (fp32) → topk → renorm opzionale, e per OLMoE
`norm_topk_prob: false`.

**Avvertenza sulla fattibilità**, misurata: la quantizzazione ternaria
diretta (PTQ) di OLMoE dà **51 % di errore di ricostruzione**,
indipendente dalla dimensione del gruppo (60→500 dà 50,9 %→51,3 %),
confermato su un peso OLMoE reale al 51,7 %. La similarità coseno crolla
da 0,899 a 0,107 su 28 strati (con la cautela che quella catena non
includeva residui né norm). **Convertire senza riaddestrare non
funziona**: serve una fase di adattamento.

## 8. Regole ingegneristiche dell'utente — non negoziabili

Testuali, dalla richiesta originale:

> Non fingere il supporto. Non ripiegare in silenzio su 9B parametri
> FP16. Non chiamare "1,58 bit" qualcosa che internamente resta int8 per
> tutta la vita. Non contare la RAM di sistema come risparmio di VRAM
> senza dirlo. Non nascondere allocazioni alla telemetria. Non aggiungere
> kernel segnaposto e dichiarare la funzionalità completa. Non rompere i
> modelli BlackForge esistenti. Non cancellare AdamW/SGD. Non riscrivere
> moduli funzionanti se non serve. Preferisci astrazioni pulite e
> riusabili agli hack specifici del modello. Compila spesso. Esegui i
> test dopo ogni sottosistema. Correggi i warning introdotti dal tuo
> codice. Usa assert per forme dei tensori e ipotesi di memoria.
> Documenta le decisioni importanti nel codice.

E:

> Se scopri che uno dei miei approcci è tecnicamente sbagliato, NON
> implementarlo alla cieca. Spiega il problema con memoria/matematica/
> codice concreti e implementa l'architettura più vicina che preserva
> gli obiettivi.

Conseguenza pratica per chi prende in mano il lavoro: **ogni numero di
prestazioni va marcato MISURATO o STIMATO**, e il §3 di questo documento
è pieno di STIMATO. La prima cosa che serve su una GPU vera è misurare,
non fidarsi.

## 9. Documentazione

`docs/blackbit.md` è il documento vivo: piano, registro delle decisioni,
risultati misurati, e §8.1 con l'elenco esplicito di **cosa non è
implementato**. Va aggiornato a ogni cambiamento sostanziale — è la
richiesta esplicita dell'utente, non una cortesia.
