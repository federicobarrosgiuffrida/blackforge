# BlackForge

BlackForge è un linguaggio di programmazione **verticale**, dedicato
esclusivamente al Machine Learning, scritto interamente in C++.

Non è un linguaggio general-purpose e non è un clone di Python: è pensato
per descrivere, compilare ed eseguire workload di ML — pretraining,
fine-tuning, LoRA, forecasting — con tensori, precisioni numeriche e
target hardware come cittadini di prima classe della sintassi.

```blackforge
target nvidia.blackwell

precision {
    storage bf16
    compute fp8.e4m3
    accumulate fp32
}

model TinyModel {
    input bf16[batch, 4096]

    input
        |> linear(4096)
        |> silu
        |> linear(4096)
}
```

## Stato del progetto

Il compilatore copre l'intera catena lettura→validazione→esecuzione,
incluso l'addestramento (pretraining, fine-tuning, LoRA) e il
forecasting, sia su CPU sia (per l'inferenza) su GPU CUDA. È comunque
un progetto giovane: il sottoinsieme del linguaggio è volutamente
piccolo (un solo tipo di layer, quattro operazioni), non generazione di
codice nativo, non ottimizzazioni del grafo. La tabella seguente
riflette lo stato **reale** del codice in questo repository, non
obiettivi futuri.

| Componente | Stato |
|---|---|
| Lexer (tokenizzazione, diagnostica riga/colonna, recupero da errori) | ✅ Completato |
| CLI `blackforge check <file>` | ✅ Completato (analisi lessicale e sintattica) |
| Parser / AST (`target`, `precision`, `model`, pipeline `\|>`) | ✅ Completato per il sottoinsieme documentato in [docs/language.md](docs/language.md) |
| Analisi semantica e controllo tipi numerici (dtype, target, operazioni note) | ✅ Completato per il sottoinsieme attuale |
| Controllo forme tensoriali | ✅ Inferenza reale lungo la pipeline (via IR); vincoli locali via analisi semantica |
| Rappresentazione interna (IR) | ✅ Completata (Value/Operation/Module, IR builder con inferenza di forma e dtype) |
| Backend CPU di riferimento (tensori, elementwise, matmul, linear, attivazioni) | ✅ Completato per le operazioni attualmente nel linguaggio |
| Esecuzione (`blackforge run`) | ✅ Esegue un modello con input sintetico e pesi deterministici |
| Autodiff / backward | ✅ Formule analitiche per linear/matmul/addBias/silu/relu/gelu, verificate con gradient checking numerico |
| Loss | 🟡 Solo errore quadratico medio (MSE); altre (es. cross-entropy) richiedono prima una sintassi per dichiarare il tipo di compito |
| Optimizer (SGD, AdamW) | ✅ Entrambi implementati e testati (incl. weight decay disaccoppiato di AdamW) |
| Checkpoint (salvataggio/caricamento pesi) | ✅ Formato binario proprietario BlackForge, con round-trip testato |
| Pass manager / ottimizzazioni (fusione, dead code elimination) | ⏳ Pianificato (ancora rimandato: nessuna ottimizzazione genuina applicabile con l'attuale insieme di operazioni) |
| Backend CUDA (tensori device, add/addBias/matmul via cuBLAS/silu/relu/gelu, esecuzione) | ✅ Implementato e testato su GPU reale (RTX 5060, sm_120) per le operazioni attualmente nel linguaggio |
| Rilevamento GPU (`blackforge devices`) | ✅ Elenca le GPU NVIDIA visibili tramite il driver CUDA |
| Selezione dispositivo (`blackforge run --device cpu\|cuda`) | ✅ Implementata |
| Tensor Core / precisioni FP8/BF16/TF32 reali su GPU | ⏳ Pianificato — il backend CUDA oggi calcola in float32 (SGEMM), come il backend CPU: nessun uso di Tensor Core o dei formati ridotti come precisione di calcolo reale ancora |
| Supporto multi-architettura CUDA oltre Blackwell (sm_120) | ⏳ Non testato: il progetto compila solo per l'architettura configurata in `BLACKFORGE_CUDA_ARCHITECTURES` |
| Precisioni FP8 (e4m3/e5m2), FP16, BF16, TF32, FP32 | 🟡 Riconosciute e validate nel linguaggio; sia il backend CPU sia quello CUDA calcolano sempre in float32 come riferimento (nessuna emulazione/uso reale dei formati ridotti ancora) |
| Grammatica `dataset { ... }` e `train { ... }` | ✅ Completata (parser, analisi semantica con validazione incrociata dei riferimenti a model/dataset) |
| Formato dataset su disco + caricamento a batch | ✅ Formato binario proprietario BlackForge, con batching (incl. wraparound tra epoche) testato |
| Pretraining (`blackforge train`) | ✅ Addestra un modello da zero su CPU, verificato end-to-end (la loss scende davvero, non solo "compila") |
| Fine-tuning (`blackforge train --from-checkpoint`) | ✅ Riprende l'addestramento da un checkpoint esistente |
| LoRA (`train { lora { rank alpha } }`) | ✅ Adapter a basso rango su ogni layer `linear`, pesi di base congelati; verificato con gradient checking e con un test che conferma che i pesi di base restano invariati dopo l'addestramento |
| Forecasting (`forecast { model horizon }`, `blackforge forecast`) | ✅ Rollout autoregressivo (l'output di un passo diventa l'input del successivo); richiede che l'ultima dimensione di input e output del modello coincidano e un checkpoint pre-allenato |
| Training/fine-tuning/LoRA su GPU | ⏳ Pianificato: l'autodiff esiste solo sul backend CPU per ora (milestone 6); il backend CUDA (milestone 7) ha solo la forward pass |
| CLI completa (`check`, `build`, `run`, `train`, `forecast`, `benchmark`, `inspect`, `devices`) | ✅ Tutti e 6 i comandi della visione originale implementati, più `forecast`/`devices` |
| Benchmark (`blackforge benchmark`) | ✅ Hardware rilevato, precisione dichiarata, forma, tempo medio, throughput, memoria stimata, iterazioni/warmup configurabili; con `--device cuda` confronta automaticamente con la CPU (speedup e scarto massimo) |
| Profiling | 🟡 Solo timing aggregato per iterazione (dentro `benchmark`); nessun breakdown per singola operazione |
| Selezione GPU (`--device cuda:N`) | ✅ Implementata (`cudaSetDevice`); resta comunque una sola GPU alla volta, non multi-GPU simultaneo |
| Pass manager / ottimizzazioni del grafo, generazione di codice nativo | ⏳ Non iniziato |
| Multi-GPU (esecuzione simultanea su più GPU) | ⏳ Pianificato |

Legenda: ✅ completato · 🟡 parzialmente implementato · ⏳ pianificato.

## Dipendenze

- CMake >= 3.24
- Un compilatore C++23: testato con **GCC 15 / MinGW** (build CPU-only)
  e con **MSVC** (build con CUDA — richiesto da `nvcc` su Windows, vedi
  nota sotto)
- Git
- Facoltativo: CUDA Toolkit + una GPU NVIDIA, per il backend CUDA. Il
  progetto compila ed è pienamente funzionante anche senza CUDA
  installato (backend CPU di riferimento).
- I test usano [GoogleTest](https://github.com/google/googletest),
  scaricato automaticamente da CMake (`FetchContent`) quando i test sono
  abilitati.

## Compilazione

### Build CPU-only (nessuna GPU richiesta)

Qualsiasi compilatore C++23 va bene (es. GCC/MinGW):

```bash
cmake -S . -B build -DBLACKFORGE_BUILD_TESTS=ON -DBLACKFORGE_ENABLE_CUDA=OFF
cmake --build build
```

### Build con backend CUDA (richiede MSVC + nvcc, testata su Windows)

> **Nota Windows/CUDA**: su Windows `nvcc` richiede **MSVC** (`cl.exe`)
> come host compiler — un toolchain MinGW/GCC non basta, e va usato per
> *l'intero* progetto quando CUDA è abilitato (non si possono linkare
> insieme oggetti compilati da MinGW e da MSVC/nvcc: hanno ABI C++
> incompatibili). Se hai installato Visual Studio (anche solo i Build
> Tools) con il carico di lavoro "Sviluppo Desktop in C++", apri un
> **Developer Command Prompt for VS** (o esegui `vcvarsall.bat x64`) e
> compila da lì con un generatore che usa `cl.exe`, ad esempio:
>
> ```bat
> "C:\Program Files\Microsoft Visual Studio\<versione>\<edizione>\VC\Auxiliary\Build\vcvarsall.bat" x64
> cmake -S . -B build_cuda -G "NMake Makefiles" -DBLACKFORGE_BUILD_TESTS=ON -DBLACKFORGE_ENABLE_CUDA=ON
> cmake --build build_cuda
> ```
>
> Il backend CUDA di questo repository è stato compilato **e testato su
> GPU reale** (NVIDIA GeForce RTX 5060, architettura Blackwell/sm_120)
> con questa esatta procedura.

Opzioni CMake principali:

| Opzione | Default | Descrizione |
|---|---|---|
| `BLACKFORGE_ENABLE_CUDA` | `ON` | Abilita il backend CUDA se `nvcc` è disponibile |
| `BLACKFORGE_BUILD_TESTS` | `OFF` | Compila la suite di test (GoogleTest) |
| `BLACKFORGE_ENABLE_WARNINGS` | `ON` | Abilita warning stretti del compilatore |
| `BLACKFORGE_CUDA_ARCHITECTURES` | `120` | Architetture CUDA target (120 = Blackwell consumer, es. RTX 50xx) |

> **Nota Windows/Controlled Folder Access**: se il repository si trova
> dentro una cartella protetta da Windows (es. `Documents`, con la
> protezione anti-ransomware "Controlled Folder Access" attiva),
> eseguibili non ancora riconosciuti dal sistema — inclusi i binari
> appena compilati di BlackForge — possono non riuscire a **creare**
> nuovi file lì (es. `saveCheckpoint`), pur riuscendo a leggerli. L'API
> restituisce un errore chiaro (`errno`/messaggio) in questo caso, non
> un crash silenzioso: se capita, salva i checkpoint in un percorso
> fuori da cartelle protette (es. `%TEMP%`) oppure aggiungi
> `blackforge.exe` alle app consentite in Windows Security → Protezione
> da ransomware.

### Eseguire i test

```bash
ctest --test-dir build
# oppure direttamente:
./build/tests/blackforge_tests
```

## Uso della CLI

```bash
blackforge check <file.bf>              # analizza il file e riporta gli errori
blackforge check <file.bf> --verbose    # mostra anche i token riconosciuti
blackforge check <file.bf> --print-ast  # mostra l'AST prodotto dal parser
blackforge check <file.bf> --print-ir   # mostra la rappresentazione interna (IR)
blackforge build <file.bf>              # compila e costruisce ogni modello (alloca i parametri), senza eseguirlo
blackforge run <file.bf>                # esegue il primo modello su CPU (batch=1)
blackforge run <file.bf> --batch 8      # come sopra, con batch size esplicito
blackforge run <file.bf> --device cuda:0  # esegue sulla GPU 0 (richiede una build con CUDA)
blackforge train <file.bf>              # addestra il modello del blocco 'train' (CPU)
blackforge train <file.bf> --from-checkpoint pesi.bfckpt  # fine-tuning (o base per LoRA)
blackforge train <file.bf> --save-checkpoint pesi.bfckpt  # salva i pesi finali
blackforge forecast <file.bf> --from-checkpoint pesi.bfckpt --batch 1  # rollout autoregressivo
blackforge benchmark <file.bf> --batch 8 --warmup 5 --iterations 20  # tempo/throughput/memoria
blackforge benchmark <file.bf> --device cuda  # come sopra + confronto automatico con la CPU
blackforge inspect <file.bf>            # riepilogo: input, pipeline, numero di parametri
blackforge devices                      # elenca i dispositivi disponibili (CPU e GPU CUDA rilevate)
blackforge --version
blackforge --help
```

`blackforge run` esegue la prima pipeline del primo modello con un
input sintetico (deterministico, non un dataset reale) e pesi generati
in modo deterministico ma statisticamente arbitrario (non Xavier/Kaiming,
non caricati da checkpoint): serve a dimostrare che l'intera catena
letto→validato→compilato→eseguito funziona, non a produrre un modello
utile. A parità di seme, `--device cpu` e `--device cuda` usano
esattamente gli stessi pesi iniziali e producono lo stesso risultato
(verificato nei test di parità CPU/GPU). `--device cuda:N` seleziona
l'indice della GPU quando ce n'è più di una (non è multi-GPU: si esegue
comunque su una sola GPU alla volta).

`blackforge train` addestra davvero il modello referenziato dal primo
blocco `train` del file, sul dataset che referenzia (caricato da disco,
non sintetico), stampando la loss media per epoca. Con un blocco
`lora { rank N alpha F }` dentro `train`, invece di allenare i pesi
originali allena un adapter a basso rango (richiede `--from-checkpoint`:
non ha senso allenare un adapter su pesi casuali). Solo sul backend CPU
per ora — l'autodiff non esiste ancora sul backend CUDA.

`blackforge forecast` esegue il modello referenziato dal primo blocco
`forecast` ripetutamente per `horizon` passi, usando l'output di ogni
passo come input del successivo (richiede `--from-checkpoint` e che
l'ultima dimensione di input e output del modello coincidano).

`blackforge benchmark` esegue warmup + iterazioni misurate della prima
pipeline del primo modello, riportando hardware, precisione dichiarata,
forma dell'input, tempo medio, throughput e una stima della memoria
(elementi × 4 byte, dato che il backend CPU memorizza sempre float32 —
non è RSS di processo misurata). Con `--device cuda` ripete la stessa
misura sulla GPU e stampa anche lo speedup e lo scarto massimo rispetto
all'output CPU (la "modalità di riferimento").

`blackforge build` compila fino alla IR e prova a costruire davvero
ogni modello (allocando i suoi parametri): a differenza di `check`, che
valida solo AST/IR, intercetta errori che emergono solo in fase di
allocazione (es. una dimensione delle feature ancora simbolica).

## Esempi

- [`examples/hello.bf`](examples/hello.bf) — sintassi indicativa di un
  modello minimale. Provalo con `blackforge check`, `blackforge run`,
  `blackforge inspect` o `blackforge benchmark` (i tensori a 4096
  dimensioni lo rendono un caso utile anche per misurare le prestazioni
  del backend CPU di riferimento).
- [`examples/tiny_regression.bf`](examples/tiny_regression.bf) —
  esempio **eseguibile end-to-end**: `model` + `dataset` + `train`,
  con il dataset sintetico incluso
  ([`tiny_regression_dataset.bfdata`](examples/tiny_regression_dataset.bfdata),
  8 esempi). Esegui `blackforge train examples/tiny_regression.bf` per
  vedere la loss scendere epoca dopo epoca.
- [`examples/tiny_forecast.bf`](examples/tiny_forecast.bf) — esempio
  di `forecast`: un layer lineare 4→4 (compatibile con il rollout
  autoregressivo). Richiede un checkpoint pre-allenato (non incluso,
  va generato con `blackforge train --save-checkpoint`, oppure con
  `blackforge::backend::cpu::saveCheckpoint` da codice C++).

## Struttura del repository

```
include/blackforge/   Header pubblici (namespace blackforge)
  backend/cpu/          Backend CPU di riferimento (tensori, ops, autodiff, optimizer, checkpoint)
  backend/cuda/          Backend CUDA (compilato solo con BLACKFORGE_ENABLE_CUDA=ON)
src/                   Implementazione del compilatore e del runtime (stessa struttura di include/)
tests/                 Test automatici (GoogleTest; tests/backend/cuda/ solo con CUDA abilitato)
examples/              Programmi .bf di esempio
docs/                  Documentazione del linguaggio
```

## Licenza

PolyForm Noncommercial License 1.0.0 — vedi [LICENSE.md](LICENSE.md).

## Roadmap

1. ✅ Lexer e CLI `check` di base
2. ✅ Parser e AST
3. ✅ Analisi semantica: tipi numerici, precisioni, forme tensoriali (base)
4. ✅ Rappresentazione interna (IR): valori, operazioni, inferenza di forma
5. ✅ Backend CPU di riferimento: tensori, elementwise, matmul, layer lineari, attivazioni, esecuzione (`blackforge run`)
6. ✅ Autodiff, loss (MSE), optimizer (SGD, AdamW), checkpoint su CPU
7. ✅ Backend CUDA: tensori device, add/addBias/matmul (cuBLAS)/silu/relu/gelu, esecuzione (`blackforge run --device cuda`), rilevamento GPU (`blackforge devices`) — testato su GPU reale. Tensor Core e precisioni ridotte reali (fp8/bf16/tf32) restano lavoro futuro.
8. ✅ Training: grammatica `dataset`/`train`/`forecast`, formato dataset su disco, pretraining, fine-tuning, LoRA (`train { lora { ... } }`) e forecasting autoregressivo (`blackforge forecast`) — tutti completati e verificati end-to-end su CPU (loss che scende davvero, pesi di base verificati congelati durante LoRA, rollout autoregressivo verificato con un modello identita'). Training/fine-tuning/LoRA su GPU restano lavoro futuro: richiedono prima l'autodiff sul backend CUDA.
9. ✅ CLI completa (`build`, `benchmark`, `inspect` aggiunti — tutti e 6 i comandi della visione originale ora esistono), benchmark con confronto CPU/GPU automatico, selezione GPU per indice (`--device cuda:N`), documentazione finale aggiornata. Profiling resta parziale (solo timing aggregato, nessun breakdown per operazione); pass manager e generazione di codice nativo non sono stati iniziati.

## Contribuire

Il progetto segue queste convenzioni:

- Tutti i commenti nel codice sono in italiano.
- Nessun placeholder permanente: ogni funzione dichiarata deve essere
  implementata o assente.
- Ogni fase deve lasciare il repository compilabile e con i test verdi.
- Un file sorgente = una responsabilità (frontend, IR, sema, backend,
  runtime sono moduli separati).
