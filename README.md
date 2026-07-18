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

Il progetto è agli inizi. La tabella seguente riflette lo stato **reale**
del codice in questo repository, non obiettivi futuri.

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
| LoRA | ⏳ Pianificato (richiede nuove operazioni IR per gli adapter a basso rango) |
| Forecasting | ⏳ Pianificato (richiede layer/loss specifici per serie temporali) |
| Training/fine-tuning su GPU (`blackforge train --device cuda`) | ⏳ Pianificato: l'autodiff esiste solo sul backend CPU per ora (milestone 6); il backend CUDA (milestone 7) ha solo la forward pass |
| Benchmark / profiling | ⏳ Pianificato |
| Multi-GPU | ⏳ Pianificato |

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
blackforge run <file.bf>                # esegue il primo modello su CPU (batch=1)
blackforge run <file.bf> --batch 8      # come sopra, con batch size esplicito
blackforge run <file.bf> --device cuda  # esegue sulla GPU (richiede una build con CUDA)
blackforge train <file.bf>              # addestra il modello del blocco 'train' (CPU)
blackforge train <file.bf> --from-checkpoint pesi.bfckpt  # fine-tuning da pesi esistenti
blackforge train <file.bf> --save-checkpoint pesi.bfckpt  # salva i pesi finali
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
(verificato nei test di parità CPU/GPU).

`blackforge train` addestra davvero il modello referenziato dal primo
blocco `train` del file, sul dataset che referenzia (caricato da disco,
non sintetico), stampando la loss media per epoca. Solo sul backend
CPU per ora — l'autodiff non esiste ancora sul backend CUDA. I comandi
`build`, `benchmark`, `inspect` descritti nella visione del progetto
non sono ancora implementati.

## Esempi

- [`examples/hello.bf`](examples/hello.bf) — sintassi indicativa di un
  modello minimale. Il compilatore oggi lo tokenizza, lo analizza
  sintatticamente e ne valida la semantica (target, precisioni, forme,
  operazioni di pipeline). Non genera ancora codice ne' lo esegue.
- [`examples/tiny_regression.bf`](examples/tiny_regression.bf) —
  esempio **eseguibile end-to-end**: `model` + `dataset` + `train`,
  con il dataset sintetico incluso
  ([`tiny_regression_dataset.bfdata`](examples/tiny_regression_dataset.bfdata),
  8 esempi). Esegui `blackforge train examples/tiny_regression.bf` per
  vedere la loss scendere epoca dopo epoca.

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
8. 🟡 Training: grammatica `dataset`/`train`, formato dataset su disco, pretraining e fine-tuning (`blackforge train`) completati e verificati end-to-end su CPU. LoRA e forecasting non ancora iniziati; training su GPU richiede prima l'autodiff sul backend CUDA.
9. Benchmark, profiling, CLI completa, documentazione finale

## Contribuire

Il progetto segue queste convenzioni:

- Tutti i commenti nel codice sono in italiano.
- Nessun placeholder permanente: ogni funzione dichiarata deve essere
  implementata o assente.
- Ogni fase deve lasciare il repository compilabile e con i test verdi.
- Un file sorgente = una responsabilità (frontend, IR, sema, backend,
  runtime sono moduli separati).
