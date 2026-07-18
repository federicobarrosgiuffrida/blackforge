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
| Backend CUDA | ⏳ Pianificato |
| Supporto Blackwell / Tensor Core | ⏳ Pianificato |
| Precisioni FP8 (e4m3/e5m2), FP16, BF16, TF32, FP32 | 🟡 Riconosciute e validate; il backend CPU calcola sempre in fp32 come riferimento (nessuna emulazione di precisione ridotta ancora) |
| Pretraining / fine-tuning / LoRA | ⏳ Pianificato |
| Forecasting | ⏳ Pianificato |
| Benchmark / profiling | ⏳ Pianificato |
| Multi-GPU | ⏳ Pianificato |

Legenda: ✅ completato · 🟡 parzialmente implementato · ⏳ pianificato.

## Dipendenze

- CMake >= 3.24
- Compilatore C++23 (testato con GCC 15 / MinGW su Windows)
- Git
- Facoltativo: CUDA Toolkit (per il futuro backend GPU). Il progetto è
  progettato per compilare anche senza CUDA installato.
- I test usano [GoogleTest](https://github.com/google/googletest),
  scaricato automaticamente da CMake (`FetchContent`) quando i test sono
  abilitati.

## Compilazione

```bash
cmake -S . -B build -DBLACKFORGE_BUILD_TESTS=ON
cmake --build build
```

Opzioni CMake principali:

| Opzione | Default | Descrizione |
|---|---|---|
| `BLACKFORGE_ENABLE_CUDA` | `ON` | Abilita il backend CUDA se `nvcc` è disponibile |
| `BLACKFORGE_BUILD_TESTS` | `OFF` | Compila la suite di test (GoogleTest) |
| `BLACKFORGE_ENABLE_WARNINGS` | `ON` | Abilita warning stretti del compilatore |
| `BLACKFORGE_CUDA_ARCHITECTURES` | `120` | Architetture CUDA target (120 = Blackwell consumer, es. RTX 50xx) |

> **Nota Windows/CUDA**: su Windows `nvcc` richiede MSVC (`cl.exe`) come
> host compiler. Se è installato solo un toolchain MinGW/GCC, il backend
> CUDA non compilerà finché non viene installato Visual Studio Build
> Tools con i componenti C++.

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
blackforge run <file.bf>                # esegue il primo modello sul backend CPU (batch=1)
blackforge run <file.bf> --batch 8      # come sopra, con batch size esplicito
blackforge --version
blackforge --help
```

`blackforge run` esegue la prima pipeline del primo modello con un
input sintetico (deterministico, non un dataset reale) e pesi generati
in modo deterministico ma statisticamente arbitrario (non Xavier/Kaiming,
non caricati da checkpoint): serve a dimostrare che l'intera catena
letto→validato→compilato→eseguito funziona, non a produrre un modello
utile. I comandi `build`, `train`, `benchmark`, `inspect` descritti
nella visione del progetto non sono ancora implementati e verranno
aggiunti man mano che le fasi corrispondenti (autodiff, training,
backend CUDA) saranno pronte.

## Esempi

- [`examples/hello.bf`](examples/hello.bf) — sintassi indicativa di un
  modello minimale. Il compilatore oggi lo tokenizza, lo analizza
  sintatticamente e ne valida la semantica (target, precisioni, forme,
  operazioni di pipeline). Non genera ancora codice ne' lo esegue.

## Struttura del repository

```
include/blackforge/   Header pubblici (namespace blackforge)
src/                   Implementazione del compilatore e del runtime
tests/                 Test automatici (GoogleTest)
examples/              Programmi .bf di esempio
docs/                  Documentazione del linguaggio
kernels/               Kernel CUDA (verranno aggiunti con il backend GPU)
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
7. Backend CUDA (kernel, Tensor Core, Blackwell)
8. Training, fine-tuning, LoRA, forecasting
9. Benchmark, profiling, CLI completa, documentazione finale

## Contribuire

Il progetto segue queste convenzioni:

- Tutti i commenti nel codice sono in italiano.
- Nessun placeholder permanente: ogni funzione dichiarata deve essere
  implementata o assente.
- Ogni fase deve lasciare il repository compilabile e con i test verdi.
- Un file sorgente = una responsabilità (frontend, IR, sema, backend,
  runtime sono moduli separati).
