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
piccolo (un solo tipo di layer, sei operazioni), non generazione di
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
| Backend CPU di riferimento (tensori, elementwise, matmul, linear, attivazioni, rmsnorm, softmax) | ✅ Completato per le operazioni attualmente nel linguaggio |
| Blocchi transformer (`embedding`, `positional_embedding`, `attention`, `feedforward`) | ✅ CPU e CUDA, forward e backward, residual/pre-norm interni all'operazione; verificati con gradient checking numerico e parità CPU/GPU — bastano per un modello linguistico decoder-only minimale (vedi [`examples/tiny_lm.bf`](examples/tiny_lm.bf)) |
| Esecuzione (`blackforge run`) | ✅ Esegue un modello con input sintetico (id di token interi se la pipeline inizia con `embedding`); pesi deterministici (casuali) di default, oppure pesi realmente allenati con `--from-checkpoint` (CPU e CUDA) |
| Autodiff / backward | ✅ Formule analitiche per linear/matmul/addBias/silu/relu/gelu/rmsnorm/softmax/embedding/positional_embedding/attention/feedforward, verificate con gradient checking numerico |
| Loss | ✅ Errore quadratico medio (MSE, per la regressione/forecasting), cross-entropy con softmax interna a target denso (`loss cross_entropy`) e a target sparso/indice di classe (`loss cross_entropy_sparse`, essenziale per vocabolari grandi), tutte verificate con gradient checking numerico e parità CPU/GPU |
| Optimizer (SGD, AdamW) | ✅ Entrambi implementati e testati (incl. weight decay disaccoppiato di AdamW) |
| Checkpoint (salvataggio/caricamento pesi) | ✅ Formato binario proprietario BlackForge, con round-trip testato |
| Pass manager / ottimizzazioni (fusione, dead code elimination) | ⏳ Pianificato (ancora rimandato: nessuna ottimizzazione genuina applicabile con l'attuale insieme di operazioni) |
| Backend CUDA (tensori device, add/addBias/matmul via cuBLAS/silu/relu/gelu/rmsnorm/softmax/embedding/attention/feedforward, esecuzione) | ✅ Implementato e testato su GPU reale (RTX 5060, sm_120) per le operazioni attualmente nel linguaggio |
| Rilevamento GPU (`blackforge devices`) | ✅ Elenca le GPU NVIDIA visibili tramite il driver CUDA |
| Selezione dispositivo (`blackforge run --device cpu\|cuda`) | ✅ Implementata |
| Tensor Core / precisioni FP8/BF16/TF32 reali su GPU | ⏳ Pianificato — il backend CUDA oggi calcola in float32 (SGEMM), come il backend CPU: nessun uso di Tensor Core o dei formati ridotti come precisione di calcolo reale ancora |
| Supporto multi-architettura CUDA oltre Blackwell (sm_120) | ⏳ Non testato: il progetto compila solo per l'architettura configurata in `BLACKFORGE_CUDA_ARCHITECTURES` |
| Precisioni FP8 (e4m3/e5m2), FP16, BF16, TF32, FP32 | 🟡 Riconosciute, validate e ora applicate come **quantizzazione simulata** (arrotondamento del valore float32 al formato dichiarato, con saturazione) dal backend CPU per `blackforge run`/`forecast`/`benchmark`: un blocco `precision` cambia davvero i numeri prodotti. Il backend CUDA calcola ancora sempre in float32 (nessun uso reale di Tensor Core o dei formati ridotti). `blackforge train` ignora sempre `precision` e resta a piena precisione float32 (la quantizzazione non è differenziabile: servirebbe uno straight-through estimator, non ancora implementato) |
| Grammatica `dataset { ... }` e `train { ... }` | ✅ Completata (parser, analisi semantica con validazione incrociata dei riferimenti a model/dataset) |
| Formato dataset su disco + caricamento a batch | ✅ Formato binario proprietario BlackForge, con batching (incl. wraparound tra epoche) testato |
| Tokenizer (`blackforge tokenizer-train`/`tokenizer-encode`, `dataset-build`) | ✅ BPE byte-level (stile GPT-2, magic `BFTOKN1`), round-trip encode/decode esatto verificato anche su UTF-8 multi-byte; `dataset-build` tokenizza un corpus di testo e produce direttamente un `.bfdata` di next-token-prediction con target **sparso** (indice di classe, non one-hot: evita di sprecare memoria proporzionale al vocabolario), pronto per `blackforge train ... loss cross_entropy_sparse` |
| Caricamento dataset efficiente per corpora grandi | ⏳ `data::Dataset` (lettura e scrittura) resta interamente in RAM: il target sparso rimuove il collo di bottiglia legato al vocabolario, ma un caricamento realmente streaming/memory-mapped per corpora da molti gigabyte non è ancora implementato |
| Pretraining (`blackforge train`) | ✅ Addestra un modello da zero (CPU o CUDA), verificato end-to-end (la loss scende davvero, non solo "compila"); rimescola gli esempi ad ogni epoca (identico su CPU/CUDA) |
| Learning rate scheduler (`lr_schedule cosine`) | ✅ Cosine annealing opzionale (learning rate costante se assente), stessa implementazione condivisa tra CPU e CUDA |
| Fine-tuning (`blackforge train --from-checkpoint`) | ✅ Riprende l'addestramento da un checkpoint esistente |
| LoRA (`train { lora { rank alpha } }`) | ✅ Adapter a basso rango su ogni layer `linear`, pesi di base congelati; verificato con gradient checking e con un test che conferma che i pesi di base restano invariati dopo l'addestramento |
| Forecasting (`forecast { model horizon }`, `blackforge forecast`) | ✅ Rollout autoregressivo (l'output di un passo diventa l'input del successivo); richiede che l'ultima dimensione di input e output del modello coincidano e un checkpoint pre-allenato |
| Autodiff su GPU (`blackforge train --device cuda`) | ✅ Kernel CUDA scritti a mano per ogni backward (matmul, addBias, silu/relu/gelu, rmsnorm, softmax, embedding, positional_embedding, attention, feedforward), loss MSE **e cross-entropy**, optimizer (SGD, AdamW) interamente su device; ogni kernel confrontato numericamente contro la controparte CPU, incluso un test di parità sull'intero training loop (stessa loss finale, stessi pesi finali, CPU vs GPU) — inclusa la pipeline completa di un modello linguistico |
| Checkpoint su GPU (`--from-checkpoint`/`--save-checkpoint` con `--device cuda`) | ✅ Stesso formato binario del backend CPU (magic `BFCKPT1` identico): un checkpoint salvato da CUDA è caricabile da CPU e viceversa, verificato con un test di interoperabilità dedicato |
| Training/fine-tuning/LoRA su GPU (oltre il percorso minimo sopra) | ⏳ Pianificato: LoRA su GPU |
| CLI completa (`check`, `build`, `run`, `train`, `forecast`, `benchmark`, `inspect`, `devices`) | ✅ Tutti e 6 i comandi della visione originale implementati, più `forecast`/`devices` |
| Benchmark (`blackforge benchmark`) | ✅ Hardware rilevato, precisione dichiarata (e realmente applicata sul backend CPU, vedi riga sopra), forma, tempo medio, throughput, memoria stimata, iterazioni/warmup configurabili; con `--device cuda` confronta automaticamente con la CPU (speedup e scarto massimo) |
| Profiling | 🟡 Timing aggregato per iterazione più breakdown per singola operazione della pipeline, ciascuna misurata separatamente sul proprio input reale (solo backend CPU: su CUDA servirebbero timer basati su cudaEvent, i lanci di kernel sono asincroni) |
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
blackforge run <file.bf> --from-checkpoint pesi.bfckpt  # esegue con pesi realmente allenati
blackforge train <file.bf>              # addestra il modello del blocco 'train' (CPU)
blackforge train <file.bf> --from-checkpoint pesi.bfckpt  # fine-tuning (CPU o CUDA; base per LoRA solo CPU)
blackforge train <file.bf> --save-checkpoint pesi.bfckpt  # salva i pesi finali (CPU o CUDA)
blackforge train <file.bf> --device cuda  # addestra sulla GPU (solo loss 'mse', niente lora/checkpoint per ora)
blackforge forecast <file.bf> --from-checkpoint pesi.bfckpt --batch 1  # rollout autoregressivo
blackforge benchmark <file.bf> --batch 8 --warmup 5 --iterations 20  # tempo/throughput/memoria
blackforge benchmark <file.bf> --device cuda  # come sopra + confronto automatico con la CPU
blackforge inspect <file.bf>            # riepilogo: input, pipeline, numero di parametri
blackforge devices                      # elenca i dispositivi disponibili (CPU e GPU CUDA rilevate)
blackforge --version
blackforge --help
```

`blackforge run` esegue la prima pipeline del primo modello con un
input sintetico (deterministico, non un dataset reale). Senza
`--from-checkpoint`, i pesi sono generati in modo deterministico ma
statisticamente arbitrario (non Xavier/Kaiming): serve a dimostrare che
l'intera catena letto→validato→compilato→eseguito funziona, non a
produrre un output utile. Con `--from-checkpoint <file>` (CPU e CUDA),
carica invece i pesi realmente allenati da quel checkpoint — è così che
si esegue un modello per davvero, dopo averlo allenato con
`blackforge train`. A parità di seme, `--from-checkpoint` assente e
**senza un blocco `precision`**, `--device cpu` e `--device cuda` usano
esattamente gli stessi pesi iniziali e producono lo stesso risultato
(verificato nei test di parità CPU/GPU); con `--from-checkpoint`,
combaciano allo stesso modo perché caricano entrambi lo stesso file.
`--device cuda:N` seleziona l'indice della GPU quando ce n'è più di una
(non è multi-GPU: si esegue comunque su una sola GPU alla volta).

Se il programma dichiara un blocco `precision { storage ... compute ...
accumulate ... }`, `blackforge run`/`forecast`/`benchmark` lo applicano
davvero sul backend CPU: ogni attivazione viene arrotondata (con
saturazione, non overflow a infinito) al formato `storage` dopo ogni
operazione, e gli operandi di ogni prodotto matriciale (`linear`)
vengono arrotondati al formato `compute` prima del calcolo — è una
**quantizzazione simulata** (i numeri restano `float`, solo con meno
bit di mantissa "effettivi"), non un vero calcolo in FP8/BF16/TF32 su
hardware dedicato. Il campo `accumulate` è analizzato ma non ancora
usato (nell'attuale insieme di operazioni non c'è un accumulatore
separato da quantizzare). Con `precision` dichiarato, `--device cpu` e `--device cuda` **non**
producono più lo stesso risultato: il backend CUDA ignora ancora
`precision` e calcola sempre in piena float32. `blackforge train`
ignora sempre `precision` a prescindere dal device: solo i percorsi di
sola inferenza (`run`/`forecast`/`benchmark`) la applicano.

`blackforge train` addestra davvero il modello referenziato dal primo
blocco `train` del file, sul dataset che referenzia (caricato da disco,
non sintetico), stampando la loss media per epoca. Con un blocco
`lora { rank N alpha F }` dentro `train`, invece di allenare i pesi
originali allena un adapter a basso rango (richiede `--from-checkpoint`:
non ha senso allenare un adapter su pesi casuali).

Con `--device cuda`, l'intero ciclo di addestramento (forward, loss,
backward, aggiornamento dei pesi) avviene su device: ogni kernel di
backward (matmul, addBias, silu/relu/gelu, rmsnorm, softmax, embedding,
positional_embedding, attention, feedforward) e ogni optimizer (SGD,
AdamW) sono implementazioni CUDA scritte a mano, verificate contro la
controparte CPU sia kernel per kernel sia sull'intero training loop
(vedi `tests/backend/cuda/`) — sia `loss mse` sia `loss cross_entropy`
sono supportate su GPU, quindi si può allenare un modello linguistico
(next-token-prediction) interamente su device.
`--from-checkpoint`/`--save-checkpoint` funzionano anche su GPU, con lo
stesso formato binario del backend CPU (un checkpoint CUDA è caricabile
da CPU e viceversa). E' comunque un **percorso minimo, non ancora alla
pari con la CPU**: nessun blocco `lora` ancora su GPU. Se il programma
richiede LoRA con `--device cuda`, il comando fallisce con un errore
esplicito — non esegue un fallback silenzioso sulla CPU né ignora la
richiesta.

`blackforge forecast` esegue il modello referenziato dal primo blocco
`forecast` ripetutamente per `horizon` passi, usando l'output di ogni
passo come input del successivo (richiede `--from-checkpoint` e che
l'ultima dimensione di input e output del modello coincidano).

`blackforge benchmark` esegue warmup + iterazioni misurate della prima
pipeline del primo modello, riportando hardware, precisione dichiarata,
forma dell'input, tempo medio, throughput e una stima della memoria
(elementi × 4 byte, dato che il backend CPU memorizza sempre float32 —
non è RSS di processo misurata). Se il programma dichiara un blocco
`precision`, il tempo misurato include anche il costo della
quantizzazione simulata applicata a ogni iterazione (vedi sotto). Con
`--device cuda` ripete la stessa misura sulla GPU (senza applicare la
quantizzazione, non ancora supportata su CUDA) e stampa anche lo
speedup e lo scarto massimo rispetto all'output CPU (la "modalità di
riferimento", anch'essa senza quantizzazione per un confronto equo).
Riporta anche un breakdown del tempo per singola operazione della
pipeline (solo backend CPU, ogni operazione misurata separatamente sul
suo input reale) — utile per capire quale operazione domina il tempo
totale.

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
- [`examples/tiny_classification.bf`](examples/tiny_classification.bf)
  — esempio **eseguibile end-to-end** di classificazione con
  `loss cross_entropy` (softmax applicata internamente su logit
  grezzi), con il dataset one-hot sintetico incluso
  ([`tiny_classification_dataset.bfdata`](examples/tiny_classification_dataset.bfdata),
  8 esempi, 2 classi). Esegui
  `blackforge train examples/tiny_classification.bf` per vedere la loss
  scendere da ~`ln(2)` (predizione casuale a 2 classi) verso zero.
- [`examples/tiny_lm.bf`](examples/tiny_lm.bf) — esempio **eseguibile
  end-to-end** di modello linguistico minimale (decoder-only,
  LLaMA-like): `embedding |> positional_embedding |> attention |>
  feedforward |> linear`, allenato con `loss cross_entropy` sul dataset
  incluso ([`tiny_lm_dataset.bfdata`](examples/tiny_lm_dataset.bfdata),
  64 sequenze di 6 token). Il compito è "shift-copy" (predire il token
  della posizione precedente): risolvibile solo grazie ad `attention`
  (embedding/feedforward, essendo per-posizione, non possono da soli
  spostare informazione tra posizioni della sequenza), quindi vedere la
  loss scendere qui dimostra che l'intera pipeline transformer sta
  imparando. Esegui `blackforge train examples/tiny_lm.bf
  --save-checkpoint tiny_lm.bfckpt` (aggiungi `--device cuda` per
  allenare sulla GPU), poi `blackforge run examples/tiny_lm.bf
  --from-checkpoint tiny_lm.bfckpt` per generare i logit con i pesi
  allenati.

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
8. ✅ Training: grammatica `dataset`/`train`/`forecast`, formato dataset su disco, pretraining, fine-tuning, LoRA (`train { lora { ... } }`) e forecasting autoregressivo (`blackforge forecast`) — tutti completati e verificati end-to-end su CPU (loss che scende davvero, pesi di base verificati congelati durante LoRA, rollout autoregressivo verificato con un modello identita'). Alla fine di questa milestone, training/fine-tuning/LoRA su GPU erano ancora lavoro futuro (l'autodiff esisteva solo su CPU): risolto in parte da lavoro successivo, vedi la riga "Autodiff su GPU" nella tabella di stato sopra.
9. ✅ CLI completa (`build`, `benchmark`, `inspect` aggiunti — tutti e 6 i comandi della visione originale ora esistono), benchmark con confronto CPU/GPU automatico, selezione GPU per indice (`--device cuda:N`), documentazione finale aggiornata. Profiling resta parziale (solo timing aggregato, nessun breakdown per operazione); pass manager e generazione di codice nativo non sono stati iniziati.
10. ✅ Addestramento di modelli linguistici: generalizzazione di `linear`/`silu`/`relu`/`gelu`/`rmsnorm`/`softmax`/`cross_entropy` a tensori di rango >= 3, e quattro nuovi blocchi di pipeline — `embedding`, `positional_embedding`, `attention` (self-attention causale multi-testa, con residual/pre-norm interni), `feedforward` (con residual/pre-norm interni) — forward e backward su CPU e CUDA, verificati con gradient checking numerico e parità CPU/GPU sull'intero training loop. `cross_entropy` ora supportata anche su `--device cuda`. `blackforge run --from-checkpoint` esteso a `run` (prima funzionava solo per `train`). Esempio end-to-end incluso ([`examples/tiny_lm.bf`](examples/tiny_lm.bf)): un modello linguistico decoder-only minimale, allenato su CPU e su GPU, con loss verificata scendere a ~zero su un compito che richiede genuinamente attention. LoRA su GPU resta lavoro futuro.

## Contribuire

Il progetto segue queste convenzioni:

- Tutti i commenti nel codice sono in italiano.
- Nessun placeholder permanente: ogni funzione dichiarata deve essere
  implementata o assente.
- Ogni fase deve lasciare il repository compilabile e con i test verdi.
- Un file sorgente = una responsabilità (frontend, IR, sema, backend,
  runtime sono moduli separati).
