# Sintassi di BlackForge — stato attuale

Questo documento descrive ciò che il compilatore riconosce **oggi**:
lexer, parser, analisi semantica, rappresentazione interna (IR) ed
esecuzione (backend CPU e CUDA, incluso l'addestramento su CPU). Non
c'è ancora generazione di codice nativo: BlackForge interpreta la IR
tramite i backend, non compila verso un binario standalone.

## Commenti

```
// commento a fine riga
/* commento
   su più righe */
```

## Identificatori e parole chiave

Un identificatore inizia con una lettera o `_`, seguito da lettere,
cifre o `_` (es. `TinyModel`, `hidden_dim`, `bf16`).

Le seguenti parole sono riservate e non possono essere usate come nomi:

```
target precision storage compute accumulate parameters forward backward
model input output dataset loss optimizer train pretrain finetune lora
forecast benchmark path labels epochs batch_size learning_rate rank
alpha horizon
```

Nomi di formati numerici (`bf16`, `fp8`, `fp16`, `fp32`, `tf32`, `e4m3`,
`e5m2`) e di operazioni (`linear`, `silu`, ...) **non** sono parole
chiave: sono identificatori ordinari, risolti in fase di analisi
semantica.

## Letterali

- **Interi**: `4096`, `0`, `128`
- **Virgola mobile**: `0.001`, `3.14`, `1e-4`, `6.02e23`
- **Stringhe**: `"dataset/train.bin"`, con escape `\"`, `\\`, `\n`, `\t`

## Punteggiatura e operatori

| Simbolo | Significato |
|---|---|
| `{` `}` | blocco |
| `(` `)` | argomenti di una chiamata |
| `[` `]` | forma di un tensore |
| `,` | separatore |
| `:` | separatore chiave/valore (uso futuro) |
| `;` | separatore di istruzione (uso futuro) |
| `.` | nome composto (`nvidia.blackwell`, `fp8.e4m3`) |
| `=` | assegnazione (uso futuro) |
| `\|>` | operatore di pipeline |

## Esempio

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

Il compilatore, tramite `blackforge check esempio.bf --verbose`, mostra
la sequenza di token riconosciuti con posizione (file:riga:colonna);
tramite `--print-ast` mostra l'albero sintattico. Controllo dei tipi,
delle forme ed esecuzione arriveranno nelle prossime milestone.

## Grammatica (sottoinsieme attualmente analizzato dal parser)

```
Program        := Declaration*
Declaration    := TargetDecl | PrecisionDecl | ModelDecl | DatasetDecl | TrainDecl | ForecastDecl

TargetDecl     := 'target' DottedName

PrecisionDecl  := 'precision' '{' PrecisionField* '}'
PrecisionField := PrecisionKw DottedName
PrecisionKw    := 'storage' | 'compute' | 'accumulate'
                | 'parameters' | 'forward' | 'backward'

ModelDecl      := 'model' Identifier '{' ModelStatement* '}'
ModelStatement := InputDecl | PipelineStmt

InputDecl      := 'input' TensorType
TensorType     := DottedName '[' ShapeDim (',' ShapeDim)* ']'
ShapeDim       := IntegerLiteral | Identifier

PipelineStmt   := PipelineSource ('|>' PipelineStage)+
PipelineSource := 'input' | Identifier
PipelineStage  := Identifier ( '(' (Arg (',' Arg)*)? ')' )?
Arg            := IntegerLiteral | FloatLiteral | StringLiteral | Identifier

DatasetDecl    := 'dataset' Identifier '{' DatasetField* '}'
DatasetField   := 'path' StringLiteral
                | 'input' TensorType
                | 'labels' TensorType

TrainDecl      := 'train' '{' TrainField* '}'
TrainField     := 'model' Identifier
                | 'dataset' Identifier
                | 'loss' Identifier
                | 'optimizer' Identifier
                | 'epochs' IntegerLiteral
                | 'batch_size' IntegerLiteral
                | 'learning_rate' (IntegerLiteral | FloatLiteral)
                | LoraField

LoraField      := 'lora' '{' LoraSubField* '}'
LoraSubField   := 'rank' IntegerLiteral | 'alpha' (IntegerLiteral | FloatLiteral)

ForecastDecl   := 'forecast' '{' ForecastField* '}'
ForecastField  := 'model' Identifier | 'horizon' IntegerLiteral

DottedName     := Identifier ('.' Identifier)*
```

Note sul recupero dagli errori: se una dichiarazione top-level non è
valida, il parser la salta fino alla prossima parola chiave
`target`/`precision`/`model`; se un'istruzione dentro un `model` non è
valida, salta fino alla prossima istruzione valida o a `}`. Questo
permette a `blackforge check` di riportare più errori sintattici in una
sola esecuzione, invece di fermarsi al primo.

## Analisi semantica (stato attuale)

Dopo il parsing, `blackforge check` valida:

- **Target**: deve essere uno tra `nvidia.blackwell`, `nvidia.hopper`,
  `nvidia.ampere`, `nvidia.ada`, `cpu`. Al massimo una dichiarazione
  `target` per programma.
- **Formati numerici**: ogni valore in `precision { ... }` e ogni dtype
  di un tensore deve essere uno tra `fp8.e4m3`, `fp8.e5m2`, `fp16`,
  `bf16`, `tf32`, `fp32`. `tf32` è rifiutato nei campi `storage` e
  `parameters` (è solo una modalità di calcolo, non un formato di
  memorizzazione). Nessun campo duplicato nello stesso blocco
  `precision`.
- **Forme tensoriali**: almeno una dimensione; le dimensioni letterali
  devono essere intere positive. Le dimensioni simboliche (es. `batch`)
  sono accettate senza ulteriori vincoli per ora — la loro risoluzione
  a runtime arriverà con l'esecuzione. *(L'inferenza della forma
  attraverso una pipeline, es. calcolare che `linear(4096)` produce
  effettivamente `[batch, 4096]`, richiede l'IR e non è ancora
  implementata: qui si controllano solo le forme dichiarate
  esplicitamente.)*
- **Modelli**: nome univoco nel programma; esattamente un `input`
  dichiarato.
- **Pipeline**: la sorgente deve essere `input` (con un `input`
  effettivamente dichiarato nel modello). Il linguaggio non supporta
  ancora binding con nome per risultati intermedi, quindi qualunque
  altro identificatore come sorgente è un errore di "non definito".
- **Operazioni**: le fasi di pipeline devono essere tra le operazioni
  note — `linear(n)` (un intero positivo), `silu`, `relu`, `gelu`
  (zero argomenti). Questo elenco crescerà quando il backend implementerà
  davvero le operazioni.
- **Dataset**: nome univoco nel programma; esattamente un campo `path`,
  un `input` e un `labels`, entrambi tensori validi (stesse regole di
  forma/dtype di un `model`).
- **Train**: esattamente un `model` (che deve riferirsi a un modello
  dichiarato nel programma), un `dataset` (idem), una `loss` (solo
  `mse` per ora), un `optimizer` (`sgd` o `adamw`), `epochs` e
  `batch_size` (interi positivi); `learning_rate` è opzionale (default
  `0.001` se omesso) e deve essere positivo se presente. I riferimenti
  a `model`/`dataset` sono risolti per nome in tutto il programma,
  indipendentemente dall'ordine delle dichiarazioni. Il blocco `lora`
  opzionale richiede `rank` (intero positivo); `alpha` è opzionale
  (default `rank`, cioè fattore di scala 1) e deve essere positivo se
  presente.
- **Forecast**: esattamente un `model` (deve esistere) e un `horizon`
  (intero positivo). La compatibilità tra forma di input e di output
  del modello (necessaria per il rollout autoregressivo) non è
  verificabile qui: richiede la IR, quindi viene controllata a runtime
  da `blackforge forecast`.

## Rappresentazione interna (IR)

Dopo l'analisi semantica, `blackforge check` costruisce la IR
(`blackforge::ir`, visibile con `--print-ir`): un `Module` per
programma, con un `ModelIR` per ogni `model`. Ogni `ModelIR` è un
piccolo grafo di `Value` (formato numerico + forma) collegati da
`Operation` lungo una `Pipeline`.

A differenza dell'analisi semantica di base, qui la forma **viene
davvero propagata**: `linear(n)` sostituisce l'ultima dimensione del
tensore in ingresso con `n` (le altre, incluse quelle simboliche come
`batch`, sono preservate), mentre `silu`/`relu`/`gelu` non modificano
forma né formato numerico essendo operazioni elementwise. Questo
permette di rilevare errori come "linear applicato a un tensore senza
dimensioni" che l'analisi semantica locale non può vedere.

Non esiste ancora un pass manager con ottimizzazioni (fusione,
eliminazione di operazioni morte): con l'attuale insieme minimo di
operazioni non c'è ancora nulla di genuino da ottimizzare.

## Backend CPU di riferimento ed esecuzione

`blackforge run <file.bf>` esegue davvero un modello: costruisce la IR,
genera un tensore di input sintetico (le dimensioni simboliche come
`batch` sono risolte al valore passato con `--batch`, default 1) e
attraversa la prima pipeline del primo modello applicando le operazioni
una a una sul backend CPU (`blackforge::backend::cpu`):

- `linear(n)`: prodotto matriciale `[batch, in] x [in, n]` più bias
  `[n]`, con pesi generati in modo deterministico (seme fisso, non una
  strategia di inizializzazione statisticamente valida come
  Xavier/Kaiming);
- `silu`, `relu`, `gelu`: applicate elemento per elemento.

Se il programma dichiara un blocco `precision { storage ... compute ...
accumulate ... }` (vedi `include/blackforge/backend/cpu/quantize.hpp`),
`run` applica una **quantizzazione simulata**: ogni attivazione viene
arrotondata (con saturazione, non overflow a infinito) al formato
`storage` dopo ogni operazione, e gli operandi di ogni `linear` vengono
arrotondati al formato `compute` prima del prodotto matriciale. I
numeri restano rappresentati come `float` C++: si tratta di
un'emulazione via arrotondamento della mantissa (bit-esatta per BF16,
via frexp/ldexp per TF32/FP16/FP8), non di un vero calcolo hardware in
formato ridotto. `accumulate` è analizzato ma non ancora usato (non c'è
un accumulatore separato da quantizzare nell'attuale insieme di
operazioni). Senza un blocco `precision`, il comportamento resta quello
originale: calcolo sempre in piena float32.

Limitazioni note, esplicite:

- Il backend CUDA non applica ancora la quantizzazione: calcola sempre
  in float32 indipendentemente da `precision`. Solo il backend CPU la
  applica per ora.
- `blackforge train` ignora sempre `precision` (a prescindere dal
  device): la quantizzazione non è differenziabile così com'è
  implementata (servirebbe uno straight-through estimator per il
  backward, non ancora presente), quindi l'addestramento resta sempre
  a piena precisione float32 per non comprometterne la correttezza
  verificata via gradient checking.
- Viene eseguita solo la prima pipeline del primo modello del file.
- `blackforge run` usa `Executor`, che rigenera pesi casuali ad ogni
  chiamata (comodo per ispezionare rapidamente un'esecuzione, ma non
  allenabile). Per l'addestramento vero e proprio esiste un tipo
  separato, `Model` (vedi sotto), che possiede i propri parametri e li
  mantiene tra una chiamata e l'altra.

## Autodiff, loss, optimizer, checkpoint (motore CPU)

`blackforge::backend::cpu` include ora un motore di addestramento
completo, utilizzabile oggi solo come API C++ (non ancora da sintassi
`.bf`: le parole chiave `dataset`, `loss`, `optimizer`, `train` sono
lessate ma non hanno ancora grammatica — arriverà con la milestone di
training/fine-tuning/LoRA):

- **`Model`**: costruito da un `ir::ModelIR`, possiede i pesi (`Parameter`,
  con `value` e `grad`) e li mantiene tra `forward()` e `backward()`.
  `forward()` salva le attivazioni intermedie necessarie a
  `backward()`. `backward(outputGrad)` **accumula** i gradienti (non li
  azzera): va chiamato `zeroGrad()` prima di ogni step.
- **Autodiff**: formule di backward scritte a mano per ogni operazione
  (`matmulBackward`, `addBiasBackward`, `siluBackward`, `reluBackward`,
  `geluBackward`) — non un autodiff generico basato su un grafo di
  espressioni, dato il piccolo insieme fisso di operazioni. Verificate
  con *gradient checking* numerico (differenze finite) nei test.
- **Loss**: solo `meanSquaredError` (errore quadratico medio) per ora.
  Altre loss (es. cross-entropy) richiedono prima che il linguaggio
  possa dichiarare il tipo di compito (classificazione vs regressione).
- **Optimizer**: `SGD` (senza momento: `param -= lr * grad`) e `AdamW`
  (Loshchilov & Hutter 2019, con weight decay disaccoppiato dal
  gradiente).
- **Checkpoint**: `saveCheckpoint`/`loadCheckpoint` in un formato
  binario proprietario di BlackForge (magic `BFCKPT1`, non compatibile
  con formati esterni come safetensors), che salva ogni parametro per
  nome e forma.

Limitazione esplicita: l'inizializzazione dei pesi resta deterministica
ma non statisticamente valida (niente Xavier/Kaiming). Una strategia di
inizializzazione seria è lavoro futuro, cosi' come il caricamento di
pesi pre-allenati esterni.

## Backend CUDA

`blackforge::backend::cuda` implementa, sulla GPU, le stesse operazioni
del backend CPU di riferimento — compilato solo quando
`BLACKFORGE_ENABLE_CUDA=ON` e testato su hardware reale (NVIDIA GeForce
RTX 5060, Blackwell/sm_120):

- **`DeviceTensor`**: tensore RAII in memoria device (`cudaMalloc`/
  `cudaFree`), con `fromHost`/`toHost` per il trasferimento da/verso
  `runtime::Tensor`. Non copiabile (evita doppie `cudaFree`),
  spostabile.
- **Operazioni** (`blackforge/backend/cuda/ops.hpp`): `add`, `addBias`,
  `silu`, `relu`, `gelu` come kernel CUDA scritti a mano (un thread per
  elemento); `matmul` tramite **cuBLAS** (`cublasSgemm`), non un kernel
  scritto a mano — per un'operazione critica per le prestazioni come il
  prodotto matriciale, usare una libreria NVIDIA ufficiale è più
  affidabile e performante di reinventarla, come indicato negli
  obiettivi del progetto. `linear = addBias(matmul(...))`, come su CPU.
- **`Executor`** (in `blackforge::backend::cuda`, da non confondere con
  `blackforge::backend::cpu::Executor`): stessa interfaccia della
  controparte CPU, stessa inizializzazione dei pesi a parità di seme —
  cosa che permette di verificare la correttezza del backend GPU
  confrontandolo direttamente con quello CPU (vedi i test di parità in
  `tests/backend/cuda/executor_tests.cu`).
- **`enumerateDevices()`**: elenca le GPU NVIDIA visibili al driver
  (nome, compute capability, memoria totale), usato da
  `blackforge devices`. Restituisce un elenco vuoto (non lancia
  eccezioni) se non ci sono GPU: l'assenza di hardware CUDA è un esito
  normale del rilevamento, non un errore.

### Uso dalla CLI

```bash
blackforge devices                      # elenca CPU e le eventuali GPU CUDA rilevate
blackforge run modello.bf --device cuda # esegue sulla GPU invece che sulla CPU
```

### Nota sulla trasposizione per cuBLAS

cuBLAS lavora in column-major, mentre i tensori di BlackForge sono
memorizzati row-major. Il codice sfrutta il fatto che un buffer
row-major `[r, c]` è, byte per byte, identico alla sua trasposta
column-major `[c, r]`: per calcolare `C = A @ B` (row-major) si chiede
a cuBLAS di calcolare `C^T = B^T @ A^T` (column-major) sugli stessi
buffer, scambiando l'ordine degli operandi e le dimensioni `m`/`n`. È
la tecnica standard per usare cuBLAS con dati row-major; è verificata
nei test confrontando il risultato con il backend CPU su matrici non
quadrate (dove uno scambio errato di `m`/`n` produrrebbe una forma o
dei valori visibilmente sbagliati).

### Limitazioni esplicite

- Come il backend CPU, calcola sempre in **float32**: nessun uso reale
  di FP8/BF16/TF32 come precisione di calcolo, nessun uso dei Tensor
  Core. Il linguaggio riconosce e valida questi formati, ma
  l'esecuzione GPU non li sfrutta ancora — è il prossimo passo naturale
  per un backend Blackwell.
- Un nuovo handle cuBLAS viene creato e distrutto ad ogni chiamata a
  `matmul`: corretto ma non ottimale; il riuso di un handle persistente
  è un'ottimizzazione futura a basso rischio.
- Nessun supporto multi-GPU, stream CUDA o esecuzione asincrona ancora.
- Testato solo per l'architettura configurata in
  `BLACKFORGE_CUDA_ARCHITECTURES` (120, Blackwell consumer); altre
  architetture non sono state verificate.

## Training (pretraining, fine-tuning, LoRA, forecasting)

`dataset { ... }` e `train { ... }` collegano il linguaggio al motore
di addestramento CPU (autodiff/loss/optimizer/checkpoint, vedi sopra).
`blackforge train <file.bf>` esegue il primo blocco `train` trovato:

```blackforge
model TinyRegression {
    input bf16[batch, 4]
    input |> linear(2)
}

dataset ToyData {
    path "dati/train.bfdata"
    input bf16[batch, 4]
    labels bf16[batch, 2]
}

train {
    model TinyRegression
    dataset ToyData
    loss mse
    optimizer adamw
    epochs 100
    batch_size 8
    learning_rate 0.1
}
```

### Formato dataset su disco

`path` punta a un file nel formato binario proprietario di BlackForge
(`blackforge::data`, magic `BFDATA1`, non compatibile con formati
esterni come safetensors/numpy): numero di esempi, forma di un singolo
esempio di input e di labels, poi tutti gli input concatenati seguiti
da tutti i target concatenati, come float32. Non esiste ancora uno
strumento BlackForge per generare questo file da sorgenti reali (CSV,
immagini, ...): va scritto con `blackforge::data::saveDataset()` da
codice C++ (vedi `examples/tiny_regression.bf` e il dataset che lo
accompagna per un esempio minimo completo).

Nota sui letterali stringa: come in C/C++/Python, `\` in una stringa
BlackForge introduce un escape. Un percorso Windows con backslash va
scritto con `/` (es. `"C:/dati/train.bfdata"`) o con `\\` raddoppiati.

### Cosa fa `blackforge train`

1. Compila il file (lexer → parser → analisi semantica → IR); si
   ferma con errore se una qualunque fase fallisce.
2. Risolve i riferimenti `model`/`dataset` del blocco `train` nella IR
   e nell'AST.
3. Carica il dataset da disco.
4. Costruisce un `backend::cpu::Model` dal modello referenziato (pesi
   nuovi, oppure caricati da `--from-checkpoint` per il fine-tuning).
5. Per ogni epoca, itera i batch del dataset (in ordine, senza
   mescolamento/shuffle ancora) eseguendo forward → loss → backward →
   step dell'optimizer, stampando la loss media dell'epoca.
6. Se `--save-checkpoint <path>` è specificato, salva i pesi finali.

### LoRA

Aggiungendo un blocco `lora` dentro `train`:

```blackforge
train {
    model TinyRegression
    dataset ToyData
    loss mse
    optimizer adamw
    epochs 50
    batch_size 8
    learning_rate 0.01
    lora {
        rank 4
        alpha 8.0
    }
}
```

`blackforge train ... --from-checkpoint base.bfckpt` diventa allora un
fine-tuning LoRA: **richiede** `--from-checkpoint` (rifiutato altrimenti
con un errore esplicito — allenare un adapter su pesi casuali non ha
senso). Ogni layer `linear` del modello riceve due matrici a basso
rango A `[inFeatures, rank]` e B `[rank, outFeatures]`, inizializzate
rispettivamente con piccoli valori deterministici e a **zero** (cosí il
contributo dell'adapter è esattamente nullo all'inizio: il modello si
comporta come i soli pesi di base finché A/B non si allenano — la
convenzione standard di LoRA). L'uscita del layer diventa:

```
output = linear(x, W, b) + (alpha / rank) * (x @ A) @ B
```

con `W`/`b` **congelati** (nessun gradiente accumulato, non restituiti
dall'optimizer): solo A e B vengono allenati. Un checkpoint salvato
durante un addestramento LoRA (`--save-checkpoint`) contiene sia i pesi
di base sia gli adapter — è autosufficiente, caricabile con
`blackforge run`/`blackforge forecast` senza bisogno di sapere che è
stato allenato con LoRA.

### Forecasting

```blackforge
model DriftModel {
    input bf16[batch, 4]
    input |> linear(4)
}

forecast {
    model DriftModel
    horizon 8
}
```

`blackforge forecast <file.bf> --from-checkpoint pesi.bfckpt --batch N`
esegue un **rollout autoregressivo**: genera un input sintetico iniziale
con la forma dichiarata dal modello (dimensioni simboliche risolte a
`--batch`), poi applica il modello `horizon` volte, usando l'output di
ogni passo come input del passo successivo. Questo richiede che
l'ultima dimensione (le feature) di input e output del modello
coincidano — altrimenti l'output non potrebbe diventare il prossimo
input — e viene verificato a runtime (non nell'analisi semantica, che
non ha accesso alla IR con le forme inferite): un modello come
`linear(2)` con input a 4 feature viene rifiutato con un errore chiaro.
Come per `train`, `--from-checkpoint` è obbligatorio.

### Limitazioni esplicite

- Solo sul backend CPU: l'autodiff non esiste ancora sul backend CUDA
  (milestone 7 ha implementato solo la forward pass su GPU) — questo
  vale anche per LoRA e forecasting.
- Solo il primo blocco `train`/`forecast` del file viene eseguito.
- Nessuno shuffle dei dati tra le epoche; nessun batch parziale (un
  dataset con `numEsempi % batch_size != 0` scarta gli esempi in eccesso).
- Nessun learning rate scheduler, nessuna validazione/early stopping.
- Il forecasting usa un input sintetico iniziale (nessun modo ancora di
  fornire una sequenza "seed" reale da cui partire).

## CLI: comandi completi

```
blackforge check <file>      Analizza (lessico, sintassi, semantica); --verbose, --print-ast, --print-ir
blackforge build <file>      Compila e costruisce ogni modello (alloca i parametri) senza eseguirlo
blackforge run <file>        Esegue il primo modello; --batch N, --device cpu|cuda|cuda:N
blackforge train <file>      Addestra (CPU); --from-checkpoint, --save-checkpoint
blackforge forecast <file>   Rollout autoregressivo (CPU); --from-checkpoint (obbligatorio), --batch N
blackforge benchmark <file>  Tempo/throughput/memoria; --device, --batch, --warmup, --iterations
blackforge inspect <file>    Riepilogo: input, pipeline, numero di parametri per ogni modello
blackforge devices           Elenca i dispositivi disponibili (CPU e GPU CUDA rilevate)
```

Tutti i comandi restituiscono un codice di uscita coerente: `0` se
l'operazione riesce, `1` per un errore rilevato durante l'analisi o
l'esecuzione, `2` per un uso scorretto della CLI (argomenti mancanti o
invalidi, comando sconosciuto) — prima ancora di provare a leggere il
file.

### `blackforge build` vs `blackforge check`

`check` valida solo l'AST e la IR (nessuna allocazione). `build` fa un
passo in più: costruisce davvero un `backend::cpu::Model` per ogni
modello del programma, cioè alloca i suoi parametri. Questo intercetta
errori che `check` non può vedere — per esempio una dimensione delle
feature ancora simbolica in ingresso a un `linear` (serve un numero
concreto per allocare la matrice dei pesi) — e stampa il numero di
parametri di ogni modello.

### `blackforge inspect`

Mostra, per ogni modello del programma: la forma/dtype dell'input, il
numero totale di parametri allenabili (e la loro dimensione stimata in
MiB come float32), e l'intera pipeline con la forma prodotta da ogni
operazione. Utile come `model.summary()` di altri framework, per capire
rapidamente la forma di un modello senza eseguirlo.

### `blackforge benchmark`

Misura la prima pipeline del primo modello: `--warmup N` iterazioni
scartate (default 5, per escludere costi una tantum come le prime
allocazioni) seguite da `--iterations N` iterazioni misurate (default
20) con `std::chrono::steady_clock`. Riporta hardware (per la CPU: solo
"backend di riferimento", nessun rilevamento del modello di processore;
per la GPU: nome e indice via il driver CUDA), la precisione dichiarata
nel modello, la forma dell'input, tempo medio, throughput
(campioni/secondo) e una stima della memoria (elementi di input +
attivazioni intermedie + parametri, × 4 byte — una stima teorica, non
memoria di processo misurata). Se il modello dichiara un blocco
`precision`, la misura sul backend CPU applica davvero la
quantizzazione simulata a ogni iterazione (il tempo riportato include
quindi anche il suo costo); il confronto `--device cuda` calcola invece
sempre in piena float32 su entrambi i lati (CUDA non applica ancora
`precision`), per restare un confronto equo tra le due esecuzioni.

Con `--device cuda`, dopo la misura CPU esegue la stessa misura sulla
GPU e stampa anche lo speedup (`tempoCPU / tempoGPU`) e lo scarto
assoluto massimo tra l'output GPU e l'output calcolato dalla CPU con lo
stesso seme (la "modalità di riferimento" richiesta per un benchmark
credibile): un valore vicino a zero (tipicamente `1e-5`–`1e-6` per la
somma di molti prodotti in float32) conferma che GPU e CPU calcolano lo
stesso risultato, non solo che sono ugualmente veloci.

**Limitazione**: nessun profiling per singola operazione (solo il
tempo totale di una `forward()` completa); nessuna deduplicazione o
analisi statistica oltre alla media.
