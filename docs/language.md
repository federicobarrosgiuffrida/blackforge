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
  note — `linear(n)` (un intero positivo), `silu`, `relu`, `gelu`,
  `rmsnorm`, `softmax` (zero argomenti), `embedding(vocab, dim)` (due
  interi positivi), `positional_embedding(maxSeq)`, `attention(numHeads)`,
  `feedforward(hiddenDim)` (un intero positivo ciascuna). Questo elenco
  crescerà quando il backend implementerà davvero le operazioni.
- **Dataset**: nome univoco nel programma; esattamente un campo `path`,
  un `input` e un `labels`, entrambi tensori validi (stesse regole di
  forma/dtype di un `model`).
- **Train**: esattamente un `model` (che deve riferirsi a un modello
  dichiarato nel programma), un `dataset` (idem), una `loss` (`mse` o
  `cross_entropy`), un `optimizer` (`sgd` o `adamw`), `epochs` e
  `batch_size` (interi positivi); `learning_rate` è opzionale (default
  `0.001` se omesso) e deve essere positivo se presente. `lr_schedule`
  è opzionale (unico valore supportato: `cosine`); se assente il
  learning rate resta costante. I riferimenti a `model`/`dataset` sono
  risolti per nome in tutto il programma, indipendentemente
  dall'ordine delle dichiarazioni. Il blocco `lora` opzionale richiede
  `rank` (intero positivo); `alpha` è opzionale (default `rank`, cioè
  fattore di scala 1) e deve essere positivo se presente.
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
`batch`, sono preservate), mentre `silu`/`relu`/`gelu`/`rmsnorm`/
`softmax` non modificano forma né formato numerico (elementwise, o
comunque riga per riga nel caso di `rmsnorm`/`softmax`). Questo
permette di rilevare errori come "linear applicato a un tensore senza
dimensioni" che l'analisi semantica locale non può vedere.

`embedding(vocab, dim)` aggiunge una nuova dimensione finale `dim` (es.
`[batch, seq]` → `[batch, seq, dim]`): è l'unica operazione che cambia
il *rango* del tensore, non solo una dimensione esistente.
`positional_embedding(maxSeq)`, `attention(numHeads)` e
`feedforward(hiddenDim)` non alterano la forma (sono operazioni
additive/residuali: l'uscita ha esattamente la stessa forma
dell'ingresso) — la loro validazione di forma "vera" (es. `dim %
numHeads == 0`, o che la dimensione delle feature sia concreta e non
simbolica) avviene un livello più giù, alla costruzione di
`backend::{cpu,cuda}::Model` (vedi sotto), non qui nell'IR.

Non esiste ancora un pass manager con ottimizzazioni (fusione,
eliminazione di operazioni morte): con l'attuale insieme minimo di
operazioni non c'è ancora nulla di genuino da ottimizzare.

## Backend CPU di riferimento ed esecuzione

`blackforge run <file.bf>` esegue davvero un modello: costruisce la IR,
genera un tensore di input sintetico (le dimensioni simboliche come
`batch` sono risolte al valore passato con `--batch`, default 1) e
attraversa la prima pipeline del primo modello applicando le operazioni
una a una sul backend CPU (`blackforge::backend::cpu`):

- `linear(n)`: prodotto matriciale `[..., in] x [in, n]` più bias `[n]`
  (generalizzato a rango >= 2: per un ingresso a rango > 2, es.
  `[batch, seq, in]`, appiattisce temporaneamente a `[batch*seq, in]`
  prima del prodotto e ripristina la forma dopo — lo stesso layout flat
  riga-maggiore rende l'operazione a costo zero). Senza
  `--from-checkpoint`, i pesi sono generati in modo deterministico (seme
  fisso, non una strategia di inizializzazione statisticamente valida
  come Xavier/Kaiming). Con `--from-checkpoint <file>` (CPU e CUDA), i
  pesi sono invece quelli realmente allenati caricati da quel
  checkpoint: internamente `run` costruisce un `Model` (che possiede i
  propri parametri) al posto dell'`Executor` (che li rigenera casuali ad
  ogni chiamata) e ci carica il file con `loadCheckpoint`;
- `silu`, `relu`, `gelu`: applicate elemento per elemento;
- `rmsnorm`: normalizzazione RMS (Zhang & Sennrich, 2019) riga per riga
  (rango >= 2: l'ultima dimensione sono le "feature", tutte le altre
  sono "righe" indipendenti), `y = x / sqrt(mean(x^2) + eps)` con
  `eps = 1e-6` fisso. **Senza** il fattore di scala `gamma` allenabile
  della formulazione più comune (es. LLaMA): è normalizzazione pura,
  senza parametri propri — una scelta di scope deliberata per non dover
  estendere ancora il formato dei checkpoint e l'interazione con LoRA
  per un singolo layer. Backward verificato con gradient checking
  numerico (vedi `tests/backend/cpu/autodiff_tests.cpp`).
- `softmax`: softmax riga per riga (rango >= 2, stessa generalizzazione
  di `rmsnorm`), `y_j = exp(x_j - max) / sum_k exp(x_k - max)`
  (sottrazione del massimo per stabilità numerica). Operazione di
  pipeline a sé stante (utile per ottenere probabilità esplicite in
  uscita da un modello, es. per l'inferenza), distinta dalla softmax
  applicata internamente da `loss cross_entropy` (che usa una formula
  combinata softmax+cross-entropy più efficiente per l'addestramento —
  le due implementazioni non sono collegate). Backward verificato con
  gradient checking numerico.
- `embedding`, `positional_embedding`, `attention`, `feedforward`: vedi
  la sezione dedicata "Blocchi transformer" più sotto.

### Blocchi transformer: `embedding`, `positional_embedding`, `attention`, `feedforward`

Disponibili sia sul backend CPU sia su CUDA, come qualunque altra
operazione di pipeline — questi quattro blocchi sono ciò che serve per
costruire un modello linguistico minimale in stile decoder-only
(LLaMA-like), oltre a `linear`:

- **`embedding(vocab, dim)`**: lookup di embedding. L'ingresso è un
  tensore `[batch, seq]` di **id di token**, rappresentati come float
  non negativi (non c'è ancora un dtype intero/indice dedicato nel
  linguaggio: gli id vengono arrotondati con `std::lround` internamente,
  una semplificazione pragmatica esplicita). L'uscita è `[batch, seq,
  dim]`. Lancia un errore se un id è fuori da `[0, vocab)`. Il
  parametro allenabile è la tabella `[vocab, dim]`; il gradiente rispetto
  agli id non esiste (sono indici, non differenziabili) — solo la
  tabella riceve gradiente, via scatter-add (più occorrenze dello stesso
  token nello stesso batch accumulano il contributo di ogni occorrenza).
- **`positional_embedding(maxSeq)`**: aggiunge un embedding posizionale
  allenabile `table[s, :]` a ogni posizione `s` della sequenza (additivo,
  forma invariata: `[batch, seq, dim] → [batch, seq, dim]`). La
  dimensione `dim` della tabella `[maxSeq, dim]` è dedotta a runtime
  dall'ultima dimensione dell'ingresso (tipicamente l'uscita di
  `embedding` che la precede), non è un argomento separato.
- **`attention(numHeads)`**: self-attention causale multi-testa
  (mascheramento autoregressivo: la posizione `s` può guardare solo alle
  posizioni `≤ s`), **con residual e pre-norm interni all'operazione**
  (`y = x + Wout · MultiHeadAttn(Q, K, V da RMSNorm(x))`, proiezioni
  Q/K/V/Out senza bias, come in LLaMA). `dim` (l'ultima dimensione
  dell'ingresso) deve essere divisibile per `numHeads`, controllato alla
  costruzione del modello (non nell'analisi semantica, che non conosce
  ancora `dim` in quel punto). Forma invariata.
- **`feedforward(hiddenDim)`**: blocco feed-forward pre-norm con
  residual (`y = x + Linear2(SiLU(Linear1(RMSNorm(x))))`). Forma
  invariata.

**Perché residual e pre-norm sono "dentro" l'operazione invece che
espressi nella sintassi `.bf`**: la pipeline di BlackForge
(`input |> op1 |> op2 |> ...`) è oggi una sequenza strettamente lineare,
senza un modo di esprimere una connessione residua o un binding con
nome per un risultato intermedio a cui tornare più avanti. Piuttosto che
riprogettare la sintassi della pipeline per supportare grafi generici (un
cambiamento di scope molto più ampio), `attention` e `feedforward`
incapsulano l'intero blocco transformer standard (norm → trasformazione
→ residual) come un'unica operazione atomica — la stessa scelta fatta da
molte implementazioni di riferimento di Transformer block.

Un modello linguistico minimale completo:

```blackforge
model TinyLM {
    input bf16[batch, 6]
    input |> embedding(8, 8) |> positional_embedding(6) |> attention(2) |> feedforward(16) |> linear(8)
}
```

Vedi [`examples/tiny_lm.bf`](../examples/tiny_lm.bf) per un esempio
completo ed eseguibile (con dataset incluso), allenato con
`loss cross_entropy` su un compito minimale di "shift-copy" (predire il
token della posizione precedente: risolvibile solo con attention, non da
embedding/feedforward presi da soli, che sono per-posizione e non
possono spostare informazione tra posizioni della sequenza).

Sia il backward di `attention` sia quello di `feedforward` **ricalcolano
il forward internamente** a partire da `input` + pesi (stessa convenzione
già usata da `softmaxBackward`), invece di mettere in cache stati
intermedi aggiuntivi: mantiene invariata la struttura di caching di
`Model` (che deve solo salvare `cachedInput` per ogni layer, come già
faceva per `linear`).

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
- **Loss**: `meanSquaredError` (regressione) e `softmaxCrossEntropy`
  (classificazione multiclasse, softmax applicata internamente) — vedi
  la sezione "Training" più sotto per i dettagli, entrambe raggiungibili
  da sintassi `.bf` tramite il campo `loss` di un blocco `train`.
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
  `rmsnorm`/`softmax` sono kernel scritti a mano con un blocco per riga
  del batch e una riduzione in shared memory (somma dei quadrati per
  `rmsnorm`; massimo poi somma degli esponenziali per `softmax`,
  numericamente stabile), verificati anche con `features` maggiore del
  numero di thread per blocco, per esercitare davvero il ciclo
  grid-stride dentro il kernel. `embedding`/`positional_embedding`/
  `attention`/`feedforward` (`src/backend/cuda/ops_transformer.cu`)
  mirrorano esattamente la logica del backend CPU: `attention` decompone
  la multi-head attention in un loop host-side su `(batch, testa)` che
  estrae/riscrive fette di tensore e richiama gli stessi primitivi 2D
  già testati (`matmul`, `matmulTransposeB`, `softmax`), invece di un
  kernel monolitico fuso — riduce la superficie di codice nuovo e non
  verificato per un'operazione complessa come l'attention causale.
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

### Tensor Core BF16 reale (`precision { compute bf16 }`)

A differenza della quantizzazione simulata del backend CPU (che
arrotonda un valore già calcolato in float32, quindi non è
differenziabile e resta solo-inferenza), `compute bf16` su
`--device cuda` usa **Tensor Core BF16 reali** via **cuBLASLt**
(`matmulBf16`/`matmulBf16Backward`, `src/backend/cuda/ops_tensorcore.cu`):
gli operandi vengono convertiti in BF16 prima del prodotto matriciale,
con accumulo e uscita in float32 (lo schema "mixed precision" standard
usato per l'addestramento di modelli linguistici — BF16 ha lo stesso
range di esponente di FP32, quindi a differenza di FP16 non serve loss
scaling per evitare overflow/underflow). Il backward differenzia
esattamente l'operazione BF16 che il forward ha davvero eseguito (non
un proxy FP32 arrotondato dopo il fatto), quindi è **genuinamente
allenabile**, non solo utilizzabile per l'inferenza:

```blackforge
precision {
    compute bf16
}

model TinyLM {
    input bf16[batch, 8]
    input |> embedding(300, 16) |> positional_embedding(8) |> attention(2) |> feedforward(32) |> linear(300)
}
```

`cuda::Model` applica Tensor Core BF16 a ogni layer **`linear`** della
pipeline **e** alle proiezioni Q/K/V/Out interne ad `attention` e ai
due layer interni a `feedforward` (`selfAttentionBf16`/
`feedForwardBf16`, vedi la sezione successiva) — sia in `forward()` sia
in `backward()`, sia per `blackforge train --device cuda` sia per
`blackforge run --device cuda --from-checkpoint`. Il nucleo
dell'attention vera e propria (Q@K^T scalato, maschera, softmax,
probs@V) resta sempre float32: solo le proiezioni lineari (la parte
dominante del costo computazionale) passano per il Tensor Core.
Verificato con gradient checking numerico
(`CudaAutodiffTest.MatmulBf16Backward...`,
`CudaAutodiffTest.SelfAttentionBf16Backward...`,
`CudaAutodiffTest.FeedForwardBf16Backward...`), confronto diretto con
la controparte FP32 e un training loop end-to-end sull'intera pipeline
di un modello linguistico (stessa traiettoria di loss della versione
FP32).

Limitazioni esplicite di questa prima versione:

- Solo **BF16**: `FP8` (e4m3/e5m2) richiederebbe fattori di scala per
  vivere nel suo range dinamico molto più limitato (tipico
  dell'addestramento a bassa precisione reale, es. Transformer Engine
  di NVIDIA) — un sotto-progetto sostanzialmente più complesso di BF16,
  non ancora affrontato. `TF32`/`FP16` restano solo quantizzazione
  simulata (solo CPU, solo inferenza, vedi sopra).
- `executor::Executor` (usato da `blackforge run` senza
  `--from-checkpoint`, pesi casuali di sola ispezione) non applica
  ancora Tensor Core: solo `Model` (pesi reali, allenabili o caricati
  da checkpoint) lo fa.
- Testato solo per l'architettura configurata in
  `BLACKFORGE_CUDA_ARCHITECTURES` (120, Blackwell consumer); altre
  architetture non sono state verificate.

### Prestazioni del backend CUDA: pool di memoria, attention fusa, Tensor Core esteso

Allenare un modello reale (~23M parametri, `TinyStories`, `batch_size
8`) con la versione iniziale del backend CUDA — corretta, verificata,
ma mai ottimizzata per throughput — risultava stimato in **ore** per
un addestramento completo. Cinque interventi mirati, ciascuno
verificato con l'intera suite di test prima di passare al successivo,
hanno ridotto il tempo per step da uno stimato oltre 400ms a
**~39,6ms** (misurato su un benchmark a regime, 1000 step a
`batch_size 8`, per isolare il costo per-step dall'overhead fisso di
avvio) — un fattore **~10×**. Un confronto diretto con un modello
PyTorch identico (stessi 23.152.640 parametri,
`scaled_dot_product_attention` con autocast BF16, stesso hardware:
16,08ms/step) mostra che BlackForge resta comunque **~2,5× più lento**
di PyTorch dopo tutti e cinque gli interventi:

1. **Pool di memoria device per-device** (`device_pool.hpp`/`.cu`):
   quasi ogni operazione allocava un `DeviceTensor` intermedio con
   `cudaMalloc` e lo liberava subito dopo con `cudaFree` — operazioni
   relativamente costose e sincronizzanti, ripetute migliaia di volte
   per singolo step. Sostituito con una free list a bucket esatti,
   indicizzata per device CUDA (essenziale per il training multi-GPU) e
   dimensione in byte: un buffer liberato resta pronto per la prossima
   richiesta della stessa dimensione sullo stesso device, invece di
   tornare al driver. Guadagno modesto da solo (~20%): non era il collo
   di bottiglia dominante.
2. **Attention batchizzata** (superata dal punto 4, vedi sotto):
   `selfAttention`/`selfAttentionIncremental`/`selfAttentionBackward`
   calcolavano Q@K^T/maschera/softmax/probs@V con un doppio loop
   host-side `for batch: for testa:`, lanciando ~9 kernel separati per
   OGNI combinazione (batch,testa) — con `batch_size=8` e 8 teste, 64
   iterazioni per singolo layer. Sostituito temporaneamente con
   `batchedQK`/`batchedMask`/`batchedPV`, un kernel ciascuno per TUTTE
   le combinazioni. Guadagno reale ma modesto (~20-25%): misurato che
   il collo di bottiglia dominante non era il numero di lanci di
   kernel — comunque materializzava l'intera matrice di score
   `[batch,numHeads,seq,seq]` in memoria globale, poi sostituito dal
   nucleo fuso del punto 4.
3. **Tensor Core BF16 esteso ad attention/feedforward** (vedi sopra):
   le proiezioni Q/K/V/Out/W1/W2 giravano ancora in FP32 su cuBLAS
   classico. Estendere `matmulBf16` anche lì (non solo a `linear`) ha
   dato un guadagno molto piu' significativo dei primi due — conferma
   che il collo di bottiglia dominante era il calcolo stesso, non
   l'allocazione né il numero di lanci di kernel.
4. **Attention fusa a online softmax** (`fused_attention.hpp`/`.cu`,
   stile FlashAttention): un confronto diretto contro un modello
   PyTorch identico (stessi 23.152.640 parametri, stesso hardware,
   `torch.nn.functional.scaled_dot_product_attention` con autocast
   BF16) ha mostrato BlackForge ancora ~4 volte più lento nonostante i
   tre interventi sopra — la causa: `batchedQK`/`batchedMask`/
   `softmax`/`batchedPV` materializzavano comunque l'intera matrice di
   score in memoria globale con 4 lanci di kernel separati, mentre
   `scaled_dot_product_attention` usa un kernel fuso a tiling che non
   la materializza mai. Sostituito con un kernel unico a "softmax
   online" (Milakov & Gimelshein 2018, lo stesso principio numerico di
   FlashAttention): un blocco CUDA per ogni combinazione
   (batch,testa,riga di query), che scorre le posizioni chiave fino al
   limite causale mantenendo massimo e somma correnti, senza mai
   scrivere la matrice di score in memoria globale — le posizioni oltre
   il limite causale non vengono nemmeno visitate. Il backward
   ricalcola gli score invece di salvarli, usando l'identità
   `D = dot(output, dOutput)` per il termine di correzione della
   softmax-backward (lo stesso trucco di FlashAttention per evitare di
   materializzare le probabilità); `dK`/`dV` sono accumulati con
   `atomicAdd` (più righe di query contribuiscono alla stessa chiave).
   La riduzione del prodotto scalare usa uno shuffle a due livelli
   (intra-warp via `__shfl_down_sync`, poi un solo `__syncthreads` per
   la riduzione finale tra warp) invece della classica riduzione ad
   albero in shared memory. Guadagno reale ma non risolutivo da solo:
   il gap rispetto a PyTorch scende da ~3,9× a ~2,8×.
5. **Tiling multi-riga** (stesso file, `tiledFusedAttentionForwardKernel`/
   `tiledFusedAttentionBackwardKernel`): anche con il kernel fuso del
   punto 4, ogni blocco (una riga di query sola) ricaricava K/V dalla
   memoria globale in modo indipendente — con `newLen` righe che
   condividono lo stesso K/V, un fattore `newLen` di traffico di
   memoria in più del necessario. Un vero FlashAttention carica un tile
   di K/V (`kBc = 32` posizioni) **una sola volta** in shared memory e
   lo riusa per un intero gruppo di righe di query (`kBr = 8`) nello
   stesso blocco, prima di passare al tile successivo — la
   sincronizzazione (`__syncthreads`) avviene una volta per tile
   (`~totalLen/kBc` volte) invece che una volta per singola posizione
   chiave. Un warp per riga (nessuna riduzione tra warp necessaria,
   solo `__shfl_down_sync` dentro il warp proprietario). Guadagno reale
   ma modesto (~12%): il gap scende da ~2,8× a **~2,5× più lento di
   PyTorch**.

L'incrocio dei risultati (guadagni via via più piccoli nonostante
interventi via via più aggressivi sull'attention) suggerisce che
l'attention non è più il collo di bottiglia dominante a questo punto —
probabilmente lo sono altre parti della pipeline non ancora esaminate
(embedding lookup con il suo round-trip host, il trasferimento
host→device per batch, l'optimizer). Identificarle richiederebbe una
profilazione reale (Nsight Compute o simili), non altre iterazioni alla
cieca sul kernel di attention: lavoro futuro esplicito, non
intrapreso in questa sessione.

Le cinque ottimizzazioni sono puramente di **strategia di
esecuzione**: nessun cambiamento alla matematica, verificato
dall'intera suite CUDA (391 test) rimasta verde ad ogni passo, inclusi
gradient checking e parità CPU/GPU per attention e feedforward.

#### Audit completo e cache dei piani cuBLASLt: un risultato nullo, riportato onestamente

Dopo il punto 5, un audit sistematico dell'intero repository (non solo
dell'attention) ha classificato ogni fonte di overhead individuata per
impatto atteso. La più promettente sulla carta: `bf16Gemm`
(`ops_tensorcore.cu`) ricreava da zero, ad **ogni singola chiamata**,
l'intero stato cuBLASLt — descrittore dell'operazione, tre layout di
matrice, preferenze, ricerca euristica dell'algoritmo
(`cublasLtMatmulAlgoGetHeuristic`) e allocazione/deallocazione del
workspace — nonostante la forma (M,K,N, trasposizioni) sia identica ad
ogni step per la stragrande maggioranza delle chiamate (proiezioni
Q/K/V/Out/W1/W2, sempre sulle stesse dimensioni). Sostituito con una
cache per-device (`GemmPlan`, chiave = tutte e sei le dimensioni grezze
più i flag di trasposizione, per non confondere le forme usate da
`matmulBf16Backward` per `dA`/`dB`): la prima chiamata con una
combinazione di forma crea e salva descrittore/layout/algoritmo/
workspace (quest'ultimo tramite il pool di memoria del punto 1, non più
`cudaMalloc` diretto), le successive fanno solo lookup e chiamano
`cublasLtMatmul`.

Risultato misurato (stesso benchmark a regime, 1000 step,
`batch_size 8`): **nessun miglioramento misurabile** — 43,84s prima,
43,88-43,98s dopo (tre run, differenza dentro il rumore di misura).
L'overhead per-chiamata di cuBLASLt non è quindi il collo di bottiglia
dominante a questa scala di problema (batch=8, seq=128, dim=512):
probabilmente il tempo speso nel kernel GEMM stesso, o overhead altrove
nella pipeline (sincronizzazioni host, un lancio di kernel per
parametro nell'optimizer), domina abbastanza da nascondere il costo
host-side rimosso qui. La modifica resta comunque corretta e a costo
zero (nessuna regressione sui 391 test CUDA né sui 282 test CPU) e
verrà mantenuta: a batch size più grandi, con più chiamate GEMM per
step, il beneficio potrebbe diventare misurabile — ma non era, come
stimato inizialmente, il singolo intervento a più alto impatto.
`batchedQK`/`batchedMask`/`batchedPV` (punto 2) sono stati rimossi dal
codice dopo essere stati superati dal nucleo fuso del punto 4, non
lasciati come codice morto.

#### Sincronizzazioni host rimosse dall'hot loop di addestramento: ~4% reale

Secondo intervento dell'audit: tre sincronizzazioni host bloccanti per
OGNI singolo step di addestramento, identificate leggendo `loss.cu`,
`ops_elementwise.cu` e `autodiff.cu`. Stesso protocollo del punto
precedente — uno stub temporaneo usa-e-getta (non committato) per
isolare l'impatto reale prima di investire in un refactor:

1. **Lettura della loss scalare** (`cudaMemcpy` in `meanSquaredError`/
   `softmaxCrossEntropy`/`softmaxCrossEntropySparse`): il valore veniva
   letto sull'host ad OGNI chiamata (1000 volte nel benchmark), anche se
   serve solo per la stampa di fine epoca — ogni lettura forza la CPU ad
   aspettare che l'intera pipeline GPU accumulata fino a quel punto
   finisca, impedendo qualunque overlap CPU/GPU tra uno step e il
   successivo.
2. **Validazione dei token id** (`embeddingLookup`/
   `embeddingLookupBackward`): un `tokenIds.toHost()` completo (l'intero
   batch) ad ogni forward E ad ogni backward, per controllare che ogni id
   sia in `[0, vocabSize)` prima di lanciare il kernel (un kernel CUDA
   non può lanciare eccezioni).
3. **Validazione degli indici di classe** (`softmaxCrossEntropySparse`):
   stesso pattern, un `targetIndices.toHost()` completo ad ogni chiamata.

Stub temporaneo di tutte e tre insieme: **41,0-41,4s** (tre run) contro
43,84-43,98s prima — un **~6-7% reale**, la seconda volta in questo
audit che una misura diretta smentisce l'intuizione iniziale (qui
l'overhead di sincronizzazione conta, ma meno di quanto la sola
presenza di tre round-trip per step avrebbe fatto pensare). Isolando il
solo punto 1 (validazioni riattivate): **42,9s**, cioè la lettura della
loss vale da sola solo ~2%; il resto (~4-5%) viene dalle due
validazioni.

Implementazione reale (non lo stub usa-e-getta, che avrebbe rotto la
sicurezza): per il punto 1, un accumulatore device (`DeviceTensor` di un
solo elemento) sommato via `atomicAdd` da un nuovo kernel
(`sumReduceAccumulateKernel`) ad ogni step, letto sull'host una sola
volta a fine epoca invece che una volta per step
(`meanSquaredErrorAccumulate`/`softmaxCrossEntropyAccumulate`/
`softmaxCrossEntropySparseAccumulate` in `loss.hpp`). Per i punti 2 e
3, NESSUNA validazione è stata rimossa: spostata invece sull'host, sui
dati del batch (`batch.input`/`batch.target`, entrambi ancora
`runtime::Tensor` residenti su CPU) PRIMA del caricamento su device —
stessi valori che il round-trip avrebbe scaricato dal device, zero
round-trip aggiuntivo, stessa tempistica dell'eccezione (lanciata nello
stesso punto logico del ciclo, non rimandata). Usata solo quando il
primo layer della pipeline è `embedding`
(`Model::firstLayerEmbeddingVocabSize()`): in quel caso
`Model::forward()`/`backward()` ricevono `inputRangeTrusted=true` e
usano `embeddingLookupPreValidated`/`embeddingLookupBackwardPreValidated`
solo per quel primo layer (ogni altro layer, e ogni chiamata generica
fuori dall'hot loop di training — `blackforge run`, `generate`, i
test — continua a validare internamente come prima, invariato).

Risultato misurato con l'implementazione reale (stesso benchmark):
**42,0-42,3s** (due run) contro 43,84-43,98s prima — **~4% reale**, un
guadagno onesto ma non risolutivo, verificato senza indebolire nessuna
garanzia di sicurezza (391 test CUDA verdi, inclusi i test di parità
CPU/GPU e multi-GPU, che confermano la matematica identica a meno del
riordino dell'accumulo in virgola mobile).

#### Il pool di memoria (punto 1, stage originale) non funzionava mai davvero: -34%, il singolo intervento più grande dell'intera sessione

Terzo intervento dell'audit, ma scoperto per caso mentre si leggeva
`device_tensor.cu` in preparazione al quarto (fusione
optimizer/zeroGrad, vedi sotto): `DeviceTensor::operator=(DeviceTensor&&)`
chiamava `cudaFree(data_)` DIRETTAMENTE sul buffer precedente invece di
`devicePoolRelease(...)` — esattamente l'operazione che il distruttore
fa correttamente due righe sopra. Risultato: ogni assegnamento a un
`DeviceTensor` già inizializzato bypassava del tutto il pool di memoria
introdotto al punto 1 della sezione precedente (quello misurato "~20%,
guadagno modesto, non il collo di bottiglia dominante").

L'assegnamento-spostamento è il pattern PIÙ comune nell'intero
forward/backward: `current = embeddingLookup(...)` / `current =
selfAttentionBf16(...)` / ecc. in `Model::forward()` (una volta per
layer: 19 per il modello TinyStories usato nel benchmark),
`layer.cachedInput = current.clone()` (idem, 19 volte),
`gradCurrent = ...Backward(...)` in `Model::backward()` (19 volte), e
soprattutto `param->grad = DeviceTensor::zeros(...)` in
`Model::zeroGrad()` (~68 volte, una per parametro). Circa 125
`cudaFree` reali per singolo step di addestramento, invece di altrettante
`devicePoolRelease` — il pool veniva svuotato ad ogni singolo step
sul percorso più caldo del codice, quindi la stragrande maggioranza
delle `devicePoolAcquire()` mancava sempre la cache e doveva fare una
`cudaMalloc` vera. Il memory pool "completato" al punto 1 della sezione
precedente non stava, di fatto, quasi mai facendo il suo lavoro.

Fix: una riga, `cudaFree(data_)` → `devicePoolRelease(data_,
elementCount() * sizeof(float))`, esattamente simmetrico al
distruttore. Nessun cambiamento di semantica (RAII identico, solo la
strategia di deallocazione cambia). Risultato misurato (stesso
benchmark a regime): **27,5-27,7s** contro 42,0-42,3s prima — **-34%**,
il singolo guadagno più grande di tutta la sessione, più grande di
tutti e cinque gli interventi della sezione precedente messi insieme.
Il gap rispetto a PyTorch scende da ~2,5x a **~1,7x più lento**
(27,5ms/step contro 16,08ms/step). Verificato: 391 test CUDA verdi
(nessun cambiamento di comportamento osservabile, solo quale
allocatore libera un buffer).

#### Optimizer e zeroGrad fusi ("multi-tensor"): nessun guadagno aggiuntivo, atteso col senno di poi

Quarto intervento dell'audit, quello originariamente pianificato prima
della scoperta del punto precedente: `SGD::step()`/`AdamW::step()`
lanciavano un kernel per OGNI parametro (~68 lanci per step sul modello
del benchmark) e `Model::zeroGrad()` faceva altrettante
`DeviceTensor::zeros()` (che ora, dopo il fix del punto precedente,
passano correttamente dal pool). Sostituiti con un solo lancio di
kernel ciascuno, stile "multi-tensor apply" di PyTorch
(`foreach`/`fused`): un array di metadati per-parametro
(puntatore+dimensione, per `SgdTarget`/`AdamWTarget`/`ZeroTarget` in
`optimizer.cu`/`model.cu`) caricato su device ad ogni chiamata (poche
centinaia di byte, nessuna dipendenza da un risultato GPU pendente,
quindi nessuna sincronizzazione bloccante), poi UN kernel con
`blockIdx.y` a scegliere il parametro invece di un lancio per
parametro.

Risultato misurato (stesso benchmark a regime): **27,4-27,8s**,
indistinguibile dai 27,5-27,7s di prima — **nessun guadagno aggiuntivo
rilevabile**. Col senno di poi, atteso: il fix del pool al punto
precedente aveva gia' reso il percorso per-parametro economico (pool
hit invece di vera `cudaMalloc`/`cudaFree`), quindi il numero di lanci
di kernel di per se' non era piu' il collo di bottiglia dominante a
questo punto — lo stesso schema gia' visto con la cache cuBLASLt (una
teoria ragionevole, smentita dalla misura diretta). Modifica mantenuta
comunque: codice piu' pulito (un solo punto di lancio invece di un
loop), nessuna regressione, verificato sui 391 test CUDA.

#### Cache pesi BF16 + clone in flatten2D: investigati, non implementati

Quinto intervento dell'audit, l'ultimo esaminato in questa sessione.
Due idee distinte, entrambe misurate/stimate PRIMA di scrivere codice
definitivo (stesso protocollo di tutta questa sezione):

1. **Cache dei pesi convertiti in BF16**: `matmulBf16`/`matmulBf16Backward`
   (`ops_tensorcore.cu`) convertono l'operando `b` (il peso) da FP32 a
   BF16 ad OGNI chiamata (`toBf16`), anche se il peso cambia una sola
   volta per step (dopo `optimizer->step()`) — forward e backward
   insieme lo riconvertono piu' volte per step senza necessita'. Prima
   di implementare una cache corretta (che richiede invalidazione
   affidabile: il puntatore di `param->value` resta stabile per tutta
   la vita del modello — l'optimizer scrive in-place — quindi non basta
   la sola identita' del puntatore come chiave, serve un segnale
   esplicito di "questo peso e' cambiato"), un esperimento usa-e-getta
   (cache MAI invalidata, deliberatamente scorretta, scartata subito
   dopo la misura — la loss infatti non scendeva piu', conferma che il
   trucco stava davvero bypassando gli aggiornamenti) ha misurato il
   **tetto massimo** di guadagno possibile: **26,4s** contro 27,4-27,8s
   — un ~4-5%. Una cache CORRETTA (con invalidazione reale) avrebbe
   inevitabilmente un overhead proprio (bookkeeping, controllo di
   validita' ad ogni chiamata) che eroderebbe parte di questo tetto —
   probabilmente vicino al rumore di misura di questo benchmark (~1-2%
   tra run identiche). Dato il rischio concreto (un bug di invalidazione
   qui produrrebbe un training che sembra funzionare ma converge su pesi
   sbagliati — un bug di correttezza silenzioso, non solo di prestazioni)
   a fronte di un guadagno atteso piccolo e incerto, non implementata in
   questa sessione.

2. **`flatten2D` e la sua `clone()`** (`autodiff.cu`/`model.cu`): appiattisce
   un tensore a rango > 2 a `[righe, features]` per poterlo passare a
   `matmulBackward`/`matmulBf16Backward` (primitivi puramente 2D),
   sempre via `clone()` (una copia device-to-device) perche' il
   tensore originale deve restare valido e riutilizzabile dal chiamante
   dopo la chiamata (commento esplicito nel codice). Circa 82 chiamate
   per step di backward sul modello del benchmark. Stima (non misurata
   con un esperimento usa-e-getta, per il rischio di corruzione di
   memoria di un hack scorretto su un puntatore condiviso): al più
   qualche decina di MB di traffico device-to-device ridondante per
   step, dell'ordine di ~0,1-0,2ms su banda GPU moderna — sotto lo
   0,5-1% del tempo per step attuale (27,5ms), quindi sotto il rumore di
   misura di questo stesso benchmark. Correggerlo richiederebbe comunque
   verificare individualmente ciascuno dei ~7 punti di chiamata per
   accertare che il tensore originale non serva più dopo (rischio di un
   bug "use after move" se fatto in fretta). Non implementato: il
   guadagno atteso è sotto la soglia di rilevabilità di questo
   benchmark, a fronte di un rischio di correttezza non nullo.

**Conclusione dell'audit (prima parte)**: dopo cinque interventi (cache
cuBLASLt: 0%, sync host rimosse: ~4%, fix del pool di memoria: **-34%**,
optimizer/zeroGrad fusi: 0%, cache BF16/flatten2D: non implementati,
guadagno atteso sotto il rumore), il gap rispetto a PyTorch è sceso da
**~2,5x a ~1,7x più lento** (39,6ms/step → 27,5ms/step, contro
16,08ms/step di PyTorch). Il profilo di overhead a quel punto sembrava
piatto — nessun singolo colpevole dominante — ma una seconda lettura
mirata del percorso caldo (non basata su stime, ma su un conteggio
diretto dei GEMM e dei lanci di kernel per step) ha trovato un'altra
fonte di overhead strutturale, grande quanto il fix del pool: il
backward ricalcolava l'intero forward.

#### Il backward ricalcolava l'intero forward: -19%

`selfAttentionBackward`/`feedForwardBackward` (`autodiff.cu`) sono
sempre state scritte per essere "stateless": dato solo `input` e i
pesi, ricalcolano da capo `rmsnorm`, le proiezioni Q/K/V, l'intero
nucleo fuso dell'attention (`fusedAttentionForward`), e per il
feedforward `linear1`+`silu` — perché `Model` cachava solo
`layer.cachedInput`, non le attivazioni intermedie del layer. Contando
i GEMM reali per step sul modello del benchmark: circa 180, contro i
~90-100 di un training loop PyTorch equivalente (che cachea le
attivazioni del forward invece di ributtarle via) — quasi il doppio
del lavoro del solo forward, pagato due volte ad ogni step.

Fix: due nuove coppie forward/backward — `selfAttentionForwardCached`/
`selfAttentionBackwardCached` e `feedForwardForwardCached`/
`feedForwardBackwardCached` (`ops.hpp`/`ops_transformer.cu`,
`autodiff.hpp`/`autodiff.cu`) — che popolano/consumano una cache di
attivazioni (`SelfAttentionCache`: `normedFlat`, `q`, `k`, `v`,
`attnOutput`, le statistiche online-softmax `m`/`l`; `FeedForwardCache`:
`normed`, `preActivation`, `hidden`), salvata in nuovi campi di
`LayerState` (`model.hpp`) e popolata da `Model::forward()`, poi letta
da `Model::backward()` invece di richiamare le funzioni "stateless"
originarie. Il costo aggiuntivo nel forward è minimo (un solo `clone()`
per layer, per preservare `attnOutput` a rango 3 dato che il backward
ne ha bisogno) — enormemente più economico del secondo forward completo
che elimina. Le funzioni originarie (`selfAttention`/`feedForward` e le
relative backward "stateless") restano invariate e in uso da
`executor.cu`/`blackforge run` (inferenza, nessun backward necessario)
e dai test di gradient checking.

Risultato misurato (stesso benchmark a regime): **22,2-22,4s** contro
27,5-27,7s prima — **-19%**, il secondo guadagno più grande di tutta
la sessione dopo il fix del pool. Il gap rispetto a PyTorch scende da
~1,7x a **~1,4x più lento** (22,3ms/step contro 16,08ms/step).
Verificato: 391 test CUDA verdi, inclusi i test di parità CPU/GPU (la
versione CPU di riferimento continua a ricalcolare tutto, quindi la
parità bit-per-bit tra le due implementazioni conferma che la cache non
ha introdotto alcuna differenza numerica) e il test di parità
multi-GPU.

**Conclusione aggiornata**: il gap rispetto a PyTorch è ora **~1,4x più
lento** (22,3ms/step vs 16,08ms/step), sceso da ~2,5x iniziale
attraverso sei interventi verificati. I prossimi in coda (cache pesi
BF16 a livello di Model con invalidazione affidabile dopo
`optimizer->step()`, proiezione QKV fusa, epilogo bias di cuBLASLt,
CUDA graph sull'intero step) restano il percorso più diretto verso la
parità o il sorpasso.

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

Il campo `loss` accetta tre valori:

- **`mse`** (errore quadratico medio): `mean((prediction - target)^2)`
  su tutti gli elementi. Adatta alla regressione, incluso il
  forecasting; `prediction`/`target` possono avere qualunque forma,
  purché coincidano.
- **`cross_entropy`** (cross-entropy con softmax interna, target
  **denso**): pensata per la classificazione multiclasse. Richiede
  `prediction`/`target` a rango >= 2 `[..., classi]` (es. `[batch,
  classi]` per la classificazione, `[batch, seq, classi]` per la
  next-token-prediction di un modello linguistico: la loss è la media
  su tutte le "righe", non solo sul batch); `prediction` sono i logit
  grezzi del modello (l'operazione applica una softmax internamente, in
  modo numericamente stabile), `target` è una distribuzione di
  probabilità per riga (tipicamente one-hot). Il gradiente restituito è
  la forma chiusa standard per softmax+cross-entropy combinati
  (`softmax(logits) - target`, diviso per il numero di righe), non
  richiede quindi un backward separato per la softmax.
- **`cross_entropy_sparse`** (stessa formula, target **sparso**):
  matematicamente identica a `cross_entropy`, ma `target` ha un rango in
  MENO di `prediction` (`[..., classi]` → `[...]`) e contiene, per ogni
  riga, l'**indice** della classe corretta invece di un vettore one-hot
  denso. Per un vocabolario grande (next-token-prediction con decine di
  migliaia di token) questo evita di materializzare mai un target
  `classi` volte più grande del necessario — è il formato prodotto da
  `blackforge dataset-build` (vedi la sezione "Tokenizer" più sopra) ed
  è la scelta corretta per allenare un modello linguistico su un
  vocabolario realistico.

Tutte e tre sono implementate sia in `blackforge::backend::cpu` sia in
`blackforge::backend::cuda` (vedi `loss.hpp`) e verificate con gradient
checking numerico (differenze finite centrali) e parità CPU/GPU.

### Formato dataset su disco

`path` punta a un file nel formato binario proprietario di BlackForge
(`blackforge::data`, magic `BFDATA1`, non compatibile con formati
esterni come safetensors/numpy): numero di esempi, forma di un singolo
esempio di input e di labels, poi tutti gli input concatenati seguiti
da tutti i target concatenati, come float32. Per dataset generici (CSV,
immagini, ...) va ancora scritto con `blackforge::data::saveDataset()`
da codice C++ (vedi `examples/tiny_regression.bf` e il dataset che lo
accompagna per un esempio minimo completo); per un **corpus di testo**,
`blackforge dataset-build` genera questo file automaticamente a partire
da un tokenizer addestrato (vedi la sezione "Tokenizer" più sotto).

Nota sui letterali stringa: come in C/C++/Python, `\` in una stringa
BlackForge introduce un escape. Un percorso Windows con backslash va
scritto con `/` (es. `"C:/dati/train.bfdata"`) o con `\\` raddoppiati.

### Tokenizer (`blackforge tokenizer-train`, `blackforge dataset-build`)

`blackforge::tokenizer` implementa un tokenizer **BPE byte-level**
(Byte Pair Encoding, Sennrich et al. 2016), nello stile di GPT-2: il
vocabolario di base sono i 256 valori di byte possibili — quindi
qualunque testo UTF-8 (o dato binario arbitrario) è rappresentabile
senza bisogno di un token `<unk>` — esteso con token appresi fondendo
iterativamente le coppie adiacenti più frequenti in un corpus di
training. Tre id di token speciali (`pad`=256, `bos`=257, `eos`=258)
sono riservati subito dopo i 256 byte base, con id fissi indipendenti
dalla dimensione del vocabolario allenato; i merge appresi partono
sempre da id 259 in su.

```bash
# Addestra un tokenizer BPE su un corpus di testo, vocabolario di 4096 token
blackforge tokenizer-train corpus.txt --vocab-size 4096 --output tok.bftok

# Ispeziona la tokenizzazione di un file (stampa gli id prodotti)
blackforge tokenizer-encode tok.bftok testo.txt

# Tokenizza il corpus e costruisce un dataset .bfdata pronto per
# 'blackforge train': finestre non sovrapposte di 'seq-len' token,
# target SPARSO shift-by-one (l'indice del token successivo per ogni
# posizione, forma [seqLen] — non un vettore one-hot [seqLen, vocabSize]:
# usa 'loss cross_entropy_sparse', non 'cross_entropy'
blackforge dataset-build corpus.txt tok.bftok --seq-len 128 --output corpus.bfdata
```

Il file `.bftok` (`blackforge::tokenizer::saveTokenizer`/`loadTokenizer`,
magic `BFTOKN1`, formato proprietario non compatibile con
tiktoken/sentencepiece/HuggingFace tokenizers) salva solo la lista dei
merge appresi, nell'ordine di apprendimento: l'intero stato del
tokenizer (vocabolario, priorità di fusione) è interamente
ricostruibile da quella lista.

Limitazioni esplicite dell'implementazione attuale:

- **Pre-tokenizzazione semplificata**: il corpus viene diviso in chunk
  su run di byte "parola" (lettere ASCII/cifre/byte >= 0x80, così le
  sequenze UTF-8 multi-byte restano raggruppate) e run di spazi
  (attaccati come prefisso alla parola successiva, convenzione stile
  GPT-2); niente regex Unicode-aware sofisticata come le implementazioni
  di riferimento — sufficiente per un round-trip `encode`/`decode`
  sempre esatto (proprietà verificata nei test), non ottimale per la
  qualità linguistica dei confini tra token.
- **Training O(numMerge × numChunkUnici)** per iterazione, non la
  struttura a coda di priorità + lista concatenata delle implementazioni
  ottimizzate per corpora da gigabyte: correttezza prima delle
  prestazioni.
- `dataset-build` (come `data::Dataset` in generale, sia in scrittura
  sia in lettura via `blackforge::data::loadDataset`) costruisce
  l'intero dataset **in RAM** prima di scriverlo/leggerlo — anche coi
  target sparsi (vedi sotto), un corpus da molti gigabyte non ci sta.
  Caricamento realmente streaming/memory-mapped (che non richieda mai
  l'intero corpus in memoria) resta lavoro futuro.

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

### Generazione autoregressiva con cache K/V (`blackforge generate`)

`forecast` (sopra) è pensato per modelli generici dove l'intero output
di un passo diventa l'input del passo successivo — non si applica a un
modello linguistico, dove l'input è una sequenza di **id di token** che
**cresce** ad ogni passo (non viene sostituita) e l'output è una
distribuzione sul vocabolario da cui va scelto il prossimo token.
`blackforge generate` è il comando dedicato a questo caso:

```bash
blackforge generate modello.bf --from-checkpoint pesi.bfckpt \
    --tokenizer tok.bftok --prompt "il gatto" --max-new-tokens 50 \
    --device cuda
```

Ricalcolare l'intera pipeline da capo ad ogni nuovo token (come farebbe
un `forward()` ingenuo su una sequenza che cresce di 1 ad ogni passo)
costerebbe O(N³) per generare N token in totale (O(N) chiamate, ciascuna
O(N²) per via dell'attention). `Model::forwardIncremental` (CPU **e
CUDA**, vedi `include/blackforge/backend/cpu/model.hpp` e
`include/blackforge/backend/cuda/model.hpp` — stessa interfaccia,
stessa semantica su entrambi i backend) mantiene invece una **cache
K/V** per ogni layer `attention` della pipeline (`ops::KVCache`):
ogni chiamata processa SOLO i token nuovi (l'intero prompt alla prima
chiamata, un solo token ad ogni chiamata successiva), calcola Q/K/V
solo per quelli, li accumula nella cache, e fa attendere le nuove query
all'intera cache — riducendo il costo totale a O(N²) (la parte
quadratica nel numero di token è intrinseca all'attention causale
stessa, non ulteriormente eliminabile senza cambiare l'architettura).
`embedding`/`positional_embedding`/`feedforward`/`linear` non hanno
bisogno di cache (sono per-posizione, nessuna dipendenza tra posizioni):
vengono applicati solo ai token nuovi ad ogni chiamata, cosa già
sufficiente ed esatta. `positional_embedding` usa la posizione
**assoluta** del token nella sequenza generata finora
(`addPositionalEmbeddingAt`, non semplicemente la posizione locale
nella chiamata corrente).

**Verificato per correttezza, non solo per velocità**: la cache è
un'ottimizzazione (riorganizzazione del calcolo), non
un'approssimazione — `ModelTest.ForwardIncrementalCorrispondeAForward
CompletoTokenPerToken` (CPU) e `CudaModelTest.ForwardIncrementalCorrispon
deAForwardCompletoTokenPerToken` (CUDA) verificano che generare token per
token con la cache produca, posizione per posizione, esattamente lo
stesso risultato che si otterrebbe ricalcolando l'intera sottosequenza
da capo con `forward()` ad ogni passo; `CudaModelTest.ForwardIncremental
CorrispondeAllaVersioneCpuAParitaDiSeme` verifica inoltre che, a parità
di seme, CPU e CUDA producano lo stesso output token per token —
confermato anche end-to-end via CLI reale su GPU (RTX 5060): lo stesso
checkpoint genera lo stesso identico testo con `--device cpu` e
`--device cuda`.

La decodifica è sempre **greedy** (argmax sulla distribuzione
dell'ultima posizione): nessuna strategia di campionamento
(temperatura, top-k, nucleus/top-p) è ancora implementata — la stessa
sequenza in ingresso produce sempre la stessa continuazione,
deterministica. Se la sequenza generata supera la lunghezza massima che
il modello supporta (`positional_embedding`'s `maxSeq`), la generazione
si interrompe con un messaggio esplicito invece di fallire in modo
oscuro.

### Modello linguistico mascherato — MLM (`bidirectional_attention`, `loss cross_entropy_masked`)

Tutto quanto visto finora (`attention`, `forecast`, `generate`) è
pensato per la next-token-prediction **causale** (autoregressiva): ogni
posizione vede solo se stessa e le precedenti. Un modello linguistico
**mascherato** (MLM, stile BERT) è diverso: alcune posizioni
dell'ingresso vengono sostituite con un token `<mask>`, e il compito è
ricostruire il token originale usando il contesto **sia a sinistra sia
a destra** — richiede quindi un'attention che veda l'**intera
sequenza**, non solo il passato.

```blackforge
model TinyMLM {
    input bf16[batch, 8]
    input |> embedding(300, 16) |> positional_embedding(8) |> bidirectional_attention(2) |> feedforward(32) |> linear(300)
}

dataset MaskedCorpus {
    path "corpus_mlm.bfdata"
    input bf16[batch, 8]
    labels bf16[batch, 8]
}

train {
    model TinyMLM
    dataset MaskedCorpus
    loss cross_entropy_masked
    optimizer adamw
    epochs 150
    batch_size 8
    learning_rate 0.05
}
```

- **`bidirectional_attention(numHeads)`**: stessa firma, stessi
  parametri (Wq/Wk/Wv/Wout, `[dim, dim]`, senza bias), stesso residual
  e pre-norm di `attention` — l'UNICA differenza è che la maschera
  causale non viene applicata: ogni posizione attende all'intera
  sequenza. Internamente, CPU condivide lo stesso nucleo di calcolo di
  `attention` (`selfAttentionImpl`/`selfAttentionBackwardImpl` in
  `ops.cpp`/`autodiff.cpp`), parametrizzato da un flag `causal`, cosi'
  le due varianti non possono divergere per un bug di copia-incolla.
- **`loss cross_entropy_masked`**: come `cross_entropy_sparse` (target
  sparso, un indice di classe per posizione), ma con un significato
  speciale per l'indice **-1**: "ignora questa posizione" (nessun
  contributo alla loss né al gradiente). Solo le posizioni
  effettivamente mascherate durante la preparazione del dataset
  contribuiscono all'addestramento — esattamente il compito MLM, non
  next-token-prediction su ogni posizione.
- **`blackforge dataset-build ... --mlm --mask-prob P`**: costruisce un
  dataset MLM invece che causale. Ogni posizione di ogni finestra viene
  mascherata con probabilità `P` (`0 < P <= 1`, default `0.15`, lo
  stesso valore usato dal BERT originale): se mascherata, l'input in
  quella posizione diventa `Tokenizer::kPadId` (id 256, riusato come
  token `<mask>` — `encode()` non lo produce mai su testo reale, quindi
  non c'è ambiguità con un token "vero") e il target è l'id del token
  ORIGINALE; se non mascherata, l'input resta il token originale e il
  target è `-1`.

Verificato end-to-end con la CLI reale (non solo unit test): un
corpus tokenizzato di questa sessione, dataset MLM costruito con
`--mlm --mask-prob 0.25`, modello allenato con `cross_entropy_masked` —
loss da 5.62 a 0.0017 in 150 epoche, a dimostrazione che il modello
impara davvero a ricostruire i token mascherati dal contesto
bidirezionale.

**Limitazioni esplicite**:

- `bidirectional_attention` esiste solo sul backend **CPU** per ora.
  `cuda::Model`/`cuda::Executor` **rifiutano esplicitamente** (errore
  chiaro, non un risultato quietamente sbagliato) una pipeline che la
  contiene: `bidirectional_attention non e' ancora implementato sul
  backend CUDA (solo su CPU)`.
- Nessuna generazione (`blackforge generate`) per un modello MLM: non
  ha senso, un modello bidirezionale non genera autoregressivamente
  (`Model::forwardIncremental` rifiuta esplicitamente una pipeline con
  `bidirectional_attention`, stesso principio).
- Il token `<mask>` riusa `Tokenizer::kPadId` invece di un id dedicato:
  una semplificazione pragmatica (evita di cambiare il layout del
  vocabolario, quindi la compatibilità di ogni `.bftok`/dataset già
  generato), documentata esplicitamente, non un'omissione nascosta.

### Addestramento su GPU (`blackforge train --device cuda`)

`blackforge::backend::cuda` include ora un motore di addestramento
completo (autodiff, loss, optimizer), interamente su device: forward,
loss, backward e l'aggiornamento dei pesi non lasciano mai la GPU
tranne per lo scalare finale della loss (necessario solo per stamparlo
a schermo) e per l'I/O di checkpoint (che deve comunque avvenire
sull'host). Ogni kernel di backward — `matmul`, `addBias`,
`silu`/`relu`/`gelu`, `rmsnorm`, `softmax` — è un kernel CUDA scritto a
mano, non un wrapper attorno al backend CPU: correttezza-prima-che-prestazioni,
quindi `matmulBackward` usa un kernel "ingenuo" (un thread per elemento
di output, loop di riduzione interno) invece di cuBLAS con
trasposizioni, la stessa struttura a triplo loop del backend CPU già
verificato, solo parallelizzata. Ogni kernel è confrontato
numericamente contro la controparte CPU (`tests/backend/cuda/
autodiff_tests.cu`, `loss_tests.cu`, `optimizer_tests.cu`), e l'intero
training loop è verificato end-to-end: uno stesso modello, allenato
indipendentemente su CPU e su GPU con lo stesso seme, stessi dati,
stesso optimizer, deve arrivare a una loss finale e a pesi finali
numericamente equivalenti (`CudaModelTest.
TrainingLoopCorrispondeAllaVersioneCpuAParitaDiSeme`).

**Percorso minimo, non ancora alla pari con la CPU.** Queste sono
limitazioni esplicite di questa prima versione, non omissioni
nascoste: se richieste, `blackforge train --device cuda` fallisce con
un errore chiaro (mai un fallback silenzioso sulla CPU, mai un
risultato quietamente sbagliato):

- Nessun blocco `lora`: nessun adapter a basso rango implementato su
  `cuda::Model`. Usa `--device cpu` per LoRA.

Sia `loss mse` sia `loss cross_entropy` sono supportate su
`--device cuda` (`softmaxCrossEntropy` è un kernel CUDA scritto a mano,
un blocco per riga con riduzione in shared memory, verificato per
parità numerica con la controparte CPU) — necessario per allenare un
modello linguistico (next-token-prediction) davvero sulla GPU, non solo
per la regressione.

`--from-checkpoint`/`--save-checkpoint` **sono** supportati su
`--device cuda`, con lo stesso formato binario del backend CPU
(`blackforge::backend::cuda::checkpoint`, magic `BFCKPT1` identico):
un checkpoint salvato da un training CUDA è caricabile da un `Model`
CPU e viceversa (verificato in
`CudaTrainRunnerTest.UnCheckpointSalvatoDaCudaECaricabileDaCpuEViceversa`),
quindi si può allenare su GPU e continuare il fine-tuning su CPU (o
viceversa) senza conversioni.
- L'inizializzazione dei pesi usa lo stesso generatore deterministico
  del backend CPU (non Xavier/Kaiming), come per `Executor`/`Model`
  CPU.

### Addestramento multi-GPU (`blackforge train --devices 0,1,...`)

Per un LLM di dimensioni realistiche, una singola GPU non basta:
`blackforge train` supporta l'addestramento **data-parallelo** su più
GPU dello stesso processo con il flag `--devices` (alternativo a
`--device`, esclusivo di CUDA):

```bash
blackforge train modello.bf --devices 0,1 --save-checkpoint pesi.bfckpt
```

**Come funziona**: una replica completa del modello per ogni indice in
`--devices` (stesso seme di inizializzazione, quindi pesi iniziali
identici su ogni GPU). Ad ogni step, `batch_size` viene diviso in
altrettanti shard disgiunti di uguale dimensione — un requisito
esplicito: `batch_size` deve essere divisibile per il numero di GPU,
altrimenti `blackforge train` fallisce con un errore chiaro invece di
approssimare con shard sbilanciati. Ogni replica calcola forward e
backward **indipendentemente** sul proprio shard e sulla propria GPU;
poi il gradiente di ogni parametro viene **mediato tra tutte le
repliche** (ALL-REDUCE) prima dello step dell'optimizer, cosicché ogni
replica applichi esattamente lo stesso aggiornamento e resti
sincronizzata con le altre — matematicamente equivalente ad allenare
con l'intero batch su una singola GPU (perché ogni operazione del
modello, `linear`/`rmsnorm`/`softmax`/`attention`/..., agisce
per-esempio, senza alcuna dipendenza incrociata dentro il batch), non
un'approssimazione. Solo la replica 0 viene salvata su checkpoint
(tutte restano sincronizzate per costruzione).

**Strategia di all-reduce — staging via host, non NCCL.** NCCL (la
libreria standard NVIDIA per la comunicazione collettiva tra GPU) non
supporta nativamente Windows, dove questo progetto viene sviluppato e
testato. L'all-reduce è quindi implementato copiando il gradiente di
ogni parametro dalla GPU alla RAM (una copia per replica), sommandolo e
mediandolo sulla CPU, e ricopiandolo su ciascuna GPU. È più lento di un
all-reduce diretto GPU-to-GPU (P2P/NVLink) o di NCCL, ma **funziona su
qualunque combinazione di GPU**, senza richiedere che supportino il
peer access (NVLink o lo stesso switch PCIe) — rilevante per GPU
affittate su cloud, dove il collegamento fisico tra le schede non è
garantito. Una scelta esplicita di correttezza e portabilità prima
delle prestazioni, coerente con il resto del progetto.

**Un bug genuino trovato e corretto lungo il percorso**: `cublasHandle_t`
e `cublasLtHandle_t` sono legati al contesto/device CUDA attivo al
momento della creazione — riusarne uno creato per il device 0 mentre è
attivo il device 1 è undefined behaviour. Prima di questa milestone,
`ops_gemm.cu`/`ops_tensorcore.cu` mantenevano un **singolo handle per
processo** (corretto per una sola GPU, ma silenziosamente sbagliato non
appena si alterna il device attivo tra repliche). Corretto con una
mappa `device -> handle`, un handle creato pigramente per ciascun
device incontrato (vedi `sharedHandle()`/`sharedLtHandle()`).

**Verificato per correttezza, non su hardware multi-GPU reale**: questa
è l'unica milestone di questo progetto per cui manca ancora la verifica
finale su hardware fisicamente multi-GPU — la macchina di sviluppo ha
una sola GPU (RTX 5060). Per rendere comunque possibile una
verifica rigorosa, `runMultiGpuTraining` permette deliberatamente indici
**ripetuti** in `--devices` (es. `{0, 0}`): due o più repliche sullo
stesso hardware fisico non offrono alcun vantaggio di velocità, ma
esercitano per intero la logica di sharding/all-reduce/sincronizzazione
dell'optimizer su una singola GPU disponibile. Con questo caso limite,
`CudaMultiGpuTrainRunnerTest.ProduceUnaLossFinaleEquivalenteAllaVersioneSingolaGpu`
e `CudaMultiGpuTrainRunnerTest.ParitaConPiuBatchPerEpocaEPiuEpoche`
verificano che l'addestramento a 2 repliche produca, epoca per epoca,
la stessa identica traiettoria di loss di un addestramento a singola
GPU sull'intero batch (entro la tolleranza numerica attesa da un
diverso ordine di somma in virgola mobile) — anche con più batch per
epoca e centinaia di step di ottimizzazione. La logica è quindi
verificata; la parallelizzazione reale su hardware fisicamente distinto
resta da confermare quando sarà disponibile (RTX PRO 4500/RTX PRO 6000
Blackwell pianificate).

**Limitazioni esplicite**: stesse limitazioni di `--device cuda` (niente
`lora`); `--devices` richiede almeno 2 indici (per una singola GPU si
usa `--device cuda:N`); nessun supporto multi-GPU per `run`/`generate`/
`forecast`/`benchmark` (solo `train`, dove il parallelismo dei dati ha
senso).

### Shuffling e learning rate scheduler

Ad ogni epoca, `blackforge train` rimescola l'ordine degli esempi del
dataset (`data::Dataset::shuffle`, Fisher-Yates seminato con il numero
di epoca: riproducibile, ma diverso epoca per epoca) prima di costruire
i batch — comportamento identico su CPU e CUDA (stesso seme per
epoca), verificato producendo la stessa identica traiettoria di loss
sui due backend. Con un solo batch per epoca (`batch_size ==
numEsempi`, il caso più comune negli esempi di questo repository) lo
shuffling non ha alcun effetto osservabile (l'ordine dentro un batch
completo non cambia il risultato); con più batch per epoca cambia
davvero quali esempi finiscono in quale batch da un'epoca all'altra.

Il campo opzionale `lr_schedule cosine` dentro `train` attiva il
cosine annealing (Loshchilov & Hutter, "SGDR", 2016): il learning rate
decade da `learning_rate` (prima epoca) a 0 (ultima epoca) seguendo
mezza onda di coseno, invece di restare costante. Implementato una sola
volta (`blackforge::backend::cosineAnnealingLearningRate`, condivisa
tra i train runner CPU e CUDA per evitare che le due implementazioni
divergano) e applicato tramite `Optimizer::setLearningRate()`, che
cambia il learning rate senza toccare altro stato dell'optimizer (per
AdamW, i momenti accumulati e il contatore di step restano invariati).
Senza `lr_schedule`, il comportamento resta quello originale: learning
rate costante per l'intero addestramento.

```blackforge
train {
    model TinyRegression
    dataset ToyData
    loss mse
    optimizer adamw
    epochs 100
    batch_size 8
    learning_rate 0.1
    lr_schedule cosine
}
```

### Limitazioni esplicite (comuni a CPU e GPU)

- Solo il primo blocco `train`/`forecast` del file viene eseguito.
- Nessun batch parziale (un dataset con `numEsempi % batch_size != 0`
  scarta gli esempi in eccesso).
- Nessuna validazione/early stopping; l'unico scheduler disponibile è
  `lr_schedule cosine` (vedi sopra).
- Il forecasting usa un input sintetico iniziale (nessun modo ancora di
  fornire una sequenza "seed" reale da cui partire), ed esiste solo sul
  backend CPU per ora.

## CLI: comandi completi

```
blackforge check <file>       Analizza (lessico, sintassi, semantica); --verbose, --print-ast, --print-ir
blackforge build <file>       Compila e costruisce ogni modello (alloca i parametri) senza eseguirlo
blackforge run <file>         Esegue il primo modello; --batch N, --device cpu|cuda|cuda:N
blackforge train <file>       Addestra (CPU o CUDA); --device, --from-checkpoint/--save-checkpoint (entrambi i device)
blackforge forecast <file>    Rollout autoregressivo generico (CPU); --from-checkpoint (obbligatorio), --batch N
blackforge generate <file>    Genera testo con cache K/V (CPU o GPU via --device, greedy); --from-checkpoint, --tokenizer, --prompt,
                               --max-new-tokens
blackforge benchmark <file>   Tempo/throughput/memoria; --device, --batch, --warmup, --iterations
blackforge inspect <file>     Riepilogo: input, pipeline, numero di parametri per ogni modello
blackforge devices            Elenca i dispositivi disponibili (CPU e GPU CUDA rilevate)
blackforge tokenizer-train <corpus.txt>            Addestra un tokenizer BPE; --vocab-size, --output
blackforge tokenizer-encode <tok.bftok> <testo.txt> Codifica un file di testo, stampa gli id
blackforge dataset-build <corpus.txt> <tok.bftok>  Costruisce un dataset .bfdata da un corpus; --seq-len, --output
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

Oltre al tempo aggregato, riporta anche un **breakdown per singola
operazione** della pipeline (solo backend CPU): ogni operazione viene
misurata separatamente, sul suo input reale (catturato eseguendo
un'intera pipeline una volta, poi ri-applicando solo quella singola
operazione per `--warmup`/`--iterations` volte) — utile per capire
quale operazione domina il tempo totale (tipicamente `linear`, per via
del prodotto matriciale), non solo il totale stesso. Non esiste ancora
su CUDA: un breakdown accurato per singolo kernel richiederebbe timer
basati su `cudaEvent` (i lanci di kernel sono asincroni, un semplice
`steady_clock` attorno a ogni chiamata misurerebbe anche tempo di
attesa non significativo, non solo l'esecuzione del kernel).

**Limitazione**: nessuna deduplicazione o analisi statistica oltre alla
media (né per il tempo aggregato né per il breakdown per operazione).
