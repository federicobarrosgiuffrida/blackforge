# Sintassi di BlackForge — stato attuale

Questo documento descrive solo ciò che il compilatore riconosce **oggi**
(lexer + parser + analisi semantica di base). Verrà esteso con la
rappresentazione interna (IR) e l'esecuzione man mano che le milestone
successive vengono completate. Non c'è ancora generazione di codice o
esecuzione: il compilatore sa solo dire se un programma è lessicalmente,
sintatticamente e semanticamente valido.

## Commenti

```
// commento a fine riga
/* commento
   su più righe */
```

## Identificatori e parole chiave

Un identificatore inizia con una lettera o `_`, seguito da lettere,
cifre o `_` (es. `TinyModel`, `batch_size`, `bf16`).

Le seguenti parole sono riservate e non possono essere usate come nomi:

```
target precision storage compute accumulate parameters forward backward
model input output dataset loss optimizer train pretrain finetune lora
forecast benchmark
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
Declaration    := TargetDecl | PrecisionDecl | ModelDecl

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

Limitazioni note, esplicite:

- Il backend CPU calcola sempre in **float32**, indipendentemente dal
  formato numerico dichiarato (`bf16`, `fp8`, ...): serve a verificare
  la correttezza funzionale, non a riprodurre la precisione reale
  dell'hardware. L'emulazione dei formati ridotti arriverà con il
  backend CUDA.
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
