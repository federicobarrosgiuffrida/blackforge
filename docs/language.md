# Sintassi di BlackForge — stato attuale

Questo documento descrive solo ciò che il compilatore riconosce **oggi**
(lexer + parser). Verrà esteso con l'analisi semantica e l'esecuzione
man mano che le milestone successive vengono completate. Non c'è ancora
alcun controllo di tipi, forme o validità dei target/precisioni: il
parser costruisce l'AST, non lo valida.

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
semantica (non ancora implementata).

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
