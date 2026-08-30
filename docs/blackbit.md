# BlackBit — motore MoE ternario a bassa memoria dentro BlackForge

Questo documento è il piano di lavoro (e il registro delle decisioni
architetturali) per **BlackBit-9B-A3B**: un modello linguistico MoE da
~9B parametri totali / ~3B attivi per token, pensato per essere
**preaddestrato a parametri pieni** su una singola RTX 5060 da 8 GB.

Non è un modello descritto nel linguaggio BlackForge (`.bf`): la
sintassi attuale è una pipeline lineare `|>` senza diramazioni, quindi
non può esprimere routing MoE, GQA o embedding condivise. BlackBit è
quindi un **sottosistema nativo** (`include/blackforge/blackbit/`,
`src/blackbit/`) che riusa le astrazioni esistenti del motore
(`runtime::Tensor`, `backend::cuda::DeviceTensor`, il pool di memoria
device, `matmulTransposeB`, l'attention fusa, `Optimizer`,
`softmaxCrossEntropySparse`) senza toccare i modelli esistenti.

---

## 1. Cosa esiste già nel repository (inventario dell'ispezione)

| Sottosistema | Dove | Riutilizzabile per BlackBit? |
|---|---|---|
| Tensore host | `runtime::Tensor` — `shape + vector<float>` | Sì, come formato di riferimento CPU e per l'I/O dei checkpoint |
| Tensore device | `backend::cuda::DeviceTensor` — RAII su `float*` | Sì. **Limite**: è `float32-only`, non ha dtype |
| Allocatore | `backend::cuda::devicePoolAcquire/Release` — free list a bucket esatti, per-device | Sì, è già l'inizio di un'arena. **Manca**: telemetria e budget |
| GEMM | `matmul` (cuBLAS SGEMM), `matmulBf16`/`matmulBf16CachedWeight` (cuBLASLt, Tensor Core BF16), `matmulTransposeB` (kernel a mano) | Sì. `matmulTransposeB` è esattamente la forma che serve a `TernaryLinear` |
| Attention | `fusedAttentionForward/Backward` — online softmax stile FlashAttention, `O(seq)` di stato, mai la matrice score in memoria globale | Sì, è la base per GQA. **Manca**: gruppi Q/KV |
| Autodiff | Funzioni libere `xxxBackward(...)` per operazione, orchestrate a mano da `Model::backward()` in un loop sui layer | Sì. **Non c'è un grafo dinamico**: il backward è già "a catena esplicita", il che rende lo *streaming backward* molto più facile da innestare che in un autograd a nastro |
| Parametri | `struct Parameter { name, value, grad }` — `grad` è un `DeviceTensor` **persistente**, allocato una volta e azzerato da `zeroGrad()` | **No**: è esattamente il "buffer di gradienti a dimensione modello" che BlackBit deve eliminare |
| Optimizer | `Optimizer` (interfaccia), `SGD`, `AdamW` con stato `m,v` per nome parametro | L'**interfaccia** sì; `AdamW` no (stato 2× i parametri) |
| Precisione | `sema::DType {FP8_E4M3, FP8_E5M2, FP16, BF16, TF32, FP32}`, `cpu::quantize` (arrotondamento *simulato*, i dati restano float32) | Il *dtype enum* va esteso con `TERNARY_1P58`. La quantizzazione simulata **non** è ciò che serve: BlackBit vuole storage davvero compresso |
| Checkpoint | `BFCKPT1`: magic + per parametro (nome, shape, float32) | Formato nuovo necessario (`BFBIT`), ma stessa filosofia e stesso stile di I/O |
| Cross-entropy | `softmaxCrossEntropySparse` — nessun target denso `[.., vocab]` | Sì, indispensabile con vocab 65536 |
| Checkpointing attivazioni | Non esiste. `SelfAttentionCache`/`FeedForwardCache` fanno l'**opposto**: memorizzano di più per ricalcolare di meno | Le cache restano per i modelli esistenti; BlackBit userà una politica opposta e configurabile |
| Rilevamento GPU | `enumerateDevices()` con compute capability | Sì, base per l'astrazione di capability Blackwell |

### Vincolo dell'ambiente di sviluppo

Il container in cui questo lavoro viene svolto **non ha né il CUDA
Toolkit né una GPU** (`nvcc` e `nvidia-smi` assenti). Di conseguenza:

* tutto ciò che è **verificato da test eseguiti davvero** vive nel
  percorso CPU / device-agnostico;
* i kernel CUDA vengono scritti condividendo con la CPU **le stesse
  funzioni di codifica/decodifica** (header `__host__ __device__`), in
  modo che la parte che i test coprono sia letteralmente la stessa che
  gira su GPU;
* nessun percorso CUDA verrà dichiarato "funzionante" senza essere stato
  compilato ed eseguito su hardware reale. Lo stato di ogni percorso è
  tracciato esplicitamente in fondo a questo documento.

---

## 2. Configurazione target e conti di memoria

```
vocab_size 65536   hidden 3072   layers 28
heads 24   kv_heads 6   head_dim 128
experts 8   experts_per_tok 2   expert_hidden 3968
weight_dtype TERNARY_1P58   norm/router BF16   tie_embeddings true
```

Parametri per layer:

| Blocco | Formula | Parametri |
|---|---|---|
| `q_proj` | 3072 × (24·128) | 9 437 184 |
| `k_proj` | 3072 × (6·128) | 2 359 296 |
| `v_proj` | 3072 × (6·128) | 2 359 296 |
| `o_proj` | (24·128) × 3072 | 9 437 184 |
| 8 esperti × (gate+up+down) | 8 · 3 · 3072 · 3968 | 292 552 704 |
| router | 3072 × 8 | 24 576 |
| 2 RMSNorm | 2 · 3072 | 6 144 |
| **totale layer** | | **316 176 384** |

* 28 layer → **8 852 938 752**
* embedding condivisa (tied) 65536 × 3072 → **201 326 592**
* norm finale → 3 072
* **Totale ≈ 9,054 G parametri**, di cui ternari 9,054 G − (router
  0,7 M + norm 0,2 M) ≈ **9,054 G ternari**.

Attivi per token: 23 592 960 (attention) + 2 · 36 569 088 (2 esperti su
8) + router ≈ 96 755 000 per layer × 28 = **2,709 G**, più la proiezione
di uscita legata all'embedding 0,201 G → **≈ 2,91 G attivi/token**.
Obiettivo `~9B / ~3B` centrato senza cambiare le dimensioni proposte.

### Memoria a regime (stima di progetto, da confermare con il benchmark)

Numeri prodotti da `blackbit::estimateTrainingMemory` (seq 512,
micro-batch 1, rango optimizer 32, stato in BF16), non stimati a mano:

| Voce | GiB |
|---|---|
| Pesi ternari impacchettati | **1,691** |
| Scale per gruppo (gruppo 160, FP32) | 0,217 |
| Norm + router (BF16) | 0,002 |
| Stato optimizer low-rank (r = 32, m e v, BF16) | 0,642 |
| Attivazioni con ricalcolo per layer | 0,221 |
| Picco gradiente (un blocco di 512 righe) | 0,0076 |
| Workspace (tile dequantizzati + logit a blocchi) | 0,0116 |
| **Totale stimato** | **2,792** |
| *Per confronto: approccio ordinario (master BF16 + grad FP32 + AdamW FP32)* | *118,1* |

Il margine rispetto agli 8 GB è voluto: serve per seq_len maggiori (a
4096 le attivazioni salvate diventano ~1,8 GB) e per la frammentazione
reale dell'allocatore.

Due voci meritano una nota, perché sono state corrette rispetto alla
prima stesura del piano dopo aver fatto i conti davvero:

* **Le scale costano 0,217 GiB**, non "trascurabile": una scala FP32
  ogni 160 pesi sono 0,2 bit/peso, cioè il 13 % del costo dei pesi
  stessi. Passarle a BF16 le dimezza; è la prima ottimizzazione
  disponibile se il budget si stringe.
* **Il picco di gradiente non è "una matrice per volta"**. La matrice
  più grande di BlackBit-9B è la tabella di embedding condivisa,
  65536 × 3072: il suo gradiente denso è **805 MB**, un decimo del
  budget totale per un buffer che vive microsecondi. Ma
  `dW[n,k] = Σ_m dY[m,n]·X[m,k]`: la riga *n* dipende solo dalla colonna
  *n* di `dY`, quindi il gradiente si calcola e si consuma a **blocchi
  di righe**. Con blocchi da 512 righe il picco scende a 7,8 MB. Il
  backward in streaming (§6) è quindi tilizzato su due assi, non uno.

Il numero che **non** deve mai comparire in questa tabella:
9,054 G × 2 B = **18,1 GB** di master copy BF16, e
9,054 G × 4 B = 36,2 GB di gradienti densi. Entrambi da soli superano la
VRAM disponibile di 2–4×: è la ragione per cui i requisiti 3, 6 e 7
esistono.

---

## 3. Formato ternario impacchettato (requisito 1)

### Scelta: base 3, 5 trit per byte, 4 byte per word

Un trit ha 3 stati; `3^5 = 243 ≤ 256`, quindi **5 pesi ternari entrano
esattamente in un byte** con codice `t0 + 3·t1 + 9·t2 + 27·t3 + 81·t4`
(dove `t ∈ {0,1,2}` rappresenta `{-1,0,+1}`).

* Densità: 8 bit / 5 pesi = **1,6 bit/peso** (l'ottimo teorico è
  log₂3 = 1,585: efficienza 99,1 %).
* **Decodifica senza divisioni**: una LUT di 256 elementi mappa un byte
  nei suoi 5 trit. Su GPU la LUT sta in constant memory o in shared
  memory (256 B, oppure 256 × 4 B se srotolata) ed è letta in broadcast.
* **Accessi coalescenti**: i byte sono raggruppati in `uint32`, quindi
  una singola load a 32 bit produce 20 pesi contigui e un warp legge
  128 B contigui (una transazione). Nessun trit attraversa mai il
  confine di un byte, quindi non serve alcuno shift a cavallo di parole.
* **Allineamento**: ogni riga logica è impacchettata indipendentemente e
  paddata a un numero intero di `uint32` — l'inizio di ogni riga è
  allineato a 4 byte, requisito minimo per le load vettoriali future
  (`uint4` con padding a 16 B è una variante immediata se servirà).

Alternative scartate:

* **2 bit/peso** (`{-1,0,+1,unused}`): decodifica banale ma è il 25 % di
  memoria in più — su 9B pesi sono **+450 MB**, cioè metà del budget di
  ottimizzatore. Chiamarlo "1,58 bit" sarebbe inoltre falso.
* **base 3 su interi a 32 bit** (20 trit per `uint32`, 1,6 bit/peso
  identici): stessa densità ma la decodifica richiede 20 `%3` in catena
  di dipendenze, contro 4 lookup indipendenti. Nessun vantaggio.
* **int8 per peso**: 5× la memoria. È esattamente ciò che il progetto
  vieta.

### Scale

Una scala per **gruppo** di `groupSize` pesi contigui lungo l'ultima
dimensione logica (default 160 = 32 byte = 8 word, così un gruppo cade
sempre su un confine di word). Il peso reale è
`w[i] = t[i] · scale[gruppo(i)]`. Le scale sono memorizzate in float32
in memoria (BF16 sul disco/GPU quando il requisito 15 lo consente):
9,054 G / 160 = 56,6 M scale, trascurabili rispetto ai pesi.

La quantizzazione usa il criterio *absmean* di BitNet b1.58:
`scale = mean(|w|)` sul gruppo, `t = round(w / scale)` saturato a
`[-1, 1]`. È un compromesso noto e già validato in letteratura.

### Layout della matrice di `TernaryLinear`

Il peso è memorizzato **`[outFeatures, inFeatures]`** (trasposto
rispetto al `linear` esistente, che usa `[in, out]`). Motivi concreti:

1. i gruppi di scala cadono lungo la dimensione di riduzione `K`, che è
   il raggruppamento statisticamente sensato;
2. `Y = X @ Wᵀ` è esattamente la firma di `matmulTransposeB(a[M,K],
   b[N,K]) -> [M,N]`, che **esiste già** sia su CPU sia su CUDA;
3. la dequantizzazione a tile produce un blocco `[nTile, K]` contiguo,
   cioè righe intere di `W`: letture perfettamente sequenziali del
   buffer impacchettato.

---

## 4. `TernaryLinear` (requisito 2)

Storage e compute sono cose diverse. Storage: `TERNARY_1P58`. Compute:
oggi float32/BF16, domani FP4 su Blackwell.

```
per ogni tile di nTile righe di W (default 128):
    dequantizza il tile -> buffer [nTile, K] nel *compute dtype*
    Y[:, tile] = matmulTransposeB(X, tileDequantizzato)
    (il buffer del tile viene riusato dal tile successivo)
```

Picco di memoria del dequant: `nTile × K × sizeof(compute)`, cioè
128 × 3968 × 4 B = 2 MB — **non** `N × K` (48 MB per una matrice di
esperto, 800 MB per l'embedding). Il buffer del tile è **uno solo**,
riusato: nessun `cudaMalloc` per tile grazie al pool esistente.

La firma pubblica di `TernaryLinear::forward/backward` non menziona il
tile né il dtype di calcolo: sostituire il percorso "dequant + GEMM" con
un GEMM ternario/FP4 nativo è un cambio interno, invisibile al modello.

### Gradiente rispetto a un peso ternario

Estimatore straight-through (BitNet): la quantizzazione è trattata come
identità nel backward, quindi `dW_latente = dY ᵀ @ X` come per un
lineare normale. Questo gradiente è **denso e transitorio**: nasce, viene
proiettato/consumato, e muore prima che nasca quello successivo (§6).

---

## 5. Nota critica: "nessuna master copy" e "parametri non congelati" sono in tensione

Va detto esplicitamente perché cambia il progetto.

L'addestramento ternario in letteratura (BitNet, BitNet b1.58) mantiene
pesi *latenti* in fp16 e quantizza a ogni forward. Sono **18 GB** per 9B
parametri: è esattamente la master copy che il requisito 3 vieta, e a
ragione.

Ma senza *nessuna* informazione continua, un aggiornamento tipico
(`lr · ĝ ≈ 1e-4`) contro una griglia con passo `scale ≈ 0,03` è ~300
volte più piccolo del passo: arrotondato al valore ternario più vicino
**sparisce**. Il modello smetterebbe di imparare in silenzio.

Il requisito 9 (residuo low-rank `Δ = A·B` consolidato periodicamente in
`T`) propone la risposta giusta, ma il requisito 7 da solo non basta:
**se `T` non si muove mai e si allena solo `Δ = A·B`, l'architettura è
matematicamente un adapter LoRA su una base congelata** — cioè
esattamente ciò che "pretraining a parametri pieni, senza congelare la
base" esclude.

La sintesi adottata, che soddisfa entrambi i vincoli senza master copy:

```
W = scale ⊙ T            (T ternario, canonico, persistente)
grad denso dW  --proiezione low-rank-->  stato Adam O(r·(m+n))
               --ricostruzione a tile-->  ΔW transitorio
               --arrotondamento STOCASTICO su griglia ternaria--> T cambia
```

L'arrotondamento stocastico (requisito 8) è il pezzo che rende il tutto
non banale: un aggiornamento di `1e-4` su una griglia di passo `0,03`
non viene buttato via, diventa un **flip con probabilità 0,0033**. In
attesa è esatto: `E[T_nuovo · scale] = W + ΔW`. Su 292 M pesi per layer,
"probabilità 0,0033" significa ~1 M di flip effettivi per step: `T`
impara davvero, a parametri pieni, senza che esista da nessuna parte una
copia continua completa.

Il residuo low-rank del requisito 9 resta e serve: riduce la **varianza**
di questo processo accumulando il sub-passo prima di consolidarlo. È
implementato dietro flag sperimentale, come richiesto, e non è
obbligatorio perché il percorso sopra funziona già da solo.

---

## 6. Backward in streaming (requisito 6)

`Parameter::grad` persistente è il problema: 9B float32 = 36 GB.

L'autodiff di BlackForge **non è un nastro dinamico**: `Model::backward`
è già un loop esplicito sui layer che chiama funzioni `xxxBackward`. Non
serve quindi riscrivere un motore autograd — serve cambiare *chi possiede
il gradiente* e *quanto vive*:

```cpp
// astrazione nuova, minima
struct GradientSink {
    // Consuma il gradiente di UN parametro e ne rilascia subito il buffer.
    virtual void consume(ParameterHandle&, DeviceTensor&& grad) = 0;
};
```

`BlackBitModel::backwardStreaming(sink)` percorre i layer a ritroso; per
ogni matrice calcola `dW`, chiama `sink.consume(...)` (che proietta,
aggiorna lo stato low-rank e applica il flip su `T`) e **libera il
buffer nello stesso istante**, prima di calcolare il gradiente
successivo. Il picco è `O(max |W_i|)` = 48,8 MB, non `O(Σ|W_i|)` =
36 GB.

Il percorso ordinario (`cpu::Model`/`cuda::Model` con
`Parameter::grad` persistente e `Optimizer::step`) **resta invariato**:
BlackBit aggiunge una modalità, non ne sostituisce una.

Telemetria richiesta dal requisito 17: un contatore globale di byte di
gradiente vivi contemporaneamente, con il massimo raggiunto → il
benchmark può stampare `FULL MODEL GRADIENT BUFFER: NO` **dimostrandolo
con un numero**, non dichiarandolo.

---

## 7. Ordine di implementazione

| Fase | Contenuto | Milestone |
|---|---|---|
| 1 | Piano (questo documento), `BlackBitConfig` + contabilità parametri/memoria, `TernaryTensor` + pack/unpack + serializzazione, test di round-trip esatto | A |
| 2 | `TernaryLinear` con dequant a tile, forward/backward, test su rete minuscola | B |
| 3 | `MoERouter` / `MoEExpert` / `MoELayer`, top-2 sparso, load balancing, metriche | — |
| 4 | GQA senza duplicazione K/V, blocco BlackBit, LM BlackBit-Tiny che riduce la cross-entropy | C |
| 5 | `GradientSink`, backward in streaming, telemetria del picco gradiente | D |
| 6 | Optimizer proiettato low-rank + arrotondamento stocastico | E |
| 7 | Ricalcolo attivazioni, arene, budget VRAM | — |
| 8 | Checkpoint impacchettato versionato, config tiny/small/medium/9B, CLI `benchmark blackbit` | F |
| 9 | Percorso CUDA (kernel dequant+GEMM, dispatch MoE, GQA) e prova su hardware reale | G, H |

---

## 8. Stato dei percorsi (aggiornato ad ogni fase)

| Percorso | Stato | Verificato da |
|---|---|---|
| Impacchettamento/decodifica ternaria | implementato | test unitari eseguiti (CPU) |
| Contabilità parametri/memoria BlackBit | implementato | test unitari eseguiti (CPU) |
| `TernaryLinear` (dequant a tile, forward/backward, STE) | implementato | 9 test unitari eseguiti (CPU), incluso il confronto con il gradiente numerico |
| Arrotondamento stocastico + RNG a contatore | implementato | 6 test unitari eseguiti (CPU), incluso quello di non distorsione su 200 000 campioni |
| Telemetria di memoria per arena + budget | implementato | verificata dai test di `TernaryLinear` |
| Aggiornamento diretto dei trit (`TernarySgdSink`) | implementato | rete minuscola che riduce la loss del 15 %+ con soli pesi ternari |
| MoE | da fare | — |
| GQA | da fare | — |
| Backward in streaming | da fare | — |
| Optimizer low-rank | da fare | — |
| Kernel CUDA BlackBit | da fare | **non compilabile in questo ambiente** (nessun `nvcc`) |
