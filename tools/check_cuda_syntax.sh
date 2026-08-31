#!/usr/bin/env bash
# Verifica il C++ dei file .cu di BlackBit SENZA nvcc.
#
# Perche' esiste: gli ambienti di sviluppo senza GPU non hanno il toolkit
# CUDA, quindi `cmake --build` compila soltanto il ramo CPU e i .cu non
# vengono mai analizzati. Un errore di battitura in un kernel resterebbe
# invisibile fino alla prima build su una macchina con GPU.
#
# Come funziona: la sintassi di lancio `kernel<<<griglia, blocchi>>>(...)`
# non e' C++, quindi viene rimossa (il kernel diventa una chiamata di
# funzione normale, e gli argomenti restano controllati contro la firma).
# Gli header di tools/cuda_syntax_stub/ dichiarano quel tanto di CUDA che
# serve a completare l'analisi.
#
# Cosa cattura: sintassi, tipi, argomenti dei kernel, membri inesistenti,
# name lookup, accessi privati.
# Cosa NON cattura: errori specifici di nvcc (uso di funzioni host nel
# device, pressione sui registri, __launch_bounds__ non soddisfatte) e
# qualunque errore di esecuzione. NON sostituisce una build CUDA vera.

set -uo pipefail
cd "$(dirname "$0")/.."

STUB_DIR="tools/cuda_syntax_stub"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

FILES=("$@")
if [ ${#FILES[@]} -eq 0 ]; then
    # Tutto il codice CUDA del repository: i .cu, i .cpp che finiscono
    # nel target solo con CUDA, e main.cpp per il suo ramo condizionale.
    mapfile -t FILES < <(ls src/blackbit/*.cu src/backend/cuda/*.cu \
        src/blackbit/cuda_benchmark.cpp src/blackbit/cuda_checkpoint.cpp src/blackbit/cuda_train.cpp)
fi

STATUS=0
for source in "${FILES[@]}"; do
    stripped="$WORK/$(basename "$source" .cu).cpp"
    python3 - "$source" "$stripped" <<'PY'
import re, sys
source, target = sys.argv[1], sys.argv[2]
text = open(source).read()
# Rimuove la configurazione di lancio. Non-greedy: `f<<<static_cast<unsigned
# int>(n), 1>>>(...)` contiene '<' e '>' interni, ma il primo '>>>' e' la
# fine vera della configurazione.
# `f<<<griglia, blocchi>>>(args)` diventa
# `(blackforgeLaunchConfiguration(griglia, blocchi), f)(args)`: una
# espressione con virgola che vale `f` e poi la chiama. Non basta
# cancellare la configurazione, perche' le variabili che compaiono solo
# li' dentro (tipicamente il conteggio di righe passato a gridFor)
# diventerebbero falsi -Wunused-variable.
text, count = re.subn(
    r'([A-Za-z_][A-Za-z0-9_:]*)\s*<<<(.*?)>>>\s*\(',
    r'(::blackforgeLaunchConfiguration(\2), \1)(',
    text, flags=re.DOTALL)
# Mantiene la numerazione di riga allineata al .cu originale, cosi' i
# messaggi di g++ puntano alla riga giusta del sorgente vero.
open(target, 'w').write('#line 1 "%s"\n' % source + text)
print("  %s: %d lanci di kernel riscritti" % (source, count))
PY
    # EXTRA_FLAGS permette di analizzare anche i rami sotto
    # `#if BLACKFORGE_HAS_CUDA` dei file che non sono compilati solo per
    # CUDA (src/main.cpp): EXTRA_FLAGS=-DBLACKFORGE_HAS_CUDA=1
    # shellcheck disable=SC2086
    if ! g++ -std=c++20 -fsyntax-only -Wall -Wextra ${EXTRA_FLAGS:-} \
        -I include -I "$STUB_DIR" -x c++ "$stripped"; then
        echo "FALLITO: $source"
        STATUS=1
    fi
done

if [ $STATUS -eq 0 ]; then
    echo "Sintassi C++ OK per ${#FILES[@]} file (con stub CUDA, NON una build nvcc)"
fi
exit $STATUS
