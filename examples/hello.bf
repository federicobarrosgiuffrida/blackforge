// Esempio minimo di programma BlackForge.
//
// NOTA: al momento (milestone lexer) il compilatore sa solo tokenizzare
// questo file e segnalare eventuali errori lessicali. Parser, analisi
// semantica ed esecuzione arriveranno nelle prossime milestone.

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
