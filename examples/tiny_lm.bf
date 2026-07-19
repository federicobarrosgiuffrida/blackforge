// Esempio eseguibile di modello linguistico minimale: embedding di
// token, embedding posizionale, un blocco di self-attention causale
// multi-testa e un blocco feed-forward (entrambi con residual e
// pre-norm interni), seguiti da una proiezione lineare finale sui
// logit del vocabolario.
//
// Compito: "shift-copy" su un vocabolario di 8 token e sequenze di
// lunghezza 6 -> il token in uscita alla posizione s deve essere il
// token di INGRESSO alla posizione s-1 (il primo token si copia su se
// stesso). E' un compito minimale ma non risolvibile da
// embedding/feedforward da soli, che operano posizione per posizione:
// serve attention per spostare l'informazione da una posizione della
// sequenza a un'altra, quindi vedere la loss scendere qui dimostra che
// l'intera pipeline sta davvero imparando, non solo l'ultimo layer
// lineare.
//
// Esegui con:
//   blackforge train examples/tiny_lm.bf --save-checkpoint tiny_lm.bfckpt
//   blackforge run examples/tiny_lm.bf --from-checkpoint tiny_lm.bfckpt

model TinyLM {
    input bf16[batch, 6]
    input |> embedding(8, 8) |> positional_embedding(6) |> attention(2) |> feedforward(16) |> linear(8)
}

dataset ShiftCopy {
    path "examples/tiny_lm_dataset.bfdata"
    input bf16[batch, 6]
    labels bf16[batch, 6, 8]
}

train {
    model TinyLM
    dataset ShiftCopy
    loss cross_entropy
    optimizer adamw
    epochs 300
    batch_size 16
    learning_rate 0.05
}
