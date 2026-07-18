// Esempio eseguibile di addestramento: un layer lineare che impara una
// combinazione nota di 4 feature di ingresso in 2 valori di uscita, su
// un dataset sintetico minimale (8 esempi) incluso nel repository.
//
// Esegui con:
//   blackforge train examples/tiny_regression.bf
// Fine-tuning da un checkpoint salvato:
//   blackforge train examples/tiny_regression.bf --from-checkpoint pesi.bfckpt

model TinyRegression {
    input bf16[batch, 4]
    input |> linear(2)
}

dataset ToyData {
    path "examples/tiny_regression_dataset.bfdata"
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
