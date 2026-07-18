// Esempio eseguibile di classificazione: un layer lineare che impara a
// separare 2 classi da 4 feature di ingresso, con loss cross-entropy
// (softmax applicata internamente), su un dataset sintetico minimale
// (8 esempi one-hot) incluso nel repository.
//
// Esegui con:
//   blackforge train examples/tiny_classification.bf

model TinyClassifier {
    input bf16[batch, 4]
    input |> linear(2)
}

dataset ToyClasses {
    path "examples/tiny_classification_dataset.bfdata"
    input bf16[batch, 4]
    labels bf16[batch, 2]
}

train {
    model TinyClassifier
    dataset ToyClasses
    loss cross_entropy
    optimizer adamw
    epochs 100
    batch_size 8
    learning_rate 0.1
}
