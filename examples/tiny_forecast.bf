// Esempio eseguibile di forecasting autoregressivo: un layer lineare
// che mappa 4 feature su se stesse (input e output hanno la stessa
// forma), cosi' il suo output puo' diventare l'input del passo
// successivo. Serve un checkpoint pre-allenato: 'forecast' rifiuta
// esplicitamente di partire da pesi casuali.
//
// Esegui con (richiede un checkpoint gia' salvato, es. da
// 'blackforge train --save-checkpoint'):
//   blackforge forecast examples/tiny_forecast.bf --from-checkpoint pesi.bfckpt --batch 1

model DriftModel {
    input bf16[batch, 4]
    input |> linear(4)
}

forecast {
    model DriftModel
    horizon 8
}
