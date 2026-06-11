"""
TEMPLATE di regressione: impara a prevedere un NUMERO dai TUOI dati.

Come usarlo con i tuoi dati:
  1. Prepara un file CSV dove:
       - ogni RIGA e' un esempio,
       - le prime COLONNE sono le caratteristiche di input (solo numeri),
       - l'ULTIMA colonna e' il numero che vuoi prevedere (il "target").
  2. Cambia la variabile FILE_CSV qui sotto col nome del tuo file.
  3. Lancia:  python3 esempio_regressione.py

Di default usa 'dati_esempio.csv' (prezzi di case finti). Crealo prima con:
    python3 crea_dati_esempio.py

Concetti importanti per la regressione (commentati nel codice):
  - NORMALIZZAZIONE: portiamo input e target su una scala simile, altrimenti
    la rete fatica enormemente. E' quasi sempre indispensabile.
  - L'ultimo layer usa attivazione 'linear' (l'output e' un numero libero).
  - La loss e' l'errore quadratico medio (MSE).
"""

import numpy as np
from sklearn.model_selection import train_test_split

from dense_layer import DenseLayer

# >>> CAMBIA QUI con il nome del tuo file CSV <<<
FILE_CSV = "dati_esempio.csv"


def carica_csv(percorso):
    """Legge il CSV: ritorna X (input), y (target) e i nomi delle colonne."""
    dati = np.genfromtxt(percorso, delimiter=",", names=True)
    nomi = list(dati.dtype.names)
    # Tutte le colonne tranne l'ultima = input; l'ultima = target.
    colonne_input = nomi[:-1]
    colonna_target = nomi[-1]
    X = np.column_stack([dati[c] for c in colonne_input])
    y = dati[colonna_target].reshape(-1, 1)
    return X, y, colonne_input, colonna_target


def main():
    X, y, nomi_input, nome_target = carica_csv(FILE_CSV)
    print(f"Caricati {len(X)} esempi da '{FILE_CSV}'")
    print(f"Input  : {nomi_input}")
    print(f"Target : {nome_target}\n")

    # --- Divisione in allenamento e test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- NORMALIZZAZIONE ---
    # Calcoliamo media e deviazione standard SOLO sui dati di allenamento,
    # poi usiamo gli stessi valori sul test (regola fondamentale: il test
    # non deve "influenzare" la preparazione).
    x_media, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    y_media, y_std = y_train.mean(), y_train.std()
    x_std[x_std == 0] = 1.0  # evita divisione per zero su colonne costanti

    def norm_x(X):
        return (X - x_media) / x_std

    def norm_y(y):
        return (y - y_media) / y_std

    def denorm_y(y):
        """Riporta le predizioni nella scala reale (es. euro)."""
        return y * y_std + y_media

    Xtr, ytr = norm_x(X_train), norm_y(y_train)
    Xte = norm_x(X_test)

    # --- La rete: input -> 32 neuroni -> 16 neuroni -> 1 numero ---
    n_feature = X.shape[1]
    layer1 = DenseLayer(n_feature, 32, activation="relu", seed=1)
    layer2 = DenseLayer(32, 16, activation="relu", seed=2)
    layer3 = DenseLayer(16, 1, activation="linear", seed=3)  # output: un numero

    def forward(X):
        a = layer1.forward(X)
        a = layer2.forward(a)
        return layer3.forward(a)

    learning_rate = 0.1
    epoche = 3000

    print("Alleno la rete...\n")
    for epoca in range(1, epoche + 1):
        # Forward
        pred = forward(Xtr)
        # Loss (MSE sui valori normalizzati)
        loss = np.mean((pred - ytr) ** 2)
        # Backward: derivata della MSE rispetto alla predizione
        d = 2.0 * (pred - ytr) / len(ytr)
        d = layer3.backward(d)
        d = layer2.backward(d)
        layer1.backward(d)
        # Update
        for layer in (layer1, layer2, layer3):
            layer.update(learning_rate)

        if epoca % 200 == 0 or epoca == 1:
            print(f"Epoca {epoca:4d}  |  loss (normalizzata) = {loss:.4f}")

    # --- Valutazione nella scala REALE ---
    pred_test = denorm_y(forward(Xte))
    errori = pred_test - y_test
    rmse = np.sqrt(np.mean(errori ** 2))   # errore tipico, nella scala reale
    mae = np.mean(np.abs(errori))          # errore medio assoluto

    print(f"\n>>> Errore sui dati di TEST (mai visti):")
    print(f"    RMSE = {rmse:,.0f}  (errore tipico, stessa unita' del target)")
    print(f"    MAE  = {mae:,.0f}  (errore medio assoluto)\n")

    print("Alcune predizioni sul test:")
    print(f"{'previsto':>14}   {'reale':>14}   {'errore':>12}")
    for i in range(min(8, len(y_test))):
        print(f"{pred_test[i,0]:>14,.0f}   {y_test[i,0]:>14,.0f}   {errori[i,0]:>12,.0f}")


if __name__ == "__main__":
    main()
