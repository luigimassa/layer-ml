"""
Esempio completo: una piccola rete neurale che impara il problema XOR.

XOR e' il classico esempio "non risolvibile da un solo neurone": serve
almeno un layer nascosto. Qui mettiamo due DenseLayer in fila e li
alleniamo. Dovresti vedere la loss SCENDERE epoca dopo epoca: e' il
segno che i neuroni stanno "imparando".

Prova a cambiare:
- learning_rate (es. 0.1, 1.0, 5.0)
- n_neuroni del layer nascosto (es. 2, 8, 16)
- attivazione del layer nascosto ("relu", "tanh")
- numero di epoche
...e osserva come cambia il risultato. E' il modo migliore per capire.
"""

import numpy as np

from dense_layer import DenseLayer


# --- Dati: la tavola di verita' dello XOR ----------------------------------
# Input: due bit. Output atteso: 1 se i bit sono diversi, 0 se uguali.
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])


def mse_loss(predizione, target):
    """Errore quadratico medio: quanto sbagliamo, in media."""
    return np.mean((predizione - target) ** 2)


def mse_loss_derivative(predizione, target):
    """Derivata della MSE rispetto alla predizione."""
    return 2.0 * (predizione - target) / target.shape[0]


def main():
    # Due layer: nascosto (2 input -> 4 neuroni) e output (4 -> 1 neurone).
    hidden = DenseLayer(n_input=2, n_neuroni=4, activation="tanh", seed=42)
    output = DenseLayer(n_input=4, n_neuroni=1, activation="sigmoid", seed=42)

    learning_rate = 1.0
    epoche = 5000

    print("Alleno la rete sul problema XOR...\n")
    for epoca in range(1, epoche + 1):
        # ---- Forward: dall'input alla predizione ----
        a1 = hidden.forward(X)
        pred = output.forward(a1)

        # ---- Calcolo della loss ----
        loss = mse_loss(pred, y)

        # ---- Backward: propaghiamo il gradiente all'indietro ----
        d_loss = mse_loss_derivative(pred, y)
        d_a1 = output.backward(d_loss)
        hidden.backward(d_a1)

        # ---- Update dei pesi ----
        output.update(learning_rate)
        hidden.update(learning_rate)

        if epoca % 500 == 0 or epoca == 1:
            print(f"Epoca {epoca:5d}  |  loss = {loss:.6f}")

    # --- Risultato finale ---
    a1 = hidden.forward(X)
    pred = output.forward(a1)

    print("\nRisultato dopo l'allenamento:")
    print("Input        Atteso   Predetto   Arrotondato")
    for i in range(len(X)):
        print(
            f"{X[i]}    {y[i, 0]:.0f}        "
            f"{pred[i, 0]:.4f}      {round(pred[i, 0])}"
        )


if __name__ == "__main__":
    main()
