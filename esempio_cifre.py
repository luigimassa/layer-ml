"""
Esempio CONCRETO: una rete neurale che riconosce cifre scritte a mano.

Usiamo il dataset "digits" di scikit-learn: 1797 immagini reali di cifre
(0-9), ciascuna di 8x8 pixel in scala di grigi. La rete guarda i 64 pixel
e deve indovinare quale numero e' scritto.

E' lo stesso DenseLayer di prima (dense_layer.py): cambia solo che qui
- l'input ha 64 valori (i pixel) invece di 2,
- l'output ha 10 valori (la probabilita' di ogni cifra da 0 a 9).

Schema della rete:
    64 pixel  ->  [layer nascosto: 32 neuroni, ReLU]  ->  [output: 10 neuroni]
                                                              |
                                                          softmax -> probabilita'

Lancialo con:  python3 esempio_cifre.py
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from dense_layer import DenseLayer


# ---------------------------------------------------------------------------
# Softmax + cross-entropy (per classificazione a piu' classi)
#
# softmax trasforma i 10 numeri grezzi dell'output in 10 PROBABILITA'
# che sommano a 1. La cross-entropy misura quanto la probabilita' assegnata
# alla cifra giusta e' lontana da 1.
# ---------------------------------------------------------------------------

def softmax(z):
    # Sottraiamo il massimo per stabilita' numerica (evita exp di numeri enormi).
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def cross_entropy(probabilita, y_onehot):
    # -log della probabilita' assegnata alla classe corretta, mediata sui campioni.
    eps = 1e-12  # evita log(0)
    return -np.mean(np.sum(y_onehot * np.log(probabilita + eps), axis=1))


def one_hot(y, n_classi):
    """Trasforma le etichette (es. 3) in vettori (es. [0,0,0,1,0,0,0,0,0,0])."""
    out = np.zeros((y.shape[0], n_classi))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def accuratezza(rete_forward, X, y):
    """Percentuale di cifre indovinate."""
    probs = rete_forward(X)
    predette = np.argmax(probs, axis=1)
    return np.mean(predette == y)


def disegna_cifra(immagine_flat):
    """Disegna una cifra 8x8 in ASCII art, cosi' la 'vediamo'."""
    livelli = " .:-=+*#%@"  # dal piu' chiaro al piu' scuro
    img = immagine_flat.reshape(8, 8)
    righe = []
    for riga in img:
        righe.append("".join(livelli[min(int(p / 16 * len(livelli)), len(livelli) - 1)]
                             for p in riga))
    return "\n".join(righe)


def main():
    # --- Carichiamo i dati ---
    digits = load_digits()
    X = digits.data / 16.0          # pixel 0..16 -> normalizziamo a 0..1
    y = digits.target              # etichette: la cifra giusta (0..9)
    n_classi = 10

    # Dividiamo in dati di allenamento e dati di test (mai visti durante il training).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_train_oh = one_hot(y_train, n_classi)

    # --- Costruiamo la rete: due DenseLayer in fila ---
    hidden = DenseLayer(n_input=64, n_neuroni=32, activation="relu", seed=1)
    output = DenseLayer(n_input=32, n_neuroni=10, activation="linear", seed=2)

    def forward(X):
        """Passa i dati attraverso la rete e ritorna le probabilita'."""
        a1 = hidden.forward(X)
        z2 = output.forward(a1)
        return softmax(z2)

    learning_rate = 0.5
    epoche = 300

    print("Alleno la rete a riconoscere cifre scritte a mano...\n")
    print(f"Immagini di allenamento: {len(X_train)}  |  di test: {len(X_test)}\n")

    for epoca in range(1, epoche + 1):
        # ---- Forward ----
        probs = forward(X_train)

        # ---- Loss ----
        loss = cross_entropy(probs, y_train_oh)

        # ---- Backward ----
        # Con softmax + cross-entropy il gradiente sull'output si semplifica
        # bellissimamente in (probabilita' - valore_atteso).
        d_output = probs - y_train_oh
        d_a1 = output.backward(d_output)
        hidden.backward(d_a1)

        # ---- Update ----
        output.update(learning_rate)
        hidden.update(learning_rate)

        if epoca % 50 == 0 or epoca == 1:
            acc = accuratezza(forward, X_train, y_train)
            print(f"Epoca {epoca:4d}  |  loss = {loss:.4f}  |  accuratezza training = {acc:.1%}")

    # --- Valutazione finale sui dati MAI visti ---
    acc_test = accuratezza(forward, X_test, y_test)
    print(f"\n>>> Accuratezza sui dati di TEST (mai visti): {acc_test:.1%}\n")

    # --- Mostriamo qualche predizione, disegnando la cifra ---
    print("Alcune predizioni su immagini di test:\n")
    probs_test = forward(X_test)
    for i in range(5):
        predetta = np.argmax(probs_test[i])
        vera = y_test[i]
        esito = "OK" if predetta == vera else "SBAGLIATO"
        print(disegna_cifra(X_test[i] * 16.0))
        print(f"  -> la rete dice: {predetta}  (vera: {vera})  [{esito}]\n")


if __name__ == "__main__":
    main()
