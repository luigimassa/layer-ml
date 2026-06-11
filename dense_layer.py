"""
Dense Layer (layer completamente connesso) implementato da zero con NumPy.

Un "dense layer" e' lo strato di base di una rete neurale: ogni neurone
e' collegato a TUTTI gli input. Il calcolo che fa un layer e':

    z = x @ W + b          # combinazione lineare degli input
    a = activation(z)      # funzione di attivazione (introduce non-linearita')

dove:
    x  -> input            shape (n_campioni, n_input)
    W  -> pesi             shape (n_input, n_neuroni)
    b  -> bias             shape (1, n_neuroni)
    a  -> output del layer shape (n_campioni, n_neuroni)

Questo file contiene tutto cio' che serve per ALLENARE il layer:
- forward pass  (calcolo dell'output)
- backward pass (backpropagation: calcolo dei gradienti)
- update dei pesi tramite discesa del gradiente (gradient descent)

E' scritto per essere LETTO e CAPITO, non per essere veloce.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Funzioni di attivazione
#
# Ogni attivazione e' una coppia: la funzione f(z) e la sua derivata f'(z).
# La derivata serve durante il backward pass per propagare il gradiente.
# ---------------------------------------------------------------------------

def relu(z):
    """ReLU: tiene i valori positivi, azzera quelli negativi."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivata della ReLU: 1 dove z > 0, altrimenti 0."""
    return (z > 0).astype(float)


def sigmoid(z):
    """Sigmoide: schiaccia qualsiasi numero nell'intervallo (0, 1)."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """Derivata della sigmoide: s(z) * (1 - s(z))."""
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    """Tangente iperbolica: schiaccia i valori nell'intervallo (-1, 1)."""
    return np.tanh(z)


def tanh_derivative(z):
    """Derivata della tanh: 1 - tanh(z)^2."""
    return 1.0 - np.tanh(z) ** 2


def linear(z):
    """Attivazione lineare (identita'): non cambia nulla. Utile in output."""
    return z


def linear_derivative(z):
    """Derivata dell'identita': sempre 1."""
    return np.ones_like(z)


# Dizionario che associa il NOME dell'attivazione alla coppia (funzione, derivata).
ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "linear": (linear, linear_derivative),
}


# ---------------------------------------------------------------------------
# Il Dense Layer
# ---------------------------------------------------------------------------

class DenseLayer:
    """Uno strato completamente connesso di neuroni.

    Parametri:
        n_input    : quanti valori entrano in ogni neurone
        n_neuroni  : quanti neuroni ha questo layer (= dimensione dell'output)
        activation : nome dell'attivazione ("relu", "sigmoid", "tanh", "linear")
        seed       : per rendere riproducibile l'inizializzazione casuale
    """

    def __init__(self, n_input, n_neuroni, activation="relu", seed=None):
        self.n_input = n_input
        self.n_neuroni = n_neuroni

        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Attivazione '{activation}' non valida. "
                f"Scegli tra: {list(ACTIVATIONS.keys())}"
            )
        self.activation_name = activation
        self.activation, self.activation_derivative = ACTIVATIONS[activation]

        rng = np.random.default_rng(seed)

        # Inizializzazione dei pesi (He per ReLU, Xavier per le altre).
        # Una buona inizializzazione evita che i segnali esplodano o si annullino.
        if activation == "relu":
            scala = np.sqrt(2.0 / n_input)          # He initialization
        else:
            scala = np.sqrt(1.0 / n_input)          # Xavier initialization
        self.W = rng.standard_normal((n_input, n_neuroni)) * scala

        # I bias partono a zero: e' una scelta standard e sicura.
        self.b = np.zeros((1, n_neuroni))

        # "Cache": valori salvati durante il forward, riusati nel backward.
        self.x = None   # input ricevuto
        self.z = None   # combinazione lineare prima dell'attivazione

        # Gradienti calcolati dal backward pass (usati per aggiornare i pesi).
        self.dW = None
        self.db = None

    # --- Forward pass --------------------------------------------------------
    def forward(self, x):
        """Calcola l'output del layer a partire dall'input x.

        x ha shape (n_campioni, n_input).
        Ritorna l'output con shape (n_campioni, n_neuroni).
        """
        self.x = x                       # salviamo l'input per il backward
        self.z = x @ self.W + self.b     # combinazione lineare
        return self.activation(self.z)   # applichiamo la non-linearita'

    # --- Backward pass -------------------------------------------------------
    def backward(self, d_output):
        """Backpropagation attraverso il layer.

        Riceve d_output = derivata della loss rispetto all'OUTPUT di questo
        layer (shape uguale all'output). Calcola e salva i gradienti dei pesi
        (dW, db) e ritorna la derivata della loss rispetto all'INPUT, cosi'
        il gradiente puo' continuare a propagarsi verso i layer precedenti.
        """
        n_campioni = self.x.shape[0]

        # 1) Attraversiamo l'attivazione: regola della catena.
        dz = d_output * self.activation_derivative(self.z)

        # 2) Gradienti dei parametri (media sui campioni del batch).
        self.dW = (self.x.T @ dz) / n_campioni
        self.db = np.sum(dz, axis=0, keepdims=True) / n_campioni

        # 3) Gradiente da passare al layer precedente.
        d_input = dz @ self.W.T
        return d_input

    # --- Aggiornamento dei pesi ---------------------------------------------
    def update(self, learning_rate):
        """Discesa del gradiente: sposta i pesi in direzione opposta al gradiente."""
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def __repr__(self):
        return (
            f"DenseLayer(n_input={self.n_input}, n_neuroni={self.n_neuroni}, "
            f"activation='{self.activation_name}')"
        )


if __name__ == "__main__":
    # Mini-dimostrazione del solo forward pass.
    print("Esempio rapido di forward pass:\n")
    layer = DenseLayer(n_input=3, n_neuroni=4, activation="relu", seed=0)
    print(layer)
    x = np.array([[1.0, 2.0, 3.0]])   # un campione con 3 valori
    out = layer.forward(x)
    print("\nInput  :", x)
    print("Output :", out)
    print("Shape output:", out.shape, "-> 1 campione, 4 neuroni")
