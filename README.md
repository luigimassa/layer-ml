# layer-ml — Dense Layer di neuroni da zero

Implementazione **didattica** di un *dense layer* (strato completamente
connesso) di una rete neurale, scritta da zero con NumPy. L'obiettivo non e'
la velocita' ma **capire come funziona** ogni pezzo, facendo prove.

## Cos'e' un dense layer

Ogni neurone dello strato e' collegato a **tutti** gli input. Il layer calcola:

```
z = x @ W + b          # combinazione lineare degli input
a = activation(z)       # funzione di attivazione (non-linearita')
```

| Simbolo | Significato        | Shape                       |
|---------|--------------------|-----------------------------|
| `x`     | input              | `(n_campioni, n_input)`     |
| `W`     | pesi               | `(n_input, n_neuroni)`      |
| `b`     | bias               | `(1, n_neuroni)`            |
| `a`     | output del layer   | `(n_campioni, n_neuroni)`   |

## File del progetto

- **`dense_layer.py`** — la classe `DenseLayer` con `forward`, `backward`
  (backpropagation) e `update` (discesa del gradiente), piu' le funzioni di
  attivazione (ReLU, sigmoid, tanh, lineare). Tutto commentato in italiano.
- **`esempio.py`** — una piccola rete a due layer che impara il problema
  **XOR** (esempio "giocattolo" per capire il meccanismo).
- **`esempio_cifre.py`** — esempio **concreto**: una rete che riconosce
  **cifre scritte a mano** (immagini reali 8x8). Raggiunge ~96% di
  accuratezza su immagini mai viste e disegna le cifre in ASCII art.

## Come si usa

```bash
pip install -r requirements.txt

python3 dense_layer.py     # mini-demo del solo forward pass
python3 esempio.py         # allena una rete sullo XOR (esempio giocattolo)
python3 esempio_cifre.py   # CONCRETO: riconosce cifre scritte a mano
```

### Creare e usare un layer

```python
import numpy as np
from dense_layer import DenseLayer

layer = DenseLayer(n_input=3, n_neuroni=4, activation="relu", seed=0)

x = np.array([[1.0, 2.0, 3.0]])   # 1 campione, 3 valori
out = layer.forward(x)            # -> shape (1, 4)
```

### Allenare (lo schema e' sempre questo)

```python
for epoca in range(n_epoche):
    pred = layer.forward(x)            # 1. forward
    # ... calcola la loss e la sua derivata d_loss ...
    layer.backward(d_loss)             # 2. backward (calcola i gradienti)
    layer.update(learning_rate)        # 3. update (aggiorna i pesi)
```

## Cose da provare per imparare

In `esempio.py` cambia un parametro alla volta e osserva l'effetto:

- **`learning_rate`** (0.1, 1.0, 5.0): troppo basso = impara lentamente,
  troppo alto = la loss "rimbalza" o diverge.
- **numero di neuroni** del layer nascosto (2, 8, 16).
- **attivazione** del layer nascosto (`"relu"` vs `"tanh"`).
- **numero di epoche**.

Lo XOR e' un esempio classico: **non** e' risolvibile con un solo neurone,
serve almeno un layer nascosto. E' il modo piu' semplice per vedere "imparare"
una rete neurale.
