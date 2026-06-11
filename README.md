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
- **`esempio_regressione.py`** — **template per i TUOI dati** (regressione):
  legge un CSV e impara a prevedere un numero. Vedi sotto.
- **`crea_dati_esempio.py`** — genera `dati_esempio.csv`, un dataset finto
  (prezzi di case) che mostra il formato richiesto.

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

## Usare i TUOI dati (regressione)

`esempio_regressione.py` e' un template pronto: prevede un **numero** a
partire dalle tue caratteristiche di input.

### 1. Prepara un file CSV

Una tabella dove **ogni riga e' un esempio**, le **prime colonne sono gli
input** e l'**ultima colonna e' il numero da prevedere** (il "target"):

```csv
metri_quadri,locali,eta_anni,prezzo
141.9,4,35.6,285938.5
83.2,6,42.2,200668.9
46.6,4,33.9,121668.2
```

Regole importanti:

- **Solo numeri.** Le categorie testuali vanno convertite in numeri
  (es. citta' "Roma/Milano/Napoli" -> colonne 0/1, oppure 1/2/3).
- **Niente celle vuote.** Riempi o rimuovi i valori mancanti.
- **Piu' dati = meglio.** Poche decine di righe non bastano; punta almeno
  a qualche centinaio.
- **L'ultima colonna deve essere il target.**

### 2. Collega il tuo file

In cima a `esempio_regressione.py` cambia:

```python
FILE_CSV = "dati_esempio.csv"   # -> metti qui il nome del tuo CSV
```

### 3. Allena

```bash
python3 esempio_regressione.py
```

Lo script: divide i dati in allenamento/test, **normalizza** input e target
(passaggio quasi sempre indispensabile), allena la rete e stampa l'errore
sui dati mai visti (RMSE e MAE, nella stessa unita' del target).

### Cosa controllare / regolare

- Se la loss **non scende**: abbassa il `learning_rate` (es. 0.01).
- Se scende ma **lentamente**: alzalo (es. 0.1, 0.3) o aumenta le `epoche`.
- Se va bene sul training ma **male sul test** (overfitting): usa piu' dati,
  riduci i neuroni o le epoche.
- Numero di neuroni e di layer si cambiano dove la rete viene costruita
  (`layer1`, `layer2`, `layer3`).

### Per la classificazione

Se invece dovrai prevedere una **categoria** (non un numero), la struttura e'
quella di `esempio_cifre.py`: ultimo layer con un neurone per classe,
attivazione gestita con **softmax** e loss **cross-entropy**.
