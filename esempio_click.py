"""
Previsione dei CLICK FUTURI a partire da DATA + PAROLA CHIAVE.

Caso d'uso: hai uno storico (stile Search Console) con anno, mese, giorno,
parola chiave, impressioni, click. Vuoi prevedere i CLICK di una parola
chiave in una data futura. NON usiamo le impressioni come input, perche'
in futuro non le conosci ancora: usiamo solo la data e la parola chiave.

Due idee importanti (tutte commentate sotto):

1) FEATURE DALLA DATA
   anno/mese/giorno "grezzi" dicono poco a una rete. Estraiamo invece:
   - il giorno della settimana (i click hanno un ciclo settimanale),
   - la stagionalita' mensile (rappresentata con seno/coseno, cosi'
     dicembre e gennaio risultano "vicini" come in un cerchio),
   - un indice di trend (giorni trascorsi dall'inizio dello storico).

2) TARGET ENCODING della parola chiave
   Con migliaia di parole chiave non possiamo creare una colonna per
   ciascuna (one-hot). Le rappresentiamo invece con la loro MEDIA storica
   di click (calcolata SOLO sui dati di allenamento, per non "barare").
   Aggiungiamo anche quante volte appare ogni parola (frequenza).

Uso:
    python3 crea_dati_keyword.py   # genera lo storico di esempio
    python3 esempio_click.py
Per i tuoi dati: cambia FILE_CSV e assicurati che le colonne si chiamino
anno, mese, giorno, parola_chiave, impressioni, click (impressioni puo'
restare, semplicemente non viene usata come input).
"""

import csv
import datetime

import numpy as np

from dense_layer import DenseLayer

FILE_CSV = "dati_keyword.csv"


def carica(percorso):
    """Legge il CSV e ritorna liste per ogni colonna che ci serve."""
    anni, mesi, giorni, keyword, click = [], [], [], [], []
    with open(percorso, newline="") as f:
        for r in csv.DictReader(f):
            anni.append(int(r["anno"]))
            mesi.append(int(r["mese"]))
            giorni.append(int(r["giorno"]))
            keyword.append(r["parola_chiave"])
            click.append(float(r["click"]))
    return anni, mesi, giorni, keyword, np.array(click).reshape(-1, 1)


def feature_data(anno, mese, giorno, ordinale_inizio):
    """Trasforma una data in numeri utili alla rete (feature engineering)."""
    d = datetime.date(anno, mese, giorno)
    trend = d.toordinal() - ordinale_inizio          # giorni dall'inizio
    mese_sin = np.sin(2 * np.pi * mese / 12)
    mese_cos = np.cos(2 * np.pi * mese / 12)
    dow = d.weekday()                                 # 0=lun ... 6=dom
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    return [trend, mese_sin, mese_cos, dow_sin, dow_cos]


def main():
    anni, mesi, giorni, keyword, y = carica(FILE_CSV)
    n = len(y)
    print(f"Caricate {n} righe da '{FILE_CSV}'")
    print(f"Parole chiave diverse: {len(set(keyword))}\n")

    ordinale_inizio = min(datetime.date(a, m, g).toordinal()
                          for a, m, g in zip(anni, mesi, giorni))

    # --- Divisione allenamento / test (80% / 20%) ---
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    taglio = int(n * 0.8)
    idx_train, idx_test = idx[:taglio], idx[taglio:]

    # --- TARGET ENCODING della parola chiave (solo sui dati di train) ---
    # media_kw[kw] = media dei click di quella parola chiave nel training.
    somma, conteggio = {}, {}
    for i in idx_train:
        kw = keyword[i]
        somma[kw] = somma.get(kw, 0.0) + y[i, 0]
        conteggio[kw] = conteggio.get(kw, 0) + 1
    media_globale = float(np.mean([y[i, 0] for i in idx_train]))
    media_kw = {kw: somma[kw] / conteggio[kw] for kw in somma}

    def enc_target(kw):
        # Parole mai viste in training -> media globale (fallback ragionevole).
        return media_kw.get(kw, media_globale)

    def enc_freq(kw):
        return conteggio.get(kw, 0)

    # --- Costruiamo la matrice delle feature per ogni riga ---
    def costruisci_X(indici):
        righe = []
        for i in indici:
            f = feature_data(anni[i], mesi[i], giorni[i], ordinale_inizio)
            f.append(enc_target(keyword[i]))   # media storica click della keyword
            f.append(enc_freq(keyword[i]))     # quanto e' frequente la keyword
            righe.append(f)
        return np.array(righe, dtype=float)

    X_train = costruisci_X(idx_train)
    X_test = costruisci_X(idx_test)
    y_train = y[idx_train]
    y_test = y[idx_test]

    # --- Normalizzazione (media/std calcolate sul training) ---
    x_media, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_media, y_std = y_train.mean(), y_train.std()

    Xtr = (X_train - x_media) / x_std
    Xte = (X_test - x_media) / x_std
    ytr = (y_train - y_media) / y_std

    # --- Rete: feature -> 32 -> 16 -> 1 ---
    n_feat = X_train.shape[1]
    layer1 = DenseLayer(n_feat, 32, activation="relu", seed=1)
    layer2 = DenseLayer(32, 16, activation="relu", seed=2)
    layer3 = DenseLayer(16, 1, activation="linear", seed=3)

    def forward(X):
        a = layer1.forward(X)
        a = layer2.forward(a)
        return layer3.forward(a)

    learning_rate = 0.5
    epoche = 8000
    print("Alleno la rete a prevedere i click...\n")
    for epoca in range(1, epoche + 1):
        pred = forward(Xtr)
        loss = np.mean((pred - ytr) ** 2)
        d = 2.0 * (pred - ytr) / len(ytr)
        d = layer3.backward(d)
        d = layer2.backward(d)
        layer1.backward(d)
        for layer in (layer1, layer2, layer3):
            layer.update(learning_rate)
        if epoca % 800 == 0 or epoca == 1:
            print(f"Epoca {epoca:4d}  |  loss (normalizzata) = {loss:.4f}")

    # --- Valutazione nella scala REALE (numero di click) ---
    def predici_norm(X):
        return forward((X - x_media) / x_std) * y_std + y_media

    pred_test = np.maximum(predici_norm(X_test), 0)  # i click non sono negativi
    errori = pred_test - y_test
    rmse = np.sqrt(np.mean(errori ** 2))
    mae = np.mean(np.abs(errori))
    print(f"\n>>> Errore sui dati di TEST (mai visti):")
    print(f"    RMSE = {rmse:,.1f} click   |   MAE = {mae:,.1f} click\n")

    # --- Funzione pronta per prevedere un CASO NUOVO ---
    def previsione(anno, mese, giorno, parola_chiave):
        f = feature_data(anno, mese, giorno, ordinale_inizio)
        f.append(enc_target(parola_chiave))
        f.append(enc_freq(parola_chiave))
        X = np.array([f], dtype=float)
        return float(max(predici_norm(X)[0, 0], 0))

    # --- Demo: prevediamo i click di alcune keyword in una data futura ---
    print("Esempio di previsione per il 15/06/2025:")
    for kw in ["scarpe sportivi roma", "monitor gaming milano", "lampada pelle napoli"]:
        c = previsione(2025, 6, 15, kw)
        print(f"  '{kw}'  ->  ~{c:.0f} click previsti")


if __name__ == "__main__":
    main()
