"""
Genera un dataset di esempio (finto) per la REGRESSIONE: prezzi di case.

Lo scopo e' mostrarti il FORMATO che i tuoi dati dovranno avere:
- una riga per ogni esempio (qui: una casa),
- una colonna per ogni caratteristica (feature) di input,
- l'ULTIMA colonna e' il valore numerico da prevedere (qui: il prezzo).

Lancialo una volta sola per creare il file 'dati_esempio.csv':
    python3 crea_dati_esempio.py
"""

import csv

import numpy as np

rng = np.random.default_rng(0)
n_case = 300

# Tre caratteristiche di input
mq = rng.uniform(40, 200, n_case)              # metri quadri
locali = rng.integers(1, 7, n_case)            # numero di locali
eta = rng.uniform(0, 50, n_case)               # eta' dell'immobile in anni

# Il "vero" legame tra input e prezzo (in un caso reale non lo conosci:
# e' proprio cio' che la rete deve imparare dai dati). Aggiungiamo del rumore.
rumore = rng.normal(0, 8000, n_case)
prezzo = 20000 + 1800 * mq + 12000 * locali - 900 * eta + rumore
prezzo = np.maximum(prezzo, 10000)             # niente prezzi negativi

with open("dati_esempio.csv", "w", newline="") as f:
    writer = csv.writer(f)
    # Intestazione: le prime colonne sono input, l'ultima e' il target.
    writer.writerow(["metri_quadri", "locali", "eta_anni", "prezzo"])
    for i in range(n_case):
        writer.writerow([
            round(mq[i], 1),
            int(locali[i]),
            round(eta[i], 1),
            round(prezzo[i], 2),
        ])

print(f"Creato 'dati_esempio.csv' con {n_case} righe.")
print("Formato: metri_quadri, locali, eta_anni  ->  prezzo (ultima colonna = target)")
