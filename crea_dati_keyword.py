"""
Genera un dataset di esempio (finto) in stile Search Console / Google Ads.

Ogni riga = una parola chiave in un certo giorno, con impressioni e click.
Colonne:  anno, mese, giorno, parola_chiave, impressioni, click

I dati hanno una struttura realistica che la rete dovra' imparare:
  - ogni parola chiave ha una sua "popolarita'" e un suo CTR;
  - c'e' un ciclo SETTIMANALE (meno traffico nel weekend);
  - c'e' una STAGIONALITA' annuale (alcuni mesi rendono di piu');
  - tutto con un po' di rumore casuale.

Lancialo una volta:  python3 crea_dati_keyword.py
"""

import csv
import datetime

import numpy as np

rng = np.random.default_rng(0)

# --- Costruiamo CENTINAIA di parole chiave combinando parole ---
prodotti = ["scarpe", "giacca", "zaino", "orologio", "occhiali",
            "cuffie", "tastiera", "monitor", "sedia", "lampada"]
attributi = ["uomo", "donna", "economici", "offerta", "online",
             "sportivi", "eleganti", "pelle", "bluetooth", "gaming"]
citta = ["roma", "milano", "napoli", "torino", "bologna"]

parole_chiave = [f"{p} {a} {c}" for p in prodotti for a in attributi for c in citta]
# -> 10 x 10 x 5 = 500 parole chiave diverse

# Ogni parola chiave ha una popolarita' (scala impressioni) e un CTR di base.
pop = {kw: rng.uniform(50, 3000) for kw in parole_chiave}
ctr = {kw: rng.uniform(0.01, 0.15) for kw in parole_chiave}

# --- Periodo: tutto il 2024 ---
inizio = datetime.date(2024, 1, 1)
giorni = [inizio + datetime.timedelta(days=i) for i in range(365)]

righe = []
for g in giorni:
    # Fattore settimanale: weekend piu' fiacco (sabato=5, domenica=6).
    settimana = 0.7 if g.weekday() >= 5 else 1.0
    # Stagionalita': onda annuale (picco verso meta' anno).
    stagione = 1.0 + 0.3 * np.sin(2 * np.pi * (g.timetuple().tm_yday / 365.0))

    # Ogni giorno appaiono ~40 parole chiave (estratte a caso).
    for kw in rng.choice(parole_chiave, size=40, replace=False):
        rumore_imp = rng.uniform(0.8, 1.2)
        impressioni = pop[kw] * settimana * stagione * rumore_imp
        rumore_ctr = rng.uniform(0.8, 1.2)
        click = impressioni * ctr[kw] * rumore_ctr
        righe.append([
            g.year, g.month, g.day, kw,
            int(round(impressioni)), int(round(click)),
        ])

with open("dati_keyword.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["anno", "mese", "giorno", "parola_chiave", "impressioni", "click"])
    writer.writerows(righe)

print(f"Creato 'dati_keyword.csv' con {len(righe)} righe e "
      f"{len(parole_chiave)} parole chiave diverse.")
