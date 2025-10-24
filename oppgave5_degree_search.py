# -*- coding: utf-8 -*-
"""
Oppgave 5: Prøv ulike polynomgrader og skriv ut R² og standardavvik.
Denne fila kjører uavhengig av appen.
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

# ---- Innstillinger (kan endres ved behov) ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "data", "nedborX.csv")

DEGREES = [1, 2, 3, 4, 5, 6]   # polynomgrader som testes
N_SPLITS = 5                   # k-delt kryssvalidering

# Mulige navn for målkolonnen (nedbør)
TARGET_CANDIDATES = [
    "Nedbor", "Nedbør", "nedbor", "nedbør",
    "nedbor_mnd", "Nedbor_mnd", "mm", "Mnd_mm"
]

# Mulige navn for månedskolonnen (CSV-en din har 'Month')
MONTH_CANDIDATES = [
    "Mnd", "mnd", "Month", "month"
]

def find_columns(df: pd.DataFrame):
    """
    Finn kolonnenavn for X, Y, (Mnd/Month) og målkolonnen i CSV-en.
    Case-insensitiv og tolerant for ulike navnevarianter.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    def pick_any(candidates, kind):
        for cand in candidates:
            key = cand.lower()
            if key in cols_lower:
                return cols_lower[key]
        raise KeyError(f"Kolonne ikke funnet: {kind} ({candidates})  | Tilgjengelige kolonner: {list(df.columns)}")

    # Obligatoriske feature-kolonner
    x_col = pick_any(["X", "x"], "X")
    y_col = pick_any(["Y", "y"], "Y")
    m_col = pick_any(MONTH_CANDIDATES, "Mnd/Month")

    # Målkolonne (nedbør)
    t_col = pick_any(TARGET_CANDIDATES, "Nedbor/Nedbør")

    return x_col, y_col, m_col, t_col


def evaluate_degree(df: pd.DataFrame, degree: int):
    """
    Evaluer PolynomialFeatures + LinearRegression for gitt grad
    med KFold. Returnerer R², standardavvik (σ) og MAE.
    """
    x_col, y_col, m_col, t_col = find_columns(df)

    X = df[[x_col, y_col, m_col]].astype(float).values
    y = df[t_col].astype(float).values

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Out-of-fold prediksjoner (hver observasjon predikeres i valideringsfold)
    y_pred = cross_val_predict(model, X_poly, y, cv=kf)

    r2 = r2_score(y, y_pred)
    residuals = y - y_pred
    std_dev = float(np.std(residuals))           # σ (ddof=0, populasjonsstandardavvik)
    mae = float(mean_absolute_error(y, y_pred))  # referansemetrikk

    return {
        "degree": degree,
        "r2": r2,
        "std": std_dev,
        "mae": mae
    }


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Fant ikke CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    results = []

    print("=== Oppgave 5: Vurdering av polynomgrad (KFold = {} ) ===".format(N_SPLITS))
    for d in DEGREES:
        try:
            met = evaluate_degree(df, d)
            results.append(met)
            print(f"Grad {d}:  R² = {met['r2']:.3f}   σ (std) = {met['std']:.2f} mm   MAE = {met['mae']:.2f} mm")
        except Exception as e:
            print(f"Grad {d}: Feil -> {e}")

    if not results:
        print("Ingen vurdering kunne gjennomføres.")
        return

    # Velg beste modell: høyest R², og ved likhet lavest σ
    results_sorted = sorted(results, key=lambda m: (m["r2"], -m["std"]), reverse=True)
    best = results_sorted[0]

    print("\n--- Oppsummering (best til svakest) ---")
    for m in results_sorted:
        flag = "  <-- best" if m is best else ""
        print(f"Grad {m['degree']}: R²={m['r2']:.3f}  σ={m['std']:.2f}  MAE={m['mae']:.2f}{flag}")

    print("\n>>> Anbefaling: Sett graden til '{}' i data_model.py for å øke R² og redusere σ.".format(best["degree"]))

if __name__ == "__main__":
    main()
