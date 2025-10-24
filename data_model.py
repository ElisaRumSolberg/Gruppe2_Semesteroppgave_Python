# data_model.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

def _normalize_month(df):
    """Ensure column 'Mnd' is numeric 1..12 (accepts many string variants)."""
    if 'Mnd' not in df.columns and 'Month' in df.columns:
        df = df.rename(columns={'Month': 'Mnd'})

    if 'Mnd' in df.columns and df['Mnd'].dtype == object:
        m = {
            'J':1,'F':2,'M':3,'A':4,'MA':5,'JUN':6,'JUL':7,'AU':8,'S':9,'O':10,'N':11,'D':12,
            'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'MAI':5,'JUN':6,'JUL':7,'AUG':8,
            'SEP':9,'SEPT':9,'OCT':10,'OKT':10,'NOV':11,'DEC':12,'DES':12
        }
        def _to_num(v):
            s = str(v).strip().upper()
            if s.isdigit(): return int(s)
            return m.get(s, None)
        df['Mnd'] = df['Mnd'].map(_to_num)
    return df

def load_and_train(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _normalize_month(df)

    # Ensure numeric types
    for c in ['X','Y','Mnd','Nedbor']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    needed = {"X","Y","Mnd","Nedbor"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)} (found: {list(df.columns)})")

    df = df.dropna(subset=['X','Y','Mnd','Nedbor'])

    # Train model: (X,Y,Mnd) -> Nedbor
    ns = df['Nedbor']
    X = df.drop('Nedbor', axis=1)
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, ns, test_size=0.25, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"R-squared: {r2_score(y_test, y_pred):.2f}")
    print("mean_absolute_error (mnd):", mean_absolute_error(y_test, y_pred))

    # Yearly aggregate per station for the map
    df_year = df.groupby(['X','Y']).agg({'Nedbor':'sum'}).reset_index()
    xr = df_year['X'].to_numpy()
    yr = df_year['Y'].to_numpy()
    nedbor_aar = df_year['Nedbor'].to_numpy()   # mm/year
    nedbor_mnd = nedbor_aar / 12.0              # for symbol size

    return {
        "df": df,
        "poly": poly,
        "model": model,
        "xr": xr, "yr": yr,
        "nedbor_aar": nedbor_aar,
        "nedbor_mnd": nedbor_mnd
    }
