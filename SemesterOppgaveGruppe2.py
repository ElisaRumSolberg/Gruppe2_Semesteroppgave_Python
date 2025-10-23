# -*- coding: utf-8 -*-
"""
SemesterOppgave – Gruppe 2 – Oppgave 1
Mål: Pen grunnvisning med bedre kart/stil.
"""

# --- Importer pakker ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patheffects as pe  # tekst-outline for lesbarhet
from matplotlib.path import Path     # til egendefinert marker
from matplotlib.markers import MarkerStyle  # stabil marker-normalisering
from matplotlib.widgets import Button       # reset-knapp

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error

# ---------- STIER / KONFIG ----------
skript_mappe = os.path.dirname(os.path.abspath(__file__))
data_sti  = os.path.join(skript_mappe, "data", "nedborX.csv")        # CSV: X,Y,Mnd,Nedbor
bilde_sti = os.path.join(skript_mappe, "assets", "Bergen_Kart2.PNG") # kartbilde

# Kart-aksegrenser (samme som lærer)
x_min, x_max = 0.0, 13.0
y_min, y_max = 0.0, 10.0
kart_extent  = (x_min, x_max, y_min, y_max)

# --- Basis for zoom-skala (lagres ved start) ---
INIT_DX = x_max - x_min
INIT_DY = y_max - y_min

def gjeldende_zoom_skala(ax):
    """Returnerer skala = nåværende utsnitts-bredde/høyde i forhold til start (maks av x,y)."""
    cx0, cx1 = ax.get_xlim()
    cy0, cy1 = ax.get_ylim()
    dx = (cx1 - cx0) / INIT_DX
    dy = (cy1 - cy0) / INIT_DY
    return max(dx, dy)  # bruker maks for å holde proporsjonene pene

# Fargeklasser – gul -> oransje -> rød (lys -> mørk)
colors = ['#FEF08A',  # lys gul
          '#FBBF24',  # gul-oransje
          '#F97316',  # oransje
          '#EF4444',  # rød
          '#991B1B']  # mørk rød

# ---------- HJELPEFUNKSJONER ----------
def index_from_nedbor(x: float) -> int:
    """Samme terskler som lærer – for kompatibilitet."""
    if x < 1300: return 0
    if x < 1700: return 1
    if x < 2500: return 2
    if x < 3200: return 3
    return 4

def color_from_nedbor(nedbor: float) -> str:
    """Velg farge etter årsnedbør (mm)."""
    return colors[index_from_nedbor(nedbor)]

def label_from_nedbor(nedbor: float) -> str:
    """Etikett i hundre mm (f.eks. 13 => 1300 mm/år)."""
    return str(int(nedbor / 100))

def size_from_nedbor_series(nedbor_array, min_size=700, max_size=1500):
    """Skalerer symbolstørrelse etter verdi (mm) – robust mot ekstreme verdier."""
    v = np.asarray(nedbor_array, dtype=float)
    if v.size == 0:
        return np.array([min_size], dtype=float)
    v_min = np.percentile(v, 5)
    v_max = np.percentile(v, 95)
    if v_max <= v_min:
        return np.full_like(v, (min_size + max_size) * 0.5)
    s = (v - v_min) / (v_max - v_min)
    s = np.clip(s, 0, 1)
    return min_size + s * (max_size - min_size)

def draw_label_and_ticks():
    """Setter x-aksens etiketter (måneder) på venstre plott."""
    xlabels = ['J','F','M','A','M','J','J','A','S','O','N','D']
    axGraph.set_xticks(np.linspace(1, 12, 12))
    axGraph.set_xticklabels(xlabels, fontsize=9)
    axGraph.set_ylabel("mm", fontsize=10)

def regndraape_marker(scale: float = 1.0) -> MarkerStyle:
    """
    Estetisk regndråpe – mykere topp og spissere bunn.
    Brukes som 'marker' i scatter.
    """
    verts = np.array([
        (0.00,  1.00),   # topp
        (0.45,  0.40),
        (0.35, -0.30),
        (0.00, -1.00),   # spiss
        (-0.35, -0.30),
        (-0.45,  0.40),
        (0.00,  1.00),   # tilbake til topp
        (0.00,  1.00)    # ekstra punkt for CLOSEPOLY
    ]) * float(scale)

    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CLOSEPOLY
    ]
    return MarkerStyle(Path(verts, codes))

# ---------- DATA / MODELL ----------
# Les data trygt, gi vennlig feilmelding hvis fil mangler
if not os.path.exists(data_sti):
    raise FileNotFoundError(
        f"Fant ikke CSV: {data_sti}\n"
        f"Legg 'nedborX.csv' i mappen: {os.path.dirname(data_sti)}\n"
        f"Forventede kolonner: X, Y, Mnd, Nedbor"
    )
df = pd.read_csv(data_sti)

# Tilpass kolonnenavn hvis nødvendig (Month -> Mnd)
if 'Mnd' not in df.columns and 'Month' in df.columns:
    df = df.rename(columns={'Month': 'Mnd'})

# Hvis 'Mnd' er tekst (Jan,Feb,...) -> tall 1..12
if 'Mnd' in df.columns and df['Mnd'].dtype == object:
    mnd_map = {
        'J':1,'F':2,'M':3,'A':4,'MA':5,'JUN':6,'JUL':7,'AU':8,'S':9,'O':10,'N':11,'D':12,
        'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'MAI':5,'JUN':6,'JUL':7,'AUG':8,
        'SEP':9,'SEPT':9,'OCT':10,'OKT':10,'NOV':11,'DEC':12,'DES':12
    }
    def _to_num(v):
        s = str(v).strip().upper()
        if s.isdigit():
            return int(s)
        return mnd_map.get(s, None)
    df['Mnd'] = df['Mnd'].map(_to_num)

# Sikkerhet: tall-typer
for col in ['X','Y','Mnd','Nedbor']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Sjekk kolonner og dropp NaN
nødvendige = {"X","Y","Mnd","Nedbor"}
mangler = nødvendige - set(df.columns)
if mangler:
    raise ValueError(f"Mangler kolonner i CSV: {sorted(mangler)}  (fant: {list(df.columns)})")
df = df.dropna(subset=['X','Y','Mnd','Nedbor'])

# Treningsdata: (X, Y, Mnd) -> Nedbor
ns = df['Nedbor']
X  = df.drop('Nedbor', axis=1)
poly = PolynomialFeatures(degree=3)  # (Oppgave 5: kan gjøres variabel)
X_poly = poly.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_poly, ns, test_size=0.25, random_state=42
)
model = LinearRegression().fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# Modellkvalitet (konsoll)
print(f"R-squared: {r2_score(Y_test, Y_pred):.2f}")
print("mean_absolute_error (mnd):", mean_absolute_error(Y_test, Y_pred))

# Aggregér årsnedbør per stasjon (for kart)
df_year = df.groupby(['X','Y']).agg({'Nedbor':'sum'}).reset_index()
xr = df_year['X'].to_numpy()
yr = df_year['Y'].to_numpy()
nedbor_aar = df_year['Nedbor'].to_numpy()   # mm/år
nedbor_mnd = nedbor_aar / 12.0              # mm/mnd (til størrelse)
ColorList  = [color_from_nedbor(v) for v in nedbor_aar]
SizeList   = size_from_nedbor_series(nedbor_mnd)

# ---------- FIGUR / AKSER ----------
plt.rcParams['figure.dpi'] = 120
fig = plt.figure(figsize=(10.6, 4.6))
axGraph = fig.add_axes((0.05, 0.10, 0.35, 0.82))
axMap   = fig.add_axes((0.42, 0.07, 0.56, 0.86))

# Last kartbilde (lokal PNG)
if not os.path.exists(bilde_sti):
    raise FileNotFoundError(
        f"Fant ikke kartbilde: {bilde_sti}\n"
        f"Legg bildet i: {os.path.dirname(bilde_sti)}\n"
        f"Filnavn må være nøyaktig: Bergen_Kart2.PNG"
    )
img = mpimg.imread(bilde_sti)

# Grunnoppsett
axMap.axis('off')
axMap.set_xlim(x_min, x_max)
axMap.set_ylim(y_min, y_max)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
axMap.set_title("Årsnedbør – Stor-Bergen", fontsize=11)
axGraph.set_title("Nedbør per måned (klikk for estimat)", fontsize=11)

# ---------- TEGNEFUNKSJONER ----------
def tegn_kart():
    """Tegner kart + stasjoner (dråpe) + etiketter med zoom-støtte."""
    axMap.cla()

    # Bakgrunn
    axMap.imshow(img, extent=kart_extent, aspect="auto", zorder=0)
    axMap.set_xlim(x_min, x_max)
    axMap.set_ylim(y_min, y_max)
    axMap.axis('off')
    axMap.set_title("Årsnedbør – Stor-Bergen", fontsize=11)

    # --- Zoom-skala: 1.0 ved start, <1 når vi zoomer inn ---
    z = gjeldende_zoom_skala(axMap)

    # Dråpestørrelse skaleres etter zoom (innzooming ⇒ mindre s)
    eff_size = (SizeList * 1.6) * z

    # --- VIKTIG: tegn én-én, slik at path_effects funker uten feil ---
    for i in range(len(xr)):
        axMap.scatter(
            [xr[i]], [yr[i]],
            c=[ColorList[i]],
            s=[eff_size[i]],
            marker=regndraape_marker(),   # damla
            edgecolor='white',
            linewidth=1.0,
            alpha=0.92,
            zorder=3,
            path_effects=[
                pe.withSimplePatchShadow(offset=(1.0, -1.0), alpha=0.35, rho=0.9),
                pe.Normal()
            ]
        )

    # Etiketter (hundre mm) – posisjon + font z'ye göre
    base_lab_fs = 9
    lab_fs = max(6, base_lab_fs * z)
    dy = 0.12 * z

    labels = [label_from_nedbor(n) for n in nedbor_aar]
    for i in range(len(xr)):
        t = axMap.text(xr[i], yr[i] - dy, labels[i],
                       color='white', fontsize=lab_fs, ha='center', va='center', zorder=5)
        t.set_path_effects([pe.withStroke(linewidth=1.6, foreground='black')])

def on_scroll(event):
    """Zoom med musehjul på kart-aksen (myk skalering)."""
    if event.inaxes != axMap:
        return
    # zoomfaktor: ut (opp) / inn (ned)
    scale = 1.2 if event.button == 'up' else 1/1.2

    # dagens grenser
    cur_xmin, cur_xmax = axMap.get_xlim()
    cur_ymin, cur_ymax = axMap.get_ylim()
    xdata, ydata = event.xdata, event.ydata

    # nytt område skaleres rundt musepekeren
    def _scale(lo, hi, c):
        w = (hi - lo) * scale
        return c - (c - lo) * scale, c + (hi - c) * scale

    new_xmin, new_xmax = _scale(cur_xmin, cur_xmax, xdata)
    new_ymin, new_ymax = _scale(cur_ymin, cur_ymax, ydata)

    # klipp til globale grenser
    new_xmin = max(x_min, new_xmin); new_xmax = min(x_max, new_xmax)
    new_ymin = max(y_min, new_ymin); new_ymax = min(y_max, new_ymax)

    # unngå å flippe akser
    if new_xmax - new_xmin > 0.1 and new_ymax - new_ymin > 0.1:
        axMap.set_xlim(new_xmin, new_xmax)
        axMap.set_ylim(new_ymin, new_ymax)
        plt.draw()

def on_click(event):
    """Klikk på kartet: prediker 12 måneder og tegn søyler + markering."""
    if event.inaxes != axMap:
        return

    x, y = float(event.xdata), float(event.ydata)
    months = np.arange(1, 13, dtype=float)

    # Bruk samme feature-navn som under trening (unngå sklearn-advarsel)
    vektorer_df = pd.DataFrame({
        'X':   np.full(12, x, dtype=float),
        'Y':   np.full(12, y, dtype=float),
        'Mnd': months
    })
    Xq = poly.transform(vektorer_df)  # nå uten advarsler
    y_pred = model.predict(Xq)        # mm/mnd
    aarsnedbor = float(np.sum(y_pred))

    # Venstre: bar-plott
    axGraph.cla()
    farger_bar = [color_from_nedbor(v * 12.0) for v in y_pred]
    axGraph.bar(months, y_pred, color=farger_bar, edgecolor='black', linewidth=0.6)
    draw_label_and_ticks()
    axGraph.set_title(f"Nedbør per måned – Årsnedbør ≈ {int(aarsnedbor)} mm", fontsize=10)
    axGraph.grid(axis='y', linestyle=':', alpha=0.35)

    # Høyre: kart + blå markør (klikkpunkt)
    tegn_kart()  # sørg for at zoom-skala brukes
    axMap.scatter([x], [y],
                  c='#1E3A8A', s=520, marker='o', zorder=6,
                  edgecolors='white', linewidths=1.2)  # mørkblå sirkel
    t = axMap.text(x, y, label_from_nedbor(aarsnedbor), color='white',
                   fontsize=10, ha='center', va='center', zorder=6)
    t.set_path_effects([pe.withStroke(linewidth=1.8, foreground='black')])
    axMap.set_title(f"C: ({x:.1f},{y:.1f}) – klikk blå = estimert", fontsize=10)

    plt.draw()

# ---------- STARTVISNING ----------
draw_label_and_ticks()
tegn_kart()

# UI-elementer utenfor kartet, men "bitişik"
map_pos = axMap.get_position()  # BBox i figur-koordinater (0..1)

# Øverst-høyre: hjelpetekst
fig.text(
    map_pos.x1, map_pos.y1 + 0.012,  # juster 0.008..0.016 ved behov
    "Klikk på kartet for estimat",
    ha="right", va="bottom",
    fontsize=9, color="#222",
    bbox=dict(boxstyle="round,pad=0.25", fc="#f3f4f6", ec="#9ca3af", lw=1, alpha=0.95),
    zorder=10
)

# Nederst-høyre: reset-knapp
btn_w, btn_h = 0.12, 0.06
btn_x = map_pos.x1 - btn_w
btn_y = map_pos.y0 - 0.065  # juster -0.065..-0.085 ved behov
axBtn = fig.add_axes([btn_x, btn_y, btn_w, btn_h])
btnReset = Button(axBtn, "Reset view", color="#f3f4f6", hovercolor="#e5e7eb")
for sp in axBtn.spines.values():
    sp.set_edgecolor("#9ca3af"); sp.set_linewidth(1)
btnReset.label.set_fontsize(10); btnReset.label.set_color("#111")

def reset_view(event=None):
    """Nullstill kart-utsnittet til start og tegn på nytt."""
    axMap.set_xlim(x_min, x_max)
    axMap.set_ylim(y_min, y_max)
    tegn_kart()
    plt.draw()

btnReset.on_clicked(reset_view)

# Koble hendelser
fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
