
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

from data_model import load_and_train
from views_map import load_map_image, tegn_kart
from views_legend import render_vertical_legend
from viz_utils import (
    KART_EXTENT, X_MIN, X_MAX, Y_MIN, Y_MAX,
    color_from_nedbor, label_from_nedbor, TEXT_STROKE
)

# ---- Paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "data", "nedborX.csv")
IMG_PATH = os.path.join(SCRIPT_DIR, "assets", "Bergen_Kart2.PNG")

# ---- Data & model ----
store = load_and_train(CSV_PATH)
poly = store["poly"]
model = store["model"]
xr, yr = store["xr"], store["yr"]
nedbor_aar, nedbor_mnd = store["nedbor_aar"], store["nedbor_mnd"]

# ---- Figure layout: [Legend] | [Bar] | [Map] ----

plt.rcParams['figure.dpi'] = 120
fig = plt.figure(figsize=(12.2, 4.8))

# width_ratios: Legend smal (0,12) + tynn plass (0,02) + lite diagram (0,33) + stort kart (0,53)
gs = gridspec.GridSpec(
    nrows=1, ncols=4,
    width_ratios=[0.12, 0.03, 0.31, 0.54],
    wspace=0.1
)

fig.subplots_adjust(left=0.01,
                    right=0.985,
                    top=0.92,
                    bottom=0.10)

axLegend = fig.add_subplot(gs[0, 0])    # vertikal forklaringskolonne til venstre
axSpacer = fig.add_subplot(gs[0, 1])    # ekte mellomrom mellom (aksel lukket)
axGraph  = fig.add_subplot(gs[0, 2])     # mindre diagram i midten
axMap    = fig.add_subplot(gs[0, 3])    # større kart til høyre
axSpacer.set_xticks([]); axSpacer.set_yticks([]); axSpacer.axis('off')
axMap.axis('off')
axMap.set_xlim(X_MIN, X_MAX);
axMap.set_ylim(Y_MIN, Y_MAX)

# ---- Load map image ----
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"Map image not found: {IMG_PATH}")
img = load_map_image(IMG_PATH)


# ---- Helpers ----
def draw_label_and_ticks():
    xlabels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    axGraph.set_xticks(np.arange(1, 13))
    axGraph.set_xticks(np.linspace(1, 12, 12))
    axGraph.set_xticklabels(xlabels, fontsize=9)
    axGraph.set_xlim(0.5, 12.5)
    axGraph.set_ylabel(None)
    axGraph.yaxis.set_label_coords(-0.03, 0.5)

    ymax = axGraph.get_ylim()[1]
    axGraph.set_ylim(0, np.ceil(ymax / 20) * 20 + 10)
    axGraph.tick_params(axis='x', pad=2)
    axGraph.tick_params(axis='y', pad=2)
    axGraph.grid(axis='y',
                 linestyle=':',
                 alpha=0.35)

    # >>>  Plasser «mm»-merkelappen inni diagrammet, øverst til venstre
    axGraph.text(
        0.015, 0.985, "mm",
        transform=axGraph.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.15",
                  fc="white",
                  ec="none",
                  alpha=0.8)
        )


def draw_initial_views():
    render_vertical_legend(axLegend)
    tegn_kart(axMap, img, xr, yr, nedbor_aar, nedbor_mnd)
    axGraph.set_title("Nedbør per måned ",
                      fontsize=9)

    draw_label_and_ticks()


# ---- Events ----
def on_scroll(event):
    if event.inaxes != axMap:
        return
    scale = 1.2 if event.button == 'up' else 1 / 1.2

    cur_xmin, cur_xmax = axMap.get_xlim()
    cur_ymin, cur_ymax = axMap.get_ylim()
    xdata, ydata = event.xdata, event.ydata

    def _scale(lo, hi, c):
        w = (hi - lo) * scale
        return c - (c - lo) * scale, c + (hi - c) * scale

    new_xmin, new_xmax = _scale(cur_xmin, cur_xmax, xdata)
    new_ymin, new_ymax = _scale(cur_ymin, cur_ymax, ydata)

    new_xmin = max(X_MIN, new_xmin);
    new_xmax = min(X_MAX, new_xmax)
    new_ymin = max(Y_MIN, new_ymin);
    new_ymax = min(Y_MAX, new_ymax)

    if new_xmax - new_xmin > 0.1 and new_ymax - new_ymin > 0.1:
        axMap.set_xlim(new_xmin, new_xmax)
        axMap.set_ylim(new_ymin, new_ymax)
        plt.draw()


def on_click(event):
    if event.inaxes != axMap:
        return
    x, y = float(event.xdata), float(event.ydata)
    months = np.arange(1, 13, dtype=float)

    df_query = {
        'X': np.full(12, x, dtype=float),
        'Y': np.full(12, y, dtype=float),
        'Mnd': months
    }

    Xq = poly.transform(pd.DataFrame(df_query))
    y_pred = model.predict(Xq)
    aarsnedbor = float(np.sum(y_pred))

    # Left-bar chart
    axGraph.cla()
    bar_colors = [color_from_nedbor(v * 12.0) for v in y_pred]
    axGraph.bar(months, y_pred,
                color=bar_colors,
                edgecolor='black',
                linewidth=0.6)

    draw_label_and_ticks()

    axGraph.set_title(f"Nedbør per måned – Årsnedbør ≈ {int(aarsnedbor)} mm",
                      fontsize=9)

    axGraph.grid(axis='y',
                 linestyle=':',
                 color='#bfbfbf',
                 alpha=0.35)

    # -----------------------------------------------------------------------------
    # --- OPPGAVE 3:Rød linje + rutenett for årlig gjennomsnitt (månedlig) ---
    #-----------------------------------------------------------------------------
    mnd_gjsnitt = float(np.mean(y_pred))  # = aarsnedbor / 12

    # Rød linje

    axGraph.axhline(mnd_gjsnitt,
                    color='#e60000',
                    linewidth=1.8,
                    linestyle='--',
                    path_effects=[pe.withStroke(linewidth=2.5,
                                                foreground='white')]
                    )

    # En kort tag:
    axGraph.text(8.5, mnd_gjsnitt + 4,
                 f"≈ {mnd_gjsnitt:.1f} mm/mnd",

                 va='bottom',
                 ha='left',
                 fontsize=9,
                 color='red',
                 bbox=dict(boxstyle="round,pad=0.25",
                           fc="white",
                           ec="none",
                           alpha=0.8))

    # Grid for lesbarhet
    axGraph.grid(True, axis='y',
                 linestyle=':',
                 alpha=0.4)
    axGraph.minorticks_on()

    # Redraw map (keep zoom) + click marker
    tegn_kart(axMap, img, xr, yr, nedbor_aar, nedbor_mnd)
    axMap.scatter([x], [y], c='#D63031',
                  s=520,
                  marker='o',
                  zorder=6,
                  edgecolors='white',
                  linewidths=1.2)

    t = axMap.text(x, y, label_from_nedbor(aarsnedbor),
                   color='white',
                   fontsize=10,
                   ha='center',
                   va='center',
                   zorder=6)

    t.set_path_effects(TEXT_STROKE)
    axMap.set_title(f"C: ({x:.1f},{y:.1f}) – klikk rød = estimert",
                    fontsize=9)

    plt.draw()


# ---- Help text & reset button (relative to map area) ----
def add_ui():
    map_pos = axMap.get_position()
    fig.text(map_pos.x1, map_pos.y1 + 0.012, "Klikk på kartet for estimat",
             ha="right",
             va="bottom",
             fontsize=8,
             color="#222",
             bbox=dict(boxstyle="round,pad=0.25",
                       fc="#f3f4f6",
                       ec="#9ca3af",
                       lw=1,
                       alpha=0.95),
             zorder=10)

    btn_w, btn_h = 0.12, 0.06
    btn_x = map_pos.x1 - btn_w
    btn_y = map_pos.y0 - 0.065
    axBtn = fig.add_axes([btn_x, btn_y, btn_w, btn_h])
    btn = Button(axBtn, "Reset view",
                 color="#f3f4f6",
                 hovercolor="#e5e7eb")
    for sp in axBtn.spines.values():
        sp.set_edgecolor("#9ca3af");
        sp.set_linewidth(1)

    def reset_view(_=None):
        axMap.set_xlim(X_MIN, X_MAX)
        axMap.set_ylim(Y_MIN, Y_MAX)
        tegn_kart(axMap, img, xr, yr, nedbor_aar, nedbor_mnd)
        plt.draw()

    btn.on_clicked(reset_view)


# ---- Boot ----
draw_initial_views()
add_ui()

cid1 = fig.canvas.mpl_connect('scroll_event', on_scroll)
cid2 = fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
