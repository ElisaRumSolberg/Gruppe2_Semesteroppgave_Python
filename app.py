# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

# Widget Referanser skal ikke gå til GC
UI = {}

from data_model import load_and_train
from views_map import load_map_image, tegn_kart
from views_legend import render_vertical_legend
from viz_utils import (
    X_MIN, X_MAX, Y_MIN, Y_MAX,
    label_from_nedbor, TEXT_STROKE
)
from views_graph import render_left_graph

# ---- Mod ----
view_mode = 'mnd'  # første visning, måned
last_pred = None  # 12 måneder med serier produsert ved siste klikk

# ---- Yollar  ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "data", "nedborX.csv")
IMG_PATH = os.path.join(SCRIPT_DIR, "assets", "Bergen_Kart2.PNG")

# ----Data og modell  ----
store = load_and_train(CSV_PATH)
poly = store["poly"]
model = store["model"]
xr, yr = store["xr"], store["yr"]
nedbor_aar, nedbor_mnd = store["nedbor_aar"], store["nedbor_mnd"]

# ---- Figurplassering: [Forklaring] | [Søyle] | [Kart] ----
plt.rcParams['figure.dpi'] = 120
fig = plt.figure(figsize=(12.2, 4.8))

gs = gridspec.GridSpec(
    nrows=1, ncols=4,
    width_ratios=[0.12, 0.03, 0.31, 0.54],
    wspace=0.1
)
fig.subplots_adjust(left=0.01, right=0.985, top=0.92, bottom=0.16)

axLegend = fig.add_subplot(gs[0, 0])
axSpacer = fig.add_subplot(gs[0, 1])
axGraph = fig.add_subplot(gs[0, 2])
axMap = fig.add_subplot(gs[0, 3])

axSpacer.set_xticks([]);
axSpacer.set_yticks([]);
axSpacer.axis('off')
axMap.axis('off')
axMap.set_xlim(X_MIN, X_MAX)
axMap.set_ylim(Y_MIN, Y_MAX)

# --------kartbilde---
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"Map image not found: {IMG_PATH}")
img = load_map_image(IMG_PATH)


# ----Innledende visninger --
def draw_initial_views():
    render_vertical_legend(axLegend)
    tegn_kart(axMap, img, xr, yr, nedbor_aar, nedbor_mnd)
    render_left_graph(axGraph, None, mode=view_mode)


# ---- Reset  ----
def _reset_core():
    global last_pred
    last_pred = None
    axMap.set_xlim(X_MIN, X_MAX)
    axMap.set_ylim(Y_MIN, Y_MAX)
    tegn_kart(axMap, img, xr, yr, nedbor_aar, nedbor_mnd)
    render_left_graph(axGraph, None, mode=view_mode)
    plt.draw()


# ---- Hendelser
def on_scroll(event):
    if event.inaxes != axMap:
        return
    scale = 1.2 if event.button == 'up' else 1 / 1.2

    cur_xmin, cur_xmax = axMap.get_xlim()
    cur_ymin, cur_ymax = axMap.get_ylim()
    xdata, ydata = event.xdata, event.ydata

    def _scale(lo, hi, c):
        return c - (c - lo) * scale, c + (hi - c) * scale

    new_xmin, new_xmax = _scale(cur_xmin, cur_xmax, xdata)
    new_ymin, new_ymax = _scale(cur_ymin, cur_ymax, ydata)

    new_xmin = max(X_MIN, new_xmin)
    new_xmax = min(X_MAX, new_xmax)
    new_ymin = max(Y_MIN, new_ymin)
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

    global last_pred
    last_pred = y_pred
    render_left_graph(axGraph, last_pred, mode=view_mode)

    tegn_kart(axMap, img, xr, yr, nedbor_aar, nedbor_mnd)
    axMap.scatter([x], [y], c='#D63031', s=520, marker='o', zorder=6,
                  edgecolors='white', linewidths=1.2)
    t = axMap.text(
        x, y, label_from_nedbor(aarsnedbor),
        color='white',
        fontsize=10,
        ha='center',
        va='center',
        zorder=6
    )
    t.set_path_effects(TEXT_STROKE)
    axMap.set_title(f"C: ({x:.1f},{y:.1f}) – klikk rød = estimert", fontsize=9)
    plt.draw()


# ----Brukergrensesnitt: hjelpetekst, modusknapper, tilbakestilling
def add_ui():
    # Karthjelpnotat
    map_pos = axMap.get_position()
    fig.text(
        map_pos.x1, map_pos.y1 + 0.012, "Klikk på kartet for estimat",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#222",
        bbox=dict(boxstyle="round,pad=0.25",
                  fc="#f3f4f6",
                  ec="#9ca3af",
                  lw=1,
                  alpha=0.95),
        zorder=10
    )

    # ---- Modusknapper: side om side nederst
    graph_pos = axGraph.get_position()
    bw, bh = 0.09, 0.05
    bx1 = graph_pos.x0 + 0.01
    by = graph_pos.y0 - 0.1
    bx2 = bx1 + bw + 0.015

    axBtnM = fig.add_axes([bx1, by, bw, bh])
    axBtnK = fig.add_axes([bx2, by, bw, bh])

    btnM = Button(axBtnM, "Måneder")
    btnK = Button(axBtnK, "Kvartal")

    UI["btnM_ax"], UI["btnK_ax"] = axBtnM, axBtnK
    UI["btnM"], UI["btnK"] = btnM, btnK

    def _style_active(btn_ax, btn, active=True):
        # tekstfarge
        if active:
            btn_ax.set_facecolor("#e5e7eb")
            btn.label.set_color("blue")
        else:
            btn_ax.set_facecolor("#e5e7eb")
            btn.label.set_color("black")
        # spine
        for sp in btn_ax.spines.values():
            sp.set_edgecolor("#3F51B5")
            sp.set_linewidth(1)

    def _switch_to(mode):
        global view_mode
        view_mode = mode
        _style_active(axBtnM, btnM, active=(mode == 'mnd'))
        _style_active(axBtnK, btnK, active=(mode == 'kvartal'))
        render_left_graph(axGraph, last_pred, mode=view_mode)
        plt.draw()

    btnM.on_clicked(lambda _: _switch_to('mnd'))
    btnK.on_clicked(lambda _: _switch_to('kvartal'))
    _style_active(axBtnM, btnM, active=False)
    _style_active(axBtnK, btnK, active=True)

    # ---- Reset btn
    axReset = fig.add_axes([0.88, 0.1, 0.10, 0.05])
    btnR = Button(axReset, "Reset view",
                  color="#f3f4f6",
                  hovercolor="#e5e7eb")

    UI["reset_ax"] = axReset
    UI["reset_btn"] = btnR

    for sp in axReset.spines.values():
        sp.set_edgecolor("#9ca3af")
        sp.set_linewidth(1)

    btnR.on_clicked(lambda _: _reset_core())


# ---- Boot ----
draw_initial_views()
add_ui()

cid1 = fig.canvas.mpl_connect('scroll_event', on_scroll)
cid2 = fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event',
                       lambda e: (_reset_core() if e.key in ('r', 'R') else None))

plt.show()
