# viz_utils.py
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle

# Figure/map extent
X_MIN, X_MAX = 0.0, 13.0
Y_MIN, Y_MAX = 0.0, 10.0
KART_EXTENT = (X_MIN, X_MAX, Y_MIN, Y_MAX)

# Initial spans for zoom scale
INIT_DX = X_MAX - X_MIN
INIT_DY = Y_MAX - Y_MIN

def current_zoom_scale(ax):
    """Return max(x,y) scale ratio relative to initial extent."""
    cx0, cx1 = ax.get_xlim()
    cy0, cy1 = ax.get_ylim()
    dx = (cx1 - cx0) / INIT_DX
    dy = (cy1 - cy0) / INIT_DY
    return max(dx, dy)

# ----Farger (blå→lilla rampe) og terskler (årlig mm) ----
COLORS = [
    '#B3E5FC',  # <1300
    '#64B5F6',  # 1300–1699
    '#3F51B5',  # 1700–2499
    '#7E57C2',  # 2500–3199
    '#512DA8'   # >=3200
]

def index_from_nedbor(x: float) -> int:
    if x < 1300: return 0
    if x < 1700: return 1
    if x < 2500: return 2
    if x < 3200: return 3
    return 4

def color_from_nedbor(nedbor: float) -> str:
    return COLORS[index_from_nedbor(nedbor)]

def label_from_nedbor(nedbor: float) -> str:
    """Hundrevis av etiketter (f.eks. 1300 -> '13')."""
    return str(int(nedbor / 100))

def size_from_nedbor_series(nedbor_array, min_size=700, max_size=1500):
    """Robust skalering for spredningsstørrelser (inndata i mm)."""
    v = np.asarray(nedbor_array, dtype=float)
    if v.size == 0:
        return np.array([min_size],
                        dtype=float)
    v_min = np.percentile(v, 5)
    v_max = np.percentile(v, 95)
    if v_max <= v_min:
        return np.full_like(v, (min_size + max_size) * 0.5)

    s = (v - v_min) / (v_max - v_min)
    s = np.clip(s, 0, 1)
    return min_size + s * (max_size - min_size)

def regndraape_marker(width: float = 1.35, height: float = 0.72) -> MarkerStyle:
    """Bred og kort regndråpemarkør."""
    verts = np.array([
        (0.00, 1.00),
        (0.45, 0.30),
        (0.35, -0.30),
        (0.00, -1.30),
        (-0.35, -0.30),
        (-0.45, 0.30),
        (0.00, 1.0),
        (0.00, 1.00),
    ], dtype=float)
    verts[:, 0] *= width
    verts[:, 1] *= height
    codes = [
        Path.MOVETO,
        Path.CURVE3, Path.CURVE3, Path.CURVE3,
        Path.CURVE3, Path.CURVE3, Path.CURVE3,
        Path.CLOSEPOLY,
    ]
    return MarkerStyle(Path(verts, codes))

# Text stroke utility (consistent across modules)
TEXT_STROKE = [pe.withStroke(linewidth=1.6,
                             foreground='black')]
