# views_legend.py
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from viz_utils import COLORS, regndraape_marker, size_from_nedbor_series


def render_vertical_legend(ax):
    """Venstre vertikal forklaring med kantlinje + størrelse (mm/mnd) og farge (mm/år)."""
    ax.cla()
    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.axis('off')
    ax.set_facecolor("white")  # gjør bakgrunnen ugjennomsiktig

    # ---- Outer card (with clear border) ----
    card = mpatches.FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.02",
        facecolor="white",
        edgecolor="#0f172a",  # dark border
        linewidth=1.2,
        alpha=0.98,
        transform=ax.transAxes, zorder=0
    )
    ax.add_patch(card)

    # subtle shadow
    card.set_path_effects([pe.withSimplePatchShadow(offset=(1, -1), alpha=0.15)])

    # ---------- Section 1: størrelse (mm/mnd) ----------
    ax.text(0.08, 0.95, "Størrelse (mm/mnd)",
            fontsize=8,
            fontweight="bold",
            color="#0f172a",
            transform=ax.transAxes,
            va="top")

    # Eksempel på månedlige beløp (match skalering av kartstørrelse)
    vals = np.array([20, 60, 120], dtype=float)
    svals = size_from_nedbor_series(vals, min_size=700, max_size=1500) * 0.33
    yrow = [0.86, 0.78, 0.70]  # vertical positions

    for y, v, s in zip(yrow, vals, svals):
        ax.scatter([0.20], [y], s=s,
                   c="#6366F1",
                   marker=regndraape_marker(),
                   edgecolors='black',
                   linewidths=0.6,
                   alpha=0.98,
                   transform=ax.transAxes,
                   zorder=2)

        ax.text(0.36, y, f"≈ {int(v)} mm",
                fontsize=9,
                color="#111827",
                va="center",
                transform=ax.transAxes)


    # Størrelsesmerknad
    SIZE_NOTE_Y = 0.66
    ax.text(0.08, SIZE_NOTE_Y, "Størrelse = månedlig nedbør",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#475569",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="white",
                      ec="none",
                      alpha=0.95))

    # Divider line
    ax.plot([0.08, 0.92], [0.635, 0.635],
            transform=ax.transAxes,
            linestyle='-',
            linewidth=0.9,
            color="#e5e7eb")

    # ---------- Section 2: COLOR (mm/år) ----------
    # -------- Nedbørsklasser (mm/år) --------
    TITLE_Y = SIZE_NOTE_Y - 0.1  # toppteksthøyde
    FIRST_Y = TITLE_Y - 0.078
    TITLE_PAD = 0.10  # mellomrom mellom tittelen og den første dråpen
    STEP = 0.105  #mellomrom mellom linjene
    X_DROP = 0.18  # slipp X-posisjon
    X_TEXT = 0.38  # tekst X-posisjon
    DROP_SIZE = 820

    ax.text(0.08, TITLE_Y, "Nedbørsklasser (mm/år)",
            fontsize=8,
            fontweight="bold",
            color="#0f172a",
            transform=ax.transAxes,
            va="top")

    labels = ["‹ 1300 mm", "1300–1699 mm", "1700–2499 mm", "2500–3199 mm", "≥ 3200 mm"]

    # ----- boşluk & konum ayarları (buradan oynayabilirsin)
    STEP = 0.085  # vertikalt mellomrom mellom linjer (0,09 → 0,115)
    X_DROP = 0.18  # X-posisjonen til dråpen
    X_TEXT = 0.38  # X-posisjon for etikettekst
    DROP_SIZE = 820  # dråpestørrelse

    y = FIRST_Y
    for fc, lab in zip(COLORS, labels):
        ax.scatter([X_DROP], [y],
                   s=DROP_SIZE,
                   c=fc,
                   marker=regndraape_marker(),
                   edgecolors='white',
                   linewidths=1.0,
                   alpha=0.98,
                   transform=ax.transAxes,
                   zorder=2)

        ax.text(X_TEXT, y, lab,
                fontsize=9,
                color="#111827",
                va="center",
                transform=ax.transAxes)
        y -= STEP

    # Small footnote
    FOOT_Y = (y + STEP) - 0.050
    ax.text(0.08, FOOT_Y, "Farger = årsnedbør",
            fontsize=8.5,
            color="#374151",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="white",
                      ec="none",
                      alpha=0.95)
    )
