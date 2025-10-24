# views_graph.py
import numpy as np
import matplotlib.patheffects as pe
from viz_utils import color_from_nedbor

KV_LABELS = ["Q1", "Q2", "Q3", "Q4"]

def months_to_quarters(month_values):
    """Konverterer en 12-måneders serie til 4 kvartaler (summen av alle tre månedene)."""
    a = np.asarray(month_values, dtype=float)
    if a.size != 12:
        raise ValueError("Input must have length 12")
    return np.array([
        a[0:3].sum(),
        a[3:6].sum(),
        a[6:9].sum(),
        a[9:12].sum()
    ], dtype=float)

def _style_axes(ax):
    ax.set_ylabel(None)
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)
    ax.grid(axis='y',
            linestyle=':',
            alpha=0.35)

def _draw_month_axes(ax):
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(list("JFMAMJJASOND"),
                       fontsize=9)

def _draw_quarter_axes(ax):
    ax.set_xlim(0.4, 4.6)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(KV_LABELS,
                       fontsize=10)

def render_left_graph(ax, monthly_values, mode='kvartal'):
    """
    modus: 'mnd' | 'kvartal'
    monthly_values: En matrise med lengde 12 eller Ingen
     """
    ax.cla()  # grafik temizle

    if mode == 'mnd':  # Aylık görünüm
        _draw_month_axes(ax)
        base_title = "Nedbør per måned"
        if monthly_values is not None:
            months = np.arange(1, 13, dtype=float)
            colors = [color_from_nedbor(v * 12.0) for v in monthly_values]
            ax.bar(months,
                   monthly_values,
                   color=colors,
                   edgecolor='black',
                   linewidth=0.6)
            aars = float(np.sum(monthly_values))
            ax.set_title(f"{base_title} – Årsnedbør ≈ {int(aars)} mm",
                         fontsize=9)
            # gjennomsnittlig rød stiplet linje
            m_avg = float(np.mean(monthly_values))
            ax.axhline(m_avg,
                       color='#e60000',
                       linewidth=1.4,
                       linestyle='--',
                       path_effects=[pe.withStroke(linewidth=2.2,
                                                   foreground='white')])
            # etikett til høyre, litt over linjen
            ax.text(8.5, m_avg + 4, f"≈ {m_avg:.1f} mm/mnd",
                    ha='left',
                    va='bottom',
                    fontsize=9,
                    color='red',
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="white",
                              ec="none",
                              alpha=0.85))
        else:
            ax.set_title(base_title,
                         fontsize=9)

    else:  # Kvartal styl
        _draw_quarter_axes(ax)
        base_title = "Nedbør per kvartal"
        if monthly_values is not None:
            q = months_to_quarters(monthly_values)
            xs = np.arange(1, 5, dtype=float)
            colors = [color_from_nedbor(v) for v in q]
            ax.bar(xs, q,
                   color=colors,
                   edgecolor='black',
                   linewidth=0.6)
            aars = float(np.sum(monthly_values))
            ax.set_title(f"{base_title} – Årsnedbør ≈ {int(aars)} mm",
                         fontsize=9)
        else:
            ax.set_title(base_title,
                         fontsize=9)

    _style_axes(ax)
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10
    ax.set_ylim(0, np.ceil(ymax / 20) * 20 + 10)
