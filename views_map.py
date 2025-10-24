# views_map.py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from viz_utils import (
    KART_EXTENT, X_MIN, X_MAX, Y_MIN, Y_MAX,
    current_zoom_scale,
    color_from_nedbor, label_from_nedbor, size_from_nedbor_series,
    regndraape_marker, TEXT_STROKE
)

#Valgfrie stedsetiketter
STEDER = {
    "Bergen": (6.73, 5.45),
    "Askøy": (4.10, 7.20),
    "Sotra": (3.80, 2.90),
    "Fana": (8.20, 3.80),
    "Arna": (8.70, 6.00),
    "Åsane": (6.90, 6.60),
    "Flesland": (7.55, 3.20),
    "Os": (9.00, 2.70),
    "Knarvik": (6.05, 9.30),
    "Østerøy": (9.65, 8.70),
    "Laksevåg": (5.75, 4.40),
    "Agåtnes": (2.55, 5.40),
}

def load_map_image(path):
    return mpimg.imread(path)

def tegn_kart(axMap, img, xr, yr, nedbor_aar, nedbor_mnd):
    """Draw background map + raindrops + labels, responsive to zoom."""
    axMap.cla()
    axMap.imshow(img,
                 extent=KART_EXTENT,
                 aspect="auto",
                 zorder=0)
    axMap.set_xlim(X_MIN, X_MAX)
    axMap.set_ylim(Y_MIN, Y_MAX)
    axMap.axis('off')
    axMap.set_title("Årsnedbør – Stor-Bergen", fontsize=11)

    z = current_zoom_scale(axMap)
    eff_size = (size_from_nedbor_series(nedbor_mnd) * 1.6) * z
    color_list = [color_from_nedbor(v) for v in nedbor_aar]

    # draw stations
    for i in range(len(xr)):
        axMap.scatter([xr[i]],[yr[i]],
                      c=[color_list[i]], s=[eff_size[i]],
                      marker=regndraape_marker(),
                      edgecolor='white',
                      linewidth=0.5,
                      alpha=0.95,
                      zorder=2,
                      path_effects=[pe.withSimplePatchShadow(offset=(1,-1),
                                                             alpha=0.35,
                                                             rho=0.9), pe.Normal()])

    # station labels (hundreds)
    base_lab_fs = 10
    lab_fs = max(6, base_lab_fs * z)
    dy = 0.03 * z
    labels = [label_from_nedbor(n) for n in nedbor_aar]
    for i in range(len(xr)):
        t = axMap.text(xr[i], yr[i] - dy, labels[i],
                       color='white',
                       fontsize=lab_fs,
                       ha='center',
                       va='center',
                       zorder=5,
                       clip_on=True)
        t.set_path_effects(TEXT_STROKE)

    # place names
    place_fs = max(7, int(10 * z))
    for name, (px, py) in STEDER.items():
        tt = axMap.text(px, py, name,
                        ha="center",
                        va="center",
                        fontsize=place_fs,
                        color="black",
                        zorder=4,
                        alpha=0.98,
                        clip_on=True)

        tt.set_path_effects([pe.withStroke(linewidth=1.0, foreground="white")])


        axMap.text(0.5, -0.06, "Tall = årsnedbør i hundre mm",
                   transform=axMap.transAxes,
                   ha='center',
                   fontsize=8.5,
                   color='#374151')

