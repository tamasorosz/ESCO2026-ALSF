from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.lines import Line2D

RUN = Path("results/test1_alsf_seed42_repair_verified_full")
OUT = Path("figures/elsevier_AB_clear")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RUN / "archive.csv")

df["status"] = (
    df["flag"].astype(str).str.strip().str.lower()
    .str.replace("_", "-", regex=False)
    .replace({"nonrobust": "non-robust", "noise": "nei"})
)

x = df.X1.to_numpy(float)
y = df.X2.to_numpy(float)

# A régió: Gauss-csúcs
A = (x - 2.5)**2 + (y - 2.5)**2 <= 0.18**2

# B régió: az eredeti ellipszis
theta = np.deg2rad(40)
dx, dy = x - 1.8, y - 1.87

B = (
    ((np.cos(theta)*dx + np.sin(theta)*dy)/0.24)**2
    + ((-np.sin(theta)*dx + np.cos(theta)*dy)/0.475)**2
) <= 1

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 16,
    "pdf.fonttype": 42,
    "axes.linewidth": 1.0
})

styles = [
    ("nei", "NEI", "#aaaaaa"),
    ("robust", "Locally stable", "#147d69"),
    ("non-robust", "Observed violation", "#bf542d")
]

regions = [
    ("A", A, (2.20, 2.82), (2.15, 2.87)),
    ("B", B, (1.20, 2.44), (1.15, 2.58))
]

for name, mask, xlim, ylim in regions:

    fig, ax = plt.subplots(figsize=(7.2, 6.5), dpi=150)

    # Körön kívüli háttér
    shown = (
        (x >= xlim[0]) & (x <= xlim[1]) &
        (y >= ylim[0]) & (y <= ylim[1])
    )
    outside = shown & ~mask

    ax.scatter(
        x[outside], y[outside],
        s=6, c="#bbbbbb", alpha=0.16,
        linewidths=0, rasterized=True
    )

    handles = []

    # Kizárólag a körön belüli pontok hangsúlyozása
    for flag, label, color in styles:

        selected = mask & (df.status.to_numpy() == flag)

        ax.scatter(
            x[selected], y[selected],
            s=19 if flag == "nei" else 30,
            c=color,
            alpha=0.82 if flag == "nei" else 0.90,
            edgecolors="#333333",
            linewidths=0.35,
            zorder=3,
            rasterized=True
        )

        handles.append(
            Line2D(
                [], [], marker="o", linestyle="",
                markerfacecolor=color,
                markeredgecolor="#333333",
                markersize=7,
                label=f"{label} (n={selected.sum()})"
            )
        )

    # A régió körvonala
    if name == "A":
        outline = Circle(
            (2.5, 2.5), 0.18,
            fill=False, edgecolor="#244c88", linewidth=2.7
        )
    else:
        outline = Ellipse(
            (1.8, 1.87), 0.48, 0.95, angle=40,
            fill=False, edgecolor="#244c88", linewidth=2.7
        )

    ax.add_patch(outline)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(f"Region {name}", loc="left")
    ax.minorticks_on()
    ax.legend(handles=handles, loc="upper right",
              fontsize=10, framealpha=0.97)

    for ext in ("png", "pdf"):
        path = OUT / f"AB_region_{name}.{ext}"
        fig.savefig(path, dpi=350, bbox_inches="tight")
        print("Elkészült:", path)

    plt.close(fig)
