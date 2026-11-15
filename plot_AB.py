from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

RUN = Path("results/test1_alsf_seed42_repair_verified_full")
OUT = Path("figures/elsevier_AB")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RUN / "archive.csv")
df["flag"] = (
    df["flag"].astype(str).str.strip().str.lower()
    .str.replace("_", "-", regex=False)
    .replace({"nei": "NEI", "noise": "NEI",
              "nonrobust": "non-robust"})
)

colors = {
    "NEI": "#B5B8BC",
    "robust": "#217A6B",
    "non-robust": "#C35B36"
}

labels = {
    "NEI": "Insufficient evidence",
    "robust": "Locally stable",
    "non-robust": "Observed violation"
}

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.75,
    "xtick.direction": "in",
    "ytick.direction": "in"
})

x = df["X1"].to_numpy()
y = df["X2"].to_numpy()

A = (x - 2.5)**2 + (y - 2.5)**2 <= 0.18**2

t = np.deg2rad(40)
dx, dy = x - 1.8, y - 1.87
B = (
    ((np.cos(t)*dx + np.sin(t)*dy)/0.24)**2
    + ((-np.sin(t)*dx + np.cos(t)*dy)/0.475)**2
) <= 1

regions = {"A": A, "B": B}


def points(ax, data, cx, cy, large=False):
    for flag in ("NEI", "robust", "non-robust"):
        d = data[data["flag"] == flag]
        ax.scatter(
            d[cx], d[cy],
            s=13 if large else 4,
            c=colors[flag],
            alpha=0.27 if flag == "NEI" else 0.85,
            edgecolors="none",
            rasterized=True
        )


def make(space):
    decision = space == "decision"
    xx, yy = ("X1", "X2") if decision else ("Obj1", "Obj2")

    fig = plt.figure(figsize=(9.6, 5.6))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.7, 1], hspace=0.35, wspace=0.30
    )

    full = fig.add_subplot(gs[:, 0])
    points(full, df, xx, yy)

    if decision:
        full.set_xlim(-0.4, 6.15)
        full.set_ylim(-0.8, 5.55)
        full.add_patch(Circle(
            (2.5, 2.5), 0.18, fill=False,
            ec="#345E89", lw=1.5
        ))
        full.add_patch(Ellipse(
            (1.8, 1.87), 0.48, 0.95, angle=40,
            fill=False, ec="#345E89", lw=1.5
        ))
        full.text(2.72, 2.57, "A")
        full.text(1.40, 1.35, "B")
    else:
        full.set_xlim(-2, 62)
        ymax = max(
            120,
            df.loc[A, "Obj2"].max()*1.1 if A.any() else 120
        )
        full.set_ylim(-2, ymax)

    for row, (name, mask) in enumerate(regions.items()):
        ax = fig.add_subplot(gs[row, 1])
        selected = df.loc[mask]

        if decision:
            if name == "A":
                xlim, ylim = (2.15, 2.85), (2.10, 2.90)
            else:
                xlim, ylim = (1.20, 2.40), (1.15, 2.55)

            nearby = df[
                df["X1"].between(*xlim) &
                df["X2"].between(*ylim)
            ]
            points(ax, nearby, xx, yy, large=True)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            if name == "A":
                ax.add_patch(Circle(
                    (2.5, 2.5), 0.18, fill=False,
                    ec="#345E89", lw=1.5
                ))
            else:
                ax.add_patch(Ellipse(
                    (1.8, 1.87), 0.48, 0.95, angle=40,
                    fill=False, ec="#345E89", lw=1.5
                ))
        else:
            points(ax, selected, xx, yy, large=True)
            if len(selected):
                vx = selected[xx].to_numpy()
                vy = selected[yy].to_numpy()
                padx = max(0.7, np.ptp(vx)*0.12)
                pady = max(1.0, np.ptp(vy)*0.12)
                ax.set_xlim(vx.min()-padx, vx.max()+padx)
                ax.set_ylim(vy.min()-pady, vy.max()+pady)

        ax.set_title(
            f"Region {name} (n={len(selected)})",
            loc="left"
        )
        ax.set_xlabel(r"$x_1$" if decision else r"$f_1(\mathbf{x})$")
        ax.set_ylabel(r"$x_2$" if decision else r"$f_2(\mathbf{x})$")
        ax.minorticks_on()

    full.set_title(
        "Full design space" if decision else "Full objective space",
        loc="left"
    )
    full.set_xlabel(r"$x_1$" if decision else r"$f_1(\mathbf{x})$")
    full.set_ylabel(r"$x_2$" if decision else r"$f_2(\mathbf{x})$")
    full.minorticks_on()

    handles = [
        plt.Line2D(
            [], [], linestyle="", marker="o", markersize=5,
            markerfacecolor=colors[k], markeredgecolor="none",
            label=labels[k]
        )
        for k in ("NEI", "robust", "non-robust")
    ]

    fig.legend(
        handles=handles, loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3, frameon=False
    )

    fig.subplots_adjust(bottom=0.14)

    for ext in ("png", "pdf"):
        filename = OUT / f"AB_{space}.{ext}"
        fig.savefig(
            filename, dpi=350, bbox_inches="tight"
        )
        print("Elkészült:", filename)

    plt.close(fig)


make("decision")
make("objective")

for name, mask in regions.items():
    print(
        name,
        "pontok:", int(mask.sum()),
        df.loc[mask, "flag"].value_counts().to_dict()
    )
