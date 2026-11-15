from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.lines import Line2D

RUN = Path("results/test1_alsf_seed42_repair_verified_full")
OUT = Path("figures/original_3_new")
OUT.mkdir(parents=True, exist_ok=True)

a = pd.read_csv(RUN / "archive.csv")
p = pd.read_csv(RUN / "process.csv").sort_values("generation")

a["flag"] = (
    a.flag.astype(str).str.strip().str.lower()
    .str.replace("_", "-", regex=False)
    .replace({"nonrobust": "non-robust", "nei": "NEI", "noise": "NEI"})
)

required_a = {"X1", "X2", "Obj1", "Obj2", "flag", "cluster_id"}
required_p = {
    "generation", "n_robust_clusters", "n_non_robust_clusters",
    "n_regenerated", "n_attempts"
}

assert required_a <= set(a), "Missing archive columns"
assert required_p <= set(p), "Missing process columns"
assert (p.n_attempts >= p.n_regenerated).all()

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.labelsize": 12,
    "pdf.fonttype": 42
})

x = a[["X1", "X2"]].to_numpy(float)
ids = pd.to_numeric(
    a.cluster_id, errors="coerce"
).fillna(-1).to_numpy(int)

stable = a.flag.eq("robust").to_numpy()
violated = a.flag.eq("non-robust").to_numpy()

# Stable representatives, not certified robust cluster medoids.
reps = []
for cid in np.unique(ids[ids >= 0]):
    ii = np.flatnonzero((ids == cid) & stable)
    if len(ii) >= 3:
        pts = x[ii] / 0.1
        centre = np.median(pts, axis=0)
        reps.append(
            ii[np.argmin(np.max(abs(pts - centre), axis=1))]
        )

reps = np.array(reps, dtype=int)

A = (x[:, 0]-2.5)**2 + (x[:, 1]-2.5)**2 <= 0.18**2

theta = np.deg2rad(40)
dx, dy = x[:, 0]-1.8, x[:, 1]-1.87
B = (
    ((np.cos(theta)*dx + np.sin(theta)*dy)/0.24)**2
    + ((-np.sin(theta)*dx + np.cos(theta)*dy)/0.475)**2
) <= 1


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT / f"{name}.{ext}",
            dpi=350, bbox_inches="tight"
        )
    plt.close(fig)


def scatter(space):
    dec = space == "design_space_clusters"
    xx, yy = ("X1", "X2") if dec else ("Obj1", "Obj2")
    X, Y = a[xx].to_numpy(float), a[yy].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    ax.scatter(
        X, Y, s=6, color="#c9c9c9",
        alpha=0.35, linewidths=0, rasterized=True
    )

    ax.scatter(
        X[violated], Y[violated],
        s=10, color="#ffaa30",
        alpha=0.78, linewidths=0, rasterized=True
    )

    for j, i in enumerate(reps):
        c = plt.get_cmap("gist_rainbow")((j * 0.618034) % 1)

        ax.scatter(
            [X[i]], [Y[i]], marker="x",
            s=75, color=c, lw=2, zorder=6
        )
        ax.scatter(
            [X[i]], [Y[i]], marker="x",
            s=24, color="black", lw=1, zorder=7
        )

    if dec:
        for area in [A, B]:
            for flag, col in [
                ("robust", "#218476"),
                ("non-robust", "#c95635")
            ]:
                keep = area & a.flag.eq(flag).to_numpy()
                ax.scatter(
                    X[keep], Y[keep], s=20, color=col,
                    edgecolor="white", linewidth=0.3, zorder=8
                )

        ax.add_patch(
            Circle(
                (2.5, 2.5), 0.18,
                fill=False, ec="blue", lw=2.3, zorder=10
            )
        )
        ax.add_patch(
            Ellipse(
                (1.8, 1.87), 0.48, 0.95, angle=40,
                fill=False, ec="blue", lw=2.3, zorder=10
            )
        )

        ax.text(2.73, 2.59, "A", fontsize=21)
        ax.text(1.48, 1.30, "B", fontsize=21)
        ax.axvline(2.5, color=".6", ls=":", lw=0.7)
        ax.axhline(2.5, color=".6", ls=":", lw=0.7)
        ax.set(
            xlabel=r"$x_1$", ylabel=r"$x_2$",
            xlim=(-0.4, 6.15), ylim=(-0.8, 5.55)
        )
    else:
        ax.set(
            xlabel=r"$f_1(\mathbf{x})$",
            ylabel=r"$f_2(\mathbf{x})$",
            xlim=(-3, 62), ylim=(-2, 57.5)
        )
        ax.scatter(
            X[B], Y[B], s=19, facecolors="none",
            edgecolors="blue", linewidths=0.45, zorder=8
        )

    legend = [
        Line2D([], [], marker="o", ls="", color="#c9c9c9",
               label="Archive"),
        Line2D([], [], marker="o", ls="", color="#ffaa30",
               label="Observed violation"),
        Line2D([], [], marker="x", ls="", color="black",
               label="Locally stable representative")
    ]

    ax.legend(
        handles=legend,
        loc="lower right" if dec else "lower left",
        frameon=True, fontsize=9
    )

    fig.tight_layout()
    save(fig, space)


def evolution():
    g = p.generation.to_numpy()
    fig, axs = plt.subplots(
        2, 1, figsize=(12, 8.5), sharex=True
    )

    axs[0].stackplot(
        g,
        p.n_robust_clusters,
        p.n_non_robust_clusters,
        colors=["#4caf50", "#e45151"],
        labels=["Robust clusters", "Non-robust clusters"]
    )

    axs[0].set(
        title="Robust / non-robust per generation",
        ylabel="Clusters"
    )
    axs[0].legend(loc="upper left")

    additional = np.maximum(
        p.n_attempts - p.n_regenerated, 0
    )

    axs[1].bar(
        g, p.n_regenerated, width=0.9,
        color="#638fb9", label="Regenerated"
    )
    axs[1].bar(
        g, additional, bottom=p.n_regenerated,
        width=0.9, color="#ed9292",
        label="Additional attempts"
    )

    axs[1].set(
        title="Archive repair — regenerated offspring & resample attempts",
        ylabel="Count",
        xlabel="Generation",
        xlim=(0, 200)
    )
    axs[1].legend(loc="upper left")

    for ax in axs:
        ax.grid(ls="--", alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "test_evolution")


scatter("design_space_clusters")
evolution()
scatter("pareto_front")

print("DONE:", OUT)
print("Stable representatives:", len(reps))
print("A:", a.loc[A, "flag"].value_counts().to_dict())
print("B:", a.loc[B, "flag"].value_counts().to_dict())
