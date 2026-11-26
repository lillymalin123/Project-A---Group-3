"""Q5 figure --- residual change |dJ| versus attack magnitude t for the three attack constructions.

Run from anywhere:  python scripts/10_q5_figure.py   (after 09_q5_attack_sweep.py)
Reads results/q5_attack_sweep.csv and writes figures/q5_attack_sweep.pdf and .png: two panels (noise-free,
noisy), |dJ| against t on log-log axes, one line per attack. Filled markers: dJ > 0; open markers: dJ < 0
(the estimator fits the attacked data better than the unattacked data). Dashed reference lines show the
predicted slopes. Dotted horizontal line: tau - J0 (noise-free) or the noise scale sqrt(2 dof) of J (noisy).
Non-convergent runs (converged = False) are left out and counted in the panel title.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window: the script only writes files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config

# categorical slots 1-3 (validated all-pairs for colour-vision deficiency); the marker is a second encoding
STYLE = {"DC": ("#2a78d6", "o"), "Jacobian": ("#eb6834", "s"), "nonlinear": ("#1baf7a", "^")}
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
# predicted slopes of |dJ| for small t (derivation in docs/), drawn through the first point of that attack
#GUIDES = {"noise-free": [("DC", 2), ("Jacobian", 4)],
#          "noisy": [("DC", 1), ("Jacobian", 2), ("nonlinear", 2)]}

df = pd.read_csv(config.RESULTS_DIR / "q5_attack_sweep.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True)

for ax, noise in zip(axes, ["noise-free", "noisy"]):
    d = df[df["noise"] == noise]
    ok = d[d["converged"]]
    for attack, (color, marker) in STYLE.items():
        s = ok[ok["attack"] == attack].sort_values("t_rad")
        t, dJ = s["t_rad"].to_numpy(), s["dJ"].to_numpy()
        ax.plot(t, np.abs(dJ), color=color, lw=2, zorder=2)
        up = dJ > 0
        ax.scatter(t[up], np.abs(dJ[up]), marker=marker, s=45, color=color,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter(t[~up], np.abs(dJ[~up]), marker=marker, s=45, facecolor="white",
                   edgecolor=color, linewidth=1.5, zorder=3)
#    for attack, p in GUIDES[noise]:
#        s = ok[ok["attack"] == attack].sort_values("t_rad")
#        t0, y0 = s["t_rad"].iloc[0], abs(s["dJ"].iloc[0]) / 30
#        tg = np.array([t0, t0 * 100])  # two decades
#        ax.plot(tg, y0 * (tg / t0) ** p, ls="--", lw=1.2, color=MUTED, zorder=1)
#        ax.text(tg[-1], y0 * 100 ** p, rf"  $\propto t^{p}$", color=MUTED, fontsize=9, va="top")

    dof, tau, J0 = ok["dof"].iloc[0], ok["tau"].iloc[0], ok["J0"].iloc[0]
    level, label = ((tau - J0, r"$\tau - J_0$ (alarm, no noise)") if noise == "noise-free"
                    else (np.sqrt(2 * dof), r"$\sqrt{2\,\mathrm{dof}}$ = noise std of $J$"))
    ax.axhline(level, color=INK, lw=1, ls=":", zorder=1)
    ax.text(0.02, level, label, transform=ax.get_yaxis_transform(), color=INK, fontsize=9,
            va="bottom")

    n_nc = int((~d["converged"]).sum())
    ax.set_title(f"{noise}  ({n_nc} non-convergent run{'s' if n_nc != 1 else ''} omitted)",
                 color=INK, fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"attack magnitude t: angle shift at bus {config.ATTACK_BUS} (rad)", color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.5)
    ax.tick_params(colors=MUTED, which="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)

axes[0].set_ylabel(r"$|\Delta J| = |J - J_0|$", color=INK)
handles = [Line2D([], [], color=c, marker=mk, lw=2, markersize=7, label=a) for a, (c, mk) in STYLE.items()]
handles += [Line2D([], [], ls="", marker="o", color=MUTED, markersize=7, label=r"$\Delta J > 0$"),
            Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=MUTED,
                   markersize=7, label=r"$\Delta J < 0$")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))

config.FIGURES_DIR.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    path = config.FIGURES_DIR / f"q5_attack_sweep.{ext}"
    fig.savefig(path, dpi=200)
    print(f"wrote {path}")