"""Q3 --- how well does chi2(dof) describe the test statistic without attack? Statistics and figures.

Run from anywhere:  python scripts/07_q3_chi2_analysis.py   (after 06_q3_monte_carlo.py; takes seconds)
Reads results/q3_mc_draws.csv and results/q3_mc_reference.csv; runs no estimator.

Per sigma and estimator (results/q3_mc_summary.csv):
  dof (accounting for the constraints: m - n + c), threshold tau = 95 % quantile of chi2(dof),
  number of draws, of converged and of non-convergent AC runs,
  sample mean and variance of J against the theory, with z-scores (|z| > 2: significant deviation),
  empirical false-alarm rate P(J > tau) with a 95 % Wilson interval, against the nominal alpha and the
  theoretical rate, the empirical 95 % quantile of J (a simulation-calibrated threshold), a KS p-value.
  Theory: AC -> central chi2(dof) (exact only for a linear model); DC -> noncentral chi2(dof, lambda),
  lambda = J of the DC estimator on noise-free AC data (exact: linear estimator, fixed model mismatch).
  AC only: the nonlinearity effect J_AC - J_lin_AC (J_lin_AC = linearised estimator on the same noise,
  exactly chi2(dof)): mean, standard deviation, largest |.|, and std / sqrt(2 dof) = its size relative
  to the natural spread of J.
Treatment of non-convergent AC runs: excluded from every AC statistic and counted in n_nonconvergent;
  far_nonconv_as_alarm gives the false-alarm rate if they were counted as alarms instead (an operator
  would at least notice them).
Figures:
  figures/q3_qq_ac.png         Q-Q plot of J_AC against chi2(dof), one panel per sigma, with the 95 %
                               pointwise band of a true chi2 sample of the same size
  figures/q3_false_alarm.png   empirical false-alarm rate against sigma: AC (left), DC with its
                               noncentral-chi2 prediction (right)
Exits non-zero if a self-check fails.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, chi2, ncx2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from common import check
from monte_carlo import chi2_summary, qq_points

# colours: categorical slots 1 and 2 of the reference palette (dataviz skill), neutral ink for references
AC_COLOR, DC_COLOR, REF_COLOR, INK, MUTED = "#2a78d6", "#eb6834", "#52514e", "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "figure.facecolor": "white", "savefig.dpi": 200, "legend.frameon": False})

for p in (config.MC_DRAWS_CSV, config.MC_REFERENCE_CSV):
    if not p.exists():
        sys.exit(f"{p} is missing: run scripts/06_q3_monte_carlo.py first.")
draws = pd.read_csv(config.MC_DRAWS_CSV)
ref = pd.read_csv(config.MC_REFERENCE_CSV).set_index("sigma")
sigmas = ref.index.to_numpy()

# --- self-checks ----------------------------------------------------------------------------------
print("Self-checks")
counts = draws.groupby("sigma").size()
ok = [
    check(np.allclose(sorted(sigmas), sorted(config.MC_SIGMAS)), f"draws exist for every sigma in config: {[float(s) for s in sigmas]}"),
    check(bool((counts == ref["n_draws"]).all()), f"{int(ref['n_draws'].iloc[0])} draws per sigma, none missing"),
    check(bool(draws.loc[~draws["converged_ac"], "J_ac"].isna().all()) and bool(draws["J_dc"].notna().all()),
          "non-convergent AC runs carry NaN (cannot enter a statistic by accident); every DC run has a J"),
    check(bool(np.allclose(ref["tau_ac"], chi2.ppf(1 - ref["alpha"], ref["dof_ac"]))),
          "thresholds in the file are the (1 - alpha) quantiles of chi2(dof)"),
    check("J_lin_ac" in draws and bool(draws["J_lin_ac"].notna().all()),
          "every draw has the linearised statistic J_lin_AC (scripts/06 is up to date)"),
]

# --- summary table ----------------------------------------------------------------------------------
rows = []
for s in sigmas:
    d, r = draws[draws["sigma"] == s], ref.loc[s]
    n_nc = int((~d["converged_ac"]).sum())
    ac = chi2_summary(d["J_ac"], r["dof_ac"], r["tau_ac"])
    nl = (d["J_ac"] - d["J_lin_ac"]).dropna()  # nonlinearity effect, converged runs only
    rows.append({"estimator": "AC", "sigma": s, "dof": r["dof_ac"], "tau": r["tau_ac"], "lambda": 0.0,
                 "n_draws": len(d), "n_nonconvergent": n_nc, **ac, "far_nominal": r["alpha"],
                 "nonlin_mean": nl.mean(), "nonlin_std": nl.std(), "nonlin_maxabs": nl.abs().max(),
                 "nonlin_std_rel": nl.std() / np.sqrt(2 * r["dof_ac"]),
                 "far_nonconv_as_alarm": (ac["far"] * ac["n"] + n_nc) / len(d),
                 "max_dVa_over_sigma": d["max_dVa_ac"].max() / s, "max_dVm_over_sigma": d["max_dVm_ac"].max() / s,
                 "max_iterations": d["iterations_ac"].max()})
    dc = chi2_summary(d["J_dc"], r["dof_dc"], r["tau_dc"], lam=r["lambda_dc"])
    rows.append({"estimator": "DC", "sigma": s, "dof": r["dof_dc"], "tau": r["tau_dc"], "lambda": r["lambda_dc"],
                 "n_draws": len(d), "n_nonconvergent": 0, **dc, "far_nominal": r["alpha"],
                 "far_nonconv_as_alarm": dc["far"]})
summary = pd.DataFrame(rows)
ok.append(check(bool((summary["n"] + summary["n_nonconvergent"] == summary["n_draws"]).all()),
                "every draw is either in the statistics or counted as non-convergent"))

config.RESULTS_DIR.mkdir(exist_ok=True)
out_csv = config.RESULTS_DIR / "q3_mc_summary.csv"
summary.to_csv(out_csv, index=False, float_format="%.6g")

print(f"\nDegrees of freedom: AC {int(ref['dof_ac'].iloc[0])} = 518 - 113 + 30 (measurements - states + "
      f"zero-injection constraints P and Q at 15 buses);  DC {int(ref['dof_dc'].iloc[0])} = 259 - 56 + 15")
print(f"Thresholds (alpha = {ref['alpha'].iloc[0]}): AC tau = {ref['tau_ac'].iloc[0]:.1f}, "
      f"DC tau = {ref['tau_dc'].iloc[0]:.1f}\n")
fmt = {"mean": "{:.1f}", "mean_theory": "{:.1f}", "z_mean": "{:+.1f}", "var": "{:.0f}", "var_theory": "{:.0f}",
       "z_var": "{:+.1f}", "q95": "{:.1f}", "ks_p": "{:.3f}"}
show = summary[["estimator", "sigma", "n", "n_nonconvergent", "mean", "mean_theory", "z_mean", "var", "var_theory",
                "z_var", "q95", "ks_p"]].copy()
for c, f in fmt.items():
    show[c] = show[c].map(lambda v, f=f: "" if pd.isna(v) else f.format(v))
show["FA rate [95% CI]"] = [f"{100 * a:.1f}% [{100 * b:.1f}, {100 * c:.1f}]  (theory {100 * t:.1f}%)"
                            for a, b, c, t in summary[["far", "far_lo", "far_hi", "far_theory"]].to_numpy()]
print(show.to_string(index=False))
print("\nNonlinearity effect on the AC statistic, J_AC - J_lin_AC (same noise; J_lin_AC is exactly chi2):")
for _, row in summary[summary.estimator == "AC"].iterrows():
    print(f"  sigma = {row['sigma']:<5g} mean {row['nonlin_mean']:+.3f}, std {row['nonlin_std']:.3f}, "
          f"max |.| {row['nonlin_maxabs']:.3f}  ->  std is {100 * row['nonlin_std_rel']:.2f} % of the chi2 spread "
          f"sqrt(2 dof) = {np.sqrt(2 * row['dof']):.1f}")
print(f"\nwrote {out_csv}")

# --- figure 1: Q-Q plot of the AC statistic ---------------------------------------------------------------
config.FIGURES_DIR.mkdir(exist_ok=True)
dof, tau = ref["dof_ac"].iloc[0], ref["tau_ac"].iloc[0]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), sharex=True, sharey=True)
for ax, s in zip(axes.ravel(), sigmas):
    th, emp = qq_points(draws.loc[draws["sigma"] == s, "J_ac"], dof)
    n = len(emp)
    i = np.arange(1, n + 1)
    band_lo, band_hi = chi2.ppf(beta.ppf(0.025, i, n + 1 - i), dof), chi2.ppf(beta.ppf(0.975, i, n + 1 - i), dof)
    ax.fill_between(th, band_lo, band_hi, color=AC_COLOR, alpha=0.15, linewidth=0,
                    label="95% band for a true χ² sample")
    lims = [th.min(), th.max()]
    ax.plot(lims, lims, color=REF_COLOR, lw=1, ls="--", label="y = x (perfect χ²)")
    ax.plot(th, emp, color=AC_COLOR, lw=2, label="AC estimator")
    ax.axhline(tau, color=MUTED, lw=0.8, ls=":")
    ax.axvline(tau, color=MUTED, lw=0.8, ls=":")
    row = summary[(summary.estimator == "AC") & (summary.sigma == s)].iloc[0]
    ax.set_title(f"σ = {s:g}", color=INK, fontsize=10, loc="left")
    ax.text(0.97, 0.04, f"mean {row['mean']:.1f} (χ²: {row['mean_theory']:.0f})\n"
                        f"var {row['var']:.0f} (χ²: {row['var_theory']:.0f})\n"
                        f"false alarms {100 * row['far']:.1f}%  ·  {int(row['n_nonconvergent'])} non-conv.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
for ax in axes[1]:
    ax.set_xlabel(f"χ²({int(dof)}) quantile")
for ax in axes[:, 0]:
    ax.set_ylabel("sorted J of the AC estimator")
axes[0, 0].legend(loc="upper left", fontsize=8)
fig.suptitle(f"Q–Q plot of the AC test statistic without attack ({int(ref['n_draws'].iloc[0])} draws per σ); "
             f"dotted: threshold τ = {tau:.1f}", fontsize=10, color=INK, x=0.02, ha="left")
fig.tight_layout()
qq_png = config.FIGURES_DIR / "q3_qq_ac.png"
fig.savefig(qq_png)
plt.close(fig)

# --- figure 2: false-alarm rate against sigma ----------------------------------------------------------
fig, (axa, axd) = plt.subplots(1, 2, figsize=(8.0, 3.4))
for ax, est, color in ((axa, "AC", AC_COLOR), (axd, "DC", DC_COLOR)):
    sm = summary[summary.estimator == est]
    yerr = np.clip(100 * np.vstack([sm["far"] - sm["far_lo"], sm["far_hi"] - sm["far"]]), 0, None)  # rounding at 0 % / 100 %
    ax.errorbar(sm["sigma"], 100 * sm["far"], yerr=yerr, fmt="o", color=color, ms=6, lw=1.5, capsize=3,
                label=f"{est}: empirical (95% CI)")
    ax.axhline(100 * ref["alpha"].iloc[0], color=REF_COLOR, lw=1, ls="--", label="nominal 5%")
    ax.set_xscale("log")
    ax.set_xticks(sigmas, [f"{s:g}" for s in sigmas])
    ax.minorticks_off()
    ax.set_xlabel("noise standard deviation σ (p.u. / rad)")
s_fine = np.geomspace(sigmas.min() * 0.8, sigmas.max() * 1.25, 200)
lam0, s0 = ref["lambda_dc"].iloc[0], sigmas[0]
pred = ncx2.sf(ref["tau_dc"].iloc[0], ref["dof_dc"].iloc[0], lam0 * (s0 / s_fine) ** 2)
axd.plot(s_fine, 100 * pred, color=DC_COLOR, lw=1.5, ls="--", label="noncentral χ² prediction")
axa.set_ylim(0, max(15, 100 * summary.loc[summary.estimator == "AC", "far_hi"].max() * 1.2))
axd.set_ylim(0, 105)
axa.set_ylabel("false-alarm rate without attack (%)")
axa.set_title("AC estimator", loc="left", fontsize=10, color=INK)
axd.set_title("DC estimator on the same draws", loc="left", fontsize=10, color=INK)
axa.legend(loc="upper left", fontsize=8)
axd.legend(loc="center left", fontsize=8)
fig.tight_layout()
fa_png = config.FIGURES_DIR / "q3_false_alarm.png"
fig.savefig(fa_png)
plt.close(fig)
print(f"wrote {qq_png} and {fa_png}")

if not all(ok):
    sys.exit("some checks FAILED")
