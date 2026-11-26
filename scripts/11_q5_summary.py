"""Q5 summary --- slopes, Lecture 6 prediction, hidden magnitudes and estimate shift for every attack.

Run from anywhere:  python scripts/11_q5_summary.py   (after 09_q5_attack_sweep.py)
Reads results/q5_attack_sweep.csv and writes results/q5_summary.csv, one row per (noise, attack):
  slope_pred, slope_fit  predicted and fitted slope of log|dJ| against log t for t <= SLOPE_FIT_T_MAX
  dJ_sim, dJ_pred        dJ at t = PREDICTION_T from the sweep and from Lecture 6 (slide 9): the residual changes
                         by S d with d = a - [h(x_hat + c) - h(x_hat)], so dJ = 2 r'W S d + (S d)'W (S d)
  t_hidden_chi2          largest t still hidden, log-interpolated: J < tau (noise-free), |dJ| < sqrt(2 dof) (noisy)
  t_hidden_lnr           largest t with largest normalized residual < LNR_THRESHOLD
  shift_ratio            estimated angle change at ATTACK_BUS divided by t, at t = SHIFT_RATIO_T
  n_nonconverged, max_iterations
In the t_hidden columns, inf = hidden over the whole sweep and NaN = already above the threshold at the
smallest t (the noisy baseline itself fails the test).
Self-check: dJ_pred matches dJ_sim within PREDICTION_REL_TOL for the noise-free DC and Jacobian attacks and the
noisy DC attack, where the linearised residual model is complete. Exits non-zero otherwise.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from ac_state_estimation import build_model, estimate_ac, load_measurements, residual_sensitivity
from attacks import (ac_state_shift, angle_shift, build_ac_jacobian_attack, build_ac_nonlinear_attack,
                     build_dc_attack, predicted_dJ)
from common import check
from dc_state_estimation import dc_measurement_matrix, dc_subset, dc_zero_injection_matrix, embed_dc_vector

# predicted slope of |dJ| for small t (derivation in docs/): leak ~ t (DC) or t^2 (Jacobian); noise-free
# dJ = |S d|^2, with noise the cross term 2 r'W S d dominates; nonlinear with noise: -g'G^-1 g / 2 ~ t^2
SLOPE_PRED = {("noise-free", "DC"): 2, ("noise-free", "Jacobian"): 4, ("noise-free", "nonlinear"): np.nan,
              ("noisy", "DC"): 1, ("noisy", "Jacobian"): 2, ("noisy", "nonlinear"): 2}
CHECKED = [("noise-free", "DC"), ("noise-free", "Jacobian"), ("noisy", "DC")]


def first_crossing(t, y, level):
    """Largest t before y first reaches level, interpolated linearly in log-log between the two grid points.
    inf if y never reaches level, NaN if it is already there at the smallest t."""
    above = np.flatnonzero(y >= level)
    if above.size == 0:
        return np.inf
    i = above[0]
    if i == 0:
        return np.nan
    lt0, lt1, ly0, ly1 = np.log10([t[i - 1], t[i], y[i - 1], y[i]])
    return float(10 ** (lt0 + (np.log10(level) - ly0) * (lt1 - lt0) / (ly1 - ly0)))


# --- Lecture 6 prediction of dJ at t = PREDICTION_T --------------------------------------------------
m = build_model()
noisy = load_measurements(config.MEASUREMENTS_CSV)
cases = {"noise-free": noisy.assign(value=noisy["true"]), "noisy": noisy}
H_dc, C_dc = dc_measurement_matrix(m, dc_subset(noisy)), dc_zero_injection_matrix(m)
c = config.PREDICTION_T * angle_shift(m, config.ATTACK_BUS)

pred = {}
for noise, meas in cases.items():
    est0 = estimate_ac(m, meas)  # x_hat on the unattacked data, as in the sweep
    S = residual_sensitivity(est0)
    a_nl = build_ac_nonlinear_attack(m, meas, est0["Vm"], est0["Va"], ac_state_shift(m, c))
    attacks = {"DC": embed_dc_vector(meas, build_dc_attack(H_dc, c, C_dc)),
               "Jacobian": build_ac_jacobian_attack(est0["H"], ac_state_shift(m, c), est0["C"]),
               "nonlinear": a_nl}
    for name, a in attacks.items():
        cross, quad = predicted_dJ(S, est0["weights"], a - a_nl, est0["residual"])
        pred[(noise, name)] = cross + quad

# --- summary from the sweep ------------------------------------------------------------------------
sweep = pd.read_csv(config.RESULTS_DIR / "q5_attack_sweep.csv")
rows = []
for (noise, attack), s in sweep.groupby(["noise", "attack"], sort=False):
    ok_s = s[s["converged"]].sort_values("t_rad")
    t, dJ = ok_s["t_rad"].to_numpy(), ok_s["dJ"].to_numpy()
    small = t <= config.SLOPE_FIT_T_MAX
    slope_fit = (np.polyfit(np.log10(t[small]), np.log10(np.abs(dJ[small])), 1)[0]
                 if np.abs(dJ[small]).max() > config.SWEEP_DJ_TOL else np.nan)  # NaN: dJ is rounding noise
    level = (ok_s["tau"].iloc[0] - ok_s["J0"].iloc[0] if noise == "noise-free"
             else np.sqrt(2 * ok_s["dof"].iloc[0]))
    at_pred = ok_s.iloc[np.argmin(np.abs(np.log(t / config.PREDICTION_T)))]
    at_shift = ok_s.iloc[np.argmin(np.abs(np.log(t / config.SHIFT_RATIO_T)))]
    rows.append({
        "noise": noise, "attack": attack,
        "slope_pred": SLOPE_PRED[(noise, attack)], "slope_fit": slope_fit,
        "t_pred": at_pred["t_rad"], "dJ_sim": at_pred["dJ"], "dJ_pred": pred[(noise, attack)],
        "t_hidden_chi2": first_crossing(t, np.abs(dJ), level),
        "t_hidden_lnr": first_crossing(t, ok_s["largest_norm_resid"].to_numpy(), config.LNR_THRESHOLD),
        "shift_ratio": at_shift["dVa_attack_bus_rad"] / at_shift["t_rad"],
        "n_nonconverged": int((~s["converged"]).sum()), "max_iterations": int(ok_s["iterations"].max()),
    })

out = pd.DataFrame(rows)
config.RESULTS_DIR.mkdir(exist_ok=True)
out.to_csv(config.RESULTS_DIR / "q5_summary.csv", index=False)
print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print(f"\nwrote {config.RESULTS_DIR / 'q5_summary.csv'}")

# --- self-checks -------------------------------------------------------------------------------------
print("\nSelf-checks")
ok = []
for noise, attack in CHECKED:
    r = out[(out["noise"] == noise) & (out["attack"] == attack)].iloc[0]
    rel = abs(r["dJ_pred"] - r["dJ_sim"]) / abs(r["dJ_sim"])
    ok.append(check(rel < config.PREDICTION_REL_TOL,
                    f"{noise} {attack}: Lecture 6 predicts dJ = {r['dJ_pred']:.4g}, simulation {r['dJ_sim']:.4g} "
                    f"(relative error {rel:.1e})", "residual change = S d, Lecture 6 slide 9"))
if not all(ok):
    sys.exit("some checks FAILED")