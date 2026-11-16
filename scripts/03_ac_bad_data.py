"""Bad-data detection for the AC estimator on the frozen baseline measurements (no attack).

Run from anywhere:  python scripts/03_ac_bad_data.py   (after 01_generate_measurements.py)
Writes results/ac_bad_data.csv (one row): chi-square statistic, degrees of freedom, threshold, p-value,
alarm flag, largest normalized residual and which measurement it belongs to.
An ALARM here is a result, not an error: with no attack it is a false alarm and happens in about
alpha = 5 % of noise draws. The script exits non-zero only if a self-check fails.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from ac_state_estimation import (build_model, chi_square_test, degrees_of_freedom, estimate_ac,
                                 load_measurements, normalized_residuals, residual_variances)
from common import check

m = build_model()
meas = load_measurements(config.MEASUREMENTS_CSV)
est = estimate_ac(m, meas)

test = chi_square_test(est)
rn = normalized_residuals(est)
var = residual_variances(est)
w = est["weights"]
worst = int(np.nanargmax(rn))
above = int(np.nansum(rn > config.LNR_THRESHOLD))
n_crit = int(np.isnan(rn).sum())

print("Self-checks")
ok = [
    check(test["dof"] == est["n_meas"] - est["n_states"] + est["n_constraints"] == degrees_of_freedom(est),
          f"dof = m - n + c = {est['n_meas']} - {est['n_states']} + {est['n_constraints']} = {test['dof']}"),
    # linearised theory: E[J] = sum_i w_i * Omega_ii must equal the degrees of freedom
    check(abs(np.sum(w * var) - test["dof"]) < config.DOF_CHECK_TOL * test["dof"],
          f"sum(w_i * Omega_ii) = {np.sum(w * var):.3f} equals dof (residual covariance consistent with the dof)"),
    check(bool(np.all(var <= 1.0 / w + 1e-12)),
          "every residual variance is at most its measurement variance (Omega_ii <= sigma_i^2)"),
]

row = meas.iloc[worst]
print(f"\nChi-square test (alpha = {config.ALPHA}): J = {test['J']:.1f}, dof = {test['dof']}, "
      f"threshold = {test['tau']:.1f}, p = {test['p_value']:.3f} -> {'ALARM' if test['alarm'] else 'passes'}")
print(f"Largest normalized residual: {rn[worst]:.2f} ({row['type']} {row['side']} index {row['element']}); "
      f"{above} measurement(s) above {config.LNR_THRESHOLD}; {n_crit} critical measurement(s)")
out = pd.DataFrame([{
    "J": test["J"], "m": est["n_meas"], "n": est["n_states"], "c": est["n_constraints"], "dof": test["dof"],
    "alpha": config.ALPHA, "tau": test["tau"], "p_value": test["p_value"], "alarm": test["alarm"],
    "largest_norm_resid": rn[worst], "lnr_type": row["type"], "lnr_side": row["side"],
    "lnr_element_0based": int(row["element"]), "lnr_threshold": config.LNR_THRESHOLD,
    "n_above_lnr_threshold": above, "n_critical": n_crit,
}])
config.RESULTS_DIR.mkdir(exist_ok=True)
out.to_csv(config.RESULTS_DIR / "ac_bad_data.csv", index=False)
print(f"wrote {config.RESULTS_DIR / 'ac_bad_data.csv'}")
if not all(ok):
    sys.exit("some checks FAILED")
