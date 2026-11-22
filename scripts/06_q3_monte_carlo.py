"""Q3 --- Monte Carlo of the chi-square test WITHOUT attack: about 2000 noise draws per sigma, AC and DC.

Run from anywhere:  python scripts/06_q3_monte_carlo.py   (after 01_generate_measurements.py)
Run time: roughly 10 minutes (about 60 ms per draw for both estimators).

For every sigma in config.MC_SIGMAS and every draw k = 1..config.MC_N_DRAWS:
  z = h(x_true) + N(0, sigma^2)   on all 518 measurements (the 'true' column of the frozen baseline table),
  AC estimator on all 518 rows, DC estimator on its 259 rows of the SAME draw (project brief, Sec. 4),
  record J_AC, J_DC, convergence, iterations and the AC estimation error, and
  J_lin_AC = the statistic of the AC estimator LINEARISED at x_true for the same noise (exactly chi2(dof));
  J_AC - J_lin_AC isolates the effect of the nonlinearity of h on the test statistic.
Random numbers: numpy SeedSequence(config.MC_SEED).spawn(len(MC_SIGMAS)), one independent child stream per
sigma, so each sigma is reproducible on its own. The frozen baseline draw (config.SEED) is not touched.
Non-convergent AC runs (estimate_ac raises RuntimeError) are KEPT in the file with converged_ac = False and
NaN AC entries; scripts/07 excludes them from the statistics and reports how many there were.

Writes (one script, one result; the statistics and figures are scripts/07_q3_chi2_analysis.py):
  results/q3_mc_draws.csv       one row per (sigma, draw)
  results/q3_mc_reference.csv   one row per sigma: dof and 95 % thresholds of both estimators, and the
                                DC noncentrality lambda (J of the DC estimator on noise-free AC data)
Exits non-zero if a self-check fails.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from ac_state_estimation import build_model, h_of_x, load_measurements
from common import check
from monte_carlo import linearised_residual_operator, noisy_copy, reference_values, run_draw

m = build_model()
meas = load_measurements(config.MEASUREMENTS_CSV)
S_lin = linearised_residual_operator(m, meas)  # built once: independent of sigma

# --- reference values (no randomness) and self-checks ----------------------------------------------
ref = pd.DataFrame([{"sigma": s, **reference_values(m, meas, s, config.ALPHA)} for s in config.MC_SIGMAS])
print("Self-checks")
ok = [
    check(np.max(np.abs(h_of_x(m, meas, m["Vm_true"], m["Va_true"]) - meas["true"].to_numpy())) < 1e-8,
          "the 'true' column the draws are built on equals h(x_true) of the current power flow"),
    check(ref["dof_ac"].nunique() == 1 and ref["dof_dc"].nunique() == 1,
          f"dof does not depend on sigma: AC {ref['dof_ac'].iloc[0]} (= 518 - 113 + 30), "
          f"DC {ref['dof_dc'].iloc[0]} (= 259 - 56 + 15)"),
    check(np.allclose(ref["lambda_dc"] * ref["sigma"] ** 2, ref["lambda_dc"].iloc[0] * ref["sigma"].iloc[0] ** 2,
                      rtol=1e-9),
          "DC noncentrality scales as 1/sigma^2 (the mismatch is fixed, only the weights change): lambda = "
          + ", ".join(f"{l:.1f}" for l in ref["lambda_dc"])),
    check(abs(np.trace(S_lin) - ref["dof_ac"].iloc[0]) < config.DOF_CHECK_TOL * ref["dof_ac"].iloc[0]
          and np.allclose(S_lin @ S_lin, S_lin, atol=1e-8),
          f"linearised residual operator is a projection (S S = S) with trace {np.trace(S_lin):.3f} = dof_AC, "
          "so J_lin_AC is exactly chi2(dof)"),
]
if not all(ok):
    sys.exit("some checks FAILED")

# --- the draws ----------------------------------------------------------------------------------------
children = np.random.SeedSequence(config.MC_SEED).spawn(len(config.MC_SIGMAS))
rows = []
for sigma, child in zip(config.MC_SIGMAS, children):
    rng = np.random.default_rng(child)
    t0 = time.perf_counter()
    for k in range(1, config.MC_N_DRAWS + 1):
        rows.append({"sigma": sigma, "draw": k, **run_draw(m, noisy_copy(meas, sigma, rng), S_lin)})
    block = pd.DataFrame(rows[-config.MC_N_DRAWS:])
    print(f"sigma = {sigma}: {config.MC_N_DRAWS} draws in {time.perf_counter() - t0:.0f} s, "
          f"{int((~block['converged_ac']).sum())} non-convergent AC run(s), "
          f"mean J_AC = {block['J_ac'].mean():.1f}, mean J_DC = {block['J_dc'].mean():.1f}, "
          f"std(J_AC - J_lin_AC) = {(block['J_ac'] - block['J_lin_ac']).std():.3f}")

draws = pd.DataFrame(rows)
ref.insert(1, "n_draws", config.MC_N_DRAWS)
ref.insert(2, "seed", config.MC_SEED)
ref.insert(3, "alpha", config.ALPHA)
config.RESULTS_DIR.mkdir(exist_ok=True)
draws.to_csv(config.MC_DRAWS_CSV, index=False, float_format="%.10g")
ref.to_csv(config.MC_REFERENCE_CSV, index=False, float_format="%.10g")
print(f"wrote {config.MC_DRAWS_CSV} ({len(draws)} rows) and {config.MC_REFERENCE_CSV}")
