"""Q5 --- sweep of the attack magnitude for three attack constructions against the AC estimator.

Run from anywhere:  python scripts/06_q5_attack_sweep.py   (after 01_generate_measurements.py)

For each noise case (noise-free: value = true; noisy: the frozen baseline draw) the AC estimator is run once
without attack to get the attacker's x_hat, J(x_hat) and the chi-square statistic J0. Then, for every magnitude
t in config.ATTACK_MAGNITUDES and the same direction c = t * angle_shift(m, config.ATTACK_BUS):
  DC         a = H_DC c on the DC rows, zero elsewhere     (attacks.build_dc_attack)
  Jacobian   a = J(x_hat) [c; 0]                           (attacks.build_ac_jacobian_attack)
  nonlinear  a = h(x_hat + [c; 0]) - h(x_hat)              (attacks.build_ac_nonlinear_attack)
and the AC estimator is re-run on z + a.

Writes results/q5_attack_sweep.csv, one row per (noise, attack, t): J0, J, dJ = J - J0, dof, tau, alarm,
largest normalized residual, estimated angle change at ATTACK_BUS, converged, iterations, last step.
A non-convergent run is a RESULT (converged = False, statistics NaN), not an error.
Self-check: noise-free, the nonlinear attack must leave J unchanged and move the estimate by exactly t;
the script exits non-zero if it does not.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from ac_state_estimation import (build_model, chi_square_test, estimate_ac, load_measurements,
                                 normalized_residuals)
from attacks import (ac_state_shift, angle_shift, build_ac_jacobian_attack, build_ac_nonlinear_attack,
                     build_dc_attack)
from common import check
from dc_state_estimation import dc_measurement_matrix, dc_subset, dc_zero_injection_matrix, embed_dc_vector

m = build_model()
noisy = load_measurements(config.MEASUREMENTS_CSV)
cases = {"noise-free": noisy.assign(value=noisy["true"]), "noisy": noisy}
H_dc, C_dc = dc_measurement_matrix(m, dc_subset(noisy)), dc_zero_injection_matrix(m)  # network data only
c_dir = angle_shift(m, config.ATTACK_BUS)
pos = config.ATTACK_BUS - 1  # 0-based position of the attacked bus

rows = []
for noise, meas in cases.items():
    est0 = estimate_ac(m, meas)  # the attacker's x_hat: the estimate on the unattacked data
    J0 = chi_square_test(est0)["J"]
    builders = {
        "DC": lambda c: embed_dc_vector(meas, build_dc_attack(H_dc, c, C_dc)),
        "Jacobian": lambda c: build_ac_jacobian_attack(est0["H"], ac_state_shift(m, c), est0["C"]),
        "nonlinear": lambda c: build_ac_nonlinear_attack(m, meas, est0["Vm"], est0["Va"],
                                                         ac_state_shift(m, c)),
    }
    print(f"\n{noise}: J0 = {J0:.4g}")
    for attack, build in builders.items():
        for t in config.ATTACK_MAGNITUDES:
            row = {"noise": noise, "attack": attack, "t_rad": t, "J0": J0}
            try:
                est1 = estimate_ac(m, meas.assign(value=meas["value"] + build(t * c_dir)))
            except RuntimeError as err:  # non-convergence is reported, not hidden (guideline rule 4)
                print(f"  {attack:9s} t = {t:.2e}: NOT CONVERGED ({err})")
                rows.append({**row, "converged": False})
                continue
            test = chi_square_test(est1)
            rows.append({**row, "converged": True, "iterations": est1["iterations"],
                         "last_step": est1["step_norms"][-1], "J": test["J"], "dJ": test["J"] - J0,
                         "dof": test["dof"], "tau": test["tau"], "alarm": test["alarm"],
                         "largest_norm_resid": float(np.nanmax(normalized_residuals(est1))),
                         "dVa_attack_bus_rad": est1["Va"][pos] - est0["Va"][pos]})
            print(f"  {attack:9s} t = {t:.2e}: dJ = {rows[-1]['dJ']:+.3e}  "
                  f"({est1['iterations']} it, last step {est1['step_norms'][-1]:.1e})")

out = pd.DataFrame(rows)
config.RESULTS_DIR.mkdir(exist_ok=True)
out.to_csv(config.RESULTS_DIR / "q5_attack_sweep.csv", index=False)
print(f"\nwrote {config.RESULTS_DIR / 'q5_attack_sweep.csv'} ({len(out)} rows, "
      f"{int((~out['converged']).sum())} non-convergent)")

# --- self-checks -------------------------------------------------------------------------------------
print("\nSelf-checks")
nl = out[(out["noise"] == "noise-free") & (out["attack"] == "nonlinear") & out["converged"]]
ok = [
    check(len(nl) == len(config.ATTACK_MAGNITUDES), "every noise-free nonlinear run converged"),
    check(nl["dJ"].abs().max() < config.SWEEP_DJ_TOL,
          f"noise-free nonlinear attack leaves J unchanged (max |dJ| = {nl['dJ'].abs().max():.1e})",
          "a = h(x_hat + c) - h(x_hat) is consistent with the AC model (Hug & Giampapa 2012, eq. 30)"),
    check(np.allclose(nl["dVa_attack_bus_rad"], nl["t_rad"], atol=config.RECOVERY_TOL),
          f"noise-free nonlinear attack moves the bus-{config.ATTACK_BUS} estimate by exactly t"),
]
if not all(ok):
    sys.exit("some checks FAILED")