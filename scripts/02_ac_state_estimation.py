"""AC state estimation on the frozen baseline measurements (no attack).

Run from anywhere:  python scripts/02_ac_state_estimation.py   (after 01_generate_measurements.py)
Writes results/ac_se_estimate.csv: true and estimated Vm and Va for every bus.
Self-checks run first; the script exits non-zero if one fails or the estimator does not converge.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from ac_state_estimation import (build_model, estimate_ac, h_of_x, jacobian, load_measurements,
                                 zero_injection_rows)
from common import check

m = build_model()
meas = load_measurements(config.MEASUREMENTS_CSV)
nb = m["nb"]
Vm0, Va0 = m["Vm_true"], m["Va_true"]
ok = []

print("Self-checks")
ok.append(check(np.max(np.abs(h_of_x(m, meas, Vm0, Va0) - meas["true"].to_numpy())) < 1e-8,
                "frozen measurements match h(x_true) of the current power flow"))  # fails if the data file and the model (or library versions) have drifted apart
H = jacobian(m, meas, Vm0, Va0).toarray()  # Jacobian against finite differences
nonref = np.delete(np.arange(nb), m["ref"])
h0, Hfd = h_of_x(m, meas, Vm0, Va0), np.zeros_like(H)
for k in range(H.shape[1]):
    Vm1, Va1 = Vm0.copy(), Va0.copy()
    if k < nb - 1:
        Va1[nonref[k]] += config.JACOBIAN_FD_STEP
    else:
        Vm1[k - (nb - 1)] += config.JACOBIAN_FD_STEP
    Hfd[:, k] = (h_of_x(m, meas, Vm1, Va1) - h0) / config.JACOBIAN_FD_STEP
ok.append(check(np.abs(H - Hfd).max() < config.JACOBIAN_FD_TOL,
                f"analytic Jacobian = finite differences (max error {np.abs(H - Hfd).max():.1e})"))
clean = estimate_ac(m, meas.assign(value=meas["true"]))
err = max(np.abs(clean["Vm"] - Vm0).max(), np.abs(clean["Va"] - Va0).max())
ok.append(check(err < config.RECOVERY_TOL, f"noise-free data recovers the true state (max error {err:.1e})"))  # with e = 0 the WLS solution is exactly x_true

est = estimate_ac(m, meas)  # the actual result
print(f"\nAC estimator: {est['n_meas']} measurements, {est['n_states']} states, "
      f"{est['n_constraints']} constraints, converged in {est['iterations']} iterations")
print("max |dx| per iteration:", [f"{s:.1e}" for s in est["step_norms"]])
cons = zero_injection_rows(m)
viol = np.abs(h_of_x(m, cons, est["Vm"], est["Va"])).max()
ok.append(check(viol < config.CONSTRAINT_TOL, f"zero-injection constraints hold at the estimate ({viol:.1e} p.u.)"))

out = pd.DataFrame({"bus": m["bus_numbers"], "Vm_true": Vm0, "Vm_est": est["Vm"],
                    "Va_true_deg": np.degrees(Va0), "Va_est_deg": np.degrees(est["Va"])})
out["Vm_err"] = out["Vm_est"] - out["Vm_true"]
out["Va_err_deg"] = out["Va_est_deg"] - out["Va_true_deg"]
config.RESULTS_DIR.mkdir(exist_ok=True)
out.to_csv(config.RESULTS_DIR / "ac_se_estimate.csv", index=False)
print(f"max |Vm error| = {out['Vm_err'].abs().max():.4f} p.u.;  max |Va error| = {out['Va_err_deg'].abs().max():.3f} deg")
print(f"wrote {config.RESULTS_DIR / 'ac_se_estimate.csv'}")
if not all(ok):
    sys.exit("some checks FAILED")
