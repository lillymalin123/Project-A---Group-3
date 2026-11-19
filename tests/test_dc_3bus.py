"""Check src/dc_state_estimation.py against the 3-bus DC state-estimation answer key (course solution sheet).

Run from anywhere:  python tests/test_dc_3bus.py        Exits non-zero if a check fails.

Answer-key setup: buses 1, 2, 3 with bus 1 the slack (theta_1 = 0), state x = [theta_2, theta_3];
reactances x12 = 0.25, x13 = 0.5, x23 = 0.5 p.u.; m = 4 meters P12, P13, P23, P2 (injection),
all with sigma = 0.01. The sheet's numbers are printed rounded, so each comparison allows half a unit
in the last printed digit. The network is built by hand as a model dict with the keys estimate_dc reads,
so no pandapower is needed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ac_state_estimation import chi_square_test, normalized_residuals, residual_variances
from common import check
from dc_state_estimation import dc_measurement_matrix, estimate_dc

# ----------------------------------------------------------------------------- 3-bus network (0-based)
f, t = np.array([0, 0, 1]), np.array([1, 2, 2])  # branches 1-2, 1-3, 2-3
nb, nl = 3, 3
rows = np.arange(nl)
model = {
    "nb": nb, "nl": nl, "ref": 0, "f": f, "t": t,
    "x_dc": np.array([0.25, 0.5, 0.5]),
    "Cf": sp.csr_matrix((np.ones(nl), (rows, f)), shape=(nl, nb)),
    "Ct": sp.csr_matrix((np.ones(nl), (rows, t)), shape=(nl, nb)),
    "zero_inj": np.zeros(nb, dtype=bool),  # the sheet has no zero-injection constraint
}
SIGMA = 0.01
meas = pd.DataFrame({"type": ["P_flow", "P_flow", "P_flow", "P_inj"],   # P12, P13, P23, P2
                     "side": ["from", "from", "from", ""],
                     "element": [0, 1, 2, 1],
                     "sigma": SIGMA})
Z_GOOD = np.array([0.205, 0.195, 0.11, -0.09])
Z_BAD = Z_GOOD + np.array([0.0, 0.0, 0.0, 0.10])  # P2 corrupted by +0.10


def close(a, b, decimals):
    """True if a matches the sheet value b, printed with `decimals` digits."""
    return bool(np.all(np.abs(np.asarray(a) - np.asarray(b)) <= 0.5 * 10.0 ** -decimals + 1e-12))


ok = []
print("Measurement matrix")
H = dc_measurement_matrix(model, meas).toarray()
H_sheet = np.array([[-4.0, 0.0], [0.0, -2.0], [2.0, -2.0], [6.0, -2.0]])
ok.append(check(np.allclose(H, H_sheet), f"H_DC equals the sheet's H\n{H}"))
ok.append(check(close(H @ [-0.05, -0.10], [0.2, 0.2, 0.1, -0.1], 4), "z_clean = H x_true = [0.2, 0.2, 0.1, -0.1]"))

print("\nCase 1: good data")
est = estimate_dc(model, meas.assign(value=Z_GOOD))
Hd = est["H"].toarray()
G = Hd.T @ (est["weights"][:, None] * Hd)  # gain matrix H'WH
test = chi_square_test(est)
ok += [
    check(np.allclose(G, [[560000.0, -160000.0], [-160000.0, 120000.0]]), "gain G = H'WH matches the sheet"),
    check(close(est["theta"], [-0.04942, -0.10173], 5), f"x_hat = {np.round(est['theta'], 5)} rad"),
    check(close(est["residual"], [0.0073, -0.0085, 0.0054, 0.0031], 4), f"r = {np.round(est['residual'], 4)}"),
    check(close(test["J"], 1.6346, 4) and test["dof"] == 2, f"J = {test['J']:.4f}, dof = {test['dof']}"),
    check(close(np.sqrt(np.diag(np.linalg.inv(G))), [0.0017, 0.00367], 5), "std(theta_2, theta_3) = sqrt(diag G^-1)"),
    check(close(test["tau"], 5.991, 3) and not test["alarm"], f"threshold {test['tau']:.3f}: no alarm"),
]

print("\nCase 2: bad data on P2")
bad = estimate_dc(model, meas.assign(value=Z_BAD))
test_b = chi_square_test(bad)
rn = normalized_residuals(bad)
ok += [
    check(close(bad["theta"], [-0.03981, -0.10558], 5), f"x_hat = {np.round(bad['theta'], 5)} rad"),
    check(close(bad["residual"], [0.0458, -0.0162, -0.0215, 0.0377], 4), f"r = {np.round(bad['residual'], 4)}"),
    check(close(test_b["J"], 42.4038, 4) and test_b["alarm"], f"J = {test_b['J']:.4f} > threshold: alarm"),
    check(close(residual_variances(bad), [5.4e-05, 4.6e-05, 6.5e-05, 3.5e-05], 6), "Omega_ii = diag(R - H G^-1 H')"),
    check(close(rn, [6.237, 2.378, 2.664, 6.406], 3), f"normalized residuals = {np.round(rn, 3)}"),
]

print("\nBeyond the sheet: stealthy attack a = H c (Lecture 4)")
c = np.array([0.01, 0.0])  # shift theta_2 by 0.01 rad
att = estimate_dc(model, meas.assign(value=Z_GOOD + H @ c))
ok += [
    check(np.allclose(att["residual"], est["residual"]), "residual unchanged by a = H c"),
    check(np.allclose(att["theta"] - est["theta"], c), "estimate shifted by exactly c"),
]

if not all(ok):
    sys.exit("some checks FAILED")
print("\nall checks passed")
