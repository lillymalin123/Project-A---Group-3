"""
ac_state_estimation.py --- functions for AC state estimation of the IEEE 57-bus case.

Library only: no printing of results, no file writing except save_measurements. The experiments that
call these functions are scripts/01_generate_measurements.py, scripts/02_ac_state_estimation.py and
scripts/03_ac_bad_data.py (bad-data detection: chi-square test and normalized residuals).

Method (Hug & Giampapa 2012, Sec. IV-A): weighted least squares  min (z - h(x))' W (z - h(x)),
W = diag(1/sigma^2), solved by Gauss-Newton on the normal equations (H'WH) dx = H'W r. Zero active and
reactive injection at zero-injection buses is enforced as an equality constraint c(x) = 0 through the
KKT system
        [ H'WH   C' ] [dx]   [ H'W r ]
        [  C     0  ] [mu] = [  -c   ]          r = z - h(x_k),   x_{k+1} = x_k + dx.

Conventions
  * the network model comes from loadCase.load_case57() and pandapower's power flow;
  * indices are 0-BASED POSITIONS: position k = bus number k+1 (bus 2 -> position 1);
    'bus_numbers' in the model maps positions back to bus numbers;
  * branch order is pandapower's (65 lines, then 15 transformers), NOT the row order of mpc.branch;
  * power in p.u. on baseMVA, voltage magnitude in p.u., angles in RADIANS;
  * state x = [angle of every bus except the reference ; Vm of every bus], length 2*nb - 1.
"""
import warnings

import numpy as np
import pandapower as pp
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import chi2

import config
from loadCase import load_case57

warnings.filterwarnings("ignore", message=".*numba.*")


# ----------------------------------------------------------------------------- network model
def build_model(net=None):
    """Solve the AC power flow and return the network model as a dict.

    Output keys: nb, nl (numbers of buses and branches); bus_numbers (bus number of each position);
    Ybus (nb x nb), Yf, Yt (nl x nb) complex admittance matrices; Cf, Ct (nl x nb) branch-bus
    connection matrices; f, t (end positions of each branch); ref (position of the slack bus);
    zero_inj (bool per bus: no load and no generator); Vm_true [p.u.], Va_true [rad] (power-flow state).
    Reads pandapower's internal solution (net._ppc), tested with pandapower 3.5.6.
    """
    net = load_case57() if net is None else net
    pp.runpp(net)
    ppc = net._ppc
    Y = ppc["internal"]
    first = net.bus.index.min()
    # position k must be the k-th bus in pandapower's internal ordering (true for this case)
    assert np.array_equal(net._pd2ppc_lookups["bus"][net.bus.index.to_numpy()], np.arange(len(net.bus))), \
        "unexpected pandapower bus ordering"
    nb, nl = len(net.bus), ppc["branch"].shape[0]
    f = ppc["branch"][:, 0].real.astype(int)
    t = ppc["branch"][:, 1].real.astype(int)
    rows = np.arange(nl)
    has_inj = np.zeros(nb, dtype=bool)  # bus with a load or a generator (shunts do not count)
    for tbl in (net.load, net.gen, net.ext_grid):
        has_inj[tbl.bus.to_numpy() - first] = True
    return {
        "nb": nb, "nl": nl, "bus_numbers": net.bus.index.to_numpy(),
        "Ybus": Y["Ybus"].tocsr(), "Yf": Y["Yf"].tocsr(), "Yt": Y["Yt"].tocsr(),
        "Cf": sp.csr_matrix((np.ones(nl), (rows, f)), shape=(nl, nb)),
        "Ct": sp.csr_matrix((np.ones(nl), (rows, t)), shape=(nl, nb)),
        "f": f, "t": t,
        "ref": int(net._pd2ppc_lookups["bus"][net.ext_grid.bus.iloc[0]]),
        "zero_inj": ~has_inj,
        "Vm_true": net.res_bus.vm_pu.to_numpy(), "Va_true": np.deg2rad(net.res_bus.va_degree.to_numpy()),
    }


# ----------------------------------------------------------------------------- measurement function h(x)
def _voltage(Vm, Va):
    return Vm * np.exp(1j * Va)


def _dSbus_dV(m, V):
    """Derivatives of the complex bus injections w.r.t. Va and Vm (MATPOWER dSbus_dV)."""
    Ibus = m["Ybus"] @ V
    dV, dI, dVn = sp.diags(V), sp.diags(Ibus), sp.diags(V / np.abs(V))
    dS_dVm = dV @ np.conj(m["Ybus"] @ dVn) + np.conj(dI) @ dVn
    dS_dVa = 1j * dV @ np.conj(dI - m["Ybus"] @ dV)
    return dS_dVa.tocsr(), dS_dVm.tocsr()


def _dSbr_dV(m, V):
    """Derivatives of the from-end and to-end branch flows w.r.t. Va and Vm (MATPOWER dSbr_dV)."""
    dV, dVn, out = sp.diags(V), sp.diags(V / np.abs(V)), []
    for Y, C in ((m["Yf"], m["Cf"]), (m["Yt"], m["Ct"])):
        dI, dVb = sp.diags(Y @ V), sp.diags(C @ V)
        out.append((1j * (np.conj(dI) @ C @ dV - dVb @ np.conj(Y @ dV))).tocsr())
        out.append((dVb @ np.conj(Y @ dVn) + np.conj(dI) @ C @ dVn).tocsr())
    return out  # Sf_dVa, Sf_dVm, St_dVa, St_dVm


def _stack_rows(m, meas):
    """Row of each measurement in the stacked candidate vector [Pf, Pt, Qf, Qt, Pinj, Qinj, Vm, Va]."""
    nl, nb = m["nl"], m["nb"]
    off = {("P_flow", "from"): 0, ("P_flow", "to"): nl, ("Q_flow", "from"): 2 * nl, ("Q_flow", "to"): 3 * nl,
           ("P_inj", ""): 4 * nl, ("Q_inj", ""): 4 * nl + nb, ("Vm", ""): 4 * nl + 2 * nb,
           ("Va", ""): 4 * nl + 3 * nb}
    return np.array([off[(ty, s)] + e for ty, s, e in zip(meas["type"], meas["side"], meas["element"])])


def h_of_x(m, meas, Vm, Va):
    """h(x): model value of each measurement row in ``meas`` at the state (Vm [p.u.], Va [rad])."""
    V = _voltage(Vm, Va)
    S = V * np.conj(m["Ybus"] @ V)
    Sf = (m["Cf"] @ V) * np.conj(m["Yf"] @ V)
    St = (m["Ct"] @ V) * np.conj(m["Yt"] @ V)
    cand = np.concatenate([Sf.real, St.real, Sf.imag, St.imag, S.real, S.imag, np.abs(V), np.angle(V)])
    return cand[_stack_rows(m, meas)]


def jacobian(m, meas, Vm, Va):
    """H(x) = dh/dx, shape (n_meas, 2*nb-1). Columns: angles except the reference, then all Vm."""
    nb, V = m["nb"], _voltage(Vm, Va)
    S_a, S_m = _dSbus_dV(m, V)
    Sf_a, Sf_m, St_a, St_m = _dSbr_dV(m, V)
    I, Z = sp.identity(nb, format="csr"), sp.csr_matrix((nb, nb))
    full = sp.bmat([[Sf_a.real, Sf_m.real], [St_a.real, St_m.real], [Sf_a.imag, Sf_m.imag],
                    [St_a.imag, St_m.imag], [S_a.real, S_m.real], [S_a.imag, S_m.imag],
                    [Z, I], [I, Z]], format="csr")
    cols = np.concatenate([np.delete(np.arange(nb), m["ref"]), nb + np.arange(nb)])
    return full[_stack_rows(m, meas)][:, cols].tocsr()


def zero_injection_rows(m):
    """Measurement-style table of the constraints P = Q = 0 at zero-injection buses."""
    zi = np.flatnonzero(m["zero_inj"])
    return pd.DataFrame({"type": ["P_inj"] * len(zi) + ["Q_inj"] * len(zi), "side": "",
                         "element": np.concatenate([zi, zi])})


# ----------------------------------------------------------------------------- measurements
def make_measurements(m, rng):
    """The 518-measurement set of Hug & Giampapa (2012) with seeded Gaussian noise.

    P and Q flows at both ends of every branch, P and Q injections at buses with load or generation,
    Vm and Va at every bus. ``rng`` is a numpy Generator. Returns a DataFrame with columns
    type, element (0-based branch or bus position), side ('from'/'to'/''), true, sigma, value
    (all p.u. / rad); 'true' is h(x_true), 'value' = true + N(0, sigma^2).
    """
    rows = [(ty, l, sd) for ty in ("P_flow", "Q_flow") for sd in ("from", "to") for l in range(m["nl"])]
    inj = np.flatnonzero(~m["zero_inj"])
    rows += [(ty, int(b), "") for ty in ("P_inj", "Q_inj") for b in inj]
    rows += [(ty, b, "") for ty in ("Vm", "Va") for b in range(m["nb"])]
    df = pd.DataFrame(rows, columns=["type", "element", "side"])
    df["true"] = h_of_x(m, df, m["Vm_true"], m["Va_true"])
    sig = {"Vm": config.SIGMA_VMAG, "Va": config.SIGMA_ANGLE}
    df["sigma"] = [sig.get(ty, config.SIGMA_POWER) for ty in df["type"]]
    df["value"] = df["true"] + rng.normal(0.0, df["sigma"].to_numpy())
    return df


def save_measurements(df, path):
    """Write a measurement table to CSV at full float precision."""
    df.to_csv(path, index=False, float_format="%.17g")


def load_measurements(path):
    """Read a table written by save_measurements (an empty 'side' stays '' instead of becoming NaN)."""
    return pd.read_csv(path, keep_default_na=False, dtype={"side": str})


# ----------------------------------------------------------------------------- the estimator
def estimate_ac(m, meas, enforce_zero_injection=True, tol=config.NEWTON_TOL, max_iter=config.NEWTON_MAX_ITER):
    """Weighted-least-squares AC state estimate (Gauss-Newton, flat start, reference angle = 0).

    ``meas``: measurement table with columns type, element, side, value, sigma (a subset or an attacked
    copy is fine). Returns a dict: Vm [p.u.], Va [rad] (length nb), iterations, step_norms (max |dx| per
    iteration), residual (z - h(x_hat)), H (Jacobian at x_hat), C (constraint Jacobian at x_hat),
    weights (1/sigma^2), n_meas, n_states, n_constraints.
    Raises RuntimeError if the iteration does not converge or the gain matrix is singular.
    """
    nb = m["nb"]
    z = meas["value"].to_numpy(float)
    w = 1.0 / meas["sigma"].to_numpy(float) ** 2
    W = sp.diags(w)
    Vm, Va = np.ones(nb), np.zeros(nb)
    cons = zero_injection_rows(m) if enforce_zero_injection else zero_injection_rows(m).iloc[0:0]
    not_ref = np.arange(nb) != m["ref"]
    steps = []
    for it in range(1, max_iter + 1):
        H = jacobian(m, meas, Vm, Va)
        r = z - h_of_x(m, meas, Vm, Va)
        G, g = (H.T @ W @ H).tocsc(), H.T @ (w * r)
        if len(cons):
            C, c = jacobian(m, cons, Vm, Va), h_of_x(m, cons, Vm, Va)
            K, rhs = sp.bmat([[G, C.T], [C, None]], format="csc"), np.concatenate([g, -c])
        else:
            K, rhs = G, g
        sol = spla.spsolve(K, rhs)
        if not np.all(np.isfinite(sol)):
            raise RuntimeError("singular gain matrix: measurement set not observable?")
        dx = sol[: 2 * nb - 1]
        Va[not_ref] += dx[: nb - 1]
        Vm += dx[nb - 1:]
        steps.append(float(np.max(np.abs(dx))))
        if steps[-1] < tol:
            break
    else:
        raise RuntimeError(f"AC estimator did not converge in {max_iter} iterations (last step {steps[-1]:.2e})")
    C = jacobian(m, cons, Vm, Va) if len(cons) else sp.csr_matrix((0, 2 * nb - 1))
    return {"Vm": Vm, "Va": Va, "iterations": it, "step_norms": steps,
            "residual": z - h_of_x(m, meas, Vm, Va), "H": jacobian(m, meas, Vm, Va), "C": C, "weights": w,
            "n_meas": len(z), "n_states": 2 * nb - 1, "n_constraints": len(cons)}


# ----------------------------------------------------------------------------- bad-data detection
def degrees_of_freedom(est):
    """Degrees of freedom of the chi-square test: measurements - states + independent constraints."""
    return est["n_meas"] - est["n_states"] + est["n_constraints"]


def chi_square_test(est, alpha=config.ALPHA):
    """Chi-square bad-data test on an ``estimate_ac`` result.

    Statistic J = r' W r (dimensionless). Under the model and no attack J is approximately
    chi-square with ``degrees_of_freedom`` degrees of freedom (exact only for a linear model).
    Returns a dict: J, dof, tau (threshold = (1 - alpha) quantile), p_value, alarm (True if J > tau).
    """
    J = float(est["residual"] @ (est["weights"] * est["residual"]))
    dof = degrees_of_freedom(est)
    tau = float(chi2.ppf(1.0 - alpha, dof))
    return {"J": J, "dof": dof, "tau": tau, "p_value": float(chi2.sf(J, dof)), "alarm": bool(J > tau)}


def residual_variances(est):
    """Variance of each residual, diag(Omega) with Omega = R - H N H', linearised at the estimate.

    N = G^-1 - G^-1 C' (C G^-1 C')^-1 C G^-1 is the covariance of the constrained estimate,
    G = H' W H, R = diag(sigma^2). Returns a vector in (p.u.^2 or rad^2), one entry per measurement.
    """
    H, C, w = est["H"].toarray(), est["C"].toarray(), est["weights"]
    Ginv = np.linalg.inv(H.T @ (w[:, None] * H))
    N = Ginv
    if C.shape[0]:
        N = Ginv - Ginv @ C.T @ np.linalg.solve(C @ Ginv @ C.T, C @ Ginv)
    return 1.0 / w - np.einsum("ij,jk,ik->i", H, N, H)


def normalized_residuals(est, critical_fraction=config.CRITICAL_VAR_FRACTION):
    """|r_i| / sqrt(Omega_ii) for every measurement (dimensionless, ~N(0,1) per entry without bad data).

    Entries whose residual variance is below ``critical_fraction`` of sigma_i^2 are critical
    measurements (their residual carries no information) and are returned as NaN.
    """
    var = residual_variances(est)
    crit = var < critical_fraction / est["weights"]
    out = np.abs(est["residual"]) / np.sqrt(np.where(crit, 1.0, var))
    out[crit] = np.nan
    return out
