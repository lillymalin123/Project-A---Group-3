"""
dc_state_estimation.py --- DC state estimation of the IEEE 57-bus case on the DC subset of the AC measurements.

Library only: no printing, no file writing. Works on the same measurement table as ac_state_estimation.py
(columns type, element, side, value, sigma), so both estimators see the SAME noise draw (project brief, Sec. 4).

DC measurement subset (what the DC model can explain):
  P_flow  active flow at both ends of every branch      (2 * nl = 160 rows)
  P_inj   active injection at buses with load or gen     (42 rows)
  Va      voltage angle at every bus                     (57 rows)        -> m_DC = 259
Q flows, Q injections and Vm are dropped: the DC model has no reactive power and fixes |V| = 1.

DC model (Lectures 3 and 7), linear in the state theta:
  P_flow, from end of branch l = b_l (theta_f - theta_t),   b_l = 1 / (x_l * tap_l)   (m['x_dc'])
  P_flow, to end              = - (from-end flow)            (lossless)
  P_inj at bus i              = sum of the flows leaving bus i  ->  Bbus theta
  Va at bus i                 = theta_i
Resistances, line charging, shunts and losses are ignored. x * tap equals the 'x' in data/case57-se.json, so
this H is the Lecture 7 model, only with the rows of the Hug & Giampapa measurement set.

Conventions follow ac_state_estimation.py: 0-based positions (position k = bus number k + 1), pandapower
branch order, p.u. on baseMVA, radians. State theta = angles of every bus except the reference
(theta_ref = 0), length nb - 1, columns in position order with the reference column removed.

Estimator: weighted least squares with the zero-injection constraints P_i = 0 (C theta = 0) at buses with
neither load nor generation, solved in one step through the KKT system (the model is linear, no iteration)
        [ H'WH   C' ] [theta]   [ H'W z ]
        [  C     0  ] [ mu  ] = [   0   ]          W = diag(1 / sigma^2).
The result dict has the same keys as estimate_ac, so ac_state_estimation.chi_square_test,
degrees_of_freedom, residual_variances and normalized_residuals work on it unchanged.
Degrees of freedom: m_DC - (nb - 1) + n_zero_injection = 259 - 56 + 15 = 218.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DC_TYPES = ("P_flow", "P_inj", "Va")  # measurement types the DC estimator uses


# ----------------------------------------------------------------------------- measurement subset
def dc_mask(meas):
    """Boolean array over the rows of an AC measurement table: True for the rows the DC estimator uses."""
    return meas["type"].isin(DC_TYPES).to_numpy()


def dc_subset(meas):
    """The DC rows of an AC measurement table (P flows, P injections, angles), index reset to 0..m_DC-1."""
    return meas[dc_mask(meas)].reset_index(drop=True)


def embed_dc_vector(meas, a_dc):
    """Place a vector on the DC rows (e.g. a DC attack a = H_DC c) into a full-length vector over all rows
    of the AC table, with zeros on the rows the DC model does not see (Q flows, Q injections, Vm)."""
    mask = dc_mask(meas)
    a_dc = np.asarray(a_dc, dtype=float)
    if a_dc.shape != (mask.sum(),):
        raise ValueError(f"expected {mask.sum()} DC entries, got shape {a_dc.shape}")
    out = np.zeros(len(meas))
    out[mask] = a_dc
    return out


# ----------------------------------------------------------------------------- DC model
def _dc_full_matrices(m):
    """Bf (nl x nb): from-end flows Bf theta_all; Bbus (nb x nb): injections Bbus theta_all (all buses)."""
    b = 1.0 / m["x_dc"]
    A = (m["Cf"] - m["Ct"]).tocsr()  # branch-bus incidence: +1 at the from bus, -1 at the to bus
    Bf = (sp.diags(b) @ A).tocsr()
    Bbus = (A.T @ Bf).tocsr()        # injection = sum of flows leaving the bus
    return Bf, Bbus


def _non_ref_columns(m):
    return np.delete(np.arange(m["nb"]), m["ref"])


def dc_measurement_matrix(m, meas):
    """H_DC, shape (len(meas), nb - 1), sparse. One row per row of ``meas``, which must contain only DC
    types (use dc_subset). Columns: angles of every bus except the reference, in position order."""
    bad = ~meas["type"].isin(DC_TYPES)
    if bad.any():
        raise ValueError(f"non-DC measurement types in the table: {sorted(set(meas['type'][bad]))}")
    nl, nb = m["nl"], m["nb"]
    Bf, Bbus = _dc_full_matrices(m)
    cand = sp.vstack([Bf, -Bf, Bbus, sp.identity(nb, format="csr")], format="csr")  # [Pf; Pt; Pinj; Va]
    off = {("P_flow", "from"): 0, ("P_flow", "to"): nl, ("P_inj", ""): 2 * nl, ("Va", ""): 2 * nl + nb}
    rows = np.array([off[(ty, s)] + e for ty, s, e in zip(meas["type"], meas["side"], meas["element"])])
    return cand[rows][:, _non_ref_columns(m)].tocsr()


def dc_zero_injection_matrix(m):
    """C_DC, shape (n_zero_injection, nb - 1): the constraints P_i = C theta = 0 at zero-injection buses."""
    _, Bbus = _dc_full_matrices(m)
    return Bbus[np.flatnonzero(m["zero_inj"])][:, _non_ref_columns(m)].tocsr()


# ----------------------------------------------------------------------------- the estimator
def estimate_dc(m, meas, enforce_zero_injection=True):
    """Weighted-least-squares DC state estimate (one linear solve, reference angle = 0).

    ``meas``: DC measurement table (output of dc_subset, or an attacked copy of it) with columns type,
    element, side, value, sigma. Returns a dict with the keys of estimate_ac where they make sense:
      Va [rad] (length nb, reference = 0), theta [rad] (length nb - 1, the state),
      residual (z - H theta_hat), H (H_DC, sparse), C (constraint matrix, sparse, possibly 0 rows),
      weights (1/sigma^2), n_meas, n_states (= nb - 1), n_constraints, iterations (= 1).
    Raises RuntimeError if the gain / KKT matrix is singular (measurement set not observable).
    """
    H = dc_measurement_matrix(m, meas)
    z = meas["value"].to_numpy(float)
    w = 1.0 / meas["sigma"].to_numpy(float) ** 2
    n = m["nb"] - 1
    C = dc_zero_injection_matrix(m) if enforce_zero_injection else sp.csr_matrix((0, n))
    G, g = (H.T @ sp.diags(w) @ H).tocsc(), H.T @ (w * z)
    if C.shape[0]:
        K, rhs = sp.bmat([[G, C.T], [C, None]], format="csc"), np.concatenate([g, np.zeros(C.shape[0])])
    else:
        K, rhs = G, g
    sol = spla.spsolve(K, rhs)
    if not np.all(np.isfinite(sol)):
        raise RuntimeError("singular DC gain matrix: measurement set not observable?")
    theta = sol[:n]
    Va = np.zeros(m["nb"])
    Va[_non_ref_columns(m)] = theta
    return {"Va": Va, "theta": theta, "iterations": 1,
            "residual": z - H @ theta, "H": H, "C": C, "weights": w,
            "n_meas": len(z), "n_states": n, "n_constraints": C.shape[0]}
