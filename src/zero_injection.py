"""
zero_injection.py --- Q3: how zero-injection buses enlarge a hidden attack (Hug & Giampapa 2012, Sec. V).

Library only: no printing, no file writing. Called by scripts/08_q3_zero_injection.py.

Setting: the attacker wants the estimator to report the voltage angle of one "center" bus k shifted by t
(the single-bus shift of the brief, applied to every bus in turn). A shift of theta_k changes the flows on
every line at k and the injections at k and at its neighbours. A neighbour j WITH load or generation
simply gets its injection measurement falsified too. A neighbour j with ZERO injection cannot: its
injection is a known zero (an equality constraint in both estimators), so the flows into j must still sum
to zero. The attacker must then also shift the state of j, which changes j's other lines, and so on
through every chain of zero-injection buses, until the attack is bounded by buses with injection
(Hug & Giampapa, Sec. V-A, Conclusion 1, and the subgraph procedure of Sec. V-B, steps 6a-b).

  Z(k)  = zero-injection buses reachable from k through zero-injection buses only
  DC:   unknowns theta_j, j in Z;  equations P_j = 0, j in Z    (linear, |Z| x |Z|)
  AC:   unknowns theta_j and V_j, j in Z;  equations P_j = Q_j = 0, j in Z    (2|Z| x 2|Z|), linearised at
        the estimate x_hat: C(x_hat) dx = 0. The attacker therefore needs x_hat (Hug & Giampapa, Sec. V-D).
Buses are 0-BASED POSITIONS (position p = bus number p + 1), as in ac_state_estimation.py.
"""
import numpy as np

from dc_state_estimation import dc_bus_matrices


def neighbours(m, p):
    """Positions of the buses connected to bus position p by a branch (sorted, without p)."""
    nb_ = np.concatenate([m["t"][m["f"] == p], m["f"][m["t"] == p]])
    return np.unique(nb_[nb_ != p])


def zero_injection_extension(m, center):
    """Z(center): zero-injection buses reachable from ``center`` along paths that pass through
    zero-injection buses only (sorted positions; empty if no neighbour of center is a zero-injection bus).
    Shifting all of them together with the center always keeps the attack hidden. It is an UPPER BOUND:
    Hug & Giampapa's procedure (Sec. V-B, steps 6-8) also tries smaller subgraphs, e.g. absorbing a
    flow change at one zero-injection bus by shifting only one of its neighbours."""
    zi = m["zero_inj"]
    found, frontier = set(), [j for j in neighbours(m, center) if zi[j]]
    while frontier:
        j = frontier.pop()
        if j in found or j == center:
            continue
        found.add(j)
        frontier += [k for k in neighbours(m, j) if zi[k] and k not in found]
    return np.array(sorted(found), dtype=int)


def dc_hidden_shift(m, center, t):
    """DC state shift c (rad, length nb - 1, reference column removed) that moves theta_center by t and
    keeps P = 0 at every zero-injection bus: the buses in Z(center) are shifted by the solution of
    Bbus[Z, Z] c_Z = - Bbus[Z, center] t, all other buses by 0. Returns (c, Z)."""
    Z = zero_injection_extension(m, center)
    full = np.zeros(m["nb"])
    full[center] = t
    if len(Z):
        B = dc_bus_matrices(m)[1].toarray()
        full[Z] = np.linalg.solve(B[np.ix_(Z, Z)], -B[Z, center] * t)
    return np.delete(full, m["ref"]), Z


def dc_naive_shift(m, center, t):
    """DC state shift that moves ONLY theta_center by t (no zero-injection extension), length nb - 1."""
    full = np.zeros(m["nb"])
    full[center] = t
    return np.delete(full, m["ref"])


def ac_hidden_shift(m, est, center, t):
    """AC state shift dx (length 2 nb - 1: angles without the reference, then all Vm) that moves the
    angle of ``center`` by t and keeps P = Q = 0 at the zero-injection buses to first order:
    the angles AND magnitudes of the buses in Z(center) solve C(x_hat)[rows Z, cols Z] dx_Z =
    - C(x_hat)[rows Z, col center] t. ``est`` is an estimate_ac result (its C is evaluated at x_hat).
    Returns (dx, Z)."""
    nb, ref = m["nb"], m["ref"]
    Z = zero_injection_extension(m, center)
    nonref = np.delete(np.arange(nb), ref)
    ang = {p: int(np.flatnonzero(nonref == p)[0]) for p in nonref}  # angle column of bus position p
    dx = np.zeros(2 * nb - 1)
    dx[ang[center]] = t
    if len(Z):
        zi = np.flatnonzero(m["zero_inj"])  # row order of the constraints: P at every zi bus, then Q
        rows = np.concatenate([np.searchsorted(zi, Z), len(zi) + np.searchsorted(zi, Z)])
        cols = np.concatenate([[ang[p] for p in Z], nb - 1 + Z])
        C = est["C"].toarray()
        dx[cols] = np.linalg.solve(C[np.ix_(rows, cols)], -C[rows, ang[center]] * t)
    return dx, Z


def measurement_buses(m, meas):
    """Bus position whose RTU reports each measurement row (one RTU per bus, Hug & Giampapa, Sec. VI-A):
    a from-end flow at the from bus, a to-end flow at the to bus, injections, Vm and Va at their bus."""
    el = meas["element"].to_numpy()
    side = meas["side"].to_numpy()
    is_flow = meas["type"].str.endswith("_flow").to_numpy()
    return np.where(is_flow & (side == "from"), m["f"][np.where(is_flow, el, 0)],
                    np.where(is_flow & (side == "to"), m["t"][np.where(is_flow, el, 0)], el))
