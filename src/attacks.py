import numpy as np
import config
from ac_state_estimation import h_of_x, zero_injection_rows

def angle_shift(m, bus_number):
    """Unit angle-shift direction c_dir (rad) in the DC state space.

    m           model dict from ac_state_estimation.build_model (uses nb, ref)
    bus_number  MATPOWER bus NUMBER whose angle is shifted (1-based)
    Returns     c_dir, shape (nb - 1,): 1.0 at the column of that bus, 0 elsewhere. Columns are the angles
                of every bus except the reference, in position order (same as H_DC and the angle block of
                the AC Jacobian). Scale it: c = t * c_dir, t in rad.
    Raises ValueError if bus_number is the reference bus or not in the case.
    """
    pos = bus_number - 1 # Because matlab and matpower indexes from 1
    # We gotta check if the bus we are fucking with is reference bus (it shouldnt be)
    if pos == m["ref"]:
        raise ValueError("Everything crashed... wah wah....")      

    nb = m["nb"]    # Number of busses
    cols = np.delete(np.arange(nb), m["ref"])   # Columns without reference bus

    k = np.flatnonzero(cols == pos) #   Find the column that matches the bus you want to attack
    if k.size == 0:
        raise ValueError(f"bus {bus_number} is not here sadly:(")
    c_dir = np.zeros(nb-1)  # Build c vector
    c_dir[k[0]] = 1         # Set bus you want to attack to 1 so you can multiply with different magnitudes later
    return c_dir

def _linear_attack(H, c, C):
    """Shared core of the DC and AC-Jacobian attacks: a = H c, with shape and constraint checks.
    H (n_meas x n_state), c (n_state,), C (n_constraints x n_state). Raises ValueError if the
    shapes do not match or C c != 0 beyond config.CONSTRAINT_TOL (scaled by max |c|)."""
    c = np.asarray(c, dtype=float)
    if H.shape[1] != c.shape[0]:
        raise ValueError(f"H has {H.shape[1]} columns but c has {c.shape[0]} entries")
    viol = np.abs(C @ c).max() if C.shape[0] else 0.0
    if viol > config.CONSTRAINT_TOL * max(1.0, np.abs(c).max()):
        raise ValueError(f"attack changes the injection at a zero-injection bus by {viol:.1e} p.u.")
    return H @ c

def build_dc_attack(H_dc, c, C_dc=None):
    """DC-designed false data injection attack a = H_DC c (Liu et al. 2009; Hug & Giampapa 2012).

    H_dc   DC measurement matrix, shape (m_DC, nb - 1), p.u./rad (dc_state_estimation.dc_measurement_matrix);
           row i belongs to row i of the DC measurement table, columns as in angle_shift
    c      angle shift the attacker wants the estimator to report, shape (nb - 1,), rad (e.g. t * angle_shift)
    C_dc   zero-injection constraint matrix, shape (n_zero_injection, nb - 1)
           (dc_state_estimation.dc_zero_injection_matrix); the attack must satisfy C_dc c = 0
    Returns a_dc, shape (m_DC,): the change added to each DC measurement, p.u. for P flows and injections,
           rad for angles. Lies in the range of H_dc, so the DC residual and chi-square statistic are
           unchanged and the DC estimate moves by exactly c. Use dc_state_estimation.embed_dc_vector to
           place it in the full AC measurement vector (zeros on Q flows, Q injections and Vm).
    Raises ValueError if the shapes do not match or c changes the injection at a zero-injection bus
           (beyond config.CONSTRAINT_TOL, scaled by max |c|).
    """
    return _linear_attack(H_dc, c, C_dc)

def build_ac_jacobian_attack(J, c_ac, C_ac):
    """AC-Jacobian false data injection attack a = J(x_hat) c (first-order AC attack, Q5).

    J      AC measurement Jacobian at the estimate x_hat, shape (n_meas, 2*nb - 1), p.u./rad and p.u./p.u.
           (estimate_ac(...)["H"] on the UNATTACKED measurements); columns: angles except the reference, then Vm
    c_ac   state shift, shape (2*nb - 1,): rad for the angle entries, p.u. for the Vm entries
           (e.g. ac_state_shift(m, t * angle_shift(m, bus)))
    C_ac   zero-injection constraint Jacobian at x_hat, shape (2 * n_zero_injection, 2*nb - 1)
           (estimate_ac(...)["C"]: P and Q rows); the attack must satisfy C_ac c_ac = 0
    Returns a, shape (n_meas,): change added to every AC measurement (P and Q in p.u., Vm in p.u., Va in rad).
           Nonzero also on Q and Vm rows, unlike the DC attack. Exact only to first order: the residual
           change grows as ||c||^2, because h(x_hat + c) = h(x_hat) + J c + O(||c||^2).
    Raises ValueError if the shapes do not match or C_ac c_ac != 0.
    """
    return _linear_attack(J, c_ac, C_ac)

def ac_state_shift(m, c_angle):
    """Embed an angle shift (nb-1,) [rad] into the AC state space (2*nb-1,): [c_angle; zeros(nb)].
    Column order as ac_state_estimation.jacobian: angles except the reference, then Vm of every bus [p.u.].
    The voltage magnitudes are not shifted, so all three Q5 attacks use the same direction."""
    return np.concatenate([np.asarray(c_angle, dtype=float), np.zeros(m["nb"])])

def build_ac_nonlinear_attack(m, meas, Vm_hat, Va_hat, c_ac):
    """Nonlinear AC false data injection attack a = h(x_hat + c) - h(x_hat) (Hug & Giampapa 2012, eq. 30).

    m       model dict from ac_state_estimation.build_model
    meas    measurement table (type, element, side); a gets one entry per row
    Vm_hat  estimated voltage magnitudes at x_hat, shape (nb,), p.u. (estimate_ac(...)["Vm"], UNATTACKED data)
    Va_hat  estimated voltage angles at x_hat, shape (nb,), rad (estimate_ac(...)["Va"])
    c_ac    state shift, shape (2*nb - 1,): angles except the reference [rad], then Vm of every bus [p.u.]
            (e.g. ac_state_shift(m, t * angle_shift(m, bus)))
    Returns a, shape (len(meas),): exact change of every measurement when the state moves from x_hat to
            x_hat + c (P, Q, Vm in p.u., Va in rad). Consistent with the full AC model, so the residual is
            unchanged if the estimator moves exactly to x_hat + c. Requires the attacker to know x_hat.
    Raises ValueError if c_ac has the wrong length or the shift changes the injection at a zero-injection bus.
    """
    nb = m["nb"]
    c_ac = np.asarray(c_ac, dtype=float)
    # Shape check: c_ac must have 2*nb - 1 entries
    if c_ac.shape != (2 * nb - 1,):
        raise ValueError(f"Expected entries to be {2 * nb - 1}, but twasnt")

    # Shifted state x_hat + c
    not_ref = np.arange(nb) != m["ref"]
    Va_new = Va_hat.copy()
    Va_new[not_ref] += c_ac[: nb - 1]
    Vm_new = Vm_hat + c_ac[nb - 1 :]

    # Zero-injection check at the shifted state
    cons = zero_injection_rows(m)
    viol = np.abs(h_of_x(m, cons, Vm_new, Va_new) - h_of_x(m, cons, Vm_hat, Va_hat)).max()
    if viol > config.CONSTRAINT_TOL:
        raise ValueError(" Zero-injection busses are being messed with")

    # The attack itself
    return h_of_x(m, meas, Vm_new, Va_new) - h_of_x(m, meas, Vm_hat, Va_hat)