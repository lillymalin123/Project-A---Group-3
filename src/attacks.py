import numpy as np
import config

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
    c = np.asarray(c, dtype=float)
    if H_dc.shape[1] != c.shape[0]:
        raise ValueError(f"H_dc has {H_dc.shape} columns but c has {c.shape[0]} entries :o")
    viol = np.abs(C_dc @ c).max()
    if viol > config.CONSTRAINT_TOL * max(1, np.abs(c).max()):
        raise ValueError(f"Bro... this attack changes the injection at zero-injection bus by {viol:.1e} p.u.")
    return H_dc @ c