"""
ac_operating_point.py --- Block 1 for the AC model: the 57-bus grid and its operating point.

Run:   python src/ac_operating_point.py        (loadCase.py must sit next to this file)

DC Block 1 step              ->  AC step in this file
  1.1 P = Pg - Pd            ->  specified injections: P and Q of loads, P and |V| of generators
  1.2 B (real, rank n-1)     ->  Ybus (complex, full rank: shunts, line charging, taps)
  1.3 solve B~ theta = P~    ->  nonlinear power flow  S_i = V_i conj((Ybus V)_i)  (pandapower runpp)
  1.4 slack gap, DC vs AC    ->  slack P and Q, total active losses

Writes results/ac_operating_point.csv. Exits non-zero if a check fails.

Conventions: internal index k = bus number k+1 (pandapower index = MATPOWER bus number);
power in p.u. on baseMVA unless a name says MW/MVAr, angles in radians unless a name says deg.
"""
import sys
import warnings

import numpy as np
import pandapower as pp
import pandas as pd

from loadCase import REPO_ROOT, load_case57

warnings.filterwarnings("ignore", message=".*numba.*")
TOL = 1e-6  # p.u., tolerance of the consistency checks
RESULTS_DIR = REPO_ROOT / "results"


def check(ok, what, note=""):
    """Print one [ok]/[FAIL] line; returns the boolean."""
    print("  " + ("[ok]  " if ok else "[FAIL]") + " " + what + (f"\n           {note}" if note else ""))
    return bool(ok)


net = load_case57()
pp.runpp(net)  # operating point
ppc = net._ppc
nb = len(net.bus)
base = ppc["baseMVA"]
first = net.bus.index.min()  # bus number of position 0
pos = lambda bus_numbers: np.asarray(bus_numbers) - first  # bus number -> 0-based position
ok = []

# --- 1.1 Specified injections -----------------------------------------------------------------
Pd, Qd, Pg = np.zeros(nb), np.zeros(nb), np.zeros(nb)
np.add.at(Pd, pos(net.load.bus), net.load.p_mw.to_numpy() / base)
np.add.at(Qd, pos(net.load.bus), net.load.q_mvar.to_numpy() / base)
np.add.at(Pg, pos(net.gen.bus), net.gen.p_mw.to_numpy() / base)
ref = int(pos(net.ext_grid.bus.iloc[0]))  # slack bus position
pv = pos(net.gen.bus)
pq = np.setdiff1d(np.arange(nb), np.r_[ref, pv])
print(f"{nb} buses, {ppc['branch'].shape[0]} branches; slack = bus {ref + first}, "
      f"{len(pv)} PV buses, {len(pq)} PQ buses, {int((~np.isin(np.arange(nb), np.r_[net.load.bus.to_numpy() - first, pv, ref])).sum())} "
      "buses with neither load nor generation")
print(f"scheduled generation {base * Pg.sum():.1f} MW (slack excluded), load {base * Pd.sum():.1f} MW, "
      f"reactive load {base * Qd.sum():.1f} MVAr")

# --- 1.2 Admittance matrix -----------------------------------------------------------------------
Ybus = ppc["internal"]["Ybus"].tocsr()
Y = Ybus.toarray()
ok.append(check(np.allclose(Y, Y.T), "Ybus is symmetric (no phase shifters in this case)"))
rk = np.linalg.matrix_rank(Y)
ok.append(check(rk == nb, f"rank(Ybus) = {rk} = n", "unlike B (rank n-1): shunts, charging and taps break Y*1 = 0"))
print(f"Ybus nonzeros: {Ybus.nnz} of {nb * nb}")

# --- 1.3 Operating point -----------------------------------------------------------------------
Vm = net.res_bus.vm_pu.to_numpy()
Va = np.deg2rad(net.res_bus.va_degree.to_numpy())
V = Vm * np.exp(1j * Va)
S = V * np.conj(Ybus @ V)  # complex injection at every bus, recomputed from Ybus and V
print(f"power flow converged in {ppc['iterations']} iterations")
ok.append(check(np.max(np.abs(S[pq] - (-Pd[pq] - 1j * Qd[pq]))) < TOL, "PQ buses: injection = - load"))
ok.append(check(np.max(np.abs(S[pv].real - (Pg[pv] - Pd[pv]))) < TOL, "PV buses: P = Pg - Pd"))
ok.append(check(np.allclose(Vm[pv], net.gen.vm_pu.to_numpy(), atol=TOL), "PV buses: |V| at the setpoint"))
ok.append(check(abs(Va[ref]) < 1e-12, "reference angle is 0"))
# pandapower counts shunts (buses 18, 25, 53) as demand at the bus; in S = V conj(Ybus V) they sit
# inside Ybus, so the shunt power is taken out of res_bus before comparing.
shunt = np.zeros(nb, dtype=complex)
np.add.at(shunt, pos(net.shunt.bus), (net.res_shunt.p_mw + 1j * net.res_shunt.q_mvar).to_numpy())
demand = net.res_bus.p_mw.to_numpy() + 1j * net.res_bus.q_mvar.to_numpy() - shunt  # MW + j MVAr
ok.append(check(np.max(np.abs(S * base + demand)) < TOL * base,
                "S recomputed from (Ybus, V) equals pandapower's bus results (shunt power removed)",
                "res_bus reports consumption: S = -(p_mw + j q_mvar - shunt)/baseMVA"))

# --- 1.4 What the slack supplies, losses ---------------------------------------------------------
loss = S.real.sum()
print(f"slack supplies P = {base * S.real[ref]:.1f} MW, Q = {base * S.imag[ref]:.1f} MVAr")
print(f"total active losses = {base * loss:.1f} MW (the DC model assumes 0)")
ok.append(check(loss > 0, "active losses > 0"))
print(f"Vm range {Vm.min():.3f} .. {Vm.max():.3f} p.u.;  Va range {np.degrees(Va.min()):.1f} .. {np.degrees(Va.max()):.1f} deg")

out = pd.DataFrame({"bus": net.bus.index, "Vm_pu": Vm, "Va_deg": np.degrees(Va),
                    "P_inj_MW": base * S.real, "Q_inj_MVAr": base * S.imag})
RESULTS_DIR.mkdir(exist_ok=True)
out.to_csv(RESULTS_DIR / "ac_operating_point.csv", index=False)
print(f"wrote {RESULTS_DIR / 'ac_operating_point.csv'}")
if not all(ok):
    sys.exit("some checks FAILED")