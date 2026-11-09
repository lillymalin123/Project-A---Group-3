"""AC counterpart of Block 1: the grid and its operating point (L2, L3), AC model only.

Run from the repo root:  python scripts/06_ac_operating_point.py
DC step                     -> AC step
  1.1 P = Pg - Pd           -> S_spec = (Pg - Pd) + j(-Qd): what each bus injects
  1.2 B (real, rank n-1)    -> Ybus (complex, full rank because of shunts/charging/taps)
  1.3 solve B~ theta = P~   -> solve S_i = V_i conj((Ybus V)_i) by Newton (powerflow.py)
Writes results/ac_operating_point.csv. Exits non-zero if a check fails.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.case_loading import build_network, load_case_json
from src.checks import check
from src.powerflow import solve_power_flow

ok = []
net = build_network(load_case_json(config.CASE_JSON))
nb, base = net.nb, net.baseMVA
print(f"{nb} buses, {net.nl} branches; slack = bus {net.bus_ids[net.ref]}, "
      f"PV = {len(net.pv)} buses, PQ = {len(net.pq)} buses")

# --- 1.1 Specified injections ------------------------------------------------
print(f"scheduled generation {base*net.Pg.sum():.1f} MW, load {base*net.Sload.real.sum():.1f} MW, "
      f"reactive load {base*net.Sload.imag.sum():.1f} MVAr")

# --- 1.2 Admittance matrix ----------------------------------------------------
Y = net.Ybus.toarray()
ok.append(check(np.allclose(Y, Y.T), "Ybus is symmetric (no phase shifters)"))
rk = np.linalg.matrix_rank(Y)
ok.append(check(rk == nb, f"rank(Ybus) = {rk} = n",
                "unlike B (rank n-1): shunts, line charging and taps break Y*1 = 0"))
print(f"Ybus nonzeros: {net.Ybus.nnz} of {nb*nb}")

# --- 1.3 Operating point (Newton power flow) ---------------------------------
Vm, Va, it = solve_power_flow(net)
V = Vm * np.exp(1j * Va)
S = V * np.conj(net.Ybus @ V)  # complex injection at every bus, p.u.
print(f"power flow converged in {it} Newton iterations")
Sspec = net.Pg - net.Sload.real - 1j * net.Sload.imag
pq = net.pq  # slack P and Q are free, so only PQ and PV buses are checked
ok.append(check(np.max(np.abs(S[pq] - Sspec[pq])) < 1e-8, "PQ buses: injection matches the scheduled load"))
ok.append(check(np.max(np.abs(S[net.pv].real - Sspec[net.pv].real)) < 1e-8, "PV buses: P matches the schedule"))
ok.append(check(np.allclose(Vm[net.pv], net.Vg[net.pv]), "PV buses: voltage magnitude at the setpoint"))
ok.append(check(abs(Va[net.ref]) < 1e-12, "reference angle is 0"))

# --- 1.4 What the slack supplies ------------------------------------------------
loss = S.real.sum()
print(f"slack supplies P = {base*S.real[net.ref]:.1f} MW, Q = {base*S.imag[net.ref]:.1f} MVAr")
print(f"total active losses = {base*loss:.1f} MW (the DC model assumes 0)")
ok.append(check(loss > 0, "active losses > 0", "L3 'DC SE: Assumptions' A1: what the DC model drops"))
print(f"Vm range {Vm.min():.3f} .. {Vm.max():.3f} p.u.;  Va range {np.degrees(Va.min()):.1f} .. {np.degrees(Va.max()):.1f} deg")

out = pd.DataFrame({"bus": net.bus_ids, "Vm": Vm, "Va_deg": np.degrees(Va),
                    "P_MW": base * S.real, "Q_MVAr": base * S.imag})
Path("results").mkdir(exist_ok=True)
out.to_csv("results/ac_operating_point.csv", index=False)
print("wrote results/ac_operating_point.csv")
if not all(ok):
    sys.exit("some checks FAILED")
