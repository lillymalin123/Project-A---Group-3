"""
dc_operating_point.py --- Block 1 for the DC model: the 57-bus grid and its operating point.

Run:   python src/dc_operating_point.py        (common.py must sit next to this file)

Adapted from the lecture solution (14-bus) to the IEEE 57-bus case in data/case57-se.json.
  1.1 P = Pg - Pd                net injections
  1.2 B = -Im(Ybus)              built from the branch reactances, rank n-1
  1.3 solve B~ theta~ = P~       DC power flow with the slack row/column removed
  1.4 slack gap                  DC slack injection vs. the case file (DC has no losses)
  1.5 theta_DC vs theta_AC       compared with the AC angles stored in the case file

Writes three CSV files to results/:
  dc_operating_point.csv   one row per bus     (angles, injections, AC comparison)
  dc_branch_flows.csv      one row per branch  (P_ij, P_ji, loading)
  dc_summary.csv           scalar results      (slack gap, max angle error, ...)
Exits non-zero if a check fails.

Indices: Python counts from 0; bus numbers in all output are lecture numbers (bus = index + 1).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import BASE_MVA, check, load_case, measurement_matrix, meter_labels

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def header(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


net = load_case()
lab = meter_labels(net)
n, nb = net["n"], net["nb"] # copies branch and bus counts into short names
m = n + 2 * nb                     # meters: n injections + 2 flows per branch (217 for 57 bus)
slack0 = net["slack"] - 1          # 0-based slack position
keep0 = [b - 1 for b in net["keep"]] #  lists the indices of every bus except the slack 
ok = []

header(f"Block 1 --- The {n}-bus grid and its DC operating point   (L2, L3)")
print(f"{n} buses, {nb} branches, slack = bus {net['slack']}, m = {m} meters")

# --- 1.1 Net injection P_i = P_{g,i} - P_{d,i} -------------------------------------------------
P = net["Pg"] - net["Pd"] # net power at each bus, + for generation, - for load, p.u.
print(f"Task 1.1  generation {BASE_MVA*net['Pg'].sum():.1f} MW at buses "
      f"{[int(b)+1 for b in np.flatnonzero(net['Pg'] > 0)]}; load {BASE_MVA*net['Pd'].sum():.1f} MW")

# --- 1.2 B = -Im(Y_bus) --------------------------------------------------------------------------
B = np.zeros((n, n))
for l, (i, j) in enumerate(zip(net["frm"], net["to"])):
    y = 1.0 / net["x"][l] # susceptance of branch l = 1/x_l, p.u.
    B[i-1, i-1] += y; B[j-1, j-1] += y
    B[i-1, j-1] -= y; B[j-1, i-1] -= y
ok.append(check(np.allclose(B, B.T), "B is symmetric"))
ok.append(check(np.linalg.norm(B @ np.ones(n)) < 1e-10, "B * 1 = 0, so 1 is in N(B)",
                "L3 'What are the states?'"))
rankB = np.linalg.matrix_rank(B)
ok.append(check(rankB == n - 1, f"rank(B) = {rankB} = n - 1", "L2 'The Four Fundamental Subspaces'"))

# --- 1.3 DC power flow on the reduced system ----------------------------------------------------
Btil = B[np.ix_(keep0, keep0)]
theta = np.linalg.solve(Btil, P[keep0])        # angles of the non-slack buses (rad)
theta_full = np.zeros(n)
theta_full[keep0] = theta                      # slack angle = 0
P_dc = B @ theta_full                          # injections the DC model produces
P_slack_dc = P_dc[slack0]
print(f"Task 1.3  slack injection: DC model {BASE_MVA*P_slack_dc:.1f} MW vs case file "
      f"{BASE_MVA*P[slack0]:.1f} MW (gap {BASE_MVA*(P[slack0]-P_slack_dc):.1f} MW)")
ok.append(check(abs(P_dc.sum()) < 1e-10, "sum_i P_i = 0 in the DC model: no losses",
                "L3 'DC SE: Assumptions' A1"))
ok.append(check(np.allclose(P_dc[keep0], P[keep0]), "non-slack injections reproduced exactly"))

# Same operating point through the measurement model z = H theta (consistency with common.py)
H = measurement_matrix(net)
z0 = H @ theta
ok.append(check(H.shape == (m, n - 1), f"H is {H.shape[0]} x {H.shape[1]}"))
ok.append(check(np.allclose(z0[:n], P_dc), "injection rows of H theta equal B theta"))

# Branch flows P_ij = (theta_i - theta_j) / x_ij
frm0, to0 = np.array(net["frm"]) - 1, np.array(net["to"]) - 1
P_ij = (theta_full[frm0] - theta_full[to0]) / net["x"]
ok.append(check(np.allclose(z0[n::2], P_ij) and np.allclose(z0[n+1::2], -P_ij),
                "flow rows of H theta equal (theta_i - theta_j)/x_ij, and P_ji = -P_ij"))
imax = int(np.argmax(np.abs(P_ij)))
print(f"          largest flow: branch {imax+1} ({net['frm'][imax]}-{net['to'][imax]}) "
      f"{BASE_MVA*P_ij[imax]:.1f} MW")

# --- 1.5 Compare with the AC angles in the case file --------------------------------------------
dth = theta_full - net["Va"]
dth_ac = np.abs(dth).max()
bmax = int(np.argmax(np.abs(dth))) + 1
print(f"Task 1.5  max |theta_DC - theta_AC| = {dth_ac:.4f} rad = {np.degrees(dth_ac):.2f} deg "
      f"(bus {bmax})")

# --- Write results -------------------------------------------------------------------------------
RESULTS_DIR.mkdir(exist_ok=True)
buses = pd.DataFrame({
    "bus": np.arange(1, n + 1),
    "Pg_MW": BASE_MVA * net["Pg"],
    "Pd_MW": BASE_MVA * net["Pd"],
    "P_spec_MW": BASE_MVA * P,               # Pg - Pd from the case file
    "P_dc_MW": BASE_MVA * P_dc,              # B theta (differs from P_spec only at the slack)
    "theta_dc_rad": theta_full,
    "theta_dc_deg": np.degrees(theta_full),
    "theta_ac_deg": np.degrees(net["Va"]),
    "dtheta_deg": np.degrees(dth),           # DC - AC
})
branches = pd.DataFrame({
    "branch": np.arange(1, nb + 1),
    "from_bus": net["frm"],
    "to_bus": net["to"],
    "x_pu": net["x"],
    "P_ij_MW": BASE_MVA * P_ij,
    "P_ji_MW": -BASE_MVA * P_ij,
    "meter_ij": lab[n::2],
    "meter_ji": lab[n+1::2],
})
summary = pd.DataFrame({
    "quantity": ["n_buses", "n_branches", "n_meters", "rank_B", "Pg_total_MW", "Pd_total_MW",
                 "P_slack_case_MW", "P_slack_dc_MW", "slack_gap_MW",
                 "max_abs_dtheta_rad", "max_abs_dtheta_deg", "max_dtheta_bus",
                 "max_flow_MW", "max_flow_branch"],
    "value": [n, nb, m, rankB, BASE_MVA*net["Pg"].sum(), BASE_MVA*net["Pd"].sum(),
              BASE_MVA*P[slack0], BASE_MVA*P_slack_dc, BASE_MVA*(P[slack0]-P_slack_dc),
              dth_ac, np.degrees(dth_ac), bmax,
              BASE_MVA*abs(P_ij[imax]), imax + 1],
})
for name, df in [("dc_operating_point.csv", buses), ("dc_branch_flows.csv", branches),
                 ("dc_summary.csv", summary)]:
    df.to_csv(RESULTS_DIR / name, index=False)
    print(f"wrote {RESULTS_DIR / name}")

if not all(ok):
    sys.exit("some checks FAILED")
