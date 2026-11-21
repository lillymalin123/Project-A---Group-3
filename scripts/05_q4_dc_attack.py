"""Q4 --- a DC-designed hidden attack a = H_DC c, applied to both the DC and the AC estimator.

Run from anywhere:  python scripts/05_q4_dc_attack.py
                     (after 01_generate_measurements.py, 02_ac_state_estimation.py,
                      03_ac_bad_data.py and 04_dc_state_estimation.py have all been run once)

What it does
  1. "before": reads the existing AC and DC no-attack results (results/ac_bad_data.csv,
     results/dc_bad_data.csv) rather than recomputing them -- one script, one result (guideline rule 8):
     those two numbers are already the claimed output of scripts 03 and 04.
  2. Builds c = t * angle_shift(m, config.ATTACK_BUS), t = config.ATTACK_ANGLE_SHIFT_RAD: a shift of the
     voltage angle at bus ATTACK_BUS only (reference bus fixed), as in Hug & Giampapa 2012, Sec. VI-B/C.
  3. a_dc = H_DC c on the DC subset (attacks.build_dc_attack; raises if the zero-injection constraint
     C_DC c = 0 is violated), embedded into the full 518-row AC vector with zero on Q flows, Q injections
     and Vm (dc_state_estimation.embed_dc_vector) -- THIS is "pad the remaining AC measurements with zero".
  4. "after": both estimators re-run on the SAME attacked vector (DC reads its own 259-row subset of it).
  5. Self-checks: the attack is in the range of H_DC (so by construction the DC residual, and therefore
     its chi-square statistic, must be unchanged); the zero-injection constraints still hold.

Writes:
  results/q4_attack_summary.csv      one row per (estimator, before/after): J, dof, tau, p-value, alarm,
                                      largest normalized residual, max |angle change| over all buses
  results/q4_attack_states.csv       one row per bus: AC and DC angle, before and after, for both

This is a single bus-2 case picked to illustrate the mechanism; it does not reproduce the paper's own
figures (different test system realisation, different attack set) -- see the project brief, Sec. 4.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from ac_state_estimation import build_model, chi_square_test, estimate_ac, load_measurements, normalized_residuals
from attacks import angle_shift, build_dc_attack
from common import check
from dc_state_estimation import (dc_measurement_matrix, dc_subset, dc_zero_injection_matrix, embed_dc_vector,
                                 estimate_dc)

BEFORE_AC = config.RESULTS_DIR / "ac_bad_data.csv"
BEFORE_DC = config.RESULTS_DIR / "dc_bad_data.csv"
for p in (BEFORE_AC, BEFORE_DC):
    if not p.exists():
        sys.exit(f"{p} is missing: run scripts 02, 03 and 04 first (in order), then this script.")

m = build_model()
meas = load_measurements(config.MEASUREMENTS_CSV)
dmeas = dc_subset(meas)
H_dc, C_dc = dc_measurement_matrix(m, dmeas), dc_zero_injection_matrix(m)

# --- build and apply the attack --------------------------------------------------------------------
c_dir = angle_shift(m, config.ATTACK_BUS)
c = config.ATTACK_ANGLE_SHIFT_RAD * c_dir
a_dc = build_dc_attack(H_dc.toarray(), c, C_dc.toarray())  # raises if C_dc @ c != 0
a_full = embed_dc_vector(meas, a_dc)  # zero on Q_flow, Q_inj, Vm

meas_attacked = meas.copy()
meas_attacked["value"] = meas_attacked["value"] + a_full
dmeas_attacked = dc_subset(meas_attacked)  # = dmeas with exactly a_dc added (same rows, same order)

est_ac_before, est_ac_after = estimate_ac(m, meas), estimate_ac(m, meas_attacked)
est_dc_before, est_dc_after = estimate_dc(m, dmeas), estimate_dc(m, dmeas_attacked)

# --- self-checks -------------------------------------------------------------------------------------
print("Self-checks")
ok = [
    check(np.allclose(dmeas_attacked["value"].to_numpy() - dmeas["value"].to_numpy(), a_dc),
          "attacked DC table = baseline DC table + a_dc exactly (embedding did not touch the wrong rows)"),
    check(np.abs(a_full[~meas["type"].isin(["P_flow", "P_inj", "Va"])]).max() == 0.0,
          "a_full is exactly zero on Q_flow, Q_inj and Vm (the AC-only measurements)"),
]
J_dc_before, J_dc_after = chi_square_test(est_dc_before)["J"], chi_square_test(est_dc_after)["J"]
ok.append(check(abs(J_dc_after - J_dc_before) < 1e-6 * max(1.0, J_dc_before),
                f"DC chi-square statistic is unchanged by the attack (J {J_dc_before:.3f} -> {J_dc_after:.3f}): "
                "a lies in the range of H_DC, exactly as Hug & Giampapa 2012, Sec. V-C/D construct it",
                "a residual-preserving attack must leave J, and every DC normalized residual, untouched"))
ok.append(check(np.abs(dc_zero_injection_matrix(m) @ np.delete(est_dc_after["Va"], m["ref"])).max()
                < config.CONSTRAINT_TOL, "zero-injection constraints still hold at the attacked DC estimate"))

# --- summary table ------------------------------------------------------------------------------------
before_ac, before_dc = pd.read_csv(BEFORE_AC).iloc[0], pd.read_csv(BEFORE_DC).iloc[0]


def row(name, before, est, test):
    rn = normalized_residuals(est)
    worst = int(np.nanargmax(rn))
    return {"estimator": name, "J": test["J"], "dof": test["dof"], "tau": test["tau"],
            "p_value": test["p_value"], "alarm": test["alarm"], "largest_norm_resid": rn[worst]}


rows = [
    {**row("AC", None, est_ac_before, chi_square_test(est_ac_before)), "condition": "before"},
    {**row("AC", None, est_ac_after, chi_square_test(est_ac_after)), "condition": "after"},
    {**row("DC", None, est_dc_before, chi_square_test(est_dc_before)), "condition": "before"},
    {**row("DC", None, est_dc_after, chi_square_test(est_dc_after)), "condition": "after"},
]
summary = pd.DataFrame(rows)[["estimator", "condition", "J", "dof", "tau", "p_value", "alarm", "largest_norm_resid"]]
config.RESULTS_DIR.mkdir(exist_ok=True)
summary.to_csv(config.RESULTS_DIR / "q4_attack_summary.csv", index=False)

def _row(est_name, cond):
    r = summary[(summary.estimator == est_name) & (summary.condition == cond)].iloc[0]
    return "ALARM" if r.alarm else "passes"


print(f"\nAttack: bus {config.ATTACK_BUS}, t = {config.ATTACK_ANGLE_SHIFT_RAD} rad "
      f"({np.degrees(config.ATTACK_ANGLE_SHIFT_RAD):.3f} deg), {int((np.abs(a_dc) > 1e-9).sum())} of "
      f"{len(a_dc)} DC measurements changed")
print(f"DC : J {J_dc_before:.1f} -> {J_dc_after:.1f}  "
      f"(before {_row('DC', 'before')}, after {_row('DC', 'after')})")
J_ac_before = summary[(summary.estimator == "AC") & (summary.condition == "before")].J.iloc[0]
J_ac_after = summary[(summary.estimator == "AC") & (summary.condition == "after")].J.iloc[0]
print(f"AC : J {J_ac_before:.1f} -> {J_ac_after:.1f}  "
      f"(before {_row('AC', 'before')}, after {_row('AC', 'after')})")

# --- per-bus state table --------------------------------------------------------------------------
out = pd.DataFrame({
    "bus": m["bus_numbers"],
    "Va_ac_before_deg": np.degrees(est_ac_before["Va"]), "Va_ac_after_deg": np.degrees(est_ac_after["Va"]),
    "Va_dc_before_deg": np.degrees(est_dc_before["Va"]), "Va_dc_after_deg": np.degrees(est_dc_after["Va"]),
})
out["dVa_ac_deg"] = out["Va_ac_after_deg"] - out["Va_ac_before_deg"]
out["dVa_dc_deg"] = out["Va_dc_after_deg"] - out["Va_dc_before_deg"]
out.to_csv(config.RESULTS_DIR / "q4_attack_states.csv", index=False)

moved_ac = out[out["dVa_ac_deg"].abs() > 1e-6]["bus"].tolist()
print(f"\nAC angle estimate moved at {len(moved_ac)} bus(es): {moved_ac}")
print(f"  bus {config.ATTACK_BUS} shift: DC {out.loc[out.bus==config.ATTACK_BUS,'dVa_dc_deg'].iloc[0]:.4f} deg, "
      f"AC {out.loc[out.bus==config.ATTACK_BUS,'dVa_ac_deg'].iloc[0]:.4f} deg")
print(f"wrote {config.RESULTS_DIR / 'q4_attack_summary.csv'} and {config.RESULTS_DIR / 'q4_attack_states.csv'}")
if not all(ok):
    sys.exit("some checks FAILED")
