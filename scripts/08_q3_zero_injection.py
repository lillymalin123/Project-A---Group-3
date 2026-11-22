"""Q3 --- the role of zero-injection buses: what a hidden single-bus attack must change, bus by bus.

Run from anywhere:  python scripts/08_q3_zero_injection.py   (after 01_generate_measurements.py)

For every bus k except the reference, the attacker shifts the estimated angle of k by
t = config.ATTACK_ANGLE_SHIFT_RAD (the size used in Q4) and keeps the attack hidden:
  * zero-injection bus k: its injection is a known zero, so no hidden attack can change it
    (Hug & Giampapa 2012, Sec. VI-A, the zeros in Fig. 3). Its STATE can change, but not on its own:
    P_k = 0 makes theta_k the b-weighted average of its neighbours' angles (in AC, P = Q = 0 pin angle and
    magnitude), so it only follows a shifted neighbour with injection. Every such shift is already one of
    the attacks centred on an injection bus below, so no separate attack is built for it.
  * bus k with injection: if a neighbour has zero injection, the buses Z(k) reached through zero-injection
    buses are shifted too (src/zero_injection.py) so that their injections stay zero.
Per bus it reports
  * the measurements (of the 518 / of the DC 259) the hidden attack changes, and the RTUs that carry them
    (one RTU per bus: its injections, Vm, Va and the flows leaving it; Hug & Giampapa, Sec. VI-A),
    for the DC attack a = H_DC c and the AC attack a = J(x_hat) dx;
  * for the hidden attack and for a NAIVE attack that shifts only theta_k and ignores the zero-injection
    constraint: the attack's own residual ||r(a)||^2 (DC, noise-free: 0 if hidden) and the change of the DC
    chi-square statistic on the frozen baseline draw (own residual + a draw-dependent cross term).
The AC count uses the linearised AC attack J(x_hat) dx: which measurements are touched is the same for
h(x_hat + dx) - h(x_hat) (checked for every bus: identical sets); the AC attack sizes themselves are Q5.
The AC zero-injection extension is first order: applied nonlinearly it leaves up to ~1.7e-3 p.u. at a
zero-injection bus (t = 0.0143 rad), so the AC counts are about WHICH meters, not a proof of exact stealth.
Z(k) shifts EVERY zero-injection bus reachable from k, so the counts are an upper bound on the minimum of
Hug & Giampapa's subgraph search (Sec. V-B), which can be smaller (compare with their Figs. 2-5 in that
light). Where no zero-injection bus is involved the counts are exact; e.g. bus 2 gives 15 AC measurements,
the number the paper reports for line 1 (bus 1-2).

Writes results/q3_zero_injection.csv (one row per bus) and figures/q3_zero_injection.png.
Exits non-zero if a self-check fails.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from ac_state_estimation import build_model, chi_square_test, estimate_ac, load_measurements
from common import check
from dc_state_estimation import dc_measurement_matrix, dc_subset, dc_zero_injection_matrix, estimate_dc
from zero_injection import (ac_hidden_shift, dc_hidden_shift, dc_naive_shift, measurement_buses, neighbours,
                            zero_injection_extension)

TOL = config.ATTACK_SUPPORT_TOL
t = config.ATTACK_ANGLE_SHIFT_RAD

m = build_model()
meas = load_measurements(config.MEASUREMENTS_CSV)
dmeas = dc_subset(meas)
H_dc, C_dc = dc_measurement_matrix(m, dmeas), dc_zero_injection_matrix(m)
est_ac = estimate_ac(m, meas)  # x_hat: the AC attacker linearises here
J_ac_mat = est_ac["H"]
J0_dc = chi_square_test(estimate_dc(m, dmeas))["J"]
rtu_ac, rtu_dc = measurement_buses(m, meas), measurement_buses(m, dmeas)
bn = m["bus_numbers"]


def dJ_dc(c):
    """Change of the DC chi-square statistic on the baseline draw when a = H_DC c is added."""
    return chi_square_test(estimate_dc(m, dmeas.assign(value=dmeas["value"].to_numpy() + H_dc @ c)))["J"] - J0_dc


def J_attack_alone(c):
    """||r(a)||_W^2: the statistic of the attack vector a = H_DC c on its own. The DC estimator is linear, so
    r(z + a) = r(z) + r(a) and dJ = ||r(a)||^2 + 2 r(z)' W r(a): this is the attack's own, noise-free
    contribution (0 for a hidden attack); the cross term depends on the draw and can be negative."""
    return chi_square_test(estimate_dc(m, dmeas.assign(value=H_dc @ c)))["J"]


rows = []
for k in np.delete(np.arange(m["nb"]), m["ref"]):
    row = {"bus": int(bn[k]), "zero_injection": bool(m["zero_inj"][k]),
           "n_neighbours": len(neighbours(m, k)),
           "adjacent_zero_injection": bool(m["zero_inj"][neighbours(m, k)].any())}
    if m["zero_inj"][k]:
        rows.append({**row, "hidden_attack_possible": False})
        continue
    c_hid, Z = dc_hidden_shift(m, k, t)
    c_nai = dc_naive_shift(m, k, t)
    dx, Z_ac = ac_hidden_shift(m, est_ac, k, t)
    a_dc, a_ac = H_dc @ c_hid, J_ac_mat @ dx
    ch_dc, ch_ac = np.abs(a_dc) > TOL, np.abs(a_ac) > TOL
    rows.append({**row, "hidden_attack_possible": True,
                 "n_extension": len(Z), "extension_buses": " ".join(str(int(bn[j])) for j in Z),
                 "n_meas_dc_naive": int((np.abs(H_dc @ c_nai) > TOL).sum()),
                 "n_meas_dc_hidden": int(ch_dc.sum()), "n_meas_ac_hidden": int(ch_ac.sum()),
                 "n_rtu_dc_hidden": len(np.unique(rtu_dc[ch_dc])), "n_rtu_ac_hidden": len(np.unique(rtu_ac[ch_ac])),
                 "dc_naive_constraint_violation_pu": float(np.abs(C_dc @ c_nai).max()),
                 "dc_hidden_constraint_violation_pu": float(np.abs(C_dc @ c_hid).max()),
                 "ac_hidden_constraint_violation_lin_pu": float(np.abs(est_ac["C"] @ dx).max()),
                 "J_attack_alone_dc_naive": J_attack_alone(c_nai), "J_attack_alone_dc_hidden": J_attack_alone(c_hid),
                 "dJ_dc_naive": dJ_dc(c_nai), "dJ_dc_hidden": dJ_dc(c_hid),
                 "same_extension_dc_ac": bool(np.array_equal(Z, Z_ac))})
res = pd.DataFrame(rows)
att = res[res["hidden_attack_possible"]]
ext = att[att["n_extension"] > 0]

# --- self-checks ----------------------------------------------------------------------------------
print("Self-checks")
ok = [
    check(int(res["zero_injection"].sum()) == int(m["zero_inj"].sum()),
          f"{int(res['zero_injection'].sum())} zero-injection buses, none is the reference bus"),
    check(bool((att["dc_hidden_constraint_violation_pu"] < config.CONSTRAINT_TOL).all()),
          "every extended DC shift keeps P = 0 at all zero-injection buses (C_DC c = 0)"),
    check(bool((att["ac_hidden_constraint_violation_lin_pu"] < config.CONSTRAINT_TOL).all()),
          "every extended AC shift keeps P = Q = 0 at all zero-injection buses to first order (C(x_hat) dx = 0)"),
    check(bool((att["dJ_dc_hidden"].abs() < 1e-6 * J0_dc).all()),
          f"every hidden DC attack leaves the DC chi-square statistic unchanged (max |dJ| = "
          f"{att['dJ_dc_hidden'].abs().max():.1e})"),
    check(bool((att["J_attack_alone_dc_hidden"] < 1e-6).all()),
          "every hidden DC attack contributes nothing to the residual on its own (||r(a)||^2 = 0)"),
    check(bool((ext["dc_naive_constraint_violation_pu"] > config.CONSTRAINT_TOL).all()
               and (ext["J_attack_alone_dc_naive"] > 1e-6).all()),
          f"every NAIVE attack next to a zero-injection bus violates the constraint and leaves a residual of its own "
          f"(||r(a)||^2 from {ext['J_attack_alone_dc_naive'].min():.1f} to {ext['J_attack_alone_dc_naive'].max():.0f})"),
    check(bool(att["same_extension_dc_ac"].all()), "DC and AC need the same extension buses Z(k)"),
]
b2 = res[res["bus"] == config.ATTACK_BUS].iloc[0]
ok.append(check(b2["n_extension"] == 0,
                f"bus {config.ATTACK_BUS} (the Q4/Q5 attack) has no zero-injection neighbour: shifting its angle "
                "alone already satisfies the constraints"))

config.RESULTS_DIR.mkdir(exist_ok=True)
out_csv = config.RESULTS_DIR / "q3_zero_injection.csv"
res.to_csv(out_csv, index=False, float_format="%.6g")

print(f"\nAngle shift t = {t} rad at one bus at a time; {len(res)} buses (reference excluded)")
print(f"  {int(res['zero_injection'].sum())} zero-injection buses: their injection cannot be falsified, and their state "
      "cannot be shifted on its own -- it only follows a shifted neighbour")
print(f"  {len(att)} buses with injection, {len(ext)} of them next to a zero-injection bus "
      f"-> attack must also shift {int(ext['n_extension'].min())}-{int(ext['n_extension'].max())} extra bus(es)")
for name, col in (("measurements, DC (of 259)", "n_meas_dc_hidden"), ("measurements, AC (of 518)", "n_meas_ac_hidden"),
                  ("RTUs, DC attack", "n_rtu_dc_hidden"), ("RTUs, AC attack", "n_rtu_ac_hidden")):
    print(f"  {name:26s}: {int(att[col].min())}-{int(att[col].max())}, median {att[col].median():.0f};  "
          f"next to zero-injection: median {ext[col].median():.0f}, "
          f"otherwise median {att.loc[att['n_extension'] == 0, col].median():.0f}")
print(f"  naive DC attack (zero-injection ignored): own residual ||r(a)||^2 = "
      f"{ext['J_attack_alone_dc_naive'].min():.1f} to {ext['J_attack_alone_dc_naive'].max():.0f} "
      f"(chi2 threshold {chi_square_test(estimate_dc(m, dmeas))['tau']:.0f}); on the baseline draw J changes by "
      f"{ext['dJ_dc_naive'].min():.1f} to {ext['dJ_dc_naive'].max():.0f} -- the cross term with the existing "
      "residual can be negative, so a small naive attack can hide in one draw")
print(f"  bus {config.ATTACK_BUS}: DC {int(b2['n_meas_dc_hidden'])} measurements / {int(b2['n_rtu_dc_hidden'])} RTUs, "
      f"AC {int(b2['n_meas_ac_hidden'])} measurements / {int(b2['n_rtu_ac_hidden'])} RTUs")

# --- figure ------------------------------------------------------------------------------------------
AC_COLOR, DC_COLOR, REF_COLOR, INK, MUTED = "#2a78d6", "#eb6834", "#52514e", "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.6, "savefig.dpi": 200, "legend.frameon": False})
x = np.arange(len(res))
w = 0.4
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.grid(axis="x", visible=False)
ax.bar(x - w / 2, res["n_meas_ac_hidden"].fillna(0), w, color=AC_COLOR, edgecolor="white", linewidth=0.5,
       label="AC attack: measurements to change (of 518)")
ax.bar(x + w / 2, res["n_meas_dc_hidden"].fillna(0), w, color=DC_COLOR, edgecolor="white", linewidth=0.5,
       label="DC attack: measurements to change (of 259)")
zi = res["zero_injection"].to_numpy()
ax.plot(x[zi], np.zeros(zi.sum()) + 1.5, "x", color=REF_COLOR, ms=6, mew=1.5,
        label="zero-injection bus: cannot be attacked on its own (only follows a neighbour)")
needs = res["n_extension"].fillna(0).to_numpy() > 0
top = np.maximum(res["n_meas_ac_hidden"].fillna(0), res["n_meas_dc_hidden"].fillna(0)).to_numpy()
ax.plot(x[needs], top[needs] + 3, "v", color=REF_COLOR, ms=5,
        label="attack must also shift zero-injection neighbours")
ax.set_xticks(x, res["bus"].astype(str), fontsize=6.5)
ax.set_xlim(-0.8, len(res) - 0.2)
ax.set_xlabel("bus whose angle the attacker shifts")
ax.set_ylabel("measurements the attacker must change")
ax.set_title(f"Hidden single-bus angle attack on the IEEE 57-bus system (t = {t} rad)", loc="left", fontsize=10)
ax.legend(loc="upper left", fontsize=7.5, ncol=2)
ax.set_ylim(0, top[np.isfinite(top)].max() * 1.35)
fig.tight_layout()
config.FIGURES_DIR.mkdir(exist_ok=True)
png = config.FIGURES_DIR / "q3_zero_injection.png"
fig.savefig(png)
plt.close(fig)
print(f"wrote {out_csv} and {png}")
if not all(ok):
    sys.exit("some checks FAILED")
