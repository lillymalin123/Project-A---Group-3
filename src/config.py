"""
config.py --- every numerical parameter of the AC state-estimation experiments, in one place
(project guideline rule 5). Scripts and src/ functions import these; nothing numeric is typed elsewhere.
Units: p.u. on baseMVA, radians.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root (this file lives in src/)

# Network case
CASE57_PATH = ROOT / "data" / "case57.m"  # MATPOWER IEEE 57-bus case, supplied with the brief
BASE_KV_FALLBACK = 100.0  # kV; replaces baseKV = 0 in case57.m (p.u. results do not depend on it)
F_HZ = 60  # Hz; system frequency (MATPOWER cases carry none; IEEE cases are 60 Hz)

# Random draws (record SEED in the README, rule 7)
SEED = 2026  # CHOOSE: any integer, but fix it and report it

# Measurement noise: standard deviations of the Gaussian errors (project brief, baseline)
SIGMA_POWER = 0.01  # p.u., P and Q flows and injections
SIGMA_VMAG = 0.01  # p.u., voltage magnitudes
SIGMA_ANGLE = 0.01  # rad, voltage angles

# Gauss-Newton iteration of the AC estimator
NEWTON_TOL = 1e-8  # stop when max |dx| (p.u. / rad) falls below this
NEWTON_MAX_ITER = 50

# Bad-data detection
ALPHA = 0.05  # chi-square test level: threshold = 95 % quantile of chi2(dof)
LNR_THRESHOLD = 3.0  # largest-normalized-residual test: flag |r_i|/sqrt(Omega_ii) above this (CHOOSE, common value)
CRITICAL_VAR_FRACTION = 1e-8  # Omega_ii below this fraction of sigma_i^2 = critical measurement (r_i has no information)

# Files
MEASUREMENTS_CSV = ROOT / "data" / "measurements_baseline.csv"  # frozen noisy draw, never regenerated
RESULTS_DIR = ROOT / "results"

# Tolerances of the self-checks in scripts/02_ac_state_estimation.py
JACOBIAN_FD_STEP = 1e-7  # finite-difference step for the Jacobian check
JACOBIAN_FD_TOL = 1e-4  # allowed |analytic - finite difference| (dominated by the step error)
RECOVERY_TOL = 1e-7  # noise-free data must recover the true state to this (p.u. / rad)
CONSTRAINT_TOL = 1e-9  # zero-injection constraints must hold to this (p.u.)
DOF_CHECK_TOL = 1e-6  # relative tolerance of the check sum(w_i * Omega_ii) = dof

# Attack settings
ATTACK_BUS = 2 # Location of the attack

# Size of the angle shift (rad). CHOSEN so the DC-attacked flow on the line nearest ATTACK_BUS changes by
# about 50% of its true value -- comparable to the worked example in Hug & Giampapa 2012, Sec. VI-B.
# Picked once against the frozen baseline draw; record/justify in the report, do not re-tune per draw.
ATTACK_ANGLE_SHIFT_RAD = 0.0143

# Q3 --- Monte Carlo check of the chi-square test without attack (project brief, Sec. 4)
MC_SIGMAS = (0.01, 0.02, 0.05, 0.1)  # p.u. for P, Q and Vm, rad for Va: one value for every measurement type
MC_N_DRAWS = 2000  # independent noise draws per sigma
MC_SEED = 1122  # CHOOSE: any integer, but fix it and report it (README). Separate from SEED: the baseline
#                 draw stays frozen; each sigma gets its own child stream (numpy SeedSequence.spawn)
MC_DRAWS_CSV = RESULTS_DIR / "q3_mc_draws.csv"  # one row per (sigma, draw), written by scripts/06
MC_REFERENCE_CSV = RESULTS_DIR / "q3_mc_reference.csv"  # one row per sigma: dof, thresholds, DC noncentrality
FIGURES_DIR = ROOT / "figures"

# Q3 --- zero-injection buses and the measurements a hidden attack must change
ATTACK_SUPPORT_TOL = 1e-9  # |a_i| above this (p.u. / rad) counts as a measurement the attacker must change

# Q5 sweep: magnitudes t of the angle shift at ATTACK_BUS (rad), 3 points per decade from 1e-4 to 1 rad
ATTACK_MAGNITUDES = [10 ** (k / 3) for k in range(-12, 1)]
SWEEP_DJ_TOL = 1e-6  # self-check: the noise-free nonlinear attack must change J by less than this

SLOPE_FIT_T_MAX = 1e-3     # rad; fit the slope of log|dJ| against log t only below this (small-attack regime)
PREDICTION_T = 1e-4        # rad; t at which dJ is predicted (Lecture 6) and compared; must be in ATTACK_MAGNITUDES
SHIFT_RATIO_T = 1e-2       # rad; t at which the estimated angle change / t is reported
PREDICTION_REL_TOL = 0.05  # self-check: |dJ_pred - dJ_sim| / |dJ_sim| must be below this