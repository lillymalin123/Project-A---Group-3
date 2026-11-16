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
