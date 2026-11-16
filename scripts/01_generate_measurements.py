"""Generate the baseline noisy measurement set ONCE and freeze it in data/ (guideline rule 7).

Run from anywhere:  python scripts/01_generate_measurements.py
Power flow -> true measurements h(x_true) -> add N(0, sigma^2) noise from config.SEED.
Writes data/measurements_baseline.csv (518 rows). Refuses to overwrite: the file is committed and
every later result (AC, DC, attacks) uses this same draw.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # src/ holds flat modules
import config
from ac_state_estimation import build_model, make_measurements, save_measurements

if config.MEASUREMENTS_CSV.exists():
    sys.exit(f"{config.MEASUREMENTS_CSV} already exists: it is frozen. Delete it by hand only if you "
             "really intend to redraw (then re-run every script).")
m = build_model()
meas = make_measurements(m, np.random.default_rng(config.SEED))
save_measurements(meas, config.MEASUREMENTS_CSV)
print(f"wrote {config.MEASUREMENTS_CSV}: {len(meas)} measurements, seed {config.SEED}")
print(meas["type"].value_counts().to_string())
