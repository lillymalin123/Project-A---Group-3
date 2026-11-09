"""
loadCase.py --- load the MATPOWER IEEE 57-bus case as a pandapower network.

The case is read from data/case57.m (the full AC case). case57-se.json is not
used here: it only holds the reduced DC data (x, Pg, Pd, Va) and lacks the
resistances, line charging, reactive loads and voltage setpoints an AC net needs.
"""
import re
from pathlib import Path

import numpy as np
from pandapower.converter.pypower import from_ppc

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE57_PATH = REPO_ROOT / "data" / "case57.m"


def _read_matrix(text, name):
    """Numeric rows of `mpc.<name> = [ ... ];` from a MATPOWER file, as a 2-D array."""
    m = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, re.S)
    if m is None:
        raise ValueError(f"mpc.{name} not found")
    rows = []
    for line in m.group(1).splitlines():
        line = line.split("%")[0].replace(";", "").strip()
        if line:
            rows.append([float(v) for v in line.split()])
    return np.array(rows)


def load_case57(path=CASE57_PATH, f_hz=60):
    """
    Load a MATPOWER case file (.m) and return it as a pandapower net.

    Inputs:  path  path to the MATPOWER .m file (default data/case57.m)
             f_hz  system frequency in Hz (MATPOWER cases carry none; 60 for IEEE)
    Output:  pandapowerNet. Bus indices are 0-based (MATPOWER bus b -> net bus b-1);
             loads/gens in MW/MVAr, line parameters in ohm as converted by pandapower.
    """
    text = Path(path).read_text()
    ppc = {
        "version": "2",
        "baseMVA": float(re.search(r"mpc\.baseMVA\s*=\s*([\d.eE+-]+)", text).group(1)),
        "bus": _read_matrix(text, "bus"),
        "gen": _read_matrix(text, "gen"),
        "branch": _read_matrix(text, "branch"),
    }
    return from_ppc(ppc, f_hz=f_hz)


if __name__ == "__main__":
    net = load_case57()
    print(net)
