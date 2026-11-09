from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import average_values, count_values, sum_values
from src.io_utils import save_json


def main() -> None:
    with (REPO_ROOT / "results/01_validated.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)

    values = [float(v) for v in payload["values"]]
    summary = {
        "count": count_values(values),
        "total": sum_values(values),
        "average": average_values(values),
    }
    save_json(REPO_ROOT / "results/02_summary.json", summary)


if __name__ == "__main__":
    main()
