from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io_utils import save_csv


def main() -> None:
    with (REPO_ROOT / "results/02_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    save_csv(
        REPO_ROOT / "results/03_summary.csv",
        ["count", "total", "average"],
        [summary["count"], summary["total"], summary["average"]],
    )


if __name__ == "__main__":
    main()
