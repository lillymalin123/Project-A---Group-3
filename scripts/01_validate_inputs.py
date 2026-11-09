from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io_utils import load_case_values, save_json


def main() -> None:
    values = load_case_values(REPO_ROOT / "data/case_inputs.json")
    if not values:
        raise ValueError("Input case list must not be empty")
    save_json(
        REPO_ROOT / "results/01_validated.json",
        {"status": "ok", "count": len(values), "values": values},
    )


if __name__ == "__main__":
    main()
