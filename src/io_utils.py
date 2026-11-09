from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def load_case_values(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [float(item["value"]) for item in payload["cases"]]


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_csv(path: Path, headers: Iterable[str], row: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(headers))
        writer.writerow(list(row))
