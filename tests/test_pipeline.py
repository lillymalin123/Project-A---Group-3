from __future__ import annotations

import csv
import json
import subprocess
import unittest
from pathlib import Path


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        scripts = [
            "01_validate_inputs.py",
            "02_compute_summary_json.py",
            "03_export_summary_csv.py",
        ]
        for script in scripts:
            subprocess.run(
                ["python", str(cls.repo_root / "scripts" / script)],
                check=True,
            )

    def test_results_files_exist(self) -> None:
        self.assertTrue((self.repo_root / "results/01_validated.json").exists())
        self.assertTrue((self.repo_root / "results/02_summary.json").exists())
        self.assertTrue((self.repo_root / "results/03_summary.csv").exists())

    def test_summary_matches_quoted_numbers(self) -> None:
        with (self.repo_root / "results/02_summary.json").open("r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["total"], 60.0)
        self.assertEqual(summary["average"], 20.0)

    def test_csv_matches_json_summary(self) -> None:
        with (self.repo_root / "results/02_summary.json").open("r", encoding="utf-8") as f:
            summary = json.load(f)

        with (self.repo_root / "results/03_summary.csv").open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        self.assertEqual(int(row["count"]), summary["count"])
        self.assertEqual(float(row["total"]), summary["total"])
        self.assertEqual(float(row["average"]), summary["average"])


if __name__ == "__main__":
    unittest.main()
