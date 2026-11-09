# Project-A---Group-3

This repository follows a simple, reproducible structure:

- `README.md` is the entry point
- `LICENSE` and `CITATION.cff` are in the root
- `environment/` contains dependency specification
- `src/` contains reusable functions (one job each)
- `scripts/` contains ordered scripts (`01_...`, `02_...`, `03_...`) with one result per script
- `data/` stores committed case inputs
- `results/` stores generated CSV/JSON outputs
- `figures/` stores generated figures used in the report
- `docs/` stores derivations and decisions
- `tests/` verifies scripts and quoted report numbers

## Quick start

```bash
python scripts/01_validate_inputs.py
python scripts/02_compute_summary_json.py
python scripts/03_export_summary_csv.py
python -m unittest discover -s tests
```
