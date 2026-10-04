# Exported embedded model: `decision_tree`

- Header: `models/decision_tree_model.h`
- Data kind: REAL; trained 2026-10-02T10:41:55
- Input order: moisture_pct, temperature_c, humidity_pct
- Calibration: soil_raw_dry = 620 → 0 %, soil_raw_wet = 270 → 100 %

Decision tree after merging redundant branches: 5 comparisons, 6 leaves, depth 3. At most 3 comparisons run per decision.

## Decision logic

- if `moisture_pct <= 33`:
  - if `temperature_c <= 22.6`:
    - if `moisture_pct <= 29.57`:
      - → **WATER** (32 training samples)
    - else (`moisture_pct > 29.57`):
      - → **NO_WATER** (17 training samples)
  - else (`temperature_c > 22.6`):
    - → **WATER** (150 training samples)
- else (`moisture_pct > 33`):
  - if `moisture_pct <= 43.29`:
    - if `temperature_c <= 30.3`:
      - → **NO_WATER** (62 training samples)
    - else (`temperature_c > 30.3`):
      - → **WATER** (41 training samples)
  - else (`moisture_pct > 43.29`):
    - → **NO_WATER** (202 training samples)

## Python ↔ C consistency check

Not run (use without `--no-verify` and with g++ installed).
