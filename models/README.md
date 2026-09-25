# models/

Trained artefacts written by `python -m src.train` for **real** data:

| File | Content |
|---|---|
| `model.joblib` | the model selected during training (scikit-learn, joblib) |
| `candidates/*.joblib` | every candidate fitted on the training split |
| `metadata.json` | feature order, calibration, seed, split, cross-validation, data-file hashes, library versions |
| `evaluation.json` | held-out test metrics (written by `python -m src.evaluate`) |
| `embedded_model.json` / `.md` | exported decision logic + Python-vs-C check (written by `python -m src.export_embedded`) |

`models/demo/` holds the same files for the **synthetic demo** run (`--demo`). Demo artefacts are
never written into this folder or into `firmware/`, so they cannot be mistaken for real ones.

**Status: no model has been trained on real data yet, so there are no real artefacts here.**
