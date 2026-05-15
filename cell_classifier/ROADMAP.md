# cell_classifier roadmap

## v0.1 (current)

- src/ layout, `pyproject.toml`, optional-dep extras
- BaseModel + registry; RandomForestModel only (EBM/BART are guarded stubs)
- Two pipeline modes (validation, production)
- Two tuning protocols (tune_inner_cv, nested_cv)
- sklearn `Pipeline([imputer, estimator])` inside each model
- Idempotency via SHA-256 of resolved config
- Sweep over data axes (N, baseline_cycle, db_version, feature_subset)

## v0.2 (next)

- **Calibration**: `CalibratedClassifierCV` option for production probas;
  Brier score + reliability curve in validation
- **Per-class metrics**: precision/recall for class 0 (bad) and confusion
  matrix in `summary.json`
- **Cross-algorithm sweep**: sweep YAMLs gain a `model` axis; paired-t
  comparison across model pairs
- **Real EBM body**: `interpret.glassbox.ExplainableBoostingClassifier`,
  with its native explain_local SHAP
- **Real BART body**: `pymc-bart` posterior; `predict_proba_samples()`
  returns the full posterior

## Beyond v0.2

- Feature importance stability across seeds/folds (bootstrap CIs on perm
  importance)
- Cell-level error analysis dashboard (which cells are systematically
  misclassified)
- Survival-analysis framing (use right-censored cells, not just discard
  them) — would change the data layer and probably the model interface
- Online/incremental retraining as new annotations arrive
