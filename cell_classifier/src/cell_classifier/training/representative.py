"""Representative HP selection across seeds / seeds x folds.

Used by `production.py` to collapse a validation run's per-seed (or per-fold)
hyperparameter set into a single representative set for production training.
Strategy: per-HP mode for non-float types (int, bool, str, None), per-HP
median for float types. Dtype is inferred from a template HP dict — typically
the model class's `suggest_params(trial=DummyTrial())` output is unavailable
without an Optuna study, so we use the first row of the loaded HP table as
the template instead.

Also exposes (de)serialization helpers so HP values round-trip through CSV
without losing the None / "None" distinction (e.g., `class_weight=None` vs
`class_weight="None"`). Each HP cell stores the JSON literal of the value.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_STRATEGY_LABEL = "mode_or_median_per_hp"


def serialize_hp_row(params: dict[str, Any]) -> dict[str, str]:
    """Encode each HP value as a JSON literal so CSV roundtrips losslessly."""
    return {k: json.dumps(v) for k, v in params.items()}


def deserialize_hp_row(row: dict[str, Any], hp_columns: list[str]) -> dict[str, Any]:
    """Decode each HP cell from its JSON literal. NaN cells become None."""
    out: dict[str, Any] = {}
    for k in hp_columns:
        cell = row[k]
        if isinstance(cell, float) and np.isnan(cell):
            out[k] = None
            continue
        out[k] = json.loads(cell)
    return out


def write_hp_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    index_columns: list[str],
    hp_columns: list[str],
) -> None:
    """Write a flat CSV. `index_columns` (e.g. ['seed'] or ['seed','fold']) come
    first verbatim; `hp_columns` come second JSON-encoded.
    """
    serialized = []
    for r in rows:
        out = {c: r[c] for c in index_columns}
        out.update(serialize_hp_row({c: r[c] for c in hp_columns}))
        serialized.append(out)
    pd.DataFrame(serialized, columns=[*index_columns, *hp_columns]).to_csv(
        path, index=False
    )


def read_hp_csv(
    path: Path,
    *,
    index_columns: list[str],
    hp_columns: list[str],
) -> list[dict[str, Any]]:
    """Read a CSV written by `write_hp_csv` back into JSON-decoded dicts."""
    df = pd.read_csv(path, dtype={c: str for c in hp_columns}, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for _, raw in df.iterrows():
        row: dict[str, Any] = {c: raw[c] for c in index_columns}
        for c in hp_columns:
            cell = raw[c]
            row[c] = json.loads(cell)
        rows.append(row)
    return rows


def representative_hp_set(
    hp_rows: list[dict[str, Any]],
    template: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Return (representative_params, strategy_label).

    For each key in `template`: median if the template value is a float
    (excluding bool, which subclasses int but is not numeric here); otherwise
    mode. Float-mode columns also fall back to mode when no numeric values
    remain after filtering Nones.
    """
    rep: dict[str, Any] = {}
    for k, v_template in template.items():
        values = [row[k] for row in hp_rows]
        if isinstance(v_template, float) and not isinstance(v_template, bool):
            numeric = [
                v for v in values
                if isinstance(v, (int, float)) and not isinstance(v, bool)
                and v is not None
            ]
            if numeric:
                rep[k] = float(np.median(numeric))
                continue
        counter = Counter(values)
        rep[k] = counter.most_common(1)[0][0]
    return rep, _STRATEGY_LABEL


def hp_summary(
    hp_rows: list[dict[str, Any]],
    template: dict[str, Any],
) -> dict[str, Any]:
    """Per-HP diagnostic: {mode, median (numeric only), range (numeric only), n_unique}."""
    out: dict[str, Any] = {}
    for k in template:
        values = [row[k] for row in hp_rows]
        mode_val, _ = Counter(values).most_common(1)[0]
        info: dict[str, Any] = {
            "mode": mode_val,
            "n_unique": len({json.dumps(v, sort_keys=True) for v in values}),
        }
        numeric = [
            v for v in values
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and v is not None
        ]
        if numeric:
            info["median"] = float(np.median(numeric))
            info["range"] = [float(min(numeric)), float(max(numeric))]
        out[k] = info
    return out
