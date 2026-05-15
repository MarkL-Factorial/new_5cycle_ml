"""CLI entry point.

Usage:
    python run.py --config configs/rf_n300.yaml

YAML configs may reference a base via `extends: base.yaml` (deep-merged).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .pipeline import run_experiment

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text())
    if "extends" in raw:
        parent_rel = raw.pop("extends")
        parent_path = (path.parent / parent_rel).resolve()
        parent = load_config(parent_path)
        return _deep_merge(parent, raw)
    return raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    config = load_config(args.config)
    print(f"[run] config: {args.config}")
    print(f"[run] experiment_name: {config.get('experiment_name')}")
    print(f"[run] model: {config.get('model')}")

    out_dir = Path(config["out_dir"])
    if not out_dir.is_absolute():
        out_dir = (args.config.parent.parent / out_dir).resolve()
    config["out_dir"] = str(out_dir)

    run_experiment(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
