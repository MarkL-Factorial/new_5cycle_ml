"""CLI entry point: `cell-lifetime run|production`.

Subcommands:
  - `run`         — single experiment with 80/20 stratified holdout (research surface)
  - `production`  — full production fit on trainable_n{N}, write per-cell predictions
                    to `cell_lifetime/results/run/<YYYYMMDD_HHMM>/`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from cell_lifetime.models.registry import get_model_class, registered_models


def _load_template(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _slug(model: str, task: str, N: int, db: str, baseline: int, fs: str) -> str:
    return f"{model}__{task}__N{N}__{db}_b{baseline}__{fs}"


def _resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    tmpl = _load_template(Path(args.model_config))
    config: dict[str, Any] = dict(tmpl)

    config["task"] = args.task
    config["N"] = int(args.N)
    config["db_version"] = str(args.db_version)
    config["baseline_cycle"] = int(args.baseline_cycle)
    config["feature_subset"] = str(args.feature_subset)

    # Tuning block
    tune_cfg = dict(config.get("tune", {}))
    if args.tune_n_trials is not None:
        tune_cfg["n_trials"] = int(args.tune_n_trials)
    if args.tune_inner_cv is not None:
        tune_cfg["inner_cv"] = int(args.tune_inner_cv)
    if args.tune_optimize is not None:
        tune_cfg["optimize"] = args.tune_optimize
    tune_cfg.setdefault("n_trials", 50)
    tune_cfg.setdefault("inner_cv", 5)
    tune_cfg.setdefault(
        "optimize", "roc_auc" if args.task == "classification" else "neg_mae"
    )
    config["tuning"] = {
        "protocol": "tune_inner_cv",
        "n_trials": int(tune_cfg["n_trials"]),
        "inner_cv_folds": int(tune_cfg["inner_cv"]),
        "test_frac": float(args.test_frac),
        "optimize_metric": str(tune_cfg["optimize"]),
    }

    # Preprocessing block
    pp = dict(config.get("preprocessing", {}))
    if args.imputer_strategy is not None:
        pp["imputer_strategy"] = args.imputer_strategy
    pp.setdefault("imputer_strategy", "median")
    if args.task == "regression":
        if args.target_transform is not None:
            pp["target_transform"] = args.target_transform
        pp.setdefault("target_transform", "log")
    config["preprocessing"] = pp

    # Seeds
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [1, 2, 3]
    config["seeds"] = seeds

    # Out path — anchor to the cell_lifetime/ package dir so the output
    # location is stable regardless of where the CLI is invoked from.
    # (Previously this concatenated "cell_lifetime/out" onto cwd, which
    # produced cell_lifetime/cell_lifetime/out when run from cell_lifetime/.)
    if args.out_root:
        out_root = Path(args.out_root)
    else:
        out_root = Path(__file__).resolve().parents[2] / "out"
    slug = _slug(
        config["model"], config["task"], config["N"], config["db_version"],
        config["baseline_cycle"], config["feature_subset"],
    )
    config["slug"] = slug
    config["out_root"] = str(out_root)

    # Validate model registration
    ModelClass = get_model_class(config["model"])
    declared_task = getattr(ModelClass, "task", "classification")
    if declared_task != args.task:
        raise SystemExit(
            f"[error] model {config['model']!r} is task={declared_task!r} "
            f"but --task is {args.task!r}. Registered models: {registered_models()}"
        )
    return config


def _build_run_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True, choices=["classification", "regression", "survival"])
    p.add_argument("--model-config", required=True)
    p.add_argument("--N", required=True, type=int, choices=[200, 300, 400])
    p.add_argument("--db-version", required=True)
    p.add_argument("--baseline-cycle", required=True, type=int, choices=[1, 2, 3, 4])
    p.add_argument("--feature-subset", required=True)
    p.add_argument("--tuning-protocol", default="tune_inner_cv", choices=["tune_inner_cv"])
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--tune.n-trials", dest="tune_n_trials", type=int, default=None)
    p.add_argument("--tune.inner-cv", dest="tune_inner_cv", type=int, default=None)
    p.add_argument("--tune.optimize", dest="tune_optimize", default=None)
    p.add_argument("--target-transform", default=None,
                   choices=["none", "log", "sqrt", "boxcox"])
    p.add_argument("--imputer-strategy", default=None)
    p.add_argument("--seeds", default=None,
                   help="comma-separated seed list, e.g. 1,2,3 (default 1,2,3)")
    p.add_argument("--out-root", default=None)


def _run(args: argparse.Namespace) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    config = _resolve_config(args)
    out_root = Path(config["out_root"])
    timestamp = __import__("datetime").datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_root / "runs" / config["task"] / f"{config['slug']}__{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from cell_lifetime.pipelines.validation import run_validation
    run_validation(config, out_dir=out_dir)

    latest_link = out_root / "runs" / config["task"] / config["slug"]
    if latest_link.is_symlink() or latest_link.exists():
        try:
            latest_link.unlink()
        except OSError:
            pass
    latest_link.symlink_to(out_dir.name)
    return 0


def _build_production_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--trials", type=int, default=30,
                   help="Optuna trials per ensemble member (default 30)")
    p.add_argument("--inner-cv", type=int, default=5,
                   help="K for KFold inner CV (default 5)")
    p.add_argument("--ensemble-seeds", type=int, default=5,
                   help="K for independent ensemble (default 5; K=1 disables ensembling)")
    p.add_argument("--baseline-cycle", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--db-version", default="A2.2")
    p.add_argument("--classifier-feature-subset", default="fs_a_only")
    p.add_argument("--rsf-feature-subset", default="fs_cv")
    p.add_argument("--out-root", default=None,
                   help="default: cell_lifetime/results/run")
    p.add_argument("--no-plots", action="store_true",
                   help="skip plot rendering")
    p.add_argument("--smoke", action="store_true",
                   help="fast mode: --trials 5 --inner-cv 3 --ensemble-seeds 1")


def _production(args: argparse.Namespace) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    trials = args.trials
    inner_cv = args.inner_cv
    ensemble_seeds = args.ensemble_seeds
    if args.smoke:
        trials = 5
        inner_cv = 3
        ensemble_seeds = 1

    if args.out_root:
        out_root = Path(args.out_root)
    else:
        out_root = Path(__file__).resolve().parents[2] / "results" / "run"
    out_root.mkdir(parents=True, exist_ok=True)

    # Use local wall-clock time so directory names match the operator's
    # clock (file mtimes are local; mismatched UTC names cause confusion).
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = out_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    from cell_lifetime.pipelines.production import run_production
    summary = run_production(
        out_dir=out_dir,
        trials=trials,
        inner_cv=inner_cv,
        ensemble_seeds=ensemble_seeds,
        baseline_cycle=args.baseline_cycle,
        db_version=args.db_version,
        classifier_feature_subset=args.classifier_feature_subset,
        rsf_feature_subset=args.rsf_feature_subset,
        make_plots=not args.no_plots,
    )

    # Update / create `latest` symlink pointing to the new run.
    latest = out_root / "latest"
    if latest.is_symlink() or latest.exists():
        try:
            latest.unlink()
        except OSError:
            pass
    latest.symlink_to(out_dir.name)
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, indent=2,
                     default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cell-lifetime")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run a single experiment (holdout)")
    _build_run_parser(run_p)
    prod_p = sub.add_parser(
        "production",
        help="full production fit on trainable_n{N}; predictions on all cells",
    )
    _build_production_parser(prod_p)

    args = ap.parse_args(argv)
    if args.cmd == "run":
        return _run(args)
    if args.cmd == "production":
        return _production(args)
    raise SystemExit(f"[error] unknown subcommand {args.cmd!r}")


if __name__ == "__main__":
    sys.exit(main())
