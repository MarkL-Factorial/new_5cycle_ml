"""Command-line entry point — the ONLY module that branches on mode.

Subcommands:
  cell-classifier run    --mode {validation|production} ...
  cell-classifier sweep  --sweep <yaml> [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from cell_classifier.utils.manifest import (
    hash_resolved_config, hash_resolved_config_ignoring_versions, read_manifest,
)
from cell_classifier.utils.paths import (
    make_run_dir, run_dir, run_slug, update_latest_symlink,
)
from cell_classifier.utils.seeds import SEEDS_PRESETS, resolve_seeds


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _load_template(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    tmpl = _load_template(Path(args.model_config))
    config: dict[str, Any] = dict(tmpl)

    # Required run axes
    config["mode"] = args.mode
    config["N"] = int(args.N)
    config["db_version"] = str(args.db_version)
    config["baseline_cycle"] = int(args.baseline_cycle)
    config["feature_subset"] = str(args.feature_subset)

    # tune overrides — collected here so they can be folded into the
    # `tuning` block below.
    tune_cfg = dict(config.get("tune", {}))
    if args.tune_optimize is not None:
        tune_cfg["optimize"] = args.tune_optimize
    if args.tune_n_trials is not None:
        tune_cfg["n_trials"] = int(args.tune_n_trials)
    if args.tune_inner_cv is not None:
        tune_cfg["inner_cv"] = int(args.tune_inner_cv)
    tune_cfg.setdefault("optimize", "f1")
    tune_cfg.setdefault("n_trials", 100)
    tune_cfg.setdefault("inner_cv", 5)

    # Tuning + provenance — nested blocks consumed by manifest.build_manifest.
    if args.mode == "validation":
        if args.tuning_protocol is None:
            raise SystemExit(
                "[error] --tuning-protocol is required for --mode validation "
                "(choices: nested_cv, tune_inner_cv)"
            )
        outer_cv_folds = int(args.outer_k) if args.tuning_protocol == "nested_cv" else None
        test_frac = float(args.test_frac) if args.tuning_protocol == "tune_inner_cv" else None
        config["tuning"] = {
            "protocol": args.tuning_protocol,
            "n_trials": int(tune_cfg["n_trials"]),
            "inner_cv_folds": int(tune_cfg["inner_cv"]),
            "outer_cv_folds": outer_cv_folds,
            "test_frac": test_frac,
            "optimize_metric": str(tune_cfg["optimize"]),
        }
        config["hp_provenance"] = {
            "source": None,
            "source_run_slug": None,
            "representative_strategy": None,
        }
    else:
        if args.production_params_source is None:
            raise SystemExit(
                "[error] --production-params-source is required for --mode production "
                "(choices: from_validation_run, retune)"
            )
        config["tuning"] = {
            "protocol": None,
            "n_trials": int(tune_cfg["n_trials"]),
            "inner_cv_folds": int(tune_cfg["inner_cv"]),
            "outer_cv_folds": None,
            "test_frac": None,
            "optimize_metric": str(tune_cfg["optimize"]),
        }
        # Source-run-slug + representative-strategy are determined entirely by
        # `source`. Resolving them here (not in the orchestrator) keeps the
        # idempotency hash stable across the gate-check and the manifest write.
        if args.production_params_source == "from_validation_run":
            source_run_slug = run_slug(
                model=config["model"], N=config["N"],
                db_version=config["db_version"],
                baseline_cycle=config["baseline_cycle"],
                feature_subset=config["feature_subset"],
            )
            representative_strategy = "mode_or_median_per_hp"
        else:
            source_run_slug = None
            representative_strategy = None
        config["hp_provenance"] = {
            "source": args.production_params_source,
            "source_run_slug": source_run_slug,
            "representative_strategy": representative_strategy,
        }

    # preprocessing
    preprocessing = dict(config.get("preprocessing", {}))
    if args.imputer_strategy is not None:
        preprocessing["imputer_strategy"] = args.imputer_strategy
    preprocessing.setdefault("imputer_strategy", "median")
    config["preprocessing"] = preprocessing

    # seeds
    literal = (
        [int(s) for s in args.seeds.split(",")] if args.seeds else None
    )
    config["seeds"] = resolve_seeds(preset=args.seeds_preset, literal=literal)

    # out_root
    if args.out_root is not None:
        config["out_root"] = str(args.out_root)
    config.setdefault("out_root", str(Path.cwd() / "results"))

    # slug + model_fixed_params for hashing
    config["slug"] = run_slug(
        model=config["model"], N=config["N"], db_version=config["db_version"],
        baseline_cycle=config["baseline_cycle"],
        feature_subset=config["feature_subset"],
    )
    # Stamp model_fixed_params from the class for hash determinism
    from cell_classifier.models.registry import get_model_class
    ModelClass = get_model_class(config["model"])
    config["model_fixed_params"] = dict(ModelClass.fixed_params)
    return config


# ---------------------------------------------------------------------------
# Idempotency gate
# ---------------------------------------------------------------------------

def _idempotency_check(
    lookup_dir: Path, config: dict[str, Any], force: bool, allow_version_drift: bool,
) -> str:
    """Returns one of {'fresh', 'skip', 'rerun'} or raises SystemExit.

    ``lookup_dir`` is the stable ``{out_root}/runs/{mode}/{slug}/`` path
    (a symlink to the most recent timestamped folder, or a legacy slug-only
    folder). 'rerun' means "create a fresh timestamped folder"; the prior
    run is preserved on disk and the symlink is repointed once the new run
    finishes.
    """
    prior = read_manifest(lookup_dir)
    if prior is None:
        return "fresh"

    expected = hash_resolved_config(config)
    actual = prior.get("resolved_config_sha256")
    if actual == expected:
        print(f"[skip] matches existing run: {lookup_dir}")
        return "skip"

    # Detect version-only drift
    expected_no_v = hash_resolved_config_ignoring_versions(config)
    prior_no_v = hash_resolved_config_ignoring_versions(prior)
    if expected_no_v == prior_no_v:
        msg = (
            f"[warn] resolved-config hash differs only in versions block "
            f"(prior {actual[:10]}… vs expected {expected[:10]}…)"
        )
        if allow_version_drift and not force:
            print(msg + " — accepted (--allow-version-drift)")
            return "skip"
        if force:
            print(msg + " — rerunning into a new timestamped folder (--force)")
            return "rerun"
        raise SystemExit(
            msg + ". Pass --allow-version-drift to accept, or --force to rerun."
        )

    if not force:
        raise SystemExit(
            f"[error] manifest at {lookup_dir} has different resolved_config_sha256 "
            f"(prior {actual[:10]}… vs expected {expected[:10]}…). "
            f"Pass --force to rerun (prior run is preserved)."
        )
    print(f"[force] rerunning; prior run preserved at {lookup_dir}")
    return "rerun"


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def _build_run_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", required=True, choices=["validation", "production"])
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--N", required=True, type=int, choices=[200, 300, 400])
    parser.add_argument("--db-version", required=True)
    parser.add_argument("--baseline-cycle", required=True, type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--feature-subset", required=True)
    parser.add_argument("--tuning-protocol", choices=["nested_cv", "tune_inner_cv"])
    parser.add_argument("--outer-k", type=int, default=5)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument(
        "--production-params-source",
        dest="production_params_source",
        choices=["from_validation_run", "retune"],
        default=None,
    )
    parser.add_argument("--tune.optimize", dest="tune_optimize", default=None)
    parser.add_argument("--tune.n-trials", dest="tune_n_trials", type=int, default=None)
    parser.add_argument("--tune.inner-cv", dest="tune_inner_cv", type=int, default=None)
    parser.add_argument("--imputer-strategy", default=None)
    seeds_group = parser.add_mutually_exclusive_group()
    seeds_group.add_argument(
        "--seeds-preset", choices=list(SEEDS_PRESETS), default="fresh",
    )
    seeds_group.add_argument(
        "--seeds", help="comma-separated literal seed list, e.g. 1,2,3",
    )
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-version-drift", action="store_true")


def _run(args: argparse.Namespace) -> int:
    # Line-buffer stdout/stderr so progress prints flush at each newline when
    # the CLI is invoked with output redirected to a file. Without this,
    # Python's default block-buffering hides per-seed / per-fold progress
    # for the full duration of multi-hour nested-CV runs.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    # If --seeds is set, override the default --seeds-preset
    if args.seeds is not None:
        args.seeds_preset = None
    config = _resolve_config(args)

    lookup_dir = run_dir(config["out_root"], config["mode"], config["slug"])
    decision = _idempotency_check(
        lookup_dir, config, force=args.force,
        allow_version_drift=args.allow_version_drift,
    )
    if decision == "skip":
        return 0

    timestamped_dir = make_run_dir(
        config["out_root"], config["mode"], config["slug"],
    )

    # Mode dispatch — the ONLY mode branch in the codebase.
    if config["mode"] == "validation":
        from cell_classifier.pipelines.validation import run_validation
        run_validation(config, out_dir=timestamped_dir)
    elif config["mode"] == "production":
        from cell_classifier.pipelines.production import run_production
        run_production(config, out_dir=timestamped_dir)
    else:
        raise SystemExit(f"[error] unknown mode {config['mode']!r}")

    # Point {slug} -> {slug}__{timestamp} so downstream lookups
    # (sweep aggregation, from_validation_run) see the latest run.
    update_latest_symlink(timestamped_dir, config["slug"])
    return 0


# ---------------------------------------------------------------------------
# Subcommand: sweep
# ---------------------------------------------------------------------------

def _run_sweep(args: argparse.Namespace) -> int:
    sweep_cfg = yaml.safe_load(Path(args.sweep).read_text())
    template_rel = sweep_cfg["template"]
    template_path = (Path(args.sweep).parent / template_rel).resolve()
    if not template_path.exists():
        # Fall back to configs/ alongside the sweep file
        template_path = Path(args.sweep).resolve().parent.parent / template_rel
    axes = sweep_cfg["axes"]
    fixed = sweep_cfg.get("fixed", {})
    sweep_id = sweep_cfg["sweep_id"]
    mode = sweep_cfg.get("mode", "validation")
    tuning_protocol = sweep_cfg.get("tuning_protocol")

    # Optional sweep-level overrides (apply to every combo)
    seeds_literal = fixed.get("seeds")          # e.g. [1, 2, 3]
    seeds_preset = fixed.get("seeds_preset")    # e.g. "fresh"
    tune_overrides = fixed.get("tune", {})       # {"n_trials": 5, "inner_cv": 3, "optimize": "f1"}
    outer_k = fixed.get("outer_k")               # int, for nested_cv
    test_frac = fixed.get("test_frac")           # float, for tune_inner_cv
    production_params_source = fixed.get("production_params_source")  # str, required for mode=production
    out_root = fixed.get("out_root")

    print(f"[sweep] sweep_id={sweep_id} axes={list(axes.keys())}")
    n_runs = 0
    for combo in product(*axes.values()):
        kw = dict(zip(axes.keys(), combo))
        argv = [
            "run",
            "--mode", mode,
            "--model-config", str(template_path),
            "--N", str(kw.get("N", fixed.get("N"))),
            "--db-version", str(kw.get("db_version", fixed.get("db_version"))),
            "--baseline-cycle", str(kw.get("baseline_cycle", fixed.get("baseline_cycle"))),
            "--feature-subset", str(kw.get("feature_subset", fixed.get("feature_subset"))),
        ]
        if mode == "validation" and tuning_protocol:
            argv += ["--tuning-protocol", tuning_protocol]
        if mode == "production" and production_params_source:
            argv += ["--production-params-source", production_params_source]
        if outer_k is not None:
            argv += ["--outer-k", str(outer_k)]
        if test_frac is not None:
            argv += ["--test-frac", str(test_frac)]
        if "n_trials" in tune_overrides:
            argv += ["--tune.n-trials", str(tune_overrides["n_trials"])]
        if "inner_cv" in tune_overrides:
            argv += ["--tune.inner-cv", str(tune_overrides["inner_cv"])]
        if "optimize" in tune_overrides:
            argv += ["--tune.optimize", str(tune_overrides["optimize"])]
        if seeds_literal is not None:
            argv += ["--seeds", ",".join(str(s) for s in seeds_literal)]
        elif seeds_preset is not None:
            argv += ["--seeds-preset", str(seeds_preset)]
        if out_root is not None:
            argv += ["--out-root", str(out_root)]
        if args.force:
            argv.append("--force")
        ret = main(argv)
        if ret != 0:
            print(f"[sweep] run failed; aborting at axes={kw}")
            return ret
        n_runs += 1

    # Aggregate (validation only — production has no metrics to aggregate)
    out_root = Path(fixed.get("out_root", Path.cwd() / "results"))
    if mode == "validation":
        _aggregate_sweep(out_root, sweep_id, axes, fixed)
    return 0


def _aggregate_sweep(out_root: Path, sweep_id: str, axes: dict, fixed: dict) -> None:
    """Build results/sweeps/{sweep_id}/ with metric_long.csv + manifest."""
    import pandas as pd
    sweep_dir = out_root / "sweeps" / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for combo in product(*axes.values()):
        kw = dict(zip(axes.keys(), combo))
        slug = run_slug(
            model=fixed.get("model", "random_forest"),
            N=kw.get("N", fixed.get("N")),
            db_version=kw.get("db_version", fixed.get("db_version")),
            baseline_cycle=kw.get("baseline_cycle", fixed.get("baseline_cycle")),
            feature_subset=kw.get("feature_subset", fixed.get("feature_subset")),
        )
        psm = out_root / "runs" / "validation" / slug / "per_seed_metrics.csv"
        if not psm.exists():
            continue
        df = pd.read_csv(psm)
        if "fold" in df.columns:
            df = df[df["fold"] == -1]
        df = df.assign(slug=slug, **kw)
        rows.append(df)
    if rows:
        long = pd.concat(rows, ignore_index=True)
        long.to_csv(sweep_dir / "metric_long.csv", index=False)
    (sweep_dir / "manifest.json").write_text(
        json.dumps(
            {"sweep_id": sweep_id, "axes": axes, "fixed": fixed,
             "n_runs_covered": len(rows)},
            indent=2,
        ) + "\n"
    )
    print(f"[sweep] wrote aggregate to {sweep_dir}")


# ---------------------------------------------------------------------------
# Top-level main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cell-classifier")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run a single experiment")
    _build_run_parser(run_p)

    sw_p = sub.add_parser("sweep", help="run a sweep over data axes")
    sw_p.add_argument("--sweep", required=True)
    sw_p.add_argument("--force", action="store_true")
    sw_p.add_argument("--allow-version-drift", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "run":
        return _run(args)
    if args.cmd == "sweep":
        return _run_sweep(args)
    raise SystemExit(f"[error] unknown subcommand {args.cmd!r}")


if __name__ == "__main__":
    sys.exit(main())
