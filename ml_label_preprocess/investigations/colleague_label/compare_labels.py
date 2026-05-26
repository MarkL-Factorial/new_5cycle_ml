"""Compare our N=300 labels against a colleague's GOOD/BAD labels.

Inner-joins on `cell_name` and emits agreement statistics, a mismatch
list, and three plots under `out/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
OURS_PARQUET = HERE.parent.parent / "datasets" / "A2.2_b1" / "A2.2_b1_latest" / "cell_labels.parquet"
THEIRS_PARQUET = HERE / "all_features.parquet"
OUT_DIR = HERE / "out"
ANOMALY_CELL = "0MC2-251022-004"

CATEGORY_ORDER = [
    "agree_pass",
    "agree_bad",
    "disagree_we_pass_they_bad",
    "disagree_we_bad_they_pass",
    "ours_censor",
    "ours_excluded",
]
CATEGORY_COLOR = {
    "agree_pass":                "#2ca02c",
    "agree_bad":                 "#1f77b4",
    "disagree_we_pass_they_bad": "#d62728",
    "disagree_we_bad_they_pass": "#ff7f0e",
    "ours_censor":               "#7f7f7f",
    "ours_excluded":             "#bcbd22",
}
OURS_LABEL_ORDER = ["pass", "bad", "censor", "excluded"]
COHORT_ORDER = ["AR", "0MC2", "other"]


def load_ours() -> pl.DataFrame:
    return pl.read_parquet(OURS_PARQUET).select(
        "cell_name",
        "status",
        "exclusion_reason",
        "last_fade_cycle",
        "final_retention",
        "n_regular",
        "truncation_cycle",
        "label_n300",
        "trainable_n300",
    )


def load_theirs() -> pl.DataFrame:
    return (
        pl.read_parquet(THEIRS_PARQUET)
        .select("cell_name", "label", "retention", "max_regular_cycle")
        .rename({
            "label": "colleague_label",
            "retention": "colleague_retention",
            "max_regular_cycle": "colleague_max_cycle",
        })
        .with_columns(pl.col("colleague_label").str.to_lowercase())
    )


def join_inner(ours: pl.DataFrame, theirs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    merged = theirs.join(ours, on="cell_name", how="inner")
    colleague_only = theirs.join(ours.select("cell_name"), on="cell_name", how="anti")
    if not colleague_only.is_empty():
        print(f"warning: {colleague_only.height} colleague cells not found in our labels")
    return merged, colleague_only


def assign_category(df: pl.DataFrame) -> pl.DataFrame:
    ours = pl.col("label_n300")
    theirs = pl.col("colleague_label")
    category = (
        pl.when(ours == "excluded").then(pl.lit("ours_excluded"))
        .when(ours == "censor").then(pl.lit("ours_censor"))
        .when((ours == "pass") & (theirs == "good")).then(pl.lit("agree_pass"))
        .when((ours == "bad") & (theirs == "bad")).then(pl.lit("agree_bad"))
        .when((ours == "pass") & (theirs == "bad")).then(pl.lit("disagree_we_pass_they_bad"))
        .when((ours == "bad") & (theirs == "good")).then(pl.lit("disagree_we_bad_they_pass"))
        .otherwise(pl.lit("unknown"))
        .alias("category")
    )
    return df.with_columns(category)


def add_cohort(df: pl.DataFrame) -> pl.DataFrame:
    cohort = (
        pl.when(pl.col("cell_name").str.starts_with("AR")).then(pl.lit("AR"))
        .when(pl.col("cell_name").str.starts_with("0MC2")).then(pl.lit("0MC2"))
        .otherwise(pl.lit("other"))
        .alias("cohort")
    )
    return df.with_columns(cohort)


def compute_summary(df: pl.DataFrame, colleague_only: pl.DataFrame, theirs_total: int) -> dict:
    n_total = df.height
    cat_counts = {c: 0 for c in CATEGORY_ORDER}
    for row in df.group_by("category").len().iter_rows():
        cat_counts[row[0]] = row[1]

    primary = df.filter(~pl.col("category").is_in(["ours_censor", "ours_excluded"]))
    n_primary = primary.height
    n_agree = int(primary.filter(pl.col("category").str.starts_with("agree_")).height)
    n_disagree = n_primary - n_agree
    agreement_pct = (n_agree / n_primary) if n_primary else 0.0

    confusion: dict[str, dict[str, int]] = {
        c: {l: 0 for l in OURS_LABEL_ORDER} for c in ("good", "bad")
    }
    for theirs, ours, n in (
        df.group_by(["colleague_label", "label_n300"]).len().iter_rows()
    ):
        if theirs in confusion and ours in confusion[theirs]:
            confusion[theirs][ours] = n

    per_cohort: dict[str, dict] = {}
    for cohort in COHORT_ORDER:
        sub = df.filter(pl.col("cohort") == cohort)
        sub_primary = sub.filter(~pl.col("category").is_in(["ours_censor", "ours_excluded"]))
        sub_agree = int(sub_primary.filter(pl.col("category").str.starts_with("agree_")).height)
        per_cohort[cohort] = {
            "n_total": sub.height,
            "n_primary": sub_primary.height,
            "n_agree": sub_agree,
            "n_disagree": sub_primary.height - sub_agree,
            "agreement_pct": (sub_agree / sub_primary.height) if sub_primary.height else None,
        }

    return {
        "n_colleague_total": theirs_total,
        "n_joined": n_total,
        "n_colleague_only": colleague_only.height,
        "colleague_only_cells": colleague_only["cell_name"].to_list(),
        "n_primary_pool": n_primary,
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "agreement_pct": agreement_pct,
        "category_counts": cat_counts,
        "confusion_matrix": confusion,
        "per_cohort": per_cohort,
        "anomaly_cell": ANOMALY_CELL,
        "anomaly_in_data": ANOMALY_CELL in df["cell_name"].to_list(),
    }


def write_tables(df: pl.DataFrame, colleague_only: pl.DataFrame, summary: dict) -> None:
    df_sorted = df.sort(["category", "cell_name"])
    df_sorted.write_parquet(OUT_DIR / "comparison_table.parquet")
    df_sorted.write_csv(OUT_DIR / "comparison_table.csv")

    # mismatches.csv: every row where the colleague's label can't be
    # confirmed by ours — disagree_*, plus ours_censor / ours_excluded
    # rows (where we suspend the comparison rather than agreeing).
    mismatches = df_sorted.filter(~pl.col("category").str.starts_with("agree_"))
    mismatches.write_csv(OUT_DIR / "mismatches.csv")

    colleague_only.sort("cell_name").write_csv(OUT_DIR / "colleague_only_cells.csv")

    (OUT_DIR / "summary_stats.json").write_text(json.dumps(summary, indent=2))


def plot_confusion_matrix(df: pl.DataFrame, out_path: Path) -> None:
    matrix = np.zeros((2, len(OURS_LABEL_ORDER)), dtype=int)
    for i, theirs in enumerate(("good", "bad")):
        for j, ours in enumerate(OURS_LABEL_ORDER):
            matrix[i, j] = df.filter(
                (pl.col("colleague_label") == theirs) & (pl.col("label_n300") == ours)
            ).height

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(OURS_LABEL_ORDER)), OURS_LABEL_ORDER)
    ax.set_yticks([0, 1], ["good", "bad"])
    ax.set_xlabel("ours: label_n300")
    ax.set_ylabel("colleague: label")
    ax.set_title(f"Confusion matrix — colleague vs our label_n300 (n={df.height})")
    thresh = matrix.max() / 2 if matrix.max() else 1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=11,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_retention_vs_cycle(df: pl.DataFrame, out_path: Path) -> None:
    plot_df = df.with_columns([
        pl.coalesce(pl.col("truncation_cycle"), pl.col("colleague_max_cycle")).alias("x_cycle"),
        pl.coalesce(pl.col("final_retention"), pl.col("colleague_retention")).alias("y_ret"),
    ])

    fig, ax = plt.subplots(figsize=(10, 6))
    for cat in CATEGORY_ORDER:
        sub = plot_df.filter(pl.col("category") == cat)
        if sub.is_empty():
            continue
        ax.scatter(
            sub["x_cycle"].to_numpy(),
            sub["y_ret"].to_numpy(),
            c=CATEGORY_COLOR[cat],
            label=f"{cat} (n={sub.height})",
            alpha=0.7,
            s=28,
            edgecolor="none",
        )

    ax.axvline(300, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0.85, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(305, ax.get_ylim()[0] + 0.02, "N=300", fontsize=9, alpha=0.7)
    ax.text(ax.get_xlim()[1] * 0.98, 0.855, "ret=0.85",
            ha="right", fontsize=9, alpha=0.7)

    anomaly = plot_df.filter(pl.col("cell_name") == ANOMALY_CELL)
    if not anomaly.is_empty():
        ax_x = float(anomaly["x_cycle"][0])
        ax_y = float(anomaly["y_ret"][0])
        ax.annotate(
            ANOMALY_CELL,
            xy=(ax_x, ax_y),
            xytext=(ax_x - 200, ax_y + 0.05),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
        )

    ax.set_xlabel("max regular cycle (truncation_cycle / colleague_max_cycle)")
    ax.set_ylabel("final retention")
    ax.set_title(f"Retention vs. cycle by agreement category (n={df.height})")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_agreement(df: pl.DataFrame, out_path: Path) -> None:
    cohorts = COHORT_ORDER
    buckets = ["agree", "disagree", "ours_censor", "ours_excluded"]
    counts = {b: [] for b in buckets}
    totals = []
    for cohort in cohorts:
        sub = df.filter(pl.col("cohort") == cohort)
        totals.append(sub.height)
        counts["agree"].append(int(sub.filter(pl.col("category").str.starts_with("agree_")).height))
        counts["disagree"].append(int(sub.filter(pl.col("category").str.starts_with("disagree_")).height))
        counts["ours_censor"].append(int(sub.filter(pl.col("category") == "ours_censor").height))
        counts["ours_excluded"].append(int(sub.filter(pl.col("category") == "ours_excluded").height))

    x = np.arange(len(cohorts))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = {
        "agree": "#2ca02c",
        "disagree": "#d62728",
        "ours_censor": "#7f7f7f",
        "ours_excluded": "#bcbd22",
    }
    for i, b in enumerate(buckets):
        ax.bar(x + (i - 1.5) * width, counts[b], width, label=b, color=bar_colors[b])

    ax.set_xticks(x, [f"{c}\n(n={n})" for c, n in zip(cohorts, totals)])
    ax.set_ylabel("cells")
    ax.set_title("Agreement breakdown by cohort")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ours = load_ours()
    theirs = load_theirs()
    theirs_total = theirs.height
    merged, colleague_only = join_inner(ours, theirs)
    merged = assign_category(merged)
    merged = add_cohort(merged)

    summary = compute_summary(merged, colleague_only, theirs_total)
    write_tables(merged, colleague_only, summary)

    plot_confusion_matrix(merged, OUT_DIR / "confusion_matrix.png")
    plot_retention_vs_cycle(merged, OUT_DIR / "retention_vs_cycle_scatter.png")
    plot_cohort_agreement(merged, OUT_DIR / "cohort_agreement_bar.png")

    cat_sum = sum(summary["category_counts"].values())
    assert cat_sum == merged.height, f"category counts sum to {cat_sum}, not {merged.height}"
    assert merged.height + colleague_only.height == theirs_total, \
        f"join accounting: {merged.height} + {colleague_only.height} != {theirs_total}"
    anomaly_row = merged.filter(pl.col("cell_name") == ANOMALY_CELL)
    if not anomaly_row.is_empty():
        anomaly_cat = anomaly_row["category"][0]
        print(f"anomaly cell {ANOMALY_CELL}: category={anomaly_cat} "
              f"(colleague retention={float(anomaly_row['colleague_retention'][0]):.3f}, "
              f"ours label_n300={anomaly_row['label_n300'][0]})")

    n_agree = summary["n_agree"]
    n_primary = summary["n_primary_pool"]
    pct = summary["agreement_pct"]
    print(f"colleague cells total : {theirs_total}")
    print(f"joined with ours       : {merged.height}")
    print(f"colleague-only (missing in ours): {colleague_only.height}")
    print(f"category counts: {summary['category_counts']}")
    print(f"agreement (excl censor/excluded): {pct:.1%}  ({n_agree}/{n_primary})")


if __name__ == "__main__":
    main()
