# Plot audit findings — 2026-05-20

User-led visual audit of the 19 `sustained` cells in
[`out/plots/sustained/`](out/plots/sustained/). Findings are grouped by
**behavioral pattern**, not by cell ID, because each pattern implies a
different fix in the labeling pipeline.

This file captures the *user's interpretation* of what each pattern means
physically. The algorithm response (what to do about each pattern) is
discussed separately in §"Algorithm implications" at the bottom.

---

## Pattern A — isolated measurement outliers (2–3 cycles, returns to trend)

**Cells**: `0MC20-251126-R001`, `AR-3422`, `AR3941` (×2 episodes at
~cycle 100 and ~cycle 200).

**Shape**: 1–3 regular cycles jump abruptly (up or down), then the curve
returns to the pre-jump fade trajectory within a few cycles. The
post-jump trend has the **same slope and intercept** as the pre-jump
trend — no permanent offset.

**Physical meaning**: measurement glitch or transient cycler anomaly.
The cell itself is healthy; only those few cycle records are bad.

**Correct action**: mark those specific cycles as **outliers** in the
regular-cycle list. The cell stays in the cohort, but those cycles
should not contribute to fade detection, baseline calculation, or
feature extraction.

**Note on current detector behavior**: when the outlier sits inside the
pre-window of an adjacent trigger, it contaminates the OLS fit and
produces a spurious `sustained` flag (see
`0MC20-251126-R001`: orange transient at cycle 332 is the real
event, the red sustained at cycle 333 is an artifact). The red flag is
**unnecessary**.

### `AR4084` — same family, with extra noise

First orange (transient) detection is **correct**: it reveals a cluster
of cycles where retention spuriously *increased* and then settled back.
The red sustained flag elsewhere is **unnecessary** — it's the same
"outlier contaminates pre-window" artifact as `0MC20-251126-R001`.

---

## Pattern B — rate change causing a step (cell becomes unusable)

**Cells**: `AR4135` (suspected; needs annotation cross-check).

**Shape**: a large discontinuous jump that coincides with (or is caused
by) a **rate change** recorded in the annotation. The pre- and post-step
curves are at different levels because they're measured at different
C-rates, not because the cell degraded or recovered.

**Physical meaning**: not a cell-degradation signal at all. The
discontinuity is an artifact of the protocol change.

**Correct action**: these cells are **completely unusable for ML**.
Drop the entire cell from the training cohort. The detector should
cross-reference the regime boundary list when classifying — if the jump
is colocated with a rate change, it is a different category from a
within-regime regime shift.

---

## Pattern C — within-regime regime shift (the AR4313 family)

**Cells**: `AR4142`, `AR4201`, `AR4313`. Suspected drivers: rate change
*or* temperature shift not captured at full resolution by the annotation
toolkit (or captured but below its tolerance — AR4313 is recorded as a
2.1% rate delta, see [DISCUSSION.md §1](DISCUSSION.md)).

**Shape**: clear step up (or down), with the post-step curve forming a
**new plateau / new fade trajectory** that is permanently offset from
the pre-step trajectory. Unlike Pattern B, the rate-change annotation
either doesn't exist or doesn't match the magnitude of the visible step.

**Physical meaning**: the cell is operating in a different regime after
the step. The pre-step segment is a clean, usable fade trajectory; the
post-step segment is a different cell behaviorally, even if the same
hardware.

**Correct action**: treat as a **censored cell** — use only the
pre-step segment for ML. Specifically: truncate the cell at the
sustained-jump cycle, then run fade detection on the truncated curve.
The cell's `last_fade_cycle` is whatever the pre-step segment dictates;
if it didn't fade before the step, the cell is right-censored at the
truncation cycle.

---

## Pattern D — coordinated post-step shift across multiple cells (`AR4143`–`AR4148`)

**Cells**: `AR4143`, `AR4144` (and by extension the `AR4145`–`AR4148`
cluster).

**Shape**: after the first detection (around cycle 277–281), the curve
sits on a **clearly shifted segment** — not a single-cycle bump, a
sustained new level. This is the same shape as Pattern C, just with a
smaller magnitude (−0.04 to −0.06 instead of ±0.1+).

**Physical meaning**: same as Pattern C — a regime shift. The cluster
of 6 AR cells shifting simultaneously at very similar cycles strongly
suggests a **batch-level protocol event** (same cycler, same operator
intervention, or coordinated rate change).

**Correct action**: same as Pattern C — **censor at the shift cycle**.
The usable ML segment is the pre-shift portion only.

---

## Pattern E — unusual early-life shapes (manual review required)

**Cells**: `AR4269` (cycle 16, Δ=−0.131, persist=+0.059),
`AR4389` (cycle 11, Δ=+0.036, persist=−0.317).

**Shape**: the trigger Δ and the persistence-score have **opposite
signs**, which is not the textbook step pattern. These look like
early-life instabilities (formation / break-in noise) rather than
regime shifts.

**Physical meaning**: ambiguous — could be formation artifacts, could be
genuinely defective cells with unstable early behavior. Cannot be
classified algorithmically with confidence.

**Correct action**: **manual visual inspection** of each, then
**force-exclude** from ML training via an explicit exclusion list. No
algorithmic rescue attempt.

---

## Algorithm implications (preview — discuss next)

Mapping the patterns to concrete pipeline changes:

| Pattern | Detector change | Labels.py change | Cohort change |
|---|---|---|---|
| **A** outliers | Add an *outlier-rejection* step before any sustained classification. Probably median-based pre-fit, or Hampel/MAD filter on the raw retention series. | None directly — outliers excluded from fade detection. | None. |
| **B** rate-change step | Cross-reference `regime_boundary_cycles` *before* classifying. If sustained jump is within ±K of a boundary AND the recorded rate delta is non-trivial, flag as `protocol_step` not `sustained`. | None — these cells are dropped entirely. | New exclusion category: `rate_change_step`. |
| **C / D** within-regime shift | Detector logic OK *if* outlier rejection (Pattern A) and rate-change cross-ref (Pattern B) are added first. The remaining true `sustained` flags are these. | New: truncate retention curve at sustained-jump cycle before fade detection. Add `truncation_cycle` and `truncation_reason` columns. | None — cell stays, just censored. |
| **E** ambiguous early-life | None — algorithm should not try. | None directly. | Manual `force_exclude.yaml` list with reasons. |

This implies a **layered pipeline**:

```
raw retentions
  → outlier rejection (Pattern A)
  → rate-change-aware step detection (Pattern B vs C/D)
  → manual force-exclude check (Pattern E)
  → truncation + fade detection (Pattern C/D)
  → labels
```

Open questions before implementing:

1. **Outlier rejection**: Hampel/MAD filter on retentions, or fit pre-window with
   robust regression (Theil-Sen) instead of OLS? The former gives us a
   per-cycle outlier flag we can store; the latter just makes the
   detector more robust.
2. **Rate-change tolerance**: how close (in cycles) must the sustained
   jump be to a regime boundary to count as `protocol_step`? Current
   `_is_near_boundary` uses ±2. Probably keep that, but also check the
   recorded `baseline_i_a` delta — a 2.1% delta (AR4313) is still
   Pattern C, not Pattern B.
3. **Temperature changes**: not currently in the annotation. Do we need
   to surface those from raw, or accept "we can't tell C from B for
   temperature-driven shifts" as a known limitation?
4. **Force-exclude file format**: YAML with `{cell: reason}` entries?
   Reuse the existing exclusion infrastructure in
   `battery-ml-prediction`?
