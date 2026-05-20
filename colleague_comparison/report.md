# Colleague-annotation investigation — findings

Audit of `colleague_annoation/all_features.parquet` (208 cells, "A")
against `ml_label_preprocess/datasets/A2.2_b1/` (459/470 cells, "B"),
with the per-cycle annotation registry as truth. 202 cells overlap.

**Bottom line:** B is reproducible from the registry to within float
precision. A disagrees with B on every concept we examined, and the
disagreements have **three distinct root causes**:

1. A uses different cycle indices than B (CE measured at cd_index=1
   instead of regular_cycle=5).
2. A's per-cell lifetime appears to come from a **partially stale
   snapshot** — most cells off by ~1 cycle (likely 0-vs-1 indexing),
   a sizeable minority off by hundreds (cells that kept cycling after A
   was exported).
3. A.retention does not match any simple formula reconstructible from
   the registry. The colleague almost certainly used a different
   baseline or a different upstream source — needs follow-up with the
   colleague to identify.

Each section below is one screen.

---

## 1. Coulombic efficiency

| Side | Column | Documented meaning | Verified against registry |
|---|---|---|---|
| A | `ce2` (fraction, mean 0.947) | implied: cycle-2 CE | Actually CE at **cd_index == 1** (the 2nd cd_event overall, typically the 2nd formation step). mean diff +0.00005, max\|d\| 0.062, only **2/202 cells** off by > 0.01. |
| B | `coulombic_efficiency_final` (percent, mean 99.4) | regular_cycle == 5 CE | Matches truth `ce_c5` **exactly** (max\|d\| = 0). |

A vs each candidate truth (lower mean\|d\| is better):

| Candidate | n | mean\|d\| | n with \|d\| > 0.01 |
|---|---:|---:|---:|
| `ce_c1` (regular_cycle=1)         | 202 | 0.0809 | 154 |
| `ce_c2` (regular_cycle=2)         | 202 | 0.0898 | 113 |
| `ce_c5` (regular_cycle=5)         | 202 | 0.0893 | 117 |
| **`ce_cdidx1` (2nd cd_event)**    | 202 | **0.0057** | **2** |
| `ce_form2` (2nd formation event)  | 199 | 4.095 | 83 |

A.ce2 is **not the 2nd regular-cycle CE**. It's the CE from the second
entry in the cd-event list — usually the 2nd formation step, before
"regular" cycling has even begun. CSV: `out/ce_per_cell.csv`. Plot:
[ce_distributions.png](out/ce_distributions.png).

A and B are talking about completely different cycles. The
fraction-vs-percent unit difference is a small annoyance on top.

---

## 2. Capacity retention

B is exact. `discharge_capacity_retention_final` reproduces truth
`cap_dis[c5] / cap_dis[c1]` with max\|d\| = 0 across all 202 cells.

A's `retention` (single scalar per cell, mean 0.879) **does not match
any formula we tested**. Best correlation across nine candidates:

| Candidate | n | Pearson r | mean\|d\| | bias |
|---|---:|---:|---:|---:|
| `ret_c5_truth` (c5/c1)              | 202 | +0.016 | 0.117 | -0.115 |
| `ret_clast_truth` (c_last/c1)       | 202 | +0.247 | 0.062 | +0.043 |
| `ret_min_truth` (min/c1)            | 202 | +0.164 | 0.066 | +0.048 |
| `ret_at_n200_truth`                 | 158 | +0.388 | 0.030 | -0.024 |
| **`ret_at_n300_truth`**             | 143 | **+0.405** | 0.014 | +0.001 |
| `ret_at_n400_truth`                 | 124 | +0.221 | 0.036 | +0.020 |
| `ret_at_A_snapshot_truth` *         | 202 | +0.301 | 0.052 | +0.032 |
| `final_retention` (B labels)        | 200 | +0.236 | 0.061 | +0.044 |
| `discharge_capacity_retention_final` (B features) | 202 | +0.016 | 0.117 | -0.115 |

\* retention at `A.max_regular_cycle` — the "A was a snapshot" hypothesis.

The best candidate (`ret_at_n300_truth`) only reaches r = +0.40 — too
weak to call this the definition. Bias near zero + tiny mean\|d\| only
mean A.retention values cluster in the same range as the truth; they
do **not** track per-cell.

Likely explanations (need colleague confirmation):

- A's retention may be normalised against rated/design capacity
  (e.g. 2.0 Ah for these cells), not against `cap_dis[c1]`.
- A's retention may come from a different upstream version of the same
  data (different renumbering, different cycle-counter resets).
- A's retention may be a smoothed/averaged value over a window of
  late-life cycles, not a single-point read.

CSV: `out/retention_per_cell.csv`. Plot:
[retention_definitions.png](out/retention_definitions.png).

**Recommendation:** ask the colleague how `retention` is computed. Until
then, do not treat A.retention as comparable to any B retention.

---

## 3. Fade cycle / lifetime

| Concept | A | B | Verified |
|---|---|---|---|
| Lifetime / max regular cycle | `max_regular_cycle` (mean 408) | `n_regular` (label-side) | **B matches truth on 202/202 cells. A matches truth on 9/202.** |
| Sticky 0.85 fade cycle       | *not in A*           | `last_fade_cycle`      | B's fade detector matches our locally-recomputed truth **exactly on all 148/148 cells** where both sides report a fade. |

A.max_regular_cycle vs truth is the most informative single chart in
this audit. Gap distribution `(truth - A.max_regular_cycle)`:

```
count   mean    std    min    25%   50%   75%    max
202    +48.8   85.0   -117    +1    +1   +161   +273
```

Two distinct populations:

- **~half of cells**: gap = +1 exactly. Looks like a **0-vs-1 indexing
  convention** difference (A counts from 0, B/registry from 1) on cells
  that haven't cycled since A was generated.
- **~quarter of cells**: gap > 100 (some up to +273). These cells kept
  cycling after A was exported — A is **stale** for them.
- **4 cells**: gap is negative (A claims more cycles than B). Either a
  data-refresh truncation or an extra event A counted that B didn't
  classify as a regular cycle.

A has no fade-cycle column at all — `max_regular_cycle` is lifetime,
not a fade indicator. So the original question "do A and B agree on the
fade cycle" has a trivial answer: A doesn't have one to compare.

Independent witness from the registry: of A's 71 cells labelled BAD,
**64 (90%) have a truth fade cycle**. Of A's 131 cells labelled GOOD,
**84 (64%) have a truth fade cycle** — i.e. nearly two-thirds of A's
"GOOD" cells have since faded in the truth. This is the snapshot-staleness
effect leaking into the labels (see §4).

CSV: `out/fade_per_cell.csv`. Plot: [fade_scatter.png](out/fade_scatter.png).

---

## 4. Categorical label (A.label vs B.label_n{N})

A.label ∈ {GOOD, BAD} with no stated cycle threshold. We crosstab against
B's per-N labels at N ∈ {200, 300, 400}.

Confusion at **N=300** (the best agreement — 10 disagreements / 202):

```
B.label_n300 →  bad  censor  excluded  pass
A_label
BAD              58       6         1     6
GOOD              3       0         1   127
```

Confusion at **N=200** (15 disagreements explained below):

```
B.label_n200 →  bad  censor  excluded  pass
A_label
BAD              49       6         1    15      ← A:BAD ∩ B:pass = 15
GOOD              0       0         1   130
```

Confusion at **N=400** (45 disagreements — snapshot staleness blows up):

```
B.label_n400 →  bad  censor  excluded  pass
A_label
BAD              59       6         1     5
GOOD             23       9         1    98      ← A:GOOD ∩ B:bad = 23
```

Reading the pattern:

- **A's labeling rule is `BAD iff A.retention < 0.85`** — 207/208 cells
  match this rule exactly; the single mismatch is `0MC2-251022-004`
  (whose retention=0.319 is itself a data bug). A.label is NOT a
  cycle-survival decision; it is a **point-in-time retention threshold**
  at A's snapshot cycle.
- **A and B disagree because of rule TYPE, not threshold value.** A
  asks "is retention currently below 0.85?". B asks "did retention
  ever cross 0.85 and fail to recover ≥3 times after?". Both rules
  use the same 0.85 number but answer different questions.
- **N=300 disagreements (9 cells) decompose into:**
  - 6 cells (AR4143–AR4148): cells faded at cycles 395–482, so B says
    pass@N=300. But A's snapshot is taken *after* the fade, so A's
    retention is < 0.85 and A says BAD. Neither side is wrong by its
    own rule.
  - 3 cells (AR3775/AR3858/AR4229): cells crossed 0.85 sticky-fade at
    cycles 256–295 (B says bad@N=300), but their retention later
    recovered to ≥ 0.85 (A.retention = 0.851/0.851/0.863). A's
    snapshot rule doesn't see permanently-faded; B's sticky-recovery
    rule does.
- **At N=200 the 15 A:BAD ∩ B:pass cells are mostly the same effect**:
  A's snapshot retention < 0.85 even though sticky fade hasn't fired
  before cycle 200.
- **At N=400 the 23 A:GOOD ∩ B:bad cells** include both the snapshot
  staleness (A's snapshot pre-dates fade onset) and the recovery
  effect (A snapshot taken when retention briefly recovered).

CSVs: `out/label_confusion.csv`, `out/label_disagreements.csv`. Plot:
[label_confusion.png](out/label_confusion.png).

---

## Verdict (one sentence per concept)

| Concept | Verdict |
|---|---|
| **CE** | B is exact (truth.ce_c5). A's `ce2` is CE at cd_index=1, not cycle 2 — meaningfully different. Unit difference (fraction vs percent) is secondary. |
| **Retention** | B is exact (`cap_dis[c5]/cap_dis[c1]`). A's `retention` definition could not be inferred from the data — ask the colleague. |
| **Fade cycle** | B's `last_fade_cycle` is exact against the documented sticky rule. A has no fade-cycle column; its `max_regular_cycle` is a stale lifetime, off by +1 cycle on most cells and by hundreds on cells that kept cycling. |
| **Label** | A's rule is `BAD iff A.retention < 0.85` (207/208 match; the lone exception is the corrupt `0MC2-251022-004`). It's a **point-in-time retention test**, NOT a cycle-survival decision — A doesn't reason about N. Agreement with B@N=300 is 95.4% (185/194 strict, dropping censor/excluded). The disagreements come from a rule-type mismatch (point-in-time vs sticky-crossing), not a threshold-value mismatch. |

**Use B as the source of truth** for any downstream modeling. Treat A
as a frozen reference from an earlier point — it may still be useful
for the EIS/DOP/dQdV features that B doesn't carry, but the
shared-name columns are not interchangeable.

## 5. Cells selected for manual review

11 cells across 5 disagreement categories — pick a handful to spot-check
in the cycler / annotation viewer. Each row in
[`cells_to_review.csv`](out/cells_to_review.csv) carries A's values,
B's values, and the registry truth for the same cell.

Retention curves: [`cells_to_review_curves.png`](out/cells_to_review_curves.png)
(red dotted line = 0.85 threshold, orange dashed = A.max_regular_cycle,
purple dashed = B.last_fade_cycle).

| Cell | Category | A.label | A.ret | A.max | B@300 | B@400 | truth.n | truth.fade | Why this cell |
|---|---|---|---:|---:|---|---|---:|---:|---|
| AR3775 | cat1 GOOD-vs-bad@300 | GOOD | 0.851 | 369 | bad | bad | 370 | 295 | borderline; truth fades 5 cycles before 300 |
| AR3858 | cat1 GOOD-vs-bad@300 | GOOD | 0.851 | 449 | bad | bad | 449 | 288 | A snapshot is current, but A still called it GOOD |
| AR4229 | cat1 GOOD-vs-bad@300 | GOOD | 0.863 | 387 | bad | bad | 388 | 256 | early fade (256), A still GOOD |
| AR3771 | cat2 BAD-vs-pass@200 | BAD | 0.837 | 340 | bad | bad | 341 | 279 | fades 279 — A's BAD threshold > N=200 |
| AR4147 | cat2 BAD-vs-pass@200 | BAD | 0.834 | 523 | pass | pass | 520 | 482 | passes 200/300, fades at 482, A still BAD |
| AR4168 | cat3 stale GOOD→bad@400 | GOOD | 0.874 | 459 | pass | bad | 563 | 356 | already faded at 356 well before A's snapshot at 459 |
| AR4131 | cat3 stale GOOD→bad@400 | GOOD | 0.865 | 505 | pass | bad | 518 | 317 | faded at 317 in truth; A snapshot at 505 still GOOD |
| **AR4193** | cat4 negative gap | GOOD | 0.953 | **430** | pass | censor | **313** | – | A claims 117 cycles registry doesn't see |
| **AR4194** | cat4 negative gap | GOOD | 0.950 | **430** | pass | censor | **314** | – | identical pattern to AR4193 (sibling cells?) |
| **0MC2-251022-004** | cat5 retention anomaly | GOOD | **0.319** | 571 | pass | pass | 842 | 797 | A.retention = 0.32 but truth = 0.91 — likely data corruption |
| AR4313 | cat5 retention anomaly | GOOD | **1.148** | 482 | pass | bad | 472 | 400 | A.retention > 1.0; truth ret@300 = 0.94 |

**Three rows worth investigating with the colleague first:**

1. **`0MC2-251022-004`** — A reports retention = 0.319, but registry
   says this cell is healthy (~0.91 throughout). Either a column-shift
   bug in A's export, a wrong unit (`1 - retention`?), or the colleague
   used a different upstream source for this cell.
2. **`AR4193` & `AR4194`** — both have `A.max_regular_cycle = 430` but
   the registry has only 313–314 regular cycles. Where did 117 extra
   cycles come from? Maybe rate_test or pulse cycles counted as
   regulars in A.
3. **`AR4147`** — A:BAD but the cell actually passed both N=200 and
   N=300 (fades at 482). What's A's BAD criterion? If it's "will the
   cell fade before some N", what N? Likely N=500 from this case +
   `AR3771` (fades 279, also BAD).

## 6. Deep-dive plots (recomputed from raw)

For each of the 11 review cells we loaded the renumbered step-level
parquet (`data/A2.2/parquet_renumbered/{cell}.parquet`) via the
toolkit's `load_raw_tagged`, then recomputed `capacity_charge_ah`,
`capacity_discharge_ah`, and `coulombic_efficiency` per regular cycle
by trapezoid-integrating `current·dt`. This is an independent witness
to the annotation registry.

**Sanity check passed across all 11 cells:** max abs diff vs the
registry is **0.0 Ah** on every cycle (`out/deep_dive_summary.csv`).
The registry is reproducible from raw to float precision — so every
A-vs-B disagreement is about *interpretation*, not measurement.

Per-cell 4-panel PNGs at [`out/deep_dive/`](out/deep_dive):

```
(1) capacity per cycle      |  (2) CE per cycle
    chg + dis (recomp)      |     recomputed (line)
    registry (dots overlay) |     registry (dots), A.ce2 (horiz),
                            |     B.CE_final/100 marker @ c5
─────────────────────────────────────────────────────────────────
(3) retention per cycle     |  (4) voltage profile at key cycles
    cap_dis / cap_dis[c1]   |     V vs time-within-cycle for
    0.85 threshold          |     cycles {1, 5, A.max, B.last_fade}
    A.retention horiz        |
    B markers @ c5, c_last   |
```

All panels (1-3) also carry: grey dotted refs at N=200/300/400,
orange dashed at `A.max_regular_cycle`, purple dashed at
`B.last_fade_cycle`.

### What the priority cells actually show

- **[`0MC2-251022-004`](out/deep_dive/0MC2-251022-004.png)** — raw
  data confirms 842 regular cycles of healthy decay, retention drops
  smoothly past 0.85 around cycle 797. **A.retention = 0.319 is
  unrelated to anything visible in the raw curve.** Almost certainly a
  data-integrity issue on A's side (column misalignment, wrong cell,
  or wrong unit).

- **[`AR4193`](out/deep_dive/AR4193.png) + [`AR4194`](out/deep_dive/AR4194.png)**
  — capacity curves end cleanly at regular_cycle 313 / 314 with no
  visible artefacts. The 117-cycle gap to `A.max_regular_cycle = 430`
  cannot be reconciled from the raw data. Likely A is counting
  non-regular events (formation / rate_test / pulse) toward its
  max, or its source had additional cycles that were later dropped.

- **[`AR4313`](out/deep_dive/AR4313.png)** — raw discharge capacity
  early in life sits below cycle-1's capacity, so retention briefly
  exceeds 1.0 in the recompute too. A's value 1.148 is at the same
  magnitude as the local maximum — so A's anomaly is *consistent* with
  the data, just selected at an unusual point.

- **[`AR4147`](out/deep_dive/AR4147.png)** — cell passes 200, 300, and
  starts faded sticky at cycle 482; A labels it BAD despite surviving
  past 300. The voltage profile at the fade cycle shows the rollover
  cleanly. Reinforces "A.BAD ≈ will-fade-by-N≈500" hypothesis.

- **cat1 cells ([`AR3775`](out/deep_dive/AR3775.png), [`AR3858`](out/deep_dive/AR3858.png),
  [`AR4229`](out/deep_dive/AR4229.png))** — all three show clean fade
  trajectories crossing 0.85 in cycles 256–295, well before N=300.
  A's retention values (≈0.85) are very near the threshold but A
  still called them GOOD — likely A's rule rounds up or uses a
  more permissive recovery window than B's `recovery_min=3`.

## Files

```
out/
├── ce_per_cell.csv               (202 × ~12)
├── ce_distributions.png
├── retention_per_cell.csv        (202 × ~16)
├── retention_definitions.png
├── fade_per_cell.csv             (202 × ~8)
├── fade_scatter.png
├── label_confusion.csv           (16 rows: A_label × B_label × N)
├── label_disagreements.csv       (85 rows: every off-diagonal cell, with witness)
├── label_confusion.png
├── cells_to_review.csv           (11 picks across 5 categories, full context)
├── cells_to_review_curves.png    (retention curve per pick, annotated)
├── deep_dive_summary.csv         (11 rows: recompute-vs-registry sanity)
└── deep_dive/                    (11 per-cell 4-panel PNGs)
    ├── 0MC2-251022-004.png
    ├── AR3771.png
    ├── AR3775.png
    ├── AR3858.png
    ├── AR4131.png
    ├── AR4147.png
    ├── AR4168.png
    ├── AR4193.png
    ├── AR4194.png
    ├── AR4229.png
    └── AR4313.png
```

Reproduce: see [README.md](README.md).
