# Capacity-jump detection — discussion & recommendations

Snapshot of the design discussion on 2026-05-19. Saved for review before
deciding the next step.

---

## 1. The trigger: cell AR4313

User observation: discharge retention shows a large step at cycle 267.
Cell should probably be excluded from labels that depend on properties
measured after that cycle.

### Verified data

AR4313 is classified `cycling_consistency: single_rate` because the two
recorded `regular_rate_regimes` differ by only 2.1% in baseline current:

| seg_id | n_regular_cd | baseline_i_a (A) |
|---:|---:|---:|
| 0 | 267 | 0.04414 |
| 1 | 205 | 0.04506 |

But the actual discharge capacity jumps **+15%** at the regime boundary:

| cycle | cap_dis (Ah) | retention |
|---:|---:|---:|
| 266 | 0.1040 | 0.801 |
| 267 | 0.1039 | 0.800 |
| **268** | **0.1198** | **0.922** |
| 269 | 0.1217 | 0.937 |
| 300 | 0.1217 | 0.937 |
| 400 | 0.1102 | 0.848 |
| 472 | 0.0958 | 0.737 |

A 2.1% rate change cannot produce a 15% capacity change. Something else
happened at cycle 267 — protocol change, voltage-window change,
reconditioning, temperature step — that the annotation toolkit missed
because the recorded rate delta is within its tolerance.

### Effect on current labels

Current row in `datasets/A2.2_b1/cell_labels.parquet`:

```
status              = faded
last_fade_cycle     = 400
n_recovered_crossings = 2
label_n200          = pass
label_n300          = pass     ← MISLABELED
label_n400          = bad
```

The recovery-aware fade detector in
`labels.py::_last_crossing_into_bad` treats the cycles 268+ "zombie life"
as legitimate recovery from the cycle-218 crossing. That pushes
`last_fade_cycle` from ~218 → 400.

If we honestly trust only cycles ≤ 267, fade lands at ~218 →
- `label_n200`: `pass` (218 > 200) — unchanged
- `label_n300`: `bad` (218 < 300) — **flipped from pass**
- `label_n400`: `bad` (218 < 400) — unchanged

Only `label_n300` is mislabeled, but that's enough to motivate fixing it.

---

## 2. Design tension: pathological vs. transient

User's concern, correctly raised: "we need to be careful to distinguish a
jump that we exclude the cell from a normal recovery that we include the
cell."

### Two patterns look superficially similar

| Pattern | Mechanism | Shape | Action |
|---|---|---|---|
| **Normal recovery** | RPT, calendar rest, reconditioning | Single-cycle bump (1–4%), returns to pre-bump fade trajectory within ~5–10 cycles | Keep the cell |
| **Pathological regime shift** | Protocol change, rate change, voltage-window change | Step function, no return; curve permanently offset for the whole post-jump life | Truncate the cell |

### The discriminating axis is persistence, not magnitude

A magnitude-only threshold would falsely flag legitimate large recoveries
in some chemistries. Instead: **fit a linear trend to the cycles before
the jump, extrapolate forward, measure whether the post-jump curve
returns to that line.**

- RPT bump: post-window cycles fall back to the extrapolated pre-trend.
  Median residual ≈ 0 → `transient` → keep.
- Pathological step: post-window cycles sit permanently offset from the
  extrapolated pre-trend. Median residual ≈ jump amplitude →
  `sustained` → truncate.

For AR4313 at cycle 268: pre-slope ≈ −0.0008/cycle, extrapolated
retention at cycle 268 ≈ 0.799, actual = 0.922 → residual = +0.12, and
the median residual over cycles 268–277 stays ≈ +0.14. Far above the
classification threshold.

---

## 3. Algorithm (final form, implemented in `detector.py`)

For each cell, walk regular cycles 1..N:

```
1. Trigger:   |ret[i] - ret[i-1]| >= bump_min                 → candidate
2. Pre-fit:   OLS on cycles [i - pre_window, i - 1]           → slope, intercept
3. Project:   extrap[j] = slope * cycle[j] + intercept   for j in [i, i+post_window)
4. Residual:  resid[j]  = actual[j] - extrap[j]
5. Score:     persist   = median(resid)
6. Classify:
   |persist| >= persist_min  →  'sustained'    (would warrant truncation)
   |persist| <  persist_min  →  'transient'    (RPT-like, keep)
   either window too short    →  'edge_skip'   (no judgement)
7. Direction: 'up' if Δ > 0 else 'down'  (preserved separately from sign of persist)
```

**Bi-directional symmetry**: both `bump_min` and `persist_min` use
absolute values, so upward and downward step shifts are treated
identically. The signed `jump_magnitude` and `jump_direction` columns
preserve the direction for filtering.

### Initial parameters

| Param | Default | Notes |
|---|---:|---|
| `bump_min` | 0.03 | Sensitive on purpose — RPT bumps still surface as candidates, classification filters them |
| `persist_min` | 0.03 | The discrimination knob — tune this first |
| `pre_window` | 20 cycles | Long enough for a stable fade-rate fit |
| `post_window` | 10 cycles | Long enough for RPT bumps to decay |
| `min_pre_len` / `min_post_len` | 10 / 5 | Edge guards |

---

## 4. Dry-run results (2026-05-19, 470 cells, defaults)

### Per-cell classification

| class | count |
|---|---:|
| sustained | 19 |
| transient | 21 |
| edge_skip | 30 |
| none | 400 |

### Sustained cells (full list, sorted by cell_name)

| cell | cycle | dir | Δ | persist | near boundary |
|---|---:|---|---:|---:|---|
| 0MC20-251126-R001 | 333 | up | +0.410 | +0.099 | False |
| 0MC20-260203-R001 | 54 | up | +0.055 | +0.056 | False |
| 0MC6-260323-R007 | 78 | up | +0.090 | +0.103 | True |
| AR-3422 | 136 | up | +0.042 | +0.051 | False |
| AR3941 | 111 | up | +0.049 | +0.031 | False |
| AR4084 | 297 | down | -0.054 | -0.058 | False |
| AR4135 | 24 | down | -0.549 | -0.764 | True |
| AR4142 | 83 | up | +0.360 | +0.241 | True |
| AR4143 | 277 | down | -0.063 | -0.065 | False |
| AR4144 | 278 | down | -0.041 | -0.042 | False |
| AR4145 | 280 | down | -0.064 | -0.066 | False |
| AR4146 | 279 | down | -0.042 | -0.044 | False |
| AR4147 | 281 | down | -0.064 | -0.066 | False |
| AR4148 | 281 | down | -0.053 | -0.054 | False |
| AR4201 | 438 | up | +0.322 | +0.219 | True |
| AR4256 | 302 | up | +0.571 | +0.151 | False |
| AR4269 | 16 | down | -0.131 | +0.059 | False |
| **AR4313** | **268** | **up** | **+0.122** | **+0.141** | **True** |
| AR4389 | 11 | up | +0.036 | -0.317 | False |

Key observations:

- **AR4313 lands exactly as predicted** — cycle 268, magnitude +0.12,
  persistence +0.14, regime boundary coincides. The detector works for
  the motivating case.
- **AR4143–AR4148 cluster** — six AR cells all flagged sustained-down at
  cycles 277–281 with similar magnitudes (−0.04 to −0.06). Looks like a
  batch-level protocol event. Worth verifying in the plots whether these
  are genuine regime shifts or a synchronized RPT-like dip the detector
  is over-calling.
- **AR4269 / AR4389** have small jump magnitude but very large
  persistence-score magnitude with mismatched signs — these are
  pathological in shape but not the "step function" pattern AR4313 shows.
  Could be early-life instabilities; needs eyeballing.
- **5/19 sustained cells coincide with a recorded regime boundary** —
  boundary alignment is a useful but not dominant signal; the
  persistence test is doing most of the discrimination work.

### Multi-regime single_rate cohort (the 69 cells)

| class | count |
|---|---:|
| sustained | 4 |
| transient | 6 |
| edge_skip | 14 |
| none | 45 |

45/69 unchanged confirms most are benign (rate delta <1%, no visible step).

---

## 5. Open questions for tomorrow

1. **Are the 19 sustained flags all true positives?** Eyeball the plots,
   especially:
   - **AR4143–AR4148 cluster**: confirm whether the simultaneous
     downward dips look like protocol events or a coordinated RPT.
   - **AR4135** (Δ = −0.549, cycle 24): the magnitude is huge and the
     cycle is early. May be a cycler-glitch or an early-cycle protocol
     change.
   - **AR4269 / AR4389**: the persistence-score sign disagrees with the
     trigger Δ sign — these are unusual shapes, possibly very early
     instabilities rather than regime shifts.

2. **Should we adjust thresholds?**
   - `persist_min = 0.03` flagged 19 cells. If after eyeballing some look
     like false positives, raise to 0.05.
   - `bump_min = 0.03` is sensitive on purpose; raising it can hide
     small persistent shifts. Probably leave as-is.

3. **Should we also act on `transient` cells?** The plan says no — they
   are RPT-like and keeping them is correct. But it might be worth
   listing them in the labels output as a diagnostic flag
   (`n_rpt_bumps`) for downstream filtering experiments.

4. **What about the 30 `edge_skip` cells?** These have jumps too close
   to start or end of life to fit. Tomorrow: are any of these worth
   special handling, or is "edge_skip → label unchanged" acceptable?

5. **Does truncation actually fix `label_n300` for AR4313 as predicted?**
   Before wiring it into `labels.py`, do a manual one-cell prototype:
   take AR4313's retentions[0:267], rerun `_last_crossing_into_bad`,
   check that `last_fade_cycle` lands at ~218 and the labels flip as
   expected.

6. **Beyond AR4313, what's the aggregate impact?** Of the 19 sustained
   cells, how many would have their `label_n{N}` change for at least one
   N if we truncated? This is the most important number for deciding
   whether truncation is worth the complexity — if it's 1–3 cells, maybe
   a manual override list is cleaner than a general detector.

---

## 6. Approach options for tomorrow

Pick one (or some combination):

### Option A — Manual override list (smallest blast radius)

Add a small YAML file listing specific cells with truncation cycles:
```yaml
truncations:
  AR4313:
    truncate_at_regular_cycle: 267
    reason: "sustained capacity jump at regime boundary"
```
`labels.py` loads it, truncates the retention curve before fade
detection. Audit-friendly, narrowly scoped, low risk.

### Option B — Wire the detector into labels.py (general fix)

Move `detector.py`'s `detect_jumps` into the production pipeline. For
each cell, find the earliest `sustained` candidate; if one exists,
truncate at that cycle and rerun the fade detector on the truncated
curve. Add `label_truncation_cycle` and `label_truncation_reason`
columns to `cell_labels.parquet`.

Pros: catches AR4143–AR4148 and others automatically.
Cons: more code, depends on parameter stability, larger label-shift
surface.

### Option C — Hybrid

Default to detector-driven truncation (Option B), with a YAML override
file (Option A) for cells where the user disagrees with the detector or
wants to force-include / force-exclude.

### Option D — Do nothing automated, just document

Add a `KNOWN_OUTLIERS.md` listing cells with suspicious patterns and let
downstream ML scripts decide whether to filter. No `labels.py` change.

---

## 7. Recommendation

**Start with Option A (manual override)** for AR4313 specifically — it
unblocks the labels regression today and is fully reversible. In
parallel, hand-audit the other 18 sustained cells using the plots in
`out/plots/sustained/`. After that audit, decide whether the broader
pattern is real enough to justify Option B/C, or whether the override
list stays small and Option A is sufficient.

The detector code in this folder is the **diagnostic tool**, not
necessarily the production pipeline. Keeping it standalone means we can
iterate parameters and re-run without touching production until we are
confident.

---

## 8. 圖片檢視優先順序 (Plot-audit priority)

從**最重要到次要**排列。每個資料夾回答不同的診斷問題,順序反映了
每個答案會如何影響下一步決策。

### 第一優先 — `out/plots/sustained/` (19 張圖,全看)

**為什麼最重要:** 這 19 顆 cell 是偵測器**主動標記為「應該截斷」**的對象。
如果這裡有 false positive (誤判),會讓健康的 cell 被錯誤排除,直接污染
label 品質。**這個資料夾決定整套方法是否可信。**

19 張圖內部還有優先順序:

**金標準 (必看):**
- **`AR4313.png`** — 引發整個調查的案例。先看這張,確認演算法畫出來的
  pre-fit 線和 extrapolation 跟你預期的一樣 (cycle 268 處有紅色垂直線,
  cycle 247–267 的 pre-trend 沿著實線往下走,虛線往右延伸,實際曲線
  高出虛線一大截)。這是教科書級的 `sustained` step 模式。

**高度可疑的 cluster (一定要看):**
- **`AR4143.png`, `AR4144.png`, `AR4145.png`, `AR4146.png`,
  `AR4147.png`, `AR4148.png`** — 這 6 顆 AR 系列 cell 在 cycle 277–281
  之間**同時**出現 sustained-down,幅度都差不多 (−0.04 ~ −0.06)。
  - **為什麼可疑:** 同一個 cohort、同一段時間、相似的幅度、相似的方向
    → 很像是 batch-level 的測試協議事件 (例如同一台 cycler 換了某個
    設定),**也可能**是正常的協同 RPT 行為被 `sustained` 誤判。
  - **要判斷的事:** 曲線是「永久平移下移」(真 regime shift) 還是
    「每顆 cell 各自 dip 之後恢復回 pre-trend」(假陽性)?
    如果是後者,代表 `PERSIST_MIN = 0.03` 太敏感。

- **`AR4135.png`** (cycle 24, Δ=−0.549) — 幅度大到離譜 (retention
  直接掉一半),但發生在非常早期 (cycle 24)。
  - **為什麼可疑:** 可能是 cycler glitch、early-life 短路、測量錯誤,
    **也可能**這顆 cell 真的在生命早期就壞掉了。
  - **要判斷的事:** 看完整曲線。如果 retention 掉到 ~0.4 之後就一直
    在那邊 cycling 到底,代表 cell 真壞掉 (這顆其實應該整顆排除,
    不只是截斷)。如果只是 cycle 24 一個離群點然後曲線重新爬回正常,
    那是測量錯誤,`sustained` 判定是錯的。

- **`AR4269.png`** (cycle 16, dir=down, Δ=−0.131, persist=**+0.059**)
  和 **`AR4389.png`** (cycle 11, dir=up, Δ=+0.036, persist=**−0.317**)
  - **為什麼可疑:** 這兩顆的 trigger Δ 方向跟 persistence 殘差方向
    **不一致**。AR4269 是「先掉但後來坐在 pre-trend 上方」,
    AR4389 是「先小升但後來掉到 pre-trend 下方很多」。
  - 這不是教科書上的 step 模式 → 不確定演算法做的 `sustained` 判斷
    是否合理。
  - **要判斷的事:** 看曲線形狀是不是「early-life 不穩定」(很常見,
    前 10 cycle 包含 formation/breakin,曲線本來就跳)。如果是,
    演算法在不該觸發的地方觸發了。

- **`AR4142.png`** (cycle 83, Δ=+0.360), **`AR4201.png`**
  (cycle 438, Δ=+0.322), **`AR4256.png`** (cycle 302, Δ=+0.571),
  **`0MC20-251126-R001.png`** (cycle 333, Δ=+0.410)
  - **為什麼可疑:** 這幾顆的 jump magnitude 都非常大 (0.3 ~ 0.6)。
    可能是真的 protocol 大改 (例如換電壓窗口),也可能是 measurement
    glitch。
  - **要判斷的事:** 看圖的*形狀*。如果是「正常衰退到一半突然飛回
    retention=0.9 然後再正常衰退」,那是跟 AR4313 同類型的 sustained
    step,正確。如果是「孤立的單一 cycle 跳起來」,代表 post_window
    內某幾個點剛好被拉高了 median,演算法抓到 glitch 而非真正的 step。

**第一優先內部的低優先 (順手看):**
- 其他 4–5 顆 (例如 `AR-3422`、`AR3941`、`0MC6-260323-R007`),
  幅度比較小,看圖確認是真的 step 就好。

### 第二優先 — `out/plots/transient/` (21 張圖,抽 5 張看)

**為什麼第二優先:** 這 21 顆被歸為 `transient` (「RPT 式瞬間波動,
保留 cell」)。如果這裡面有任何一張看起來像 AR4313 式的 sustained step,
就是 **false negative** → `PERSIST_MIN` 太鬆。

不需要每張都看。**隨機挑 5 張**,確認每張都長得像:
- 單一 cycle 的尖峰
- 之後 ~5 個 cycle 內回到 pre-jump trend
- **post-jump 之後沒有永久性的垂直平移**

如果 5 張裡有任何一張看起來像 step pattern,代表演算法漏抓 → 需要
降低 `PERSIST_MIN`。

### 第三優先 — `out/plots/multi_regime_audit/` (69 張圖,從 `none` 子集挑 5 張)

**為什麼第三優先:** 這個資料夾回答最一開始的問題 *「除了 AR4313,
是不是還有其他類似的 cell 我們沒注意到?」* — 收集所有 69 顆
`single_rate` 但有多個 `regular_rate_regimes` 的 cell。

**怎麼看:**
- 我們已知 4 顆 `sustained`、6 顆 `transient`、14 顆 `edge_skip`、
  45 顆 `none`。
- 4 顆 `sustained` 會跟第一優先資料夾重疊 — 已經看過了。
- **這個資料夾真正要看的是:** 從 45 顆 `none` 隨機挑 5 張,
  確認它們沒有 visible step (曲線平滑、沒有永久 offset)。這是反向
  驗證:多 regime cell 大部分都是良性的,演算法不該把它們全部掃進來。

### 第四優先 — `out/plots/false_negative_audit/` (20 張圖,全看)

**為什麼第四優先:** 這 20 張是從 `classification = none` 隨機抽樣的
cell (`n_regular ≥ 100` 確保 cell 夠成熟)。**這是檢查偵測器有沒有
漏掉 cohort 中其他明顯 step pattern 的樣本。**

**只有 20 張,全部看一遍。** 每張應該長得像:
- 平滑的單調衰退曲線,或
- 健康的 plateau (retention ~1.0,還沒明顯衰退)
- **沒有任何明顯的 step 或斷層**

如果有任何一張看起來像 AR4313 但被歸為 `none`,代表 `BUMP_MIN = 0.03`
太高 (需要提高靈敏度),或者偵測器有 bug。

---

## 9. 明天的最短路徑

如果時間很緊,只看這五件事:

1. **`sustained/AR4313.png`** — 確認金標準的視覺化跟預期一致。
2. **`sustained/AR4143.png` 到 `AR4148.png`** — 確認這個 cluster 是
   真陽性還是假陽性 (**這是是否需要調 `PERSIST_MIN` 的最大判斷點**)。
3. **`sustained/AR4135.png`、`AR4269.png`、`AR4389.png`** — 確認這些
   奇怪形狀真的需要截斷,還是 early-life 雜訊誤報。
4. **`transient/` 隨機 5 張** — 反向驗證,沒有漏抓。
5. **`false_negative_audit/` 全部 20 張** — 反向驗證,沒有漏網之魚。

第 2 和第 3 是**最有可能讓你想調參數**的地方。如果這幾張看起來都對,
明天可以直接跳到 Option A / B / C 的實作階段;如果有 false positive,
明天的第一步就是把 `PERSIST_MIN` 調高 (例如 0.03 → 0.05) 重跑一次,
再決定走哪個 Option。

---

## 10. 明天從哪裡開始 (Where to start tomorrow)

1. 打開 `out/plots/sustained/AR4313.png` — 確認演算法視覺化合理。
2. 按照第 8 節的第一優先清單逐張過,每顆 cell 標記為「true positive」/
   「false positive」/「unclear」。
3. 依照第 8 節的程序抽樣檢查第二、三、四優先的資料夾。
4. 如果發現 false positive,重跑
   `python run_investigation.py --persist-min 0.05` (或審查後決定的
   數值),然後重新檢視。
5. 決定走 Option A / B / C / D 以及參數值。
6. 把選定的 option 寫成新的 plan + commit。

---

## Files referenced

- `detector.py` — pure-function detector + selftest (passes 8 cases)
- `run_investigation.py` — CLI scanner; reuses `_common.iter_annotations`
- `out/jump_detection_report.csv` — 551 rows, one per cell+candidate
- `out/jump_detection_summary.txt` — aggregate stats
- `out/plots/` — 129 PNGs across 4 audit buckets
- `README.md` — methodology & audit procedure
- Production code (UNCHANGED): `../../labels.py`, `../../features.py`,
  `../../_common.py`, `../../preprocess.py`, `../../datasets/`

Plan file: `/home/mliao/.claude/plans/peaceful-gliding-bentley.md`
