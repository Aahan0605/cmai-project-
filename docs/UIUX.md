# UI / UX Design Specification

> Planning specification: this document includes future research work. The [README status](../README.md#project-status) and [system architecture](ARCHITECTURE.md) describe the current implementation.

**Product:** OptLab-DF results dashboard (read-only)
**Companion docs:** `docs/PRD.md`, `docs/TRD.md` (§12 API, §13 frontend), `backend/schema.sql`
**Status:** Draft v1 · 2026-10-12

---

## 1. Design principles

1. **Evidence first.** Every number shows its uncertainty (± std, CI, n seeds). A result that is within noise is labelled as such rather than ranked.
2. **One question per screen.** Each view answers a single question from the research question in the PRD.
3. **Traceable.** Any number can be clicked through to the runs and epochs behind it.
4. **Honest by default.** No rank medals, no "winner" banners. Rankings appear with an "indistinguishable from" note when intervals overlap.
5. **Calm and technical.** Dense but quiet; the data is the decoration.
6. **Works without a backend.** The same UI runs from the API or from a static JSON snapshot.

## 2. Users and tasks

| User | Top tasks |
|------|-----------|
| Course evaluator | Find the optimizer maths; see convergence analysis; check the constrained-optimization part |
| Repo visitor | Understand the question in under a minute; see the headline table; open one plot |
| Author | Spot failed or diverged runs; compare optimizers; sanity-check a number against its runs |

## 3. Information architecture

```
/                       Overview
/leaderboard            Leaderboard
/convergence            Convergence explorer
/generalization         In-distribution vs cross-dataset
/tuning                 Tuning and sensitivity
/toy                    Toy benchmarks (2-D contour viewer)
/fusion                 Constrained fusion (simplex, KKT)
/hybrid                 Hybrid optimizer and ablations
/runs                   Run list
/runs/:id               Run detail
/demo                   Upload demo (stretch, API mode only)
/about                  Method, protocol, limitations, citations
```

Primary navigation order matches the research story: Overview → Leaderboard → Convergence → Generalization → Tuning → Toy → Fusion → Hybrid. `Runs` and `About` sit in the secondary group.

## 4. Global layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│ OptLab-DF            experiment: [baseline_v1 ▾]      ◐ theme   GitHub ↗ │  header 56px
├───────────────┬──────────────────────────────────────────────────────────┤
│ Overview      │  Page title                                              │
│ Leaderboard   │  One-sentence question this page answers                 │
│ Convergence   │ ┌── filter bar ─────────────────────────────────────────┐│
│ Generalization│ │ Optimizers ▾  Dataset ▾  Metric ▾  Granularity ▾  ⟲   ││
│ Tuning        │ └───────────────────────────────────────────────────────┘│
│ Toy           │                                                          │
│ Fusion        │   content: cards, charts, tables                         │
│ Hybrid        │                                                          │
│ ───────────   │                                                          │
│ Runs          │                                                          │
│ About         │                                                          │
├───────────────┴──────────────────────────────────────────────────────────┤
│ footer: schema v1 · n runs · data snapshot date · commit hash            │
└──────────────────────────────────────────────────────────────────────────┘
```

- Sidebar 220px, collapsible to icons below 1024px, becomes a bottom sheet menu below 640px.
- Content max width 1200px, centred; 24px page padding (16px on mobile).
- Filter bar state lives in the URL query string so every view is linkable.
- Header experiment switcher changes all views; default is the latest `baseline` experiment.

## 5. Design tokens

### 5.1 Colour

Contrast ratios were computed against the stated backgrounds (WCAG 2.1 relative luminance).

| Token | Light | Dark | Contrast (own background) |
|-------|-------|------|:-------------------------:|
| `--bg` | `#FFFFFF` | `#0B1220` | — |
| `--surface` | `#F1F5F9` | `#111A2E` | — |
| `--text` | `#0F172A` | `#E2E8F0` | 17.9 / 15.2 |
| `--text-muted` | `#475569` | `#94A3B8` | 7.6 / 7.3 |
| `--accent` | `#1D4ED8` | `#60A5FA` | 6.7 / 7.4 |
| `--ok` | `#047857` | `#34D399` | 5.5 / 9.7 |
| `--warn` | `#92400E` | `#FBBF24` | 7.1 / 11.2 |
| `--danger` | `#B91C1C` | `#F87171` | 6.5 / 6.8 |
| `--border` | `#CBD5E1` | `#26324A` | decorative |

Primary button: white text on `--accent` (light, 6.7:1).

**Optimizer family colours** (Okabe–Ito based, colour-vision-safe). All chart colours have at least 3:1 contrast against their background (WCAG 1.4.11).

| Family | Light | Dark |
|--------|-------|------|
| plain | `#6B7280` | `#6B7280` |
| momentum | `#0284C7` | `#0EA5E9` |
| adaptive | `#B86E00` | `#E69F00` |
| sign-based | `#B5588C` | `#CC79A7` |
| sharpness-aware | `#009E73` | `#009E73` |
| second-order | `#0072B2` | `#0072B2` |
| hybrid | `#D55E00` | `#D55E00` |

Colour identifies the **family**. Individual optimizers inside a family are told apart by **dash style and marker shape**, so colour is never the only signal.

| Optimizer | Family | Dash | Marker |
|-----------|--------|------|--------|
| SGD | plain | solid | circle |
| SGD + Momentum | momentum | solid | square |
| NAG | momentum | dashed | triangle-up |
| RMSProp | adaptive | dotted | diamond |
| Adam | adaptive | solid | circle |
| AdamW | adaptive | dashed | square |
| RAdam | adaptive | dash-dot | triangle-down |
| AdaBelief | adaptive | long-dash | cross |
| Lion | sign-based | solid | star |
| SAM (+ base) | sharpness-aware | solid / dashed | hexagon |
| L-BFGS / Newton | second-order | solid | pentagon |
| Hybrid H1–H4 | hybrid | solid / dashed / dotted / dash-dot | thick-bordered marker |

### 5.2 Typography

- UI: Inter (system-ui fallback). Numbers in tables use tabular figures (`font-variant-numeric: tabular-nums`).
- Code and update rules: JetBrains Mono (monospace fallback). Maths rendered with KaTeX.
- Scale (px / line-height): 12/16 caption · 14/20 body-small · 16/24 body · 20/28 h3 · 24/32 h2 · 32/40 h1.
- Weight: 400 body, 500 labels, 600 headings.

### 5.3 Spacing, shape, motion

- 4px grid; common steps 4, 8, 12, 16, 24, 32, 48.
- Radius: 6 (inputs, chips), 10 (cards), 999 (pills).
- Elevation: 1px border plus a very subtle shadow on cards; no heavy shadows.
- Motion: 120–200ms ease-out for hover and expansion. Respect `prefers-reduced-motion`: the toy-trajectory animation becomes a static final frame with a scrub slider.

## 6. Screens (in implementation order)

### 6.1 Overview `/`

**Question:** *What is being asked, and what is the headline answer so far?*

```
┌──────────────────────────────────────────────────────────────────────┐
│ Which optimizer trains the most robust multimodal deepfake detector? │
│ A controlled comparison under equal tuning budgets and multiple seeds│
├───────────────┬───────────────┬───────────────┬──────────────────────┤
│ 10 optimizers │ 5 seeds each  │ 3 datasets    │ 20 trials / optimizer│
├───────────────┴───────────────┴───────────────┴──────────────────────┤
│ Headline table (top rows of the leaderboard, with ± and CI)          │
│ [ In-distribution AUC ] [ Cross-dataset AUC ] [ Gap ] [ Epochs→0.95 ]│
├──────────────────────────────────┬───────────────────────────────────┤
│ Mini convergence chart           │ Findings (hand-written, dated)    │
│                                  │ • finding with link to the view   │
├──────────────────────────────────┴───────────────────────────────────┤
│ How to read this · Method · Limitations (links to /about)            │
└──────────────────────────────────────────────────────────────────────┘
```

Data: `/experiments/{id}`, `/leaderboard`. Findings are static Markdown shipped with the snapshot, not generated.

### 6.2 Leaderboard `/leaderboard`

**Question:** *How do the optimizers rank, and is the ranking meaningful?*

```
 Dataset: [Celeb-DF v2 ▾]  Metric: [AUC ▾]  Level: [video ▾]  Family: [all ▾]
┌─────────────┬─────────┬───────────────┬──────────┬────────┬────────────┐
│ Optimizer   │ n seeds │ AUC mean ± sd │ 95% CI   │ EER    │ Time to fit│
├─────────────┼─────────┼───────────────┼──────────┼────────┼────────────┤
│ ●  AdamW    │ 5       │ ▮▮▮▮▮▯ …     │ [ , ]    │ …      │ …          │
│ ■  SGD+M    │ 5       │ …             │          │        │            │
│ …           │         │               │          │        │            │
└─────────────┴─────────┴───────────────┴──────────┴────────┴────────────┘
 ⓘ Rows whose intervals overlap the top row are marked "≈ top" (not significantly different).
```

- Columns sortable; default sort by mean of the selected metric.
- Marker + dash glyph before each optimizer name (§5.1).
- Each row shows a small dot-strip of per-seed values so spread is visible at a glance.
- Row click opens a drawer with hyperparameters, selected tuning trial, and links to the runs.
- "≈ top" badge uses the pairwise-test output; tooltip explains the rule and shows the Holm-adjusted p-value.
- Footer: toggle to show pairwise matrix (`/stats/pairwise`) as a heat table with effect sizes; cells carry numbers, not just colour.

### 6.3 Convergence explorer `/convergence`

**Question:** *Which optimizers get there faster, and which are stable?*

```
 Optimizers: [chips ×] [+ add]   Metric: ( val AUC | val loss | train loss | grad norm )
 X: ( epoch | wall-clock )   Y scale: ( linear | log )   Band: ( ±1 sd | min–max )
┌──────────────────────────────────────────────────────────────────────┐
│  mean curve per optimizer with shaded band across seeds              │
│  dotted horizontal line = target AUC; markers where each crosses it  │
└──────────────────────────────────────────────────────────────────────┘
 Summary strip: epochs-to-target · area under val-loss · divergence rate
```

- Hover shows a unified tooltip with all visible optimizers at that x.
- Legend entries are toggles (keyboard-operable). Max 8 optimizers by default; a warning appears beyond that.
- Diverged runs are listed in a note under the chart and excluded from the mean, never silently dropped.

### 6.4 Generalization `/generalization`

**Question:** *Does a faster or sharper optimizer pay for it on unseen data?*

```
┌────────────────────────────────┬─────────────────────────────────────┐
│ Scatter: x = in-dist AUC       │ Bar: generalization gap per optimizer│
│          y = cross-dataset AUC │ (in-dist − cross) with CI whiskers   │
│ points = optimizers (± error)  │                                     │
│ diagonal y = x reference       │                                     │
└────────────────────────────────┴─────────────────────────────────────┘
 Optional: leave-one-manipulation-out heat table (optimizer × method)
```

- Distance below the diagonal is the gap; the page text says so in one sentence.
- Error bars on both axes (seed std).

### 6.5 Tuning and sensitivity `/tuning`

**Question:** *Was the comparison fair, and how sensitive is each optimizer to its learning rate?*

```
 Optimizer: [AdamW ▾]   Method: ( random | grid | PSO | GA )
┌────────────────────────────────┬─────────────────────────────────────┐
│ LR sensitivity: val AUC vs lr  │ Trial table (sortable)              │
│ (log x), selected trial ringed │ budget: 20 / 20 trials used         │
├────────────────────────────────┴─────────────────────────────────────┤
│ Search method comparison: best-so-far val AUC vs trial index          │
└──────────────────────────────────────────────────────────────────────┘
```

- A visible "budget used" chip per optimizer proves equal budgets.
- Diverged trials are plotted at the bottom edge with a distinct marker.
- LR-sensitivity width ("how wide is the good region") is surfaced as a small summary stat.

### 6.6 Toy benchmarks `/toy`

**Question:** *Do the optimizers behave as the maths says on known functions?*

```
 Benchmark: [Rosenbrock ▾]   Optimizers: [chips]   Seed: [0 ▾]   ▶ play  ──●────
┌────────────────────────────────┬─────────────────────────────────────┐
│ Contour plot with trajectories │ f(x_t) − f* vs step (log y)         │
│ start ◯  end ★  global min ✕   │ steps-to-tolerance table            │
└────────────────────────────────┴─────────────────────────────────────┘
 Update rule of the selected optimizer rendered with KaTeX
```

- Animation is optional; a scrub slider is always available.
- Condition-number selector for the quadratic benchmark.
- Newton and L-BFGS appear as reference lines.

### 6.7 Constrained fusion `/fusion`

**Question:** *How much does each modality contribute, and is the constrained solution exact?*

```
 Run: [adamw · seed 0 ▾]
┌────────────────────────────────┬─────────────────────────────────────┐
│ Simplex plot (3 modalities)    │ Weights table                        │
│ or stacked bar for 2           │ solver · objective · λ · active set  │
│ solution point marked          │ KKT residual (numeric, with check ✓) │
├────────────────────────────────┴─────────────────────────────────────┤
│ Weights across optimizers and seeds (strip chart)                    │
└──────────────────────────────────────────────────────────────────────┘
 KKT conditions and the loss rendered with KaTeX
```

- Two modalities show as a stacked bar; three as a ternary plot.
- A residual above the tolerance shows as a warning chip with the value.

### 6.8 Hybrid optimizer `/hybrid` (appears after Phase 3)

**Question:** *Does the hybrid help, and which part earns its place?*

- Top: update rule (KaTeX), design rationale, and a "design frozen on <date>, commit <hash>" badge.
- Middle: hybrid vs. the best two baselines on the leaderboard metrics, equal budget stated.
- Bottom: ablation table (full, minus each component) with CI and effect size.
- A visible "Result: not distinguishable from baseline" or "Result: better / worse" label derived from the CI, never from the mean alone.

### 6.9 Runs `/runs` and Run detail `/runs/:id`

- List: filter by experiment, optimizer, seed, status; status chips (completed, failed, diverged).
- Detail: header with config summary, hyperparameters (monospace), curves (loss, AUC, lr, grad norm), evaluation table (val/test × granularity), artifact list, git commit.

### 6.10 Demo `/demo` (stretch, API mode only)

```
┌──────────────────────────────────────────────┐
│   Drop a short video here (max 25 MB)        │
│   [ Choose file ]                            │
│   Processed in a temporary folder and deleted│
└──────────────────────────────────────────────┘
 Result: fused score gauge · per-modality scores · weights used · caveat text
```

- Caveat text is always visible: detectors can fail on unseen generators; not evidence about a real person.
- Hidden when running from a static snapshot.

### 6.11 About `/about`

Method summary, evaluation protocol, tuning-budget table, seed policy, limitations, dataset acknowledgements and citations, link to the repo and docs.

## 7. Component library

| Component | Notes |
|-----------|-------|
| `OptimizerChip` | Marker + dash glyph + name; removable; keyboard focusable |
| `MetricValue` | `0.912 ± 0.006` with tooltip showing CI and n |
| `SeedStrip` | Tiny dot plot of per-seed values |
| `SignificanceBadge` | "≈ top", "worse", "better" with tooltip (effect size, adjusted p) |
| `FilterBar` | URL-synced selects and toggles |
| `ChartCard` | Title, one-line takeaway, chart, export (PNG, CSV), "view runs" link |
| `DataTable` | Sortable, sticky header, tabular numbers, row drawer |
| `StatusChip` | completed / running / failed / diverged with icon and text |
| `MathBlock` | KaTeX with copyable source |
| `EmptyState`, `ErrorState`, `Skeleton` | Defined in §8 |

## 8. States and edge cases

| State | Behaviour |
|-------|-----------|
| Loading | Skeletons shaped like the final content; no spinners over charts |
| Empty (no runs yet) | Explain which CLI command produces the data: "Run `optlab sweep --experiment baseline_v1`" |
| Partial data | Show `n seeds` honestly; badge "low n" when fewer than 3 seeds |
| Diverged runs | Counted and listed; excluded from means with an explicit note |
| API error | Inline error card with retry; the rest of the page stays usable |
| Snapshot mode | Footer shows snapshot date; demo and live-only features hidden |
| Overlapping intervals | Neutral "≈" badge; never colour a winner |
| Very long optimizer names | Truncate with tooltip; full name in the row drawer |

## 9. Accessibility

- Target WCAG 2.1 AA. Text and graphic contrast verified as in §5.1.
- Full keyboard operation: skip link, visible 2px focus ring (`--accent`), logical tab order, legend and chip toggles reachable by keyboard.
- Charts: every chart has a "View as table" toggle exposing the same data in an accessible table; each chart has a text summary above it.
- No information by colour alone: family colour plus dash plus marker, status chips with icon and text.
- Respect `prefers-reduced-motion` and `prefers-color-scheme`; theme toggle persists in `localStorage`.
- Touch targets at least 44×44px on mobile; tooltips also open on focus and tap.
- Semantic landmarks (`header`, `nav`, `main`, `footer`), one `h1` per page.

## 10. Responsive behaviour

| Breakpoint | Layout |
|------------|--------|
| ≥ 1280px | Sidebar + two-column chart grids |
| 1024–1279px | Sidebar collapsed to icons; charts stack to one column when narrower than 520px |
| 640–1023px | Top bar with menu button; filter bar wraps; tables scroll horizontally |
| < 640px | Single column; leaderboard becomes cards; contour viewer fits width; hover tooltips replaced by tap |

## 11. Microcopy rules

- Plain language first, symbols second ("learning rate", then `η`).
- Never say "best" without the qualifier "on this protocol".
- Tooltips for jargon: AUC, EER, CI, KKT, simplex, cross-dataset, generalization gap.
- Findings are dated and link to the view that supports them.
- Buttons use verbs ("Export CSV", "View runs").

## 12. Build order for the dashboard

1. Shell: layout, router, theme tokens, `FilterBar`, data hooks, snapshot/API switch.
2. Leaderboard (the main table and `MetricValue`, `SeedStrip`, `SignificanceBadge`).
3. Convergence explorer.
4. Generalization.
5. Tuning and sensitivity.
6. Toy benchmarks.
7. Fusion.
8. Run list and run detail.
9. Overview and About (content assembled from the views above).
10. Hybrid (after Phase 3).
11. Demo (stretch).
