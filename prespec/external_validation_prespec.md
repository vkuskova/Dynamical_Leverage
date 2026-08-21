# Pre-specification: external criterion validation of dynamical leverage
## Version 3 — FROZEN. (v2 amended per second methodological review,
## Aug 17, 2026. Deviations require a documented amendment in the repo.)

## Purpose and claim discipline
Answer the circularity objection (raised independently in two review
processes; venue identifiers redacted in this repository copy for
double-blind review) by testing
whether pre-event, model-derived leverage predicts OBSERVED, raw-data-computed
displacement after historical disruptions. This is **external
predictive/criterion validation, not causal validation**: association between
leverage and displacement under a system-wide shock does not establish that
perturbing an entity causes system displacement.

## Construct correspondence (susceptibility vs propagation) — resolved
E_j = sum_tau ||Phi^tau e_j||^2 is a COLUMN quantity: propagation FROM j.
Displacement of j under a common shock is a ROW quantity: susceptibility OF j.
Propagation validation proper would require many node-specific historical
shocks and is observationally infeasible here. The frozen design is therefore
**susceptibility validation with an explicit bridge**:
- Define trajectory in-leverage R_j = sum_tau ||e_j^T Phi^tau||^2 (row twin,
  same Jacobian products, standardized channels), with its pooled analogue
  R_j^pool from the single mean-state linearization.
- **Interpretation of R, made exact:** R_j is the DIRECTION-NEUTRAL incoming
  susceptibility analogue, matched to the receiving-node character of the
  displacement outcome. If the horizon-zero perturbation u has identity
  covariance, E[(e_j^T Phi^tau u)^2] = ||e_j^T Phi^tau||^2, so R_j is the
  expected response under isotropic perturbation directions. The realized
  2008 and COVID shocks need not satisfy that assumption; a null for R can
  therefore mean either no external correspondence of the susceptibility
  structure OR a realized shock direction materially different from the
  direction-neutral geometry R summarizes. Duality note (for methods):
  sum_j E_j = sum_j R_j = sum_tau ||Phi^tau||_F^2 — E and R partition the
  same total propagation energy by origin vs accumulation; R is principled,
  not an ad hoc rescue metric.
- **E is the paper's measure**; its correspondence to the outcome is mediated
  by the bridge diagnostic rho(E, R), computed and reported per system
  (seed-averaged). No text may treat susceptibility success as propagation
  validation.
- Interpretation grid, frozen: R predicts D => the fitted trajectory
  structure carries external information (the circularity answer). E also
  predicts D and rho(E,R) high => E_j inherits that support. E fails while R
  succeeds => report as a construct boundary of E_j. Both fail => decision
  class 4 below.
- Exploratory only (labeled as such, no headline): 2014 Russia-sanctions
  propagation case study (displacement of Russia's pre-2014 top partners),
  a single-shock illustration of construct B.

## Study 1 (primary): tgbn-trade, 2008-09 global trade collapse
- Fit window 1993-2007 (15 annual steps); estimator settings identical to the
  paper (MAXLAGS 3, HORIZON 8, EPOCHS 400); seeds 0-7; predictors E_j, R_j
  and pooled analogues = per-nation seed-averages.
- Panel: clean imports-attraction design, one channel per nation; top-20
  selection by attraction on the FIT WINDOW ONLY.
- **Outcome (primary), raw data only:** D_i = || p_i(post) - p_i(pre) ||_1,
  p_i(t) = nation i's inbound exporter-share profile (vector over all
  exporters, normalized to sum 1); pre = mean over 2005-2007, post = mean
  over 2008-2010. Window rationale, stated in advance: the post window
  measures the three-year crisis-and-adjustment regime beginning 2008
  (persistent compositional reorganization), not the instantaneous trough.
- Outcome (secondary, descriptive): |Delta log1p total attraction|.
- Comparators (all pre-event only): pooled-AC, pooled-R, strength centrality
  (1993-2007), attraction volume (1993-2007), dynamic communicability
  (broadcast/receive) and temporal closeness on 1993-2007 snapshots, and
  **historical volatility**: V_i = mean over fit-window years of
  || p_i(t) - p_i(t-1) ||_1 (the naive "volatile nodes move more" predictor,
  same object as the outcome, historical).
- Measurement-noise diagnostic, reported: rho(D, pre-event volume), to
  diagnose whether displacement is dominated by activity-dependent
  composition noise.
- Known limitation, stated in advance: trade is near-static
  (rho(E, AC) ~ 0.97), so Study 1 separates model-based from structural
  predictors, not E from AC. That separation is Study 2's job.

## Study 2 (discriminating): tkgl-polecat, COVID-19 onset
- Fit window: monthly bins, calendar 2018-2019 (24 bins); valenced coding A;
  top-20 actors by fit-window event volume only; seeds 0-7.
- Event: COVID-19 onset. Post = Mar 2020 - Feb 2021; pre = Mar 2019 - Feb 2020.
- **Outcome (primary), frozen mathematically:** using fit-window mean/sd per
  channel (the same standardization in which E and R are defined),
  D_i = | zbar_coop,i(post) - zbar_coop,i(pre) | +
        | zbar_conf,i(post) - zbar_conf,i(pre) |,
  where zbar is the mean standardized monthly intensity over the window.
  Standardization constants come from the FIT WINDOW ONLY and are applied
  unchanged to post-event data — no post-period restandardization. (Same
  rule in Study 1 for any standardized quantity.) This preserves intensity
  information and is expressed in the measure's own coordinates. (A normalized 2-dim (coop,conf) share profile collapses to a
  single ratio and discards intensity; rejected for the primary.)
- Outcome (secondary, descriptive): |Delta coop share|, the ratio change.
- Comparators (fit-window only): pooled-AC, pooled-R, total event volume,
  co-occurrence strength, historical volatility V_i = mean_t
  ( |Delta z_coop,i(t)| + |Delta z_conf,i(t)| ) over fit-window months.
- Confound, pre-stated: POLECAT is machine-coded from news; COVID changed the
  reporting process itself. Interpretation is frozen narrowly as
  "displacement in recorded event intensities", not behavioral change
  per se. COVID remains primary (clean exogeneity, no anticipation in the
  fit window); the Feb 2022 invasion is the pre-declared secondary event
  (fit 2018-2021, post Mar-Dec 2022), run only if Study 2 executes cleanly,
  with the converse caveat (possible pre-event escalation dynamics).

## Cross-study predictions (frozen; hierarchy revised in v3)
- **Prerequisite (trajectory structure, internal):** compute the row
  departure d_R = 1 - rho(R, R_pool) per system, seed-averaged. The E-based
  departure ordering (trade 0.037 << polecat 0.168) does not automatically
  transfer to rows; the prediction **d_R(polecat) > d_R(trade)** is frozen
  as an empirical prerequisite and reported before any external result.
- **Primary external test (construct-matched):** with
  Delta^R_s = rho_s(R, D) - rho_s(R_pool, D), the framework predicts
  **Delta^R_polecat > Delta^R_trade**, conditional on the prerequisite
  holding. Directional prediction only; no significance requirement at
  these n.
- **Secondary (bridge-dependent, prespecified):**
  Delta^E_s = rho_s(E, D) - rho_s(AC_pooled, D), same directional
  prediction, interpreted through the rho(E, R) bridge.
If the primary holds, the departure DIAGNOSTIC — not only the measure —
has external support, in its construct-matched form.

## Inference (both studies)
- Spearman rho over the 20 entities for every (predictor, outcome) pair,
  all printed in one pass (no sequential peeking).
- **95% CIs** (entity bootstrap, 10,000 resamples), reported as wide if
  wide; the **paired entity bootstrap CI is the primary uncertainty summary
  for every Delta-rho** (dependent correlations sharing D).
- **Permutation of D across entities** (10,000 draws): finite-sample null
  for each individual rho.
- **Predictor-swap permutation** (rank-standardized, entity-wise swap,
  10,000 draws): retained ONLY as an exchangeability-null diagnostic,
  explicitly NOT inference for H0: Delta-rho = 0 (swapping tests the
  stronger joint-exchangeability null, not parameter equality).
- Raw point estimates reported prominently; the design's evidentiary value
  rests on pre-specification and effect ordering, not p-value thresholds.
- n = 20 per study; no asymptotic claims; bootstrap tie artifacts at this n
  acknowledged.

## Interpretive classification (two axes; frozen)
Evidence is classified on two INDEPENDENT axes, per study and per predictor
family (R primary, E bridge-dependent). Failure on axis 2 does not negate
success on axis 1 — trade is expected in advance to show exactly
axis-1-without-axis-2.
**Axis 1 — criterion validity** (does learned dynamical structure correspond
to observed displacement?): predictor vs structural + volatility
comparators.
  - Strong: positive association, advantage over all comparators, paired-CI
    uncertainty excluding or substantially concentrated above zero.
  - Supported: positive association and point-estimate advantage;
    uncertainty not resolving the ranking.
  - Inconclusive: positive point estimate; ranking unresolved.
  - Contradictory: zero/negative, or volatility/structural comparators
    consistently outperform.
**Axis 2 — trajectory increment** (does following the trajectory add
predictive information beyond one static linearization?): R vs R_pool
(and E vs AC_pooled for the bridge version), same four grades applied to
the increment.
No positive point estimate is promoted to "validation" language beyond its
class on its axis.

## Reporting commitment (frozen; no discretion)
**All pre-specified primary results are reported in the paper regardless of
sign or class**, including Contradictory — as a stated boundary condition of
the measure. Exploratory analyses beyond this document are labeled
exploratory. This document and the analysis script are committed together.
