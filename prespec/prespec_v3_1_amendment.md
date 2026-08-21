# Pre-specification amendment v3.1 — convention audit resolution
## Appended to external_validation_prespec.md (v3 remains frozen; this amendment documents implementation corrections and the E/R convention resolution)

## 1. E_j convention, as read from the reference implementation
Audit of the measure module (`code/dynamical_leverage.py` in this
repository; sha-matched to the copy frozen at the
prior submission, i.e., the code in force for every committed number):
(a) `ec.weights = E / E.sum()` per window; the sweep scripts average
`weights` across sliding windows. **Every committed E_j is therefore a
mean of per-window normalized shares**, not a mean of raw window energies.
(b) State space: NL companion form. The perturbation is e_j in the
current-state (leading) block; the response metric is identity on the
FULL companion state (lagged copies included in the response norm).
(c) The tau = 0 self-term is included.
Definition text in the paper amended accordingly (mean-window-share
convention stated in Def. 3.1).

## 2. R convention, resolved
The as-run script accumulated RAW per-window row energies for R while E
accumulated shares; mean-of-raws and mean-of-shares differ across windows
with varying totals. **Correction (one line, applied in
run_external_validation_v3.py):** R is normalized per window by its own
window total before averaging, exactly parallel to E's convention.

## 3. Duality claim corrected
v3 stated sum_j E_j = sum_j R_j = sum_tau ||Phi^tau||_F^2. In companion
form this is FALSE: sum_j E_j sums squared norms of the first-N COLUMNS of
each Phi^tau (full-space response to current-block perturbations), while
sum_j R_j as implemented sums squared norms of the current-block ROWS
restricted to current-block columns. The identity holds only in component
form (state_dim = N). The "same denominator" justification is withdrawn;
the operative rule is simply: **each quantity is normalized by its own
window total.** R is the row ANALOGUE under the stated convention, not an
exact algebraic twin of E.

## 4. R's input/output ranges, stated as a construct choice
E: input = current block, output norm = full companion state.
R (as implemented): input range = current block, output = current block.
Restricting R's input to the current block is retained deliberately
(lagged coordinates are not independently perturbable states); the
resulting asymmetry with E's full-space response norm is a documented
construct difference, not an error. No text may describe R as the exact
transpose of E.

## 5. Prior implementation corrections (carried from the audit-fixed run)
Annual profiles for Study-1 outcomes; calendar-month bins for Study 2;
degree comparator added; independent RNG streams per statistic.

## 6. Consequences for reported numbers
E-side numbers are unchanged by this amendment. R-side numbers (R_traj,
R_pool where share-normalized, d_R, Delta^R) refresh on FINAL_RUN under
the corrected convention; the v3 freeze's reporting commitment applies
unchanged to the refreshed values, including the prerequisite
d_R(polecat) > d_R(trade) in whatever form it lands.

## 7. Repository redactions (documented deviation, metadata only)
Repository copies of the pre-specification redact submission-venue
identifiers and use this repository's module names. No analysis
content, prediction, or reporting commitment is altered.

## 8. Secondary event study: implementation declarations (before run)
The pre-declared secondary (fit Jan 2018 - Dec 2021, event Feb 2022) fixes,
prior to execution: PRE = Mar 2021 - Feb 2022 (the last ~5 days of Feb 2022
postdate the onset; declared, direction of bias is toward the post level,
i.e., against finding displacement); POST = Mar 2022 - Feb 2023 per the
pre-specification, truncated at data end (Dec 2022; 10 months); the
graph-derived comparators are computed inline on the fit window under the
committed baselines-cell conventions (a = 0.9 / max_t rho(A_t); binary
time-respecting earliest-arrival closeness; 500-actor graph by fit-window
counts). All other constructions are byte-derived from the committed
Study 2 code, including a property both studies share and which is hereby
made explicit: the PRE window lies mostly inside the fit window (10 of 12
months in each study). The outcome is computed from raw standardized data
with no fitted object anywhere in it, so the overlap cannot manufacture a
correlation between a fitted-structure predictor and the displacement; it
is a construction property of the committed design, declared rather than
altered. The fit window contains the COVID period; the window's own
departure d_E is reported as the gate for whether the trajectory-increment
comparison has room, and all results are reported regardless of sign.
