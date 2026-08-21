#!/usr/bin/env python3
# =============================================================================
# run_external_validation.py — implements external_validation_prespec.md v3
# (FROZEN Aug 17, 2026) line-for-line. Two studies:
#   Study 1: tgbn-trade, 2008-09 collapse (fit 1993-2007, seeds 0-7)
#   Study 2: tkgl-polecat, COVID onset  (fit 2018-2019 monthly, seeds 0-7)
# Predictors: E (trajectory out-leverage), R (trajectory in-leverage,
#   direction-neutral susceptibility), pooled analogues E_pool (=AC_pooled),
#   R_pool; comparators: strength, volume, historical volatility V.
# Temporal baselines (communicability/closeness) merge from their own CSV if
# present (built Aug 18); absence is reported, not fatal.
# Outcomes computed from RAW data only. All results print in ONE pass.
# Inference: paired entity bootstrap (primary for Delta-rho), permutation of
# D (individual rho), swap permutation (exchangeability diagnostic ONLY).
# Colab + repo dual environment, same conventions as run_polecat_sweep.py.
# =============================================================================
import os, sys, csv
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.func import jacrev

SEEDS = list(range(8))
MAXLAGS, HORIZON, EPOCHS = 3, 8, 400
N_BOOT, N_PERM, RNG_SEED = 10000, 10000, 20260817
TOP_K = 20
MONTH = 2592000

IN_COLAB = 'google.colab' in sys.modules or os.path.isdir('/content')
if IN_COLAB:
    if not os.path.isdir('/content/drive/MyDrive'):
        from google.colab import drive; drive.mount('/content/drive')
    TRADE_RAW   = globals().get("COLAB_TRADE",   # inherited; UNUSED in the secondary
        "/content/drive/MyDrive/WSDM_TemporalEj/rawdata")
    POLECAT_RAW = globals().get("COLAB_POLECAT",
        "/content/drive/MyDrive/WSDM_TemporalEj/rawdata_polecat")
    OUT_DIR     = globals().get("COLAB_OUT",
        "/content/drive/MyDrive/WSDM_TemporalEj/external_validation")
    CODE_DIR    = globals().get("COLAB_CODE",
        "/content/drive/MyDrive/code")
    sys.path.insert(0, CODE_DIR)
    TEMPORAL_BASELINES_CSV = globals().get("COLAB_TB",
        "/content/drive/MyDrive/WSDM_TemporalEj/baselines/temporal_baselines.csv")
else:
    try:
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        ROOT = os.getcwd()
    TRADE_RAW   = os.path.join(ROOT, "data")
    POLECAT_RAW = os.path.join(ROOT, "data")
    OUT_DIR     = os.path.join(ROOT, "results")
    sys.path.insert(0, os.path.join(ROOT, "src"))
    TEMPORAL_BASELINES_CSV = os.path.join(ROOT, "results", "temporal_baselines.csv")
os.makedirs(OUT_DIR, exist_ok=True)

try:
    from dynamical_leverage import (dynamical_leverage_from_jacobians,
                                    average_controllability_gramian_trace)
    from neural_var import fit_navar
except ImportError as e:
    raise ImportError(
        "dynamical_leverage / neural_var not importable. For a Drive-side "
        "run, copy code/dynamical_leverage.py and code/neural_var.py from "
        "this repository into your Drive code folder first.") from e

import hashlib
def rng_for(*ids):
    h = hashlib.sha256(("|".join(map(str, ids)) + f"|{RNG_SEED}").encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))

# ---------------------------------------------------------------- shared core
def companion_fn(model, N):
    def pw(wv):
        p, _ = model(wv.reshape(1, N, MAXLAGS)); return p.reshape(-1)
    jr = jacrev(pw)
    def c(wv):
        wv = torch.as_tensor(np.asarray(wv), dtype=torch.float32)
        Jb = jr(wv)
        top_ = torch.cat([Jb[:, :, MAXLAGS - 1 - l] for l in range(MAXLAGS)], 1)
        eye = torch.eye(N, dtype=top_.dtype); zero = torch.zeros(N, N, dtype=top_.dtype)
        rws = [torch.cat([eye if cc == (r - 1) else zero for cc in range(MAXLAGS)], 1)
               for r in range(1, MAXLAGS)]
        return torch.cat([top_] + rws, 0).detach().numpy()
    return c

def row_energy_from_jseq(jseq, n_components, horizon):
    """
run_secondary_feb2022.py — SECONDARY event study only (pre-declared in
prespec v3; implementation declarations in amendment section 8).

One study: tkgl-polecat, fit Jan 2018 - Dec 2021, event Feb 2022,
post Mar 2022 - data end. Machinery (fitting, E/R scoring, inference,
reporting, persistence) is inherited verbatim from the committed
run_external_validation script for parity; the trade configuration below
is inherited with it and is UNUSED in this file. Outputs:
secondary_validation_results.csv / secondary_validation_values.csv.
"""
    NL = jseq[0].shape[0]
    R = np.zeros(n_components)
    Phi = np.eye(NL)
    for tau in range(horizon):
        blk = Phi[:n_components, :n_components]   # state-block response
        R += (blk ** 2).sum(axis=1)               # row energies
        if tau < horizon - 1:
            Phi = jseq[min(tau, len(jseq) - 1)] @ Phi
    return R

def fit_EG(X, seed):
    """One fit -> (E, R, E_pool, R_pool) at channel level."""
    N, T = X.shape[1], X.shape[0]
    torch.manual_seed(seed); np.random.seed(seed)
    model, _, _ = fit_navar(X, [(0, T)], n_components=N, maxlags=MAXLAGS,
                            epochs=EPOCHS, seed=seed)
    jac = companion_fn(model, N)
    Js = [jac(X[s - MAXLAGS:s].T) for s in range(MAXLAGS, T)]
    acc_E = np.zeros(N); acc_R = np.zeros(N); nwin = 0
    for p in range(MAXLAGS - 1, T - HORIZON + 1):
        st = p + 1
        jseq = [Js[s - MAXLAGS] for s in range(st, st + HORIZON - 1)]
        ec = dynamical_leverage_from_jacobians(
            jseq, n_components=N, horizon=HORIZON, strict=True)
        acc_E += np.array([ec.weights[k] for k in range(N)])
        rwin = row_energy_from_jseq(jseq, N, HORIZON)
        acc_R += rwin / rwin.sum()          # per-window share, twin of ec.weights
        nwin += 1
    E, R = acc_E / nwin, acc_R / nwin
    A_pool = jac(np.tile(X.mean(0)[:, None], (1, MAXLAGS)))
    E_pool = np.array([average_controllability_gramian_trace(A_pool, j, HORIZON)
                       for j in range(N)])
    Phi = np.eye(A_pool.shape[0]); Rp = np.zeros(N)
    for tau in range(HORIZON):
        blk = Phi[:N, :N]; Rp += (blk ** 2).sum(axis=1)
        if tau < HORIZON - 1: Phi = A_pool @ Phi
    return E, R, E_pool, Rp

def seed_avg(X):
    packs = [fit_EG(X, s) for s in SEEDS]
    return [np.mean([p[k] for p in packs], axis=0) for k in range(4)]

# ------------------------------------------------------------------ inference
def rho(a, b): return spearmanr(a, b).correlation

def _finite(vals, label):
    v = np.array([u for u in vals if np.isfinite(u)])
    if len(v) == 0:
        print(f"    [warn] {label}: 0 finite replicates"); return None
    if len(v) < len(vals):
        print(f"    [note] {label}: {len(v)}/{len(vals)} finite replicates")
    return v

def boot_ci_rho(x, D, sid):
    g = rng_for(*sid, "rho_boot"); n = len(D); vals = []
    for _ in range(N_BOOT):
        idx = g.integers(0, n, n)
        vals.append(spearmanr(x[idx], D[idx]).correlation)
    v = _finite(vals, f"boot {sid}")
    return (np.nan, np.nan) if v is None else (np.percentile(v, 2.5), np.percentile(v, 97.5))

def boot_ci_delta(x, y, D, sid):
    """PAIRED entity bootstrap for rho(x,D)-rho(y,D): primary Delta summary."""
    g = rng_for(*sid, "delta_boot"); n = len(D); vals = []
    for _ in range(N_BOOT):
        idx = g.integers(0, n, n)
        vals.append(spearmanr(x[idx], D[idx]).correlation
                    - spearmanr(y[idx], D[idx]).correlation)
    v = _finite(vals, f"delta {sid}")
    return (np.nan, np.nan) if v is None else (np.percentile(v, 2.5), np.percentile(v, 97.5))

def perm_p_rho(x, D, sid):
    g = rng_for(*sid, "perm"); obs = rho(x, D); cnt = 0
    for _ in range(N_PERM):
        if abs(spearmanr(x, g.permutation(D)).correlation) >= abs(obs): cnt += 1
    return (cnt + 1) / (N_PERM + 1)

def swap_diag_p(x, y, D, sid):
    """Exchangeability-null DIAGNOSTIC only (prespec: NOT inference for
    H0 Delta=0). Rank-standardize, entity-wise swap with prob 1/2."""
    from scipy.stats import rankdata
    g = rng_for(*sid, "swap")
    xs = rankdata(x); ys = rankdata(y)
    obs = rho(xs, D) - rho(ys, D); cnt = 0
    n = len(D)
    for _ in range(N_PERM):
        sw = g.integers(0, 2, n).astype(bool)
        xp = np.where(sw, ys, xs); yp = np.where(sw, xs, ys)
        if abs(spearmanr(xp, D).correlation
               - spearmanr(yp, D).correlation) >= abs(obs): cnt += 1
    return obs, (cnt + 1) / (N_PERM + 1)

def load_temporal_baselines(study, entities):
    """Optional merge of Aug-18 temporal baselines: CSV rows
    study,entity,measure,value. Returns dict measure -> aligned array or {}."""
    out = {}
    if os.path.exists(TEMPORAL_BASELINES_CSV):
        rows = list(csv.DictReader(open(TEMPORAL_BASELINES_CSV)))
        for m in sorted({r['measure'] for r in rows if r['study'] == study}):
            d = {r['entity']: float(r['value']) for r in rows
                 if r['study'] == study and r['measure'] == m}
            if all(e in d for e in entities):
                out[m] = np.array([d[e] for e in entities])
    return out

FINAL_RUN = True   # baselines computed inline; assert must pass

RESULT_ROWS = []          # (study, block, name, value, ci_lo, ci_hi, p)
VALUE_ROWS = []           # (study, entity, quantity, value)

def report(study_name, entities, D, D2, predictors, diag_volume):
    print(f"\n================ {study_name}: RESULTS (one pass) ================")
    print(f"entities (n={len(entities)}): {entities}")
    for i, e in enumerate(entities):
        VALUE_ROWS.append((study_name, e, 'D_primary', f"{D[i]:.6f}"))
        VALUE_ROWS.append((study_name, e, 'D_secondary', f"{D2[i]:.6f}"))
        for k, x in predictors.items():
            VALUE_ROWS.append((study_name, e, k, f"{x[i]:.6f}"))
    if FINAL_RUN:
        need = [m for m in ("communicability_bc", "temporal_closeness")
                if m not in predictors]
        assert not need, f"FINAL_RUN: missing prespecified temporal baselines {need}"
    order = list(predictors.keys())
    print(f"\n{'predictor':22s} {'rho(x,D)':>9s} {'95% CI':>18s} {'perm p':>7s}")
    for k in order:
        x = predictors[k]
        lo, hi = boot_ci_rho(x, D, (study_name, k))
        pp = perm_p_rho(x, D, (study_name, k))
        print(f"{k:22s} {rho(x,D):+9.3f}   [{lo:+.3f}, {hi:+.3f}] "
              f"{pp:7.4f}")
        RESULT_ROWS.append((study_name, 'primary_rho', k,
                            f"{rho(x,D):.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{pp:.4f}"))
    print(f"\nsecondary outcome, descriptive: rho(x, D2):")
    for k in order:
        print(f"  {k:20s} {rho(predictors[k], D2):+.3f}")
        RESULT_ROWS.append((study_name, 'secondary_rho', k,
                            f"{rho(predictors[k], D2):.4f}", '', '', ''))
    print(f"\nmeasurement-noise diagnostic rho(D, pre-event volume): "
          f"{rho(diag_volume, D):+.3f}")
    RESULT_ROWS.append((study_name, 'noise_diag', 'rho_D_volume',
                        f"{rho(diag_volume, D):.4f}", '', '', ''))
    # Delta blocks: axis-1 (vs comparators) and axis-2 (trajectory increment)
    def delta_block(a, b, label):
        d = rho(predictors[a], D) - rho(predictors[b], D)
        lo, hi = boot_ci_delta(predictors[a], predictors[b], D, (study_name, a, b))
        obs, p = swap_diag_p(predictors[a], predictors[b], D, (study_name, a, b))
        print(f"  {label:38s} Delta {d:+.3f}  paired-CI [{lo:+.3f},{hi:+.3f}]  "
              f"(swap-diag p {p:.4f})")
        RESULT_ROWS.append((study_name, 'delta', f"{a}_vs_{b}",
                            f"{d:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{p:.4f}"))
        return d
    print("\nAxis 2 — trajectory increment (primary construct-matched first):")
    dR = delta_block('R_traj', 'R_pool', 'R vs R_pool')
    dE = delta_block('E_traj', 'E_pool(AC)', 'E vs E_pool(AC)  [bridge-dependent]')
    print("\nAxis 1 — criterion Deltas vs strongest mundane comparator:")
    for a in ('R_traj', 'E_traj'):
        for b in ('volatility_V', 'strength', 'volume'):
            delta_block(a, b, f'{a} vs {b}')
    return dR, dE


# =============================================================================
# SECONDARY EVENT STUDY (pre-declared in prespec v3): tkgl-polecat, fit
# Jan 2018 - Dec 2021, event Feb 2022, post Mar 2022 onward.
# Implementation declarations (recorded in prespec amendment BEFORE running):
#   PRE  = Mar 2021 - Feb 2022 (12 months; the last ~5 days of Feb 2022
#          postdate the event onset -- declared, direction: biases the
#          pre-mean TOWARD post levels, i.e., against finding displacement).
#   POST = Mar 2022 - Feb 2023 per prespec, truncated at data end
#          (Dec 2022): 10 months.
#   Graph-derived baselines computed INLINE on the fit window under the
#   conventions of the committed baselines cell (a = 0.9/max_t rho(A_t),
#   binary time-respecting earliest-arrival closeness, top-500 actor graph
#   by fit-window counts).
# Everything else is byte-derived from the committed Study 2 code.
# =============================================================================
print("\n" + "=" * 70)
print("SECONDARY: tkgl-polecat, fit 2018-2021, event Feb 2022 (prespec v3)")
COOP_A = frozenset({0, 1, 2, 3, 8, 13}); CONF = frozenset({4, 5, 7, 10, 11, 12, 14, 15})
src_l, dst_l, ts_l, rel_l = [], [], [], []
with open(os.path.join(POLECAT_RAW, "tkgl-polecat_edgelist.csv")) as f:
    for r in csv.DictReader(f):
        src_l.append(int(r['head'])); dst_l.append(int(r['tail']))
        ts_l.append(int(r['date'])); rel_l.append(int(r['relation_type']))
SRC = np.array(src_l); DST = np.array(dst_l)
TS = np.array(ts_l, np.int64)
assert int(np.array(rel_l).max()) < 32, "relation ids exceed 0-31 schema"
EV = np.array(rel_l) % 16
import datetime as _dt
def _mid(ts):
    d = _dt.datetime.utcfromtimestamp(int(ts))
    return d.year * 12 + (d.month - 1)
BASE_M = _dt.datetime(2018, 1, 1).year * 12 + 0
MB = np.array([_mid(t) - BASE_M for t in TS], dtype=int)   # 0 = Jan 2018
FIT_B  = list(range(0, 48))                    # Jan 2018 - Dec 2021
PRE_B  = list(range(38, 50))                   # Mar 2021 - Feb 2022
n_bins = int(MB.max()) + 1
POST_B = [b for b in range(50, 62) if b < n_bins]   # Mar 2022 -, truncated
print(f"fit {len(FIT_B)} months | pre {len(PRE_B)} | post {len(POST_B)} "
      f"(data ends at month index {n_bins-1})")

fit_mask = np.isin(MB, FIT_B)
counts_fit = np.bincount(
    np.concatenate([SRC[fit_mask], DST[fit_mask]]), minlength=SRC.max() + 1)
top2 = np.argsort(counts_fit)[::-1][:TOP_K]
pos = {int(a): j for j, a in enumerate(top2)}
ents2 = [f"actor{a}" for a in top2]

CO = np.zeros((n_bins, TOP_K)); CF = np.zeros((n_bins, TOP_K))
for s_, t_, e_, b_ in zip(SRC, DST, EV, MB):
    col = 0 if e_ in COOP_A else (1 if e_ in CONF else -1)
    if col < 0: continue
    for a in (int(s_), int(t_)):
        if a in pos:
            (CO if col == 0 else CF)[b_, pos[a]] += 1
CO, CF = np.log1p(CO), np.log1p(CF)
mu_c, sd_c = CO[FIT_B].mean(0), CO[FIT_B].std(0)
mu_f, sd_f = CF[FIT_B].mean(0), CF[FIT_B].std(0)
assert (sd_c > 1e-9).all() and (sd_f > 1e-9).all(), "dead channel in fit window"
ZC = (CO - mu_c) / (sd_c + 1e-9)
ZF = (CF - mu_f) / (sd_f + 1e-9)

X2 = np.zeros((len(FIT_B), 2 * TOP_K))
for j in range(TOP_K):
    X2[:, 2 * j] = ZC[FIT_B, j]; X2[:, 2 * j + 1] = ZF[FIT_B, j]
E2c, R2c, E2pc, R2pc = seed_avg(X2)
agg = lambda v: np.array([v[2 * j] + v[2 * j + 1] for j in range(TOP_K)])
E2, R2, E2p, R2p = map(agg, (E2c, R2c, E2pc, R2pc))
dR_ch, dE_ch = 1 - rho(R2c, R2pc), 1 - rho(E2c, E2pc)
dR_ac, dE_ac = 1 - rho(R2, R2p), 1 - rho(E2, E2p)
print(f"bridge rho(E,R) actor-level = {rho(E2,R2):+.3f}")
print(f"departures: channel-level d_E = {dE_ch:.3f}, d_R = {dR_ch:.3f} | "
      f"actor-level d_E = {dE_ac:.3f}, d_R = {dR_ac:.3f}")
print("(fit window CONTAINS COVID: d_E above is the increment-room gate)")
for _nm, _v in (('bridge_rho_E_R_actor', rho(E2,R2)),
                ('d_E_channel', dE_ch), ('d_R_channel', dR_ch),
                ('d_E_actor', dE_ac), ('d_R_actor', dR_ac)):
    RESULT_ROWS.append(('SECONDARY (polecat / Feb-2022)', 'construct', _nm,
                        f"{_v:.4f}", '', '', ''))

D2 = np.array([abs(ZC[POST_B, j].mean() - ZC[PRE_B, j].mean())
               + abs(ZF[POST_B, j].mean() - ZF[PRE_B, j].mean())
               for j in range(TOP_K)])
coop_share = lambda B, j: (np.expm1(CO[B, j]).sum()
                           / max(np.expm1(CO[B, j]).sum() + np.expm1(CF[B, j]).sum(), 1e-9))
D2_sec = np.array([abs(coop_share(POST_B, j) - coop_share(PRE_B, j))
                   for j in range(TOP_K)])

vol_fit = np.array([counts_fit[a] for a in top2], float)
from collections import defaultdict as _dd
partners = _dd(set)
_fitset = set(FIT_B)
for s_, t_, b_ in zip(SRC, DST, MB):
    if b_ in _fitset:
        if int(s_) in pos: partners[int(s_)].add(int(t_))
        if int(t_) in pos: partners[int(t_)].add(int(s_))
strength2 = np.array([len(partners[int(a)]) for a in top2], float)
volat2 = np.array([np.mean(np.abs(np.diff(ZC[FIT_B, j])))
                   + np.mean(np.abs(np.diff(ZF[FIT_B, j]))) for j in range(TOP_K)])

# ---- inline graph-derived baselines on the fit window (declared conventions)
top500 = np.argsort(counts_fit)[::-1][:500]
p500 = {int(a): j for j, a in enumerate(top500)}
assert all(int(a) in p500 for a in top2)
Gs = {b: np.zeros((500, 500)) for b in FIT_B}
for s_, t_, b_ in zip(SRC, DST, MB):
    if b_ in _fitset and int(s_) in p500 and int(t_) in p500:
        Gs[b_][p500[int(s_)], p500[int(t_)]] += 1
Gs = [np.log1p(Gs[b]) for b in sorted(FIT_B)]

def _specr(A):
    return float(np.max(np.abs(np.linalg.eigvals(A))))
_alpha = 0.9 / max(_specr(A) for A in Gs)
_Q = np.eye(500)
for A in Gs:
    _Q = _Q @ np.linalg.inv(np.eye(500) - _alpha * A)
_bc, _rc = _Q.sum(1), _Q.sum(0)
_reach = np.eye(500, dtype=bool)
_delay = np.full((500, 500), np.inf); np.fill_diagonal(_delay, 0.0)
for _t, A in enumerate(Gs, start=1):
    _adj = A > 0
    _new = (_reach @ _adj) & ~_reach
    _delay[_new & np.isinf(_delay)] = _t
    _reach = _reach | _new
with np.errstate(divide='ignore'):
    _inv = np.where(np.isinf(_delay), 0.0, 1.0 / np.maximum(_delay, 1.0))
np.fill_diagonal(_inv, 0.0)
_cl = _inv.sum(1) / 499.0
_sel = [p500[int(a)] for a in top2]

preds2 = {
    'R_traj': R2, 'E_traj': E2, 'R_pool': R2p, 'E_pool(AC)': E2p,
    'strength': strength2, 'volume': vol_fit, 'volatility_V': volat2,
    'communicability_bc': _bc[_sel], 'communicability_rc': _rc[_sel],
    'temporal_closeness': _cl[_sel],
}
res2 = report("SECONDARY (polecat / Feb-2022)", ents2, D2, D2_sec, preds2, vol_fit)


# ============================ persist results ================================
import csv as _csv
_res = os.path.join(OUT_DIR, "secondary_validation_results.csv")
with open(_res, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["study", "block", "name", "value", "ci_lo", "ci_hi", "p"])
    w.writerows(RESULT_ROWS)
_val = os.path.join(OUT_DIR, "secondary_validation_values.csv")
with open(_val, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["study", "entity", "quantity", "value"])
    w.writerows(VALUE_ROWS)
print(f"\nsaved {_res}")
print(f"saved {_val}")
