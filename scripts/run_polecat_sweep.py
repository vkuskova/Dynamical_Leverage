#!/usr/bin/env python3
# =============================================================================
# run_polecat_sweep.py — tkgl-polecat E_j sweep (Path B). Runs BOTH as a repo
# script and as a Colab cell.
#
# Config (one line): ABLATION
#   None         -> primary valenced run: coding A and coding B, 8 seeds each
#   "role_split" -> ablation (a): all 16 event types, per-actor
#                   (as-source, as-target) channels, N=40, coverage 100%.
#                   One run only (codings are irrelevant to it).
#   "collapsed"  -> ablation (b): COOP-union-CONF event filter, ONE total-
#                   activity channel per actor, N=20. Runs both codings
#                   (their event filters differ).
#
# Output: ONE accumulating CSV (results/polecat_sweep_results.csv). Re-running
# a variant replaces its rows; other variants' rows are preserved. So the three
# invocations (None, role_split, collapsed) build a single committed artifact.
#
# Environments:
#   Repo:  scripts/run_polecat_sweep.py with code/, data/, results/ siblings.
#   Colab: paste as a cell; Drive is mounted automatically; defaults below
#          point at the project Drive layout (override COLAB_* if yours
#          differs). Copy code/dynamical_leverage.py and code/neural_var.py
#          into the Drive code folder before the first Colab run.
# =============================================================================
import os, sys, csv
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.func import jacrev

# ---------------- configuration ----------------
ABLATION = globals().get("ABLATION", None)   # None | "role_split" | "collapsed"
N_SEEDS  = 8
TOP_K    = 20
MONTH    = 2592000
MAXLAGS, HORIZON, EPOCHS = 3, 8, 400

COOP_A = frozenset({0, 1, 2, 3, 8, 13})
CONF   = frozenset({4, 5, 7, 10, 11, 12, 14, 15})
COOP_B = COOP_A | {9}                        # RETREAT as cooperation
# REQUEST(6) excluded everywhere; RETREAT(9) excluded in coding A.

# ---------------- environment: Colab vs repo ----------------
IN_COLAB = 'google.colab' in sys.modules or os.path.isdir('/content')
if IN_COLAB:
    if not os.path.isdir('/content/drive/MyDrive'):
        from google.colab import drive
        drive.mount('/content/drive')
    DATA_DIR = globals().get("COLAB_DATA",
        "/content/drive/MyDrive/WSDM_TemporalEj/rawdata_polecat")
    OUT_DIR = globals().get("COLAB_OUT",
        "/content/drive/MyDrive/WSDM_TemporalEj/polecat")
    CODE_DIR = globals().get("COLAB_CODE",
        "/content/drive/MyDrive/code")
    sys.path.insert(0, CODE_DIR)
else:
    try:
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:                         # interactive shell outside Colab
        ROOT = os.getcwd()
    DATA_DIR = os.environ.get("POLECAT_DATA", os.path.join(ROOT, "data"))
    OUT_DIR  = os.environ.get("POLECAT_OUT",  os.path.join(ROOT, "results"))
    sys.path.insert(0, os.path.join(ROOT, "code"))
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

# ---------------- load raw quadruples ----------------
edge_path = os.path.join(DATA_DIR, "tkgl-polecat_edgelist.csv")
assert os.path.exists(edge_path), f"edgelist not found: {edge_path}"
src_l, dst_l, ts_l, rel_l = [], [], [], []
with open(edge_path) as f:
    for row in csv.DictReader(f):
        src_l.append(int(row["head"])); dst_l.append(int(row["tail"]))
        ts_l.append(int(row["date"])); rel_l.append(int(row["relation_type"]))
SRC = np.array(src_l); DST = np.array(dst_l)
TS = np.array(ts_l, np.int64); EV = np.array(rel_l) % 16
MB = ((TS - TS.min()) // MONTH).astype(int)
N_BINS = int(MB.max()) + 1
counts = np.bincount(np.concatenate([SRC, DST]))
TOP = np.argsort(counts)[::-1][:TOP_K]
POS = {int(a): j for j, a in enumerate(TOP)}
print(f"quadruples {len(SRC)} | monthly bins {N_BINS} | top-{TOP_K} actors")

# ---------------- panel constructions ----------------
def build_panel(variant):
    if variant in ("valence_A", "valence_B"):
        COOP = COOP_A if variant == "valence_A" else COOP_B
        X = np.zeros((N_BINS, 2 * TOP_K)); mapped = 0
        for s_, t_, e_, b_ in zip(SRC, DST, EV, MB):
            col = 0 if e_ in COOP else (1 if e_ in CONF else -1)
            if col < 0: continue
            mapped += 1
            if int(s_) in POS: X[b_, 2 * POS[int(s_)] + col] += 1
            if int(t_) in POS: X[b_, 2 * POS[int(t_)] + col] += 1
        coverage = mapped / len(EV); valenced = True
    elif variant == "ablation_role_split":
        X = np.zeros((N_BINS, 2 * TOP_K))
        for s_, t_, b_ in zip(SRC, DST, MB):
            if int(s_) in POS: X[b_, 2 * POS[int(s_)]] += 1
            if int(t_) in POS: X[b_, 2 * POS[int(t_)] + 1] += 1
        coverage = 1.0; valenced = False
    elif variant in ("ablation_collapsed_A", "ablation_collapsed_B"):
        COOP = COOP_A if variant.endswith("A") else COOP_B
        KEEP = COOP | CONF
        X = np.zeros((N_BINS, TOP_K)); mapped = 0
        for s_, t_, e_, b_ in zip(SRC, DST, EV, MB):
            if e_ not in KEEP: continue
            mapped += 1
            if int(s_) in POS: X[b_, POS[int(s_)]] += 1
            if int(t_) in POS: X[b_, POS[int(t_)]] += 1
        coverage = mapped / len(EV); valenced = False
    else:
        raise ValueError(variant)
    X = np.log1p(X)
    sd = X.std(0)
    assert (sd > 1e-9).all(), f"dead channel in {variant} panel"
    X = (X - X.mean(0)) / (sd + 1e-9)
    return X, coverage, valenced

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

def one_fit(X, seed, valenced):
    N, T = X.shape[1], X.shape[0]
    torch.manual_seed(seed); np.random.seed(seed)
    model, _, _ = fit_navar(X, [(0, T)], n_components=N, maxlags=MAXLAGS,
                            epochs=EPOCHS, seed=seed)
    jac = companion_fn(model, N)
    Js = [jac(X[s - MAXLAGS:s].T) for s in range(MAXLAGS, T)]
    Cm = sum(Js) / len(Js)
    acc = np.zeros(N); nwin = 0
    for p in range(MAXLAGS - 1, T - HORIZON + 1):
        st = p + 1
        jseq = [Js[s - MAXLAGS] for s in range(st, st + HORIZON - 1)]
        ec = dynamical_leverage_from_jacobians(
            jseq, n_components=N, horizon=HORIZON, strict=True)
        acc += np.array([ec.weights[k] for k in range(N)]); nwin += 1
    Ej = acc / nwin
    ctrl_r = np.array([average_controllability_gramian_trace(Cm, j, HORIZON)
                       for j in range(N)])
    ctrl_p = np.array([average_controllability_gramian_trace(
        jac(np.tile(X.mean(0)[:, None], (1, MAXLAGS))), j, HORIZON)
        for j in range(N)])
    gap_r = 1 - spearmanr(Ej, ctrl_r).correlation
    gap_p = 1 - spearmanr(Ej, ctrl_p).correlation
    coop_share = float(Ej[0::2].sum() / Ej.sum()) if valenced else None
    return gap_r, gap_p, coop_share

# ---------------- which variants this invocation runs ----------------
if ABLATION is None:
    RUN = ["valence_A", "valence_B"]
elif ABLATION == "role_split":
    RUN = ["ablation_role_split"]
elif ABLATION == "collapsed":
    RUN = ["ablation_collapsed_A", "ablation_collapsed_B"]
else:
    raise ValueError(f"ABLATION={ABLATION!r}")

# ---------------- run ----------------
new_rows = []
for variant in RUN:
    X, coverage, valenced = build_panel(variant)
    print(f"\n=== {variant}: coverage {coverage:.1%}, "
          f"{X.shape[0]} bins x {X.shape[1]} channels ===")
    GR, GP = [], []
    for seed in range(N_SEEDS):
        gr, gp, cs = one_fit(X, seed, valenced)
        GR.append(gr); GP.append(gp)
        cs_str = f"{cs:.2f}" if cs is not None else "n/a"
        print(f"  seed {seed}: realized gap {gr:.3f} | pooled gap {gp:.3f} | "
              f"coop share {cs_str}")
        new_rows.append([variant, seed, f"{gr:.4f}", f"{gp:.4f}",
                         f"{cs:.3f}" if cs is not None else "",
                         f"{coverage:.4f}"])
    GR, GP = np.array(GR), np.array(GP)
    print(f"  -> pooled gap:   {GP.mean():.3f} +/- {GP.std():.3f}  "
          f"[{GP.min():.3f}, {GP.max():.3f}]")
    print(f"  -> realized gap: {GR.mean():.3f} +/- {GR.std():.3f}")

# ---------------- accumulate into ONE csv (replace reran variants) ----------
csv_path = os.path.join(OUT_DIR, "polecat_sweep_results.csv")
HEADER = ["variant", "seed", "realized_gap", "pooled_gap",
          "coop_share", "valence_coverage"]
kept = []
if os.path.exists(csv_path):
    with open(csv_path) as f:
        r = csv.reader(f); next(r, None)
        kept = [row for row in r if row and row[0] not in RUN]
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(HEADER)
    w.writerows(kept); w.writerows(new_rows)
print(f"\nwrote {csv_path} "
      f"({len(kept)} preserved rows + {len(new_rows)} new rows)")
