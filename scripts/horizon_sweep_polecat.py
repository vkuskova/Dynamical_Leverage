# =============================================================================
# HORIZON SWEEP (polecat) -- robustness cell, one job only.
# Derived from the committed polecat sweep script: panel construction and
# fitting are byte-identical to the code that produced the committed
# artifacts (coding A, the canonical valenced construction). This cell fits
# 8 seeds once each and rescoring E_j, realized AC, and pooled AC at
# horizons H in {4, 8, 12, 16} from the same Jacobians. Reported (fixed
# before running, no selection): per-horizon pooled/realized departure
# distributions; pairwise Spearman between horizon variants of the
# ensemble-mean ranking; per-horizon across-seed ranking stability (28
# seed pairs). Continuity check: the H=8 rows must reproduce the committed
# coding-A departures digit-for-digit. Valence-coding comparisons and
# ablations are the COMMITTED PARENT script's outputs, not duplicated here;
# inherited config beyond this cell's needs is unused.
# =============================================================================
import os, sys, csv, numpy as np, torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results")
sys.path.insert(0, os.path.join(HERE, "..", "code"))
try:
    from dynamical_leverage import (dynamical_leverage_from_jacobians,
                                    average_controllability_gramian_trace)
    from neural_var import fit_navar
except ImportError as e:
    raise ImportError(
        "dynamical_leverage / neural_var not importable. For a Drive-side "
        "run, copy code/dynamical_leverage.py and code/neural_var.py from "
        "this repository into your Drive code folder first.") from e
from torch.func import jacrev

N_SEEDS = 8
MONTH = 2592000; TOP_K = 20
MAXLAGS, HORIZON, EPOCHS = 3, 8, 400
CODINGS = {
    "A_primary":       (frozenset({0,1,2,3,8,13}),   frozenset({4,5,7,10,11,12,14,15})),
    "B_retreat_coop":  (frozenset({0,1,2,3,8,9,13}), frozenset({4,5,7,10,11,12,14,15})),
}

# ---- load raw quadruples once ----
src, dst, ts, rel = [], [], [], []
with open(f"{RAW}/tkgl-polecat_edgelist.csv") as f:
    for row in csv.DictReader(f):
        src.append(int(row['head'])); dst.append(int(row['tail']))
        ts.append(int(row['date'])); rel.append(int(row['relation_type']))
src, dst = np.array(src), np.array(dst)
ts, rel = np.array(ts, np.int64), np.array(rel)
ev = rel % 16
mb = ((ts - ts.min()) // MONTH).astype(int); n_bins = mb.max() + 1
counts = np.bincount(np.concatenate([src, dst]))
top = np.argsort(counts)[::-1][:TOP_K]
pos = {int(a): j for j, a in enumerate(top)}

def build_X(COOP, CONF):
    X = np.zeros((n_bins, 2*TOP_K)); mapped = 0
    for s_, t_, e_, b_ in zip(src, dst, ev, mb):
        col = 0 if e_ in COOP else (1 if e_ in CONF else -1)
        if col < 0: continue
        mapped += 1
        if int(s_) in pos: X[b_, 2*pos[int(s_)] + col] += 1
        if int(t_) in pos: X[b_, 2*pos[int(t_)] + col] += 1
    X = np.log1p(X); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X, mapped / len(ev)

def companion_fn(model, N):
    def pw(wv): p,_ = model(wv.reshape(1,N,MAXLAGS)); return p.reshape(-1)
    jr = jacrev(pw)
    def c(wv):
        wv = torch.as_tensor(np.asarray(wv), dtype=torch.float32)
        Jb = jr(wv); top_ = torch.cat([Jb[:,:,MAXLAGS-1-l] for l in range(MAXLAGS)],1)
        eye = torch.eye(N,dtype=top_.dtype); zero = torch.zeros(N,N,dtype=top_.dtype)
        rws = [torch.cat([eye if cc==(r-1) else zero for cc in range(MAXLAGS)],1)
               for r in range(1,MAXLAGS)]
        return torch.cat([top_]+rws,0).detach().numpy()
    return c

HS = (4, 8, 12, 16)                     # horizon sweep, frozen

def score_at_horizons(X, Y, N, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model, _, _ = fit_navar(X, [(0, Y)], n_components=N, maxlags=MAXLAGS,
                            epochs=EPOCHS, seed=seed)
    jac = companion_fn(model, N)
    Js = [jac(X[s-MAXLAGS:s].T) for s in range(MAXLAGS, Y)]
    Cm = sum(Js) / len(Js)
    Jp = jac(np.tile(X.mean(0)[:, None], (1, MAXLAGS)))
    out = {}
    for H in HS:
        acc = np.zeros(N); nwin = 0
        for p in range(MAXLAGS-1, Y-H+1):
            st = p+1
            jseq = [Js[s-MAXLAGS] for s in range(st, st+H-1)]
            ec = dynamical_leverage_from_jacobians(jseq, n_components=N,
                                                      horizon=H, strict=True)
            acc += np.array([ec.weights[k] for k in range(N)]); nwin += 1
        Ej = acc / nwin
        cr = np.array([average_controllability_gramian_trace(Cm, j, H)
                       for j in range(N)])
        cp = np.array([average_controllability_gramian_trace(Jp, j, H)
                       for j in range(N)])
        out[H] = (Ej,
                  1 - spearmanr(Ej, cr).correlation,
                  1 - spearmanr(Ej, cp).correlation)
    return out


X, cov = build_X(*CODINGS["A_primary"])       # canonical coding, committed
Y, N = X.shape[0], X.shape[1]
print(f"horizon sweep on polecat (coding A): {Y} bins x {N} channels, "
      f"coverage {cov:.1%}, seeds {N_SEEDS}, horizons {HS}")
rows, rank_rows = [], []
EJH = {H: [] for H in HS}
for seed in range(N_SEEDS):
    res = score_at_horizons(X, Y, N, seed)
    for H in HS:
        Ej, gr, gp = res[H]
        EJH[H].append(Ej)
        rows.append(["polecat", H, seed, f"{gr:.4f}", f"{gp:.4f}"])
        print(f"  seed {seed} H={H:2d}: realized {gr:.4f} pooled {gp:.4f}")

for H in HS:
    g = np.array([float(r[4]) for r in rows if r[1]==H])
    print(f"H={H:2d}: pooled departure {g.mean():.3f} +/- {g.std():.3f} "
          f"[{g.min():.3f}, {g.max():.3f}]")

mean_rank = {H: np.argsort(np.argsort(-np.mean(EJH[H], 0))) + 1 for H in HS}
print("pairwise Spearman between horizon variants (ensemble-mean ranking):")
for a in range(len(HS)):
    for b in range(a+1, len(HS)):
        r = spearmanr(mean_rank[HS[a]], mean_rank[HS[b]]).correlation
        rank_rows.append(["polecat", HS[a], HS[b], f"{r:.4f}"])
        print(f"  H={HS[a]} vs H={HS[b]}: {r:+.3f}")

SYSTEM_NAME = "polecat"
import csv as _csv, os as _os
_out = _os.path.join(OUT, "horizon_sweep_polecat.csv")
with open(_out, "w", newline="") as f:
    w = _csv.writer(f); w.writerow(["system","horizon","seed","realized_gap","pooled_gap"])
    w.writerows(rows)
_out2 = _os.path.join(OUT, "horizon_rankcorr_polecat.csv")
with open(_out2, "w", newline="") as f:
    w = _csv.writer(f); w.writerow(["system","h1","h2","spearman"])
    w.writerows(rank_rows)
print("saved", _out, "and", _out2)

# per-horizon ACROSS-SEED ranking stability (28 seed pairs per horizon)
from itertools import combinations as _comb
stab_rows = []
print("across-seed rank stability per horizon (mean pairwise Spearman, 28 pairs):")
for H in HS:
    _rs = [spearmanr(EJH[H][a], EJH[H][b]).correlation
           for a, b in _comb(range(N_SEEDS), 2)]
    print(f"  H={H:2d}: {np.mean(_rs):.3f} +/- {np.std(_rs):.3f} "
          f"[{np.min(_rs):.3f}, {np.max(_rs):.3f}]")
    stab_rows.append([SYSTEM_NAME, H, f"{np.mean(_rs):.4f}", f"{np.std(_rs):.4f}",
                      f"{np.min(_rs):.4f}", f"{np.max(_rs):.4f}"])
_out3 = _os.path.join(OUT, f"horizon_seedstab_{SYSTEM_NAME}.csv")
with open(_out3, "w", newline="") as f:
    w = _csv.writer(f); w.writerow(["system","horizon","mean","sd","min","max"])
    w.writerows(stab_rows)
print("saved", _out3)

