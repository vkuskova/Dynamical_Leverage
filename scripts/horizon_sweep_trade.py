# ============================================================================# =============================================================================
# HORIZON SWEEP (trade) -- robustness cell, one job only.
# Derived from the committed trade sweep script: panel construction and
# fitting are byte-identical to the code that produced the committed
# artifacts. This cell fits 8 seeds once each and rescoring E_j, realized
# AC, and pooled AC at horizons H in {4, 8, 12, 16} from the same
# Jacobians. Reported (fixed before running, no selection): per-horizon
# pooled/realized departure distributions; pairwise Spearman between
# horizon variants of the ensemble-mean ranking; per-horizon across-seed
# ranking stability (28 seed pairs). Continuity check: the H=8 rows must
# reproduce the committed departure numbers digit-for-digit.
# Structural baselines, perturbation validation, and canonical-seed
# artifacts are the COMMITTED PARENT script's outputs, not duplicated here;
# any inherited config constants beyond this cell's needs are unused.
# =============================================================================
import os, sys, csv, numpy as np, torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
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

N_SEEDS = 8; CANON = 0
YEAR_MIN, TOP_K = 1993, 20
MAXLAGS, HORIZON, EPOCHS = 3, 8, 400
DELTA = 1.0

# ---------- build the clean panel ----------
rows, years_set = [], set()
with open(f"{RAW}/tgbn-trade_edgelist.csv") as f:
    for row in csv.DictReader(f):
        y = int(row['year'])
        if y < YEAR_MIN: continue
        rows.append((y, row['nation'], row['trading nation'], float(row['weight'])))
        years_set.add(y)
years = np.array(sorted(years_set)); yidx = {y:i for i,y in enumerate(years)}
nations = sorted({t for _,_,t,_ in rows})          # import side only
nidx = {n:i for i,n in enumerate(nations)}
I = np.zeros((len(nations), len(years)))
for y,s,t,w in rows: I[nidx[t], yidx[y]] += w      # attraction: shares pointing at t
attract_total = I.sum(1)
top = [nations[i] for i in np.argsort(attract_total)[::-1][:TOP_K]]
Y = len(years)
X = np.zeros((Y, TOP_K))
for j,nm in enumerate(top): X[:,j] = np.log1p(I[nidx[nm]])
# sanity: every channel must have real variance now
cvs = X.std(0) / (np.abs(X.mean(0)) + 1e-12)
assert (X.std(0) > 1e-6).all(), "dead channel in clean panel - stop"
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
N = X.shape[1]
print(f"clean panel: {Y} yrs x {N} channels (imports-attraction), "
      f"min channel CV {cvs.min():.3f} - no dead channels")
print("top-20 by attraction:", top[:8], "...")

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

print(f"horizon sweep on trade: seeds {N_SEEDS}, horizons {HS}")
rows, rank_rows = [], []
EJH = {H: [] for H in HS}
for seed in range(N_SEEDS):
    res = score_at_horizons(X, Y, N, seed)
    for H in HS:
        Ej, gr, gp = res[H]
        EJH[H].append(Ej)
        rows.append(["trade", H, seed, f"{gr:.4f}", f"{gp:.4f}"])
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
        rank_rows.append(["trade", HS[a], HS[b], f"{r:.4f}"])
        print(f"  H={HS[a]} vs H={HS[b]}: {r:+.3f}")

SYSTEM_NAME = "trade"
import csv as _csv, os as _os
_out = _os.path.join(OUT, "horizon_sweep_trade.csv")
with open(_out, "w", newline="") as f:
    w = _csv.writer(f); w.writerow(["system","horizon","seed","realized_gap","pooled_gap"])
    w.writerows(rows)
_out2 = _os.path.join(OUT, "horizon_rankcorr_trade.csv")
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

