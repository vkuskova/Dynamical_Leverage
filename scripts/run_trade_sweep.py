# =============================================================================
# tgbn-trade sweep: imports-attraction panel, 20 channels, 1993-2016.
# tgbn-trade edge weights are per-exporter SHARES, so export totals are
# constant by construction; node state is therefore log1p of a nation's
# import-attraction (sum of all exporters' shares pointing at it).
#
# One run produces every trade number in the paper:
#   (1) 8-seed sweep: pooled/realized departure distributions
#   (2) nation ranking stability across seeds (Table 1)
#   (3) baselines: strength centrality, volume, static AC
#   (4) perturbation validation on the canonical seed
# Requires data/tgbn-trade_edgelist.csv (run scripts/fetch_data.py). ~10 min.
# =============================================================================
import os, sys, csv, numpy as np, torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from dynamical_leverage import (dynamical_leverage_from_jacobians,
                                average_controllability_gramian_trace)
from neural_var import fit_navar
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

def fit_and_score(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model,_,_ = fit_navar(X, [(0,Y)], n_components=N, maxlags=MAXLAGS,
                          epochs=EPOCHS, seed=seed)
    jac = companion_fn(model, N)
    Js = [jac(X[s-MAXLAGS:s].T) for s in range(MAXLAGS, Y)]
    Cm = sum(Js) / len(Js)
    acc = np.zeros(N); nwin = 0
    for p in range(MAXLAGS-1, Y-HORIZON+1):
        st = p+1
        jseq = [Js[s-MAXLAGS] for s in range(st, st+HORIZON-1)]
        ec = dynamical_leverage_from_jacobians(jseq, n_components=N,
                                                  horizon=HORIZON, strict=True)
        acc += np.array([ec.weights[k] for k in range(N)]); nwin += 1
    Ej = acc / nwin
    ctrl_r = np.array([average_controllability_gramian_trace(Cm, j, HORIZON)
                       for j in range(N)])
    ctrl_p = np.array([average_controllability_gramian_trace(
        jac(np.tile(X.mean(0)[:,None],(1,MAXLAGS))), j, HORIZON) for j in range(N)])
    return model, jac, Ej, ctrl_r, ctrl_p

# ---------- (1)+(2) seed sweep ----------
GR, GP, RANKS = [], [], {}
canon_pack = None
for seed in range(N_SEEDS):
    model, jac, Ej, cr, cp = fit_and_score(seed)
    gr = 1 - spearmanr(Ej, cr).correlation
    gp = 1 - spearmanr(Ej, cp).correlation
    GR.append(gr); GP.append(gp)
    order = list(np.argsort(Ej)[::-1])
    RANKS[seed] = {top[i]: r+1 for r,i in enumerate(order)}
    if seed == CANON: canon_pack = (model, jac, Ej, cp)
    print(f"seed {seed}: realized gap {gr:.3f} | pooled gap {gp:.3f} | "
          f"top-3 {[top[i] for i in order[:3]]}")
GR, GP = np.array(GR), np.array(GP)
print(f"\npooled gap:   {GP.mean():.3f} +/- {GP.std():.3f}  [{GP.min():.3f},{GP.max():.3f}]")
print(f"realized gap: {GR.mean():.3f} +/- {GR.std():.3f}")

R = np.array([[RANKS[s][nm] for s in range(N_SEEDS)] for nm in top])
pw = [spearmanr([RANKS[a][n] for n in top],[RANKS[b][n] for n in top]).correlation
      for a in range(N_SEEDS) for b in range(a+1,N_SEEDS)]
pw = np.array(pw)
print(f"rank stability: mean pairwise Spearman {pw.mean():.3f} [{pw.min():.3f},{pw.max():.3f}]")
print(f"\n{'nation':30s} {'mean_rank':>9s} {'sd':>5s}")
for i in np.argsort(R.mean(1)):
    print(f"{top[i]:30s} {R[i].mean():9.1f} {R[i].std():5.2f}")

# ---------- (3) baselines on canonical seed ----------
model, jac, Ej, ctrl_p = canon_pack
strength = {nm: 0.0 for nm in top}; topset = set(top)
for y,s,t,w in rows:
    if t in topset: strength[t] += w
    if s in topset: strength[s] += w
strength = {nm: v/len(years) for nm,v in strength.items()}
ejv = np.array([Ej[j] for j in range(N)])
stv = np.array([strength[nm] for nm in top])
vol = np.array([attract_total[nidx[nm]] for nm in top])
print("\n=== canonical-seed contrasts ===")
print(f"Spearman(E_j, strength centrality): {spearmanr(ejv, stv).correlation:+.3f}")
print(f"Spearman(E_j, attraction volume):   {spearmanr(ejv, vol).correlation:+.3f}")
print(f"Spearman(E_j, static AC pooled):    {spearmanr(ejv, ctrl_p).correlation:+.3f}")

# ---------- (4) perturbation validation (canonical seed) ----------
def rollout(hist, T):
    h = torch.tensor(hist.copy(), dtype=torch.float32); outp = []
    for _ in range(T):
        p,_ = model(h.T.reshape(1, N, MAXLAGS))
        nxt = p.reshape(-1); outp.append(nxt.detach().numpy())
        h = torch.cat([h[1:], nxt.reshape(1, N)], 0)
    return np.array(outp)
starts = list(range(MAXLAGS, Y - HORIZON))
impact = np.zeros(N)
for t in starts:
    hist = X[t-MAXLAGS:t]; base = rollout(hist, HORIZON)
    for j in range(N):
        h2 = hist.copy(); h2[-1, j] += DELTA
        impact[j] += float(np.linalg.norm(rollout(h2, HORIZON) - base))
impact /= len(starts)
print("\n=== perturbation validation (clean panel, canonical seed) ===")
print(f"Spearman(impact, E_j):       {spearmanr(impact, ejv).correlation:+.3f}")
print(f"Spearman(impact, static AC): {spearmanr(impact, ctrl_p).correlation:+.3f}")
print(f"Spearman(impact, strength):  {spearmanr(impact, stv).correlation:+.3f}")
print(f"Spearman(impact, volume):    {spearmanr(impact, vol).correlation:+.3f}")

# ---------- save ----------
with open(f"{OUT}/clean_sweep.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["seed","realized_gap","pooled_gap"])
    for s in range(N_SEEDS): w.writerow([s, f"{GR[s]:.4f}", f"{GP[s]:.4f}"])
with open(f"{OUT}/clean_rankings.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["nation","mean_rank","sd"]+[f"seed{s}" for s in range(N_SEEDS)])
    for i,nm in enumerate(top):
        w.writerow([nm, f"{R[i].mean():.2f}", f"{R[i].std():.2f}"]+list(R[i]))
with open(f"{OUT}/clean_validation.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["nation","Ej","staticAC","strength","volume","impact"])
    for j,nm in enumerate(top):
        w.writerow([nm, f"{ejv[j]:.6f}", f"{ctrl_p[j]:.6f}",
                    f"{stv[j]:.4f}", f"{vol[j]:.4f}", f"{impact[j]:.6f}"])
print(f"\nsaved clean_sweep.csv, clean_rankings.csv, clean_validation.csv -> {OUT}")
