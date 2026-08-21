# =============================================================================
# Phase-diagram grid at 20 seeds. Frozen DGP and metrics.
# Writes phase_diagram_results.csv (the artifact the paper's Table 1 reads) and
# phase_diagram.png (Figure 1).
# =============================================================================
import sys, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd
# (src/ already on sys.path via the setup cell)
from scipy.stats import spearmanr, wilcoxon
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'code'))
from dynamical_leverage import dynamical_leverage_from_jacobians, average_controllability_gramian_trace
np.seterr(all='ignore')

N, HORIZON, DRIVER = 7, 8, 0
BGAIN, DIAG, NOISE = 2.0, 0.6, 0.5            # FROZEN calibration — do not change

def make_regimes(seed, structure):
    r=np.random.default_rng(seed)
    A=r.normal(0,0.04,(N,N)); np.fill_diagonal(A,r.uniform(DIAG-0.05,DIAG+0.05,N)); B=r.normal(0,0.04,(N,N))
    A1,B1=A.copy(),B.copy(); A2,B2=A.copy(),B.copy()
    for t in (1,2,3): B1[t,DRIVER]=BGAIN
    if structure in ('smooth-drift','persistent-switch','sign-flip-switch'):
        for t in (1,2,3): B2[t,DRIVER]=(-BGAIN if 'sign-flip' in structure else BGAIN)
        if structure=='persistent-switch':
            for t in (1,2,3): B2[t,DRIVER]=0.0
            for t in (1,2,3): B2[t,4]=BGAIN
    for M in (A1,A2):
        sr=np.max(np.abs(np.linalg.eigvals(M)))
        if sr>0.85: M*=0.85/sr
    return (A1,B1),(A2,B2)
def step(reg,x,alpha): A,B=reg; return A@x+alpha*np.tanh(B@x)
def jac(reg,x,alpha): A,B=reg; pre=B@x; return A+alpha*(np.diag(1-np.tanh(pre)**2)@B)
def regime_at(t,L,regs,structure,which):
    if structure=='static': return regs[0]
    if structure=='smooth-drift':
        g=min(1.0,t/L); return (g*regs[1][0]+(1-g)*regs[0][0], g*regs[1][1]+(1-g)*regs[0][1])
    return regs[which[min(t,len(which)-1)]]
def simulate(regs,structure,alpha,persistence,T=400,seed=0):
    r=np.random.default_rng(seed+7); x=r.normal(0,1.0,N); xs=[x.copy()]; which=[]; cur=0
    p={'short':0.30,'medium':0.12,'long':0.02}[persistence]
    for t in range(T):
        if structure not in ('static','smooth-drift') and r.random()<p: cur=1-cur
        reg=regime_at(t,T,regs,structure,which+[cur]); x=step(reg,x,alpha)+r.normal(0,NOISE,N)
        xs.append(x.copy()); which.append(cur)
        if not np.all(np.isfinite(x)) or np.max(np.abs(x))>1e4: return None,None
    return np.array(xs),which
def true_contrib(xs,regs,structure,which,alpha,eps):
    acc=np.zeros(N)
    for t in range(len(xs)-HORIZON+1):
        x0=xs[t]
        for j in range(N):
            e=np.zeros(N);e[j]=eps;xp=x0+e;xb=x0.copy();en=np.sum((xp-xb)**2)
            for h in range(HORIZON-1):
                reg=regime_at(t+h,len(xs),regs,structure,which);xp=step(reg,xp,alpha);xb=step(reg,xb,alpha);en+=np.sum((xp-xb)**2)
            acc[j]+=en
    return acc/acc.sum()
def cell(structure,alpha,persistence,eps,seed):
    regs=make_regimes(seed,structure); xs,which=simulate(regs,structure,alpha,persistence,seed=seed)
    if xs is None: return None
    tr=true_contrib(xs,regs,structure,which,alpha,eps)
    Js=[jac(regime_at(t,len(xs),regs,structure,which),xs[t],alpha) for t in range(len(xs)-1)]
    acc={k:[] for k in range(N)}
    for t in range(len(Js)-HORIZON+2):
        jseq=[Js[t+h] for h in range(HORIZON-1)]
        ec=dynamical_leverage_from_jacobians(jseq,n_components=N,horizon=HORIZON,strict=True)
        for k in range(N): acc[k].append(ec.weights[k])
    Ej=np.array([np.mean(acc[k]) for k in range(N)])
    Jmean=np.mean(Js,0)
    ACr=np.array([average_controllability_gramian_trace(Jmean,j,HORIZON) for j in range(N)])
    Abar=0.5*(regs[0][0]+regs[1][0]); Bbar=0.5*(regs[0][1]+regs[1][1])
    Jp=jac((Abar,Bbar),xs.mean(0),alpha)
    ACp=np.array([average_controllability_gramian_trace(Jp,j,HORIZON) for j in range(N)])
    return (spearmanr(Ej,tr).correlation,spearmanr(ACr,tr).correlation,
            spearmanr(ACp,tr).correlation,spearmanr(Ej,ACr).correlation,
            spearmanr(Ej,ACp).correlation)

NL={'low':0.3,'med':0.8,'high':1.5}
STRUCT=['static','smooth-drift','persistent-switch','sign-flip-switch']
PERS=['short','medium','long']
EPS={'small':0.02,'med':1.0,'large':6.0,'extreme':12.0}
SEEDS=range(20)                                 # <-- ONLY CHANGE: 8 -> 20
NO_SWITCH={'static','smooth-drift'}

def paired_ci(g,B=2000):
    g=np.asarray(g)
    if len(g)<2: return (np.nan,np.nan)
    bs=[np.mean(np.random.choice(g,len(g),replace=True)) for _ in range(B)]
    return (np.percentile(bs,2.5),np.percentile(bs,97.5))

rows=[]
for st in STRUCT:
  for nlk,al in NL.items():
    for pe in PERS:
      for epk,ep in EPS.items():
        rs=[cell(st,al,pe,ep,s) for s in SEEDS]; rs=[r for r in rs if r and np.all(np.isfinite(r))]
        if not rs: continue
        rs=np.array(rs); gr=rs[:,0]-rs[:,1]; gp=rs[:,0]-rs[:,2]
        cir=paired_ci(gr); cip=paired_ci(gp)
        try: wp=wilcoxon(rs[:,0],rs[:,2]).pvalue if len(rs)>=6 else np.nan
        except: wp=np.nan
        m=rs.mean(0)
        rows.append(dict(struct=st,nl=nlk,pers=pe,eps=epk,ej_t=m[0],acr_t=m[1],acp_t=m[2],ej_acr=m[3],
                         d_real=gr.mean(),d_pool=gp.mean(),d_real_lo=cir[0],d_real_hi=cir[1],
                         d_pool_lo=cip[0],d_pool_hi=cip[1],wilcoxon_p=wp,n=len(rs),
                         degenerate_control=(st in NO_SWITCH),ej_acp=m[4]))
        print(f"{st:>16s} {nlk:>4s} {pe:>6s} {epk:>7s} | Dpool {gp.mean():.3f} [{cip[0]:.3f},{cip[1]:.3f}] n={len(rs)}")
df=pd.DataFrame(rows)
import os as _os
OUT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results")
df.to_csv(_os.path.join(str(OUT_DIR),"phase_diagram_results.csv"), index=False)
print("\nsaved phase_diagram_results.csv (20 seeds)")
fig,axes=plt.subplots(len(STRUCT),len(NL),figsize=(11,12),sharex=True,sharey=True)
epk_order=list(EPS); xpos=range(len(epk_order))
for i,st in enumerate(STRUCT):
  for j,nlk in enumerate(NL):
    ax=axes[i][j]
    for pe in PERS:
      sub=[df[(df.struct==st)&(df.nl==nlk)&(df.pers==pe)&(df.eps==ek)] for ek in epk_order]
      ejt=[(s.ej_t.values[0] if len(s) else np.nan) for s in sub]
      acrt=[(s.acr_t.values[0] if len(s) else np.nan) for s in sub]
      acpt=[(s.acp_t.values[0] if len(s) else np.nan) for s in sub]
      ax.plot(xpos,ejt,'-o',label=f'E_j ({pe})',alpha=0.8)
      ax.plot(xpos,acrt,'--x',label=f'AC_real ({pe})',alpha=0.45)
      ax.plot(xpos,acpt,':s',label=f'AC_pool ({pe})',alpha=0.35,markersize=3)
    ax.set_title(f'{st}\n{nlk} nonlin',fontsize=8); ax.set_ylim(0.3,1.05)
    if i==len(STRUCT)-1: ax.set_xticks(xpos); ax.set_xticklabels(epk_order,fontsize=7,rotation=45)
    if i==0 and j==len(NL)-1: ax.legend(fontsize=4,ncol=2)
fig.suptitle('Phase diagram: fidelity to true contribution — E_j (solid) vs AC_realized (dashed) vs AC_pooled (dotted)',fontsize=10)
fig.supxlabel('perturbation amplitude'); fig.supylabel('Spearman fidelity to ground truth')
plt.tight_layout(); plt.savefig(_os.path.join(str(OUT_DIR),"phase_diagram.png"), dpi=150, bbox_inches='tight')
print("saved phase_diagram.png")