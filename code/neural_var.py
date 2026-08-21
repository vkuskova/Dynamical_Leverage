"""
neural_var.py — neural vector autoregression used for all real-system fits.
Interface: forward(x) -> (preds, None), x:(B,N,K).
Per-target MLP over the flattened (N*K) lag window, hidden=32, 1 hidden
layer, tanh. Frozen after fit.
"""
import numpy as np
import torch
import torch.nn as nn


class NAVAR(nn.Module):
    def __init__(self, n_components, maxlags, hidden=32, layers=1):
        super().__init__()
        self.N, self.K = n_components, maxlags
        def mlp():
            mods = [nn.Linear(self.N * self.K, hidden), nn.Tanh()]
            for _ in range(layers - 1):
                mods += [nn.Linear(hidden, hidden), nn.Tanh()]
            mods += [nn.Linear(hidden, 1)]
            return nn.Sequential(*mods)
        self.heads = nn.ModuleList([mlp() for _ in range(self.N)])

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B = x.shape[0]
        flat = x.reshape(B, self.N * self.K)
        preds = torch.cat([h(flat) for h in self.heads], dim=1)
        return preds, None


def fit_navar(Xz, runs, *, n_components, maxlags, hidden=32, layers=1,
              epochs=400, lr=1e-3, weight_decay=1e-4, lambda1=0.0,
              device="cpu", seed=0, verbose=False):
    """Fit on one site; build one-step (window->next) pairs WITHIN runs only.
    Returns (frozen model, final train MSE, n_pairs)."""
    torch.manual_seed(seed); np.random.seed(seed)
    N, K = n_components, maxlags
    Xin, Yout = [], []
    for (a, b) in runs:
        seg = Xz[a:b]; L = len(seg)
        for t in range(K - 1, L - 1):
            Xin.append(seg[t - K + 1:t + 1].T)   # (N,K) cols OLD->NEW
            Yout.append(seg[t + 1])              # (N,)
    Xin = torch.tensor(np.asarray(Xin), dtype=torch.float32, device=device)
    Yout = torch.tensor(np.asarray(Yout), dtype=torch.float32, device=device)
    assert len(Xin) > 0, "no training pairs — runs too short?"

    model = NAVAR(N, K, hidden=hidden, layers=layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossfn = nn.MSELoss()
    first = None
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        preds, _ = model(Xin)
        loss = lossfn(preds, Yout)
        if lambda1 > 0:
            loss = loss + lambda1 * sum(p.abs().sum() for p in model.parameters())
        loss.backward(); opt.step()
        if ep == 0: first = loss.item()
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    epoch {ep+1}: train MSE {loss.item():.5f}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    final = loss.item()
    assert final <= first + 1e-6, f"train loss did not decrease ({first:.4f}->{final:.4f})"
    return model, final, len(Xin)
