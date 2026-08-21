"""
dynamical_leverage.py
========================

Reference implementation of the dynamical leverage measure E_j, the methods
core of Paper 1. Computes, faithfully to the locked math spec:

    E_j(t, x; T) = sum_{tau=0}^{T-1} (Phi_t^tau(x) e_j)^T M (Phi_t^tau(x) e_j)

with Phi^0 = I (the tau=0 self-term retained), along a trajectory of Jacobians.

This module is ESTIMATOR-AGNOSTIC: it operates on Jacobian MATRICES, not on any
model. You supply the Jacobians (e.g. from asof_jacobian.make_torch_jacobian_fn,
DCNAR, a linear VAR, a neural ODE — anything differentiable). E_j is then linear
algebra on those matrices. This separation is the math: the measure depends only
on the A_t(x).

Commitments implemented (from the methods write-up):
  - tau=0 self-term retained (Phi^0 = I).
  - Companion form: perturb node j in the CURRENT-state block only (e_j^(0)),
    NOT every lagged copy. See node_basis_vector().
  - Two trajectory modes: 'simulated' (iterate the map) and 'observed' (use the
    realized states). The Jacobian source decides which; this module supports
    both via how you supply the Jacobian sequence.
  - Declared gauge: standardized components, M = I default; M overridable.
  - Horizon T is an explicit REQUIRED parameter (finite-horizon by design).
  - LTI self-check: in the LTI limit E_j == tr W_j(T) (average controllability);
    verify_lti_reduction() checks this against the canonical Gramian. This is the
    credibility anchor made executable AND the guard against re-introducing the
    tau-indexing bug (verify against the Gramian, never against a copy of E_j).
"""

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


# ===========================================================================
# Node perturbation in (possibly companion) coordinates
# ===========================================================================

def node_basis_vector(j: int, n_components: int, state_dim: int) -> np.ndarray:
    """
    The perturbation vector for "perturb node j" (methods spec #3).

    If state_dim == n_components: ordinary basis vector e_j in R^n.
    If state_dim == n_components * K (companion form): e_j^(0), i.e. 1 in slot j
      of the CURRENT-state (leading) block and 0 everywhere else — including all
      lagged copies of j. Matches asof_jacobian's companion layout, where the
      leading block is the current state.
    """
    if state_dim % n_components != 0:
        raise ValueError(
            f"state_dim {state_dim} is not a multiple of n_components "
            f"{n_components}; cannot place node perturbation.")
    if not (0 <= j < n_components):
        raise ValueError(f"node index j={j} out of range [0,{n_components})")
    v = np.zeros(state_dim)
    v[j] = 1.0          # leading block occupies indices [0, n_components)
    return v


def _validate_metric_pd(M: np.ndarray, *, name: str = "M",
                        sym_tol: float = 1e-8) -> None:
    """
    Validate that a supplied metric M is symmetric and positive-definite
    (methods spec #3). The construct is the quadratic form v^T M v; the gauge
    discussion requires M positive-definite, so a non-PD M is a user error that
    would otherwise silently produce meaningless non-positive "energy".

    Checks: square, symmetric (within sym_tol), and positive-definite via an
    attempted Cholesky factorization (the cheapest reliable PD test). Raises
    ValueError with a specific diagnosis on failure.
    """
    M = np.asarray(M, float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {M.shape}")
    asym = float(np.max(np.abs(M - M.T)))
    if asym > sym_tol:
        raise ValueError(
            f"{name} must be symmetric: max|M - M^T| = {asym:.2e} > {sym_tol:.0e}. "
            f"The metric defines a quadratic form and must be symmetric.")
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        eigmin = float(np.min(np.linalg.eigvalsh((M + M.T) / 2)))
        raise ValueError(
            f"{name} must be positive-definite (the construct is v^T M v): "
            f"Cholesky failed; smallest eigenvalue = {eigmin:.2e} (must be > 0). "
            f"Supply a positive-definite metric, or use the default M = I.")


# ===========================================================================
# Core: dynamical leverage from a sequence of Jacobians
# ===========================================================================

@dataclass
class DynamicalLeverage:
    E: np.ndarray                 # (n_components,) dynamical leverage per node
    weights: np.ndarray           # (n_components,) normalized E / sum(E)
    horizon: int                  # T (declared)
    n_components: int
    state_dim: int                # n (component form) or nK (companion form)
    metric: str                   # description of M
    trajectory_mode: str          # 'observed' | 'simulated' | 'lti'
    note: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["E"] = self.E.tolist()
        d["weights"] = self.weights.tolist()
        return d


def dynamical_leverage_from_jacobians(
    jacobians: Sequence[np.ndarray],
    *,
    n_components: int,
    horizon: int,
    M: Optional[np.ndarray] = None,
    trajectory_mode: str = "observed",
    strict: bool = True,
) -> DynamicalLeverage:
    """
    Compute E_j for all nodes from a sequence of Jacobian matrices along a
    trajectory.

    jacobians : list of T-1 square matrices [A_t, A_{t+1}, ..., A_{t+T-2}], each
                (d x d) with d = state_dim. These are the Jacobians at successive
                points along the trajectory. The state-transition matrices are
                their running products:
                    Phi^0 = I
                    Phi^1 = A_t
                    Phi^2 = A_{t+1} A_t
                    ...
                    Phi^{T-1} = A_{t+T-2} ... A_t
                So to reach horizon T you need T-1 Jacobians. If you pass MORE,
                only the first T-1 are used. If you pass FEWER: with strict=True
                (default) this raises; with strict=False the horizon is reduced
                to len(jacobians)+1 and noted (exploratory use only).
    n_components : number of nodes n. state_dim may be n (component form) or n*K
                (companion form); the node perturbation is placed accordingly.
    horizon T : REQUIRED. E_j is finite-horizon by design (methods spec #4).
    M : (d x d) SYMMETRIC POSITIVE-DEFINITE metric. Default I_d (the declared
                standardized gauge). Validated for symmetry and positive-
                definiteness (the construct is a quadratic form v^T M v; a non-PD
                M would yield meaningless non-positive "energy"). If state_dim =
                nK (companion) and you pass an (n x n) M, it is embedded into the
                leading block (impact measured on current components only); the
                embedded matrix is PSD on the full state by construction, and the
                quadratic form is evaluated only on current-block perturbations,
                so E_j stays well-defined.
    trajectory_mode : label only ('observed'/'simulated'); the caller decides
                which trajectory the Jacobians came from (methods spec #2).
    strict : if True (default, paper-grade), too few Jacobians for the requested
                horizon raises ValueError rather than silently reducing T. Set
                False only for exploration, where capping-with-a-note is wanted.

    Returns DynamicalLeverage with E (per node), normalized weights, and
    metadata.
    """
    if horizon < 1:
        raise ValueError("horizon T must be >= 1")
    if len(jacobians) == 0 and horizon > 1:
        raise ValueError("need at least one Jacobian for horizon > 1")

    d = jacobians[0].shape[0] if jacobians else None
    if d is not None:
        for k, A in enumerate(jacobians):
            if A.shape != (d, d):
                raise ValueError(
                    f"Jacobian {k} has shape {A.shape}, expected ({d},{d})")

    # Horizon vs available Jacobians (need T-1 for horizon T).
    note = ""
    T = horizon
    if d is not None and len(jacobians) < T - 1:
        if strict:
            raise ValueError(
                f"strict=True: requested horizon {horizon} needs {horizon-1} "
                f"Jacobians but only {len(jacobians)} provided. The trajectory "
                f"is too short for this horizon. Either lower the horizon, "
                f"provide more Jacobians, or set strict=False for exploratory "
                f"capping (NOT recommended for reported results — it would make "
                f"E_j incomparable across units with different lengths).")
        T = len(jacobians) + 1
        note = (f"horizon reduced from {horizon} to {T}: only "
                f"{len(jacobians)} Jacobians available (need T-1). "
                f"strict=False — exploratory only.")

    # state_dim: infer from Jacobians if present, else assume component form
    state_dim = d if d is not None else n_components

    # metric M
    if M is None:
        Mmat = np.eye(state_dim)
        metric_desc = "I (standardized gauge, default)"
    else:
        M = np.asarray(M, float)
        # Validate the USER-SUPPLIED matrix for symmetry + positive-definiteness
        # at its own dimension (BEFORE any embedding: the embedded full-state
        # matrix is only PSD/singular, so PD must be checked on the original).
        if M.shape == (state_dim, state_dim):
            _validate_metric_pd(M, name="M")
            Mmat = M
            metric_desc = "user M (full state_dim, validated PD)"
        elif M.shape == (n_components, n_components) and state_dim != n_components:
            _validate_metric_pd(M, name="M (component-space)")
            # embed component-space M into the leading (current) block
            Mmat = np.zeros((state_dim, state_dim))
            Mmat[:n_components, :n_components] = M
            metric_desc = "user M embedded in current-component block (validated PD)"
        else:
            raise ValueError(
                f"M shape {M.shape} incompatible with state_dim {state_dim} "
                f"or n_components {n_components}")

    # Compute E_j for each node.
    # Phi^tau e_j is built incrementally: v_0 = e_j, v_{tau} = A_{t+tau-1} v_{tau-1}.
    E = np.zeros(n_components)
    for j in range(n_components):
        v = node_basis_vector(j, n_components, state_dim)   # Phi^0 e_j = e_j
        # tau = 0 term (self-term, RETAINED): v^T M v
        acc = float(v @ Mmat @ v)
        # tau = 1 .. T-1
        for tau in range(1, T):
            v = jacobians[tau - 1] @ v          # apply A_{t+tau-1}
            acc += float(v @ Mmat @ v)
        E[j] = acc

    total = E.sum()
    weights = E / total if total > 0 else np.full(n_components, np.nan)

    return DynamicalLeverage(
        E=E, weights=weights, horizon=T, n_components=n_components,
        state_dim=state_dim, metric=metric_desc, trajectory_mode=trajectory_mode,
        note=note,
    )


# ===========================================================================
# Convenience: build the Jacobian sequence from a jac_fn along a trajectory
# ===========================================================================

def jacobian_sequence_observed(
    states: np.ndarray, jac_fn: Callable[[np.ndarray], np.ndarray],
    *, maxlags: int, start_idx: int, horizon: int,
) -> List[np.ndarray]:
    """
    Build the Jacobian sequence along the OBSERVED trajectory (methods spec #2,
    'observed' mode). states is the (T_total, n) standardized series for ONE unit,
    indexed by row = time step (row 0 is the earliest observation).

    TIME INDEXING (read carefully — an off-by-one shifts the whole trajectory):
      - `start_idx` is the index of the FIRST state at which a Jacobian is
        evaluated. The Jacobian "at step s" is computed from the lag window
        states[s-maxlags : s] (the `maxlags` rows ENDING at row s-1, i.e. strictly
        before s), which is the window the model uses to predict row s. So the
        Jacobian at step s describes the one-step map INTO observed row s.
      - The sequence runs s = start_idx, start_idx+1, ..., start_idx+horizon-2,
        producing horizon-1 Jacobians (the count needed for horizon T; Phi^0 = I
        contributes the self-term with no Jacobian).
      - Therefore the horizon spans observed rows [start_idx, start_idx+horizon-1].
        For calendar anchoring: if row r corresponds to year y0+r, the E_j you get
        is the leverage accumulated over years [y0+start_idx, y0+start_idx+horizon-1]
        evaluated along the path the unit actually took. The first usable
        start_idx is `maxlags` (you need maxlags prior rows for the first window).

    Returns up to horizon-1 Jacobians, stopping early if the series ends. (When
    fewer than horizon-1 are returned, the core function with strict=True will
    raise — the right behavior for reported results, since a short tail would
    otherwise silently shorten the horizon for that unit.)

    jac_fn convention matches asof_jacobian.make_torch_jacobian_fn: it takes an
    (n, maxlags) window and returns the (companion or component) Jacobian.
    """
    out = []
    for s in range(start_idx, start_idx + horizon - 1):
        if s + 1 > len(states) or s - maxlags < 0:
            break
        window = states[s - maxlags:s, :].T          # (n, maxlags)
        out.append(jac_fn(window))
    return out


def dynamical_leverage_observed(
    states: np.ndarray, jac_fn: Callable[[np.ndarray], np.ndarray],
    *, n_components: int, maxlags: int, start_idx: int, horizon: int,
    M: Optional[np.ndarray] = None, strict: bool = True,
) -> DynamicalLeverage:
    """
    End-to-end E_j along the OBSERVED trajectory of one unit (methods spec #2).
    Starts at `start_idx` and accumulates leverage over `horizon` steps; see
    jacobian_sequence_observed for the precise meaning of start_idx and which
    observed rows the horizon spans. Thin wrapper: build the observed Jacobian
    sequence, then call the core.

    strict (default True): if the unit's series is too short to supply horizon-1
    Jacobians from start_idx, raise rather than silently shorten the horizon for
    this unit (which would make E_j incomparable across units).
    """
    jseq = jacobian_sequence_observed(
        states, jac_fn, maxlags=maxlags, start_idx=start_idx, horizon=horizon)
    return dynamical_leverage_from_jacobians(
        jseq, n_components=n_components, horizon=horizon, M=M,
        trajectory_mode="observed", strict=strict)


# ===========================================================================
# LTI self-check: E_j == tr W_j(T)  (the credibility anchor, executable)
# ===========================================================================

def average_controllability_gramian_trace(
    A: np.ndarray, j: int, horizon: int
) -> float:
    """
    Canonical node-j average controllability over horizon T:
        tr W_j(T),  W_j(T) = sum_{tau=0}^{T-1} A^tau e_j e_j^T (A^T)^tau
    Computed directly from the Gramian definition (NOT from E_j), so it is an
    independent reference. tr W_j(T) = sum_tau ||A^tau e_j||^2.
    """
    n = A.shape[0]
    e = np.zeros(n); e[j] = 1.0
    v = e.copy()                # A^0 e_j
    acc = float(v @ v)          # tau = 0
    for _ in range(1, horizon):
        v = A @ v
        acc += float(v @ v)
    return acc


def verify_lti_reduction(
    A: np.ndarray, horizon: int, *, tol: float = 1e-9
) -> Tuple[bool, float, np.ndarray, np.ndarray]:
    """
    Verify E_j == tr W_j(T) for a constant matrix A (the LTI limit), M = I.

    Builds the Jacobian sequence as the CONSTANT A repeated (LTI: A_t = A for all
    t), computes E_j via the core, and compares to the canonical Gramian trace
    computed independently. Returns (passed, max_abs_diff, E_via_core, AC_gramian).

    This is the executable credibility anchor and the guard against the tau-
    indexing bug: it checks E_j against the canonical Gramian, never against a
    second copy of the E_j code.
    """
    n = A.shape[0]
    jseq = [A] * (horizon - 1)
    ec = dynamical_leverage_from_jacobians(
        jseq, n_components=n, horizon=horizon, M=None, trajectory_mode="lti")
    ac = np.array([average_controllability_gramian_trace(A, j, horizon)
                   for j in range(n)])
    diff = float(np.max(np.abs(ec.E - ac)))
    return (diff < tol), diff, ec.E, ac
