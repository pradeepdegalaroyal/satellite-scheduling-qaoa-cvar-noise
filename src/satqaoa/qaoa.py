"""Statevector QAOA with a global-depolarising gate-noise model and finite-shot training."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from .problem import Instance, repair_greedy


@dataclass
class QAOAConfig:
    p: int = 2
    alpha: float = 1.0          # CVaR fraction (1.0 = plain expectation)
    cnot_error: float = 0.0     # per-CNOT depolarising error
    train_shots: int = 256
    eval_shots: int = 30
    maxiter: int = 60
    n_eval_draws: int = 20


def _mixer(psi: np.ndarray, n: int, beta: float) -> np.ndarray:
    c, s = np.cos(beta), -1j * np.sin(beta)
    t = psi.reshape((2,) * n)
    for q in range(n):
        a = np.take(t, 0, axis=q)
        b = np.take(t, 1, axis=q)
        t = np.stack([c * a + s * b, s * a + c * b], axis=q)
    return t.reshape(-1)


def probabilities(inst: Instance, theta: np.ndarray, p: int, cnot_error: float) -> np.ndarray:
    """Output distribution; noise = mixing with uniform at fidelity (1-e)^(#CNOT)."""
    n = inst.n
    psi = np.full(1 << n, 1 / np.sqrt(1 << n), dtype=complex)
    g, b = theta[:p], theta[p:]
    c = inst.cost / max(inst.penalty, 1.0)
    for k in range(p):
        psi = psi * np.exp(-1j * g[k] * c)
        psi = _mixer(psi, n, b[k])
    probs = np.abs(psi) ** 2
    n_cnot = 2 * len(inst.edges) * p      # ZZ term = 2 CNOT, all-to-all connectivity
    F = (1 - cnot_error) ** n_cnot
    return F * probs + (1 - F) / (1 << n)


def _cvar(costs: np.ndarray, alpha: float) -> float:
    k = max(1, int(np.ceil(alpha * len(costs))))
    return float(np.sort(costs)[:k].mean())


def run_qaoa(inst: Instance, cfg: QAOAConfig, seed: int) -> dict:
    """Train with shot-noisy (CVaR or mean) objective, then evaluate best-of-S metrics."""
    rng = np.random.default_rng(seed)

    def objective(theta: np.ndarray) -> float:
        pr = probabilities(inst, theta, cfg.p, cfg.cnot_error)
        s = rng.choice(len(pr), size=cfg.train_shots, p=pr / pr.sum())
        return _cvar(inst.cost[s], cfg.alpha)

    best = None
    for _ in range(2):
        th0 = rng.uniform(-0.6, 0.6, 2 * cfg.p)
        r = minimize(objective, th0, method="COBYLA", options={"maxiter": cfg.maxiter, "rhobeg": 0.3})
        if best is None or r.fun < best.fun:
            best = r
    pr = probabilities(inst, best.x, cfg.p, cfg.cnot_error)
    pr = pr / pr.sum()
    raw, rep = [], []
    for _ in range(cfg.n_eval_draws):
        s = rng.choice(len(pr), size=cfg.eval_shots, p=pr)
        raw.append(inst.value[s].max() / inst.optimum)
        i = s[np.argmin(inst.cost[s])]
        rep.append(inst.weights @ repair_greedy(inst, inst.bits[i]) / inst.optimum)
    return {"raw_ratio": float(np.mean(raw)), "repaired_ratio": float(np.mean(rep)),
            "feasible_mass": float(pr[inst.feasible].sum()),
            "opt_mass": float(pr[inst.value >= inst.optimum - 1e-9].sum())}
