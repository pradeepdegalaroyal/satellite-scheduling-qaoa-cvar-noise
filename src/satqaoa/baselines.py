"""Classical baselines at matched sample budget."""
from __future__ import annotations
import numpy as np
from .problem import Instance, repair_greedy


def greedy_baseline(inst: Instance) -> float:
    """Priority-per-conflict greedy independent set; returns ratio to optimum."""
    deg = np.zeros(inst.n)
    nbr = [[] for _ in range(inst.n)]
    for i, j in inst.edges:
        deg[i] += 1
        deg[j] += 1
        nbr[i].append(j)
        nbr[j].append(i)
    x = np.zeros(inst.n, dtype=np.int8)
    for i in np.argsort(-inst.weights / (1 + deg)):
        if not any(x[k] for k in nbr[i]):
            x[i] = 1
    return float(inst.weights @ x / inst.optimum)


def random_baseline(inst: Instance, shots: int, seed: int, draws: int = 20, repair: bool = False) -> float:
    """Best of `shots` uniform bitstrings (raw feasible, or repaired lowest-cost sample)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        s = rng.integers(0, 1 << inst.n, shots)
        if repair:
            x = repair_greedy(inst, inst.bits[s[np.argmin(inst.cost[s])]])
            out.append(inst.weights @ x / inst.optimum)
        else:
            out.append(inst.value[s].max() / inst.optimum)
    return float(np.mean(out))


def anneal_baseline(inst: Instance, evals: int, seed: int) -> float:
    """Single-flip simulated annealing on the QUBO with `evals` cost evaluations; best feasible ratio."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, inst.n).astype(np.int8)

    def idx(v: np.ndarray) -> int:
        return int("".join(map(str, v)), 2)

    c = inst.cost[idx(x)]
    best = inst.value[idx(x)]
    for t in range(evals):
        T = 3.0 * (1 - t / evals) + 0.05
        k = rng.integers(inst.n)
        y = x.copy()
        y[k] ^= 1
        cy = inst.cost[idx(y)]
        if cy < c or rng.random() < np.exp((c - cy) / T):
            x, c = y, cy
            best = max(best, inst.value[idx(x)])
    return float(best / inst.optimum)
