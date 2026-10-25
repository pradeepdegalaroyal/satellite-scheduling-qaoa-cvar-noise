"""Observation-scheduling instances as maximum-weight independent set (penalty QUBO)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Instance:
    """Targets with priorities and pairwise conflicts (window overlap / slew)."""
    n: int
    weights: np.ndarray            # (n,) priorities
    edges: list[tuple[int, int]]   # conflicting target pairs
    penalty: float
    bits: np.ndarray               # (2^n, n) bit table, qubit 0 = most significant
    cost: np.ndarray               # (2^n,) QUBO cost (minimise)
    feasible: np.ndarray           # (2^n,) bool
    value: np.ndarray              # (2^n,) scheduled weight if feasible else 0
    optimum: float                 # best feasible weight (exact)


def make_instance(n: int, seed: int, extra_conflict: float = 0.15) -> Instance:
    """Random instance: interval-overlap conflicts plus random slew conflicts."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(0, 1, n)
    dur = rng.uniform(0.12, 0.3, n)
    w = rng.uniform(1, 10, n).round(2)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            overlap = start[i] < start[j] + dur[j] and start[j] < start[i] + dur[i]
            if overlap or rng.random() < extra_conflict * 0.3:
                edges.append((i, j))
    N = 1 << n
    idx = np.arange(N)
    bits = ((idx[:, None] >> (n - 1 - np.arange(n))) & 1).astype(np.int8)
    P = 1.5 * float(w.max())
    viol = np.zeros(N)
    for i, j in edges:
        viol += bits[:, i] * bits[:, j]
    val = bits @ w
    cost = -val + P * viol
    feas = viol == 0
    value = np.where(feas, val, 0.0)
    return Instance(n, w, edges, P, bits, cost, feas, value, float(value.max()))


def repair_greedy(inst: Instance, x: np.ndarray) -> np.ndarray:
    """Drop the lowest-weight endpoint of each violated conflict, then add free targets greedily."""
    x = x.copy()
    for i, j in sorted(inst.edges, key=lambda e: -(inst.weights[e[0]] + inst.weights[e[1]])):
        if x[i] and x[j]:
            x[i if inst.weights[i] < inst.weights[j] else j] = 0
    nbr = [[] for _ in range(inst.n)]
    for i, j in inst.edges:
        nbr[i].append(j)
        nbr[j].append(i)
    for i in np.argsort(-inst.weights):
        if not x[i] and not any(x[k] for k in nbr[i]):
            x[i] = 1
    return x
