"""Reproduce all experiments: python experiments/run_all.py [--quick]"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from satqaoa import (make_instance, QAOAConfig, run_qaoa, greedy_baseline,
                     random_baseline, anneal_baseline)

ROOT = Path(__file__).resolve().parents[1]
COL = {"random-30": "gray", "greedy": "g", "anneal-30": "r"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    q = ap.parse_args().quick
    seeds = [0, 1] if q else [0, 1, 2, 3, 4]
    N0 = 8 if q else 10
    sizes = [6, 8] if q else [8, 10, 12]
    errs = [0.0, 0.01] if q else [0.0, 0.001, 0.003, 0.006, 0.01]
    alphas = [1.0, 0.1] if q else [1.0, 0.5, 0.25, 0.1, 0.05]
    mi = 15 if q else 60
    out = ROOT / ("results_quick" if q else "results")
    out.mkdir(exist_ok=True)
    figd = out if q else ROOT / "figures"
    figd.mkdir(exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()

    def add(exp, n, e, seed, method, d, **kw):
        rows.append({"exp": exp, "n": n, "cnot_error": e, "seed": seed, "method": method, **d, **kw})

    def qaoa(exp, n, e, seed, p, alpha):
        inst = make_instance(n, seed)
        cfg = QAOAConfig(p=p, alpha=alpha, cnot_error=e, maxiter=mi, n_eval_draws=5 if q else 20)
        add(exp, n, e, seed, f"QAOA p={p} alpha={alpha}", run_qaoa(inst, cfg, seed), p=p, alpha=alpha)

    def classical(exp, n, e, seed):
        inst = make_instance(n, seed)
        g = greedy_baseline(inst)
        add(exp, n, e, seed, "random-30", {"feasible_mass": float(inst.feasible.mean()),
            "opt_mass": float((inst.value >= inst.optimum - 1e-9).mean()), "raw_ratio": random_baseline(inst, 30, seed),
            "repaired_ratio": random_baseline(inst, 30, seed, repair=True)})
        add(exp, n, e, seed, "greedy", {"raw_ratio": g, "repaired_ratio": g})
        sa = anneal_baseline(inst, 30, seed)
        add(exp, n, e, seed, "anneal-30", {"raw_ratio": sa, "repaired_ratio": sa})

    for s in seeds:  # noise sweep
        for e in errs:
            for al in (1.0, 0.1):
                qaoa("noise", N0, e, s, 2, al)
        classical("noise", N0, 0.0, s)
    for s in seeds:  # ablation: CVaR alpha, depth p (e=0.003)
        for al in alphas:
            qaoa("ablation", N0, 0.003, s, 2, al)
        for al in (1.0, 0.1):
            qaoa("ablation", N0, 0.003, s, 1, al)
    for s in seeds:  # scaling
        for n in sizes:
            for al in (1.0, 0.1):
                qaoa("scaling", n, 0.003, s, 2, al)
            classical("scaling", n, 0.003, s)
    df = pd.DataFrame(rows)
    df.to_csv(out / "all_runs.csv", index=False)
    m = ["raw_ratio", "repaired_ratio", "feasible_mass", "opt_mass"]
    summ = df.groupby(["exp", "n", "cnot_error", "method"])[m].agg(["mean", "std"]).reset_index()
    summ.columns = ["_".join(c).strip("_") for c in summ.columns]
    summ.to_csv(out / "summary.csv", index=False)
    with open(out / "meta.json", "w") as f:
        json.dump({"runtime_s": time.time() - t0, "seeds": seeds, "quick": q}, f)

    d = df[df.exp == "noise"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, key in enumerate(["raw_ratio", "repaired_ratio"]):
        for meth in sorted(d.method.unique()):
            g = d[d.method == meth]
            if meth.startswith("QAOA"):
                s = g.groupby("cnot_error")[key].agg(["mean", "std"])
                ax[k].errorbar(s.index, s["mean"], s["std"], marker="o", capsize=3, label=meth)
            else:
                ax[k].axhline(g[key].mean(), ls="--", color=COL[meth], label=meth)
        ax[k].set_xlabel("per-CNOT depolarising error")
        ax[k].set_ylabel(key.replace("_", " ") + " (to optimum)")
        ax[k].set_title(f"{key} vs gate noise, n={N0}")
        ax[k].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figd / "noise_sweep.png", dpi=200)
    plt.close(fig)

    d = df[(df.exp == "ablation") & df.method.str.startswith("QAOA")]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for p in sorted(d.p.unique()):
        s = d[d.p == p].groupby("alpha")["repaired_ratio"].agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], s["std"], marker="o", capsize=3, label=f"p={int(p)}")
    ax.set_xlabel("CVaR fraction alpha (1 = mean)")
    ax.set_ylabel("repaired ratio")
    ax.legend()
    ax.set_title("Ablation: CVaR alpha and depth p")
    fig.tight_layout()
    fig.savefig(figd / "ablation.png", dpi=200)
    plt.close(fig)

    d = df[df.exp == "scaling"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for meth in sorted(d.method.unique()):
        s = d[d.method == meth].groupby("n")["repaired_ratio"].agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], s["std"], marker="o", capsize=3, label=meth)
    ax.set_xlabel("targets / qubits n")
    ax.set_ylabel("repaired ratio")
    ax.legend(fontsize=7)
    ax.set_title("Scaling at 0.003 CNOT error")
    fig.tight_layout()
    fig.savefig(figd / "scaling.png", dpi=200)
    plt.close(fig)
    print(summ.to_string())
    print("done in", time.time() - t0)


if __name__ == "__main__":
    main()
