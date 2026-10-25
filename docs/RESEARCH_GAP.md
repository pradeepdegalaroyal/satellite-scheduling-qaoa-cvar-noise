# Research gap

**State of the art.** Blekos et al. (2024, doi:10.1016/j.physrep.2024.03.002) review Quantum Approximate Optimization Algorithm (QAOA) and its variants and stress that noise resilience and the question of when QAOA should be used remain open. Abbas et al. (2024, doi:10.1038/s42254-024-00770-9) argue that systematic benchmarking is essential for quantum optimisation. Brandhofer et al. (2023, doi:10.1007/s11128-022-03766-5) benchmark QAOA for constrained portfolio optimisation and note that several practical issues (shot noise, gate and readout errors, choice of optimiser) are not discussed together in the literature.

**Gap.** Space-technology optimisation (e.g. Earth-observation scheduling) is often motivated for QAOA, but few studies measure how *finite-shot training* and *gate noise that grows with conflict-graph density* jointly affect schedule quality at a small, matched sample budget against strong classical baselines.

**Hypothesis (falsifiable).** For satellite observation scheduling (max-weight independent set on a conflict graph, penalty quadratic unconstrained binary optimization (QUBO)), training QAOA (p=2) with a conditional value at risk (CVaR) objective (alpha=0.1) instead of the plain expectation yields a higher best-of-30-shots schedule weight (ratio to the exact optimum) under per-CNOT depolarising error 0.003-0.01, n=10, by at least one standard deviation over 5 instances.

**Success metric.** Best-of-30-shots scheduled weight / exact optimum (raw feasible, and after greedy repair), mean ± std over 5 instances; secondary: feasible and optimal probability mass. Scale: exact statevector simulation, n=8-12 qubits; noise is a global-depolarising model, not a hardware run.
