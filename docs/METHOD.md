# Method

- **Problem.** n targets with priorities w_i; conflicting pairs E (time-window overlap plus random slew conflicts). Cost C(x) = -sum w_i x_i + P sum_(i,j in E) x_i x_j, with P = 1.5 max w. Exact optimum by enumeration.
- **Quantum method.** QAOA of depth p (1 or 2), COBYLA (2 restarts, 60 iterations) on a 256-shot estimate of either the mean or CVaR_alpha of the cost. Noise: output = F·ideal + (1-F)·uniform with F = (1-e)^(2|E|p), i.e. 2 CNOTs per ZZ term and all-to-all connectivity, so denser graphs lose more fidelity.
- **Standard quantum baseline.** Expectation-value QAOA (alpha = 1) with the same budget.
- **Classical baselines.** Uniform random sampling (30 shots, raw and repaired), single-flip simulated annealing (30 cost evaluations), priority/degree greedy (no sampling budget; a strong reference).
- **Metrics.** Best-of-30-shots weight / optimum, raw (infeasible = 0) and repaired (lowest-cost shot plus greedy repair); feasible mass; optimum mass.
- **Ablation.** CVaR alpha in {1, 0.5, 0.25, 0.1, 0.05} and depth p in {1, 2} at e = 0.003.
- **Noise sweep.** e in {0, 0.001, 0.003, 0.006, 0.01}, n = 10. **Scaling:** n = 8, 10, 12 at e = 0.003.
- **Seeds.** 5 instance seeds (0-4), each also seeding sampling and optimiser; 20 evaluation draws per run.
