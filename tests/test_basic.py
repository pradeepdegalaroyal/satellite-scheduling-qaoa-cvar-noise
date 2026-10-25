import numpy as np
from satqaoa import make_instance, repair_greedy, QAOAConfig, run_qaoa
from satqaoa.qaoa import probabilities


def test_optimum_is_feasible():
    inst = make_instance(8, 0)
    assert inst.feasible.any() and inst.optimum > 0


def test_probabilities_normalised_and_noise_uniform():
    inst = make_instance(6, 1)
    th = np.array([0.3, 0.2, 0.1, 0.4])
    assert abs(probabilities(inst, th, 2, 0.0).sum() - 1) < 1e-9
    pr = probabilities(inst, th, 2, 0.5)
    assert np.allclose(pr, 1 / 64, atol=0.05) and abs(pr.sum() - 1) < 1e-9


def test_repair_feasible():
    inst = make_instance(8, 2)
    x = repair_greedy(inst, np.ones(8, dtype=np.int8))
    assert all(not (x[i] and x[j]) for i, j in inst.edges)


def test_run_qaoa_smoke():
    inst = make_instance(6, 0)
    r = run_qaoa(inst, QAOAConfig(p=1, maxiter=5, n_eval_draws=2), 0)
    assert 0 <= r["raw_ratio"] <= 1
