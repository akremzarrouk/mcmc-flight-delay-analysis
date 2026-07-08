import numpy as np
import pytest

from src.mcmc import compute_rhat, log_target, random_walk_metropolis, target


def test_target_matches_log_target():
    x = np.array([-3.0, -1.0, 0.0, 1.0, 4.5])
    np.testing.assert_allclose(np.log(target(x)), log_target(x))


def test_target_integrates_to_one():
    x = np.linspace(-50, 50, 200_001)
    area = np.trapezoid(target(x), x)
    assert area == pytest.approx(1.0, abs=1e-6)


def test_random_walk_metropolis_shape_and_start():
    chain = random_walk_metropolis(N=500, s=1.0, x0=2.5, seed=1)
    assert chain.shape == (501,)
    assert chain[0] == 2.5


def test_random_walk_metropolis_recovers_laplace_moments():
    chain = random_walk_metropolis(N=50_000, s=1.0, x0=0.0, seed=42)
    samples = chain[1:]
    assert samples.mean() == pytest.approx(0.0, abs=0.1)
    assert samples.std(ddof=1) == pytest.approx(np.sqrt(2), abs=0.1)


def test_random_walk_metropolis_is_deterministic_given_seed():
    chain_a = random_walk_metropolis(N=200, s=0.5, x0=0.0, seed=7)
    chain_b = random_walk_metropolis(N=200, s=0.5, x0=0.0, seed=7)
    np.testing.assert_array_equal(chain_a, chain_b)


def test_compute_rhat_near_one_for_well_mixed_chains():
    chains = [
        random_walk_metropolis(N=5_000, s=1.0, x0=x0, seed=100 + j)[1:]
        for j, x0 in enumerate([-1.0, 0.0, 1.0, 2.0])
    ]
    rhat = compute_rhat(chains)
    assert rhat < 1.05


def test_compute_rhat_large_for_stuck_chains():
    chains = [
        random_walk_metropolis(N=2_000, s=0.001, x0=x0, seed=200 + j)[1:]
        for j, x0 in enumerate([-5.0, -1.0, 1.0, 5.0])
    ]
    rhat = compute_rhat(chains)
    assert rhat > 1.05


def test_compute_rhat_of_identical_chains_is_exactly_one():
    chain = random_walk_metropolis(N=1_000, s=1.0, x0=0.0, seed=1)[1:]
    rhat = compute_rhat([chain, chain, chain])
    assert rhat == pytest.approx(1.0)
