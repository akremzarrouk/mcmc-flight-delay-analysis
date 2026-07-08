"""Random walk Metropolis sampler for the Laplace(0, 1) distribution,
plus the Gelman-Rubin convergence diagnostic.

Extracted from mcmc_metropolis_hastings.ipynb so the core logic is
importable and unit-testable outside the notebook.
"""
import numpy as np


def log_target(x):
    """Log-density of the Laplace(0,1) target: log f(x) = -log2 - |x|"""
    return -np.log(2) - np.abs(x)


def target(x):
    """True density f(x) = 0.5 * exp(-|x|)"""
    return 0.5 * np.exp(-np.abs(x))


def random_walk_metropolis(N, s, x0=0.0, seed=None):
    """
    Random Walk Metropolis sampler for f(x) = 0.5*exp(-|x|).
    Returns array of shape (N+1,) including x0.
    """
    rng = np.random.default_rng(seed)
    samples = np.empty(N + 1)
    samples[0] = x0
    for i in range(1, N + 1):
        x_star     = rng.normal(loc=samples[i - 1], scale=s)
        log_ratio  = log_target(x_star) - log_target(samples[i - 1])
        if np.log(rng.uniform()) < log_ratio:
            samples[i] = x_star
        else:
            samples[i] = samples[i - 1]
    return samples


def compute_rhat(chains):
    """Gelman-Rubin R-hat; within-chain variance uses the 1/N (population) form."""
    chain_means = np.array([np.mean(c) for c in chains])
    chain_vars  = np.array([np.mean((c - m) ** 2)
                            for c, m in zip(chains, chain_means)])
    W = np.mean(chain_vars)
    M = np.mean(chain_means)
    B = np.mean((chain_means - M) ** 2)
    return np.sqrt((B + W) / W)
