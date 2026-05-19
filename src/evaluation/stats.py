"""Statistical utilities for drug-feature null experiments."""

from __future__ import annotations

import numpy as np


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni correction over a named set of p-values.

    Returns adjusted p-values (step-down method; family-wise error rate control).
    """
    items = sorted(p_values.items(), key=lambda x: x[1])
    n = len(items)
    adjusted: dict[str, float] = {}
    cummax = 0.0
    for rank, (name, p) in enumerate(items):
        adj = p * (n - rank)
        cummax = max(cummax, adj)
        adjusted[name] = min(cummax, 1.0)
    return adjusted


def bootstrap_delta_ci(
    delta_per_drug: dict[str, float],
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap 95% CI for mean Δ using drug-level resampling.

    Args:
        delta_per_drug: per-drug Δr values (condition - baseline)
        n_bootstrap: number of bootstrap resamples
        seed: random seed for reproducibility

    Returns:
        (ci_lower, ci_upper) at the 2.5th and 97.5th percentiles
    """
    deltas = np.array(list(delta_per_drug.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = np.array([rng.choice(deltas, size=n, replace=True).mean() for _ in range(n_bootstrap)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


__all__ = ["bootstrap_delta_ci", "holm_bonferroni"]
