"""Statistical power analysis for the representation ablation decision gate.

Question: given the observed null distribution (fold std ≈ 0.023, K=10 folds,
Holm-Bonferroni over 9 non-degenerate conditions), what is the minimum detectable
effect (MDE) at 80% power? At 90% power?

If MDE > gate (0.01), we cannot statistically distinguish "null" from a small positive
effect — the experiment is underpowered and the Δ=0.01 gate claim needs qualification.

Usage:
    python experiments/03_drug_feature_null/02_representation_ablation/jobs/power_analysis.py

Output: report/data/power_analysis.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist

ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(ROOT))

EXP_DIR = Path(__file__).parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Parameters from the observed null distribution
K_FOLDS = 10
N_CONDITIONS = 9           # non-degenerate conditions for Holm-Bonferroni
N_DRUGS = 233              # drugs in per-drug Δ paired test
FOLD_STD_ESTIMATE = 0.023  # observed fold-to-fold std of per-drug r
SIGMA_PER_DRUG = 0.15      # approx per-drug r std across drugs (for paired t-test power)
N_SIM = 50_000             # Monte Carlo samples
RNG_SEED = 42


def fold_level_power(
    delta: float,
    fold_std: float = FOLD_STD_ESTIMATE,
    k: int = K_FOLDS,
    alpha: float = 0.05,
) -> float:
    """Analytical power for a one-sample t-test on k fold differences.

    H0: mean_delta = 0, H1: mean_delta = delta.
    Assumes fold deltas are iid N(delta, fold_std²).
    """
    se = fold_std / np.sqrt(k)
    ncp = delta / se  # non-centrality parameter
    df = k - 1
    crit = t_dist.ppf(1 - alpha, df)
    power = 1 - t_dist.cdf(crit, df, loc=ncp)
    return float(power)


def per_drug_power(
    delta: float,
    sigma: float = SIGMA_PER_DRUG,
    n: int = N_DRUGS,
    alpha: float = 0.05,
) -> float:
    """Analytical power for a one-sample t-test on n per-drug delta values.

    H0: mean_delta = 0, H1: mean_delta = delta.
    """
    se = sigma / np.sqrt(n)
    ncp = delta / se
    df = n - 1
    crit = t_dist.ppf(1 - alpha, df)
    power = 1 - t_dist.cdf(crit, df, loc=ncp)
    return float(power)


def holm_alpha(n_tests: int, family_alpha: float = 0.05) -> float:
    """Most conservative (first comparison) Holm-Bonferroni alpha."""
    return family_alpha / n_tests


def find_mde(
    power_fn,
    target_power: float = 0.80,
    alpha: float = 0.05,
    lo: float = 1e-4,
    hi: float = 0.10,
    tol: float = 1e-5,
) -> float:
    """Binary search for minimum detectable effect at target_power."""
    for _ in range(60):
        mid = (lo + hi) / 2
        if power_fn(mid, alpha=alpha) >= target_power:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def monte_carlo_power(
    delta: float,
    fold_std: float = FOLD_STD_ESTIMATE,
    k: int = K_FOLDS,
    n_conditions: int = N_CONDITIONS,
    family_alpha: float = 0.05,
    n_sim: int = N_SIM,
    seed: int = RNG_SEED,
) -> float:
    """Simulate Holm-corrected rejection rate for a single condition with true delta.

    All other conditions are simulated under H0 (delta=0).
    Returns P(reject H0 for the true-effect condition) under Holm-Bonferroni.
    """
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(n_sim):
        # True condition: fold deltas ~ N(delta, fold_std²)
        true_folds = rng.normal(delta, fold_std, size=k)
        true_t = true_folds.mean() / (true_folds.std(ddof=1) / np.sqrt(k))

        # Null conditions: fold deltas ~ N(0, fold_std²)
        null_ts = []
        for _ in range(n_conditions - 1):
            null_folds = rng.normal(0, fold_std, size=k)
            null_ts.append(null_folds.mean() / (null_folds.std(ddof=1) / np.sqrt(k)))

        all_ts = [true_t] + null_ts
        # Two-sided p-values via t-distribution
        df = k - 1
        p_values = [2 * (1 - t_dist.cdf(abs(t), df)) for t in all_ts]

        # Holm correction: sort p-values, compare against α/n, α/(n-1), …
        sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])
        rejected = set()
        for rank, (idx, p) in enumerate(sorted_p):
            threshold = family_alpha / (n_conditions - rank)
            if p <= threshold:
                rejected.add(idx)
            else:
                break  # Holm stops at first non-rejection

        if 0 in rejected:  # True condition is index 0
            rejections += 1

    return rejections / n_sim


def main() -> None:
    report_dir = EXP_DIR / "report" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Effective alpha after Holm correction (most conservative)
    holm_alpha_val = holm_alpha(N_CONDITIONS)
    logger.info(
        "Setup: K=%d folds, fold_std=%.3f, n_conditions=%d, Holm α₁=%.4f",
        K_FOLDS, FOLD_STD_ESTIMATE, N_CONDITIONS, holm_alpha_val,
    )

    # --- Fold-level power curve ---
    logger.info("=== Fold-level t-test power (corrected α=%.4f) ===", holm_alpha_val)
    fold_power_curve = {}
    for delta in [0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030]:
        p = fold_level_power(delta, alpha=holm_alpha_val)
        fold_power_curve[str(delta)] = round(p, 4)
        logger.info("  Δ=%.3f  power=%.3f", delta, p)

    mde_80_fold = find_mde(fold_level_power, target_power=0.80, alpha=holm_alpha_val)
    mde_90_fold = find_mde(fold_level_power, target_power=0.90, alpha=holm_alpha_val)
    logger.info("Fold-level MDE @ 80%% power: %.4f", mde_80_fold)
    logger.info("Fold-level MDE @ 90%% power: %.4f", mde_90_fold)

    # --- Per-drug level power curve ---
    logger.info("=== Per-drug t-test power (corrected α=%.4f) ===", holm_alpha_val)
    drug_power_curve = {}
    for delta in [0.003, 0.005, 0.008, 0.010, 0.015, 0.020]:
        p = per_drug_power(delta, alpha=holm_alpha_val)
        drug_power_curve[str(delta)] = round(p, 4)
        logger.info("  Δ=%.3f  power=%.3f", delta, p)

    mde_80_drug = find_mde(per_drug_power, target_power=0.80, alpha=holm_alpha_val)
    mde_90_drug = find_mde(per_drug_power, target_power=0.90, alpha=holm_alpha_val)
    logger.info("Per-drug MDE @ 80%% power: %.4f", mde_80_drug)
    logger.info("Per-drug MDE @ 90%% power: %.4f", mde_90_drug)

    # --- Monte Carlo simulation (most realistic, includes Holm correlation) ---
    logger.info("=== Monte Carlo Holm power simulation (n_sim=%d) ===", N_SIM)
    mc_power_curve = {}
    for delta in [0.003, 0.005, 0.008, 0.010, 0.015, 0.020]:
        p = monte_carlo_power(delta)
        mc_power_curve[str(delta)] = round(p, 4)
        logger.info("  Δ=%.3f  MC_power=%.3f", delta, p)

    # Gate assessment
    gate = 0.010
    gate_power_mc = monte_carlo_power(gate)
    logger.info("=" * 60)
    logger.info("Gate Δ=%.3f  Monte Carlo power: %.3f", gate, gate_power_mc)
    if gate_power_mc >= 0.80:
        logger.info("PASS: gate is detectable at ≥ 80%% power (%.1f%%)", 100 * gate_power_mc)
    else:
        logger.warning(
            "CAUTION: gate Δ=%.3f has only %.1f%% power — "
            "'null' and 'effect at gate' cannot be reliably distinguished.",
            gate, 100 * gate_power_mc,
        )

    output = {
        "setup": {
            "k_folds": K_FOLDS,
            "fold_std": FOLD_STD_ESTIMATE,
            "n_conditions_holm": N_CONDITIONS,
            "n_drugs_per_drug_test": N_DRUGS,
            "holm_alpha_1": float(holm_alpha_val),
            "gate": gate,
        },
        "fold_level": {
            "power_curve": fold_power_curve,
            "mde_80pct": round(mde_80_fold, 5),
            "mde_90pct": round(mde_90_fold, 5),
        },
        "per_drug_level": {
            "power_curve": drug_power_curve,
            "mde_80pct": round(mde_80_drug, 5),
            "mde_90pct": round(mde_90_drug, 5),
        },
        "monte_carlo": {
            "power_curve": mc_power_curve,
            "gate_power": round(gate_power_mc, 4),
            "n_sim": N_SIM,
        },
    }
    out_path = report_dir / "power_analysis.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Power analysis written to %s", out_path)


if __name__ == "__main__":
    main()
