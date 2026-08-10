"""Bootstrap confidence intervals over anchors (§9 statistics).

Used by Gate 1 (mean split-half Spearman) and Gate 2 (paired ΔLoss ordering).
When multi-seed retraining is infeasible, paired prompt/anchor-level bootstrap
CIs are the accepted significance test in this subfield.
"""
from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """(mean, lo, hi) for the mean of a 1-D sample, resampling rows w/ replacement."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(v.mean()), float(lo), float(hi)


def cluster_bootstrap_mean_ci(
    clusters, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0,
):
    """(mean, lo, hi) resampling CLUSTERS with replacement, keeping each intact.

    ``clusters`` is a sequence of sequences: one inner sequence per independent unit.
    The FU-MD unit is the story; recurrence events and generation seeds are repeated
    measurements nested inside it, so resampling them as if they were independent
    narrows the interval by roughly sqrt(events x seeds) and would let a handful of
    stories look like a population. The cluster mean is the mean over drawn stories of
    each story's own mean, so a story with more events does not get more weight.
    """
    groups = [np.asarray(c, dtype=np.float64) for c in clusters]
    groups = [g for g in groups if g.size]
    if not groups:
        return 0.0, 0.0, 0.0
    per_cluster = np.array([g.mean() for g in groups])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, per_cluster.size, size=(n_boot, per_cluster.size))
    means = per_cluster[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(per_cluster.mean()), float(lo), float(hi)


def wilson_interval(successes: int, n: int, alpha: float = 0.05):
    """(rate, lo, hi) Wilson score interval for a proportion.

    Wald breaks exactly where this project lives -- small n and rates near 0 or 1 -- by
    producing intervals that cover impossible values or collapse to zero width at 0/n.
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    from math import sqrt
    # 95% -> 1.959964; other alphas fall back to the normal quantile via erfinv.
    if abs(alpha - 0.05) < 1e-12:
        z = 1.959963984540054
    else:
        from statistics import NormalDist
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half))


def paired_bootstrap_diff_ci(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """(mean_diff, lo, hi, p_one_sided) for mean(a − b) over paired anchors.

    ``p_one_sided`` is the bootstrap fraction with diff ≤ 0 — i.e. evidence
    against H1: mean(a − b) > 0. p < 0.05 ⇒ a > b significantly.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.size, b.size)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0
    d = a[:n] - b[:n]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    p_one_sided = float((means <= 0.0).mean())
    return float(d.mean()), float(lo), float(hi), p_one_sided
