"""Wilson score interval for binomial accuracy — the CI we report everywhere."""
from __future__ import annotations

import math


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fmt_acc(successes: int, n: int) -> str:
    lo, hi = wilson_ci(successes, n)
    return f"{successes}/{n} = {successes / n:.1%} [{lo:.1%}, {hi:.1%}]" if n else "n=0"
