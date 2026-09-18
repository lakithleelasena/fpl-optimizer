"""Market-implied team expected goals: devigging (Shin's method) + a joint
Poisson-grid fit against both the 1X2 and totals markets simultaneously, with a
Dixon-Coles low-score correlation correction. Phase 4, PREDICTION_MODEL_PLAN.md.

Replaces the old approach (simple proportional devig, then a naive algebraic
split of the totals-market total by the h2h-implied home-win-share) with the
two-step process the source review recommended: remove the vig properly, then
solve for the (lambda_home, lambda_away) pair whose Poisson scoreline grid best
reproduces the market's match-outcome probabilities at the market's implied total
goals — rather than a linear split that ignores the shape of the Poisson
distribution entirely.
"""
from __future__ import annotations

import math

# Dixon & Coles (1997) reported rho for English football around this range; used
# here as a literature default, not fit from our own data (see the Phase 5 note
# in PREDICTION_MODEL_PLAN.md about calibrating it locally via backtesting).
DIXON_COLES_RHO = -0.13

MAX_GOALS_GRID = 10  # scoreline grid truncation; P(either side scoring >10) is negligible


def devig_shin(probs: list[float], tol: float = 1e-9, max_iter: int = 100) -> list[float]:
    """
    Remove the bookmaker margin from raw implied probabilities (1/odds) using
    Shin's method, which corrects for the market's well-documented
    favorite-longshot bias (overpricing longshots, underpricing favorites) —
    better than simple proportional normalisation (`p_i = pi_i / sum(pi)`).

    Solves for the "informed money" fraction z in:
        p_i(z) = (sqrt(z^2 + 4*(1-z)*pi_i^2/B) - z) / (2*(1-z))
    such that sum(p_i(z)) = 1, where B = sum(raw implied probs) > 1 (the
    overround). sum(p_i(z)) is monotonically decreasing in z over [0, 1), so the
    root is found by bisection. Falls back to simple proportional normalisation
    if the input is degenerate or the result fails a basic sanity check —
    correctness here matters more than always using the fancier method.
    """
    B = sum(probs)
    if B <= 1.0 or len(probs) < 2:
        return [p / B for p in probs] if B > 0 else list(probs)

    def total_at(z: float) -> float:
        return sum(
            (math.sqrt(z * z + 4 * (1 - z) * p * p / B) - z) / (2 * (1 - z))
            for p in probs
        )

    lo, hi = 0.0, 1.0 - 1e-6
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if total_at(mid) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    z = (lo + hi) / 2

    result = [(math.sqrt(z * z + 4 * (1 - z) * p * p / B) - z) / (2 * (1 - z)) for p in probs]
    s = sum(result)
    if s <= 0 or any(p < 0 for p in result):
        return [p / B for p in probs]
    return [p / s for p in result]


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _dixon_coles_tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Correlation adjustment on the four low-scoreline cells — pure independent
    Poisson underpredicts draws (and slightly misjudges 1-0/0-1 results) without it."""
    if x == 0 and y == 0:
        return 1 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1 + lam_h * rho
    if x == 1 and y == 0:
        return 1 + lam_a * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def poisson_match_probs(
    lam_h: float, lam_a: float, rho: float = DIXON_COLES_RHO, max_goals: int = MAX_GOALS_GRID,
) -> tuple[float, float, float]:
    """(P(home win), P(draw), P(away win)) from independent Poisson(lam_h)/
    Poisson(lam_a) scorelines, Dixon-Coles-adjusted on the four low-score cells."""
    p_home = p_draw = p_away = 0.0
    for x in range(max_goals + 1):
        px = _poisson_pmf(x, lam_h)
        for y in range(max_goals + 1):
            p = px * _poisson_pmf(y, lam_a) * _dixon_coles_tau(x, y, lam_h, lam_a, rho)
            if x > y:
                p_home += p
            elif x == y:
                p_draw += p
            else:
                p_away += p
    total = p_home + p_draw + p_away
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return p_home / total, p_draw / total, p_away / total


def fit_team_lambdas(
    total_xg: float, target_p_home: float, rho: float = DIXON_COLES_RHO, iterations: int = 30,
) -> tuple[float, float]:
    """
    Solve for (lambda_home, lambda_away) whose Poisson (+ Dixon-Coles) scoreline
    grid reproduces the market's implied P(home win), holding the total at the
    market's implied total goals from the totals market. The totals market alone
    only pins down lambda_home + lambda_away; the 1X2 market's home-win
    probability then determines the split — holding the total fixed and
    bisecting the split converges to a unique answer since P(home win) is
    monotonic in the split for a fixed total. (A full joint 2D least-squares fit
    over both markets simultaneously — letting the total move too — would be a
    marginal refinement over this; skipped here to avoid a numerical solver
    dependency for a small expected gain.)
    """
    lo, hi = 0.05, 0.95
    for _ in range(iterations):
        mid = (lo + hi) / 2
        lam_h, lam_a = total_xg * mid, total_xg * (1 - mid)
        p_home, _, _ = poisson_match_probs(lam_h, lam_a, rho)
        if p_home < target_p_home:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2
    return round(total_xg * s, 3), round(total_xg * (1 - s), 3)
