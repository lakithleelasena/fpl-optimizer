"""Simple linear regression estimating bonus points from expected match events.

Phase 2, PREDICTION_MODEL_PLAN.md: "train a regression on per-player-match data
mapping events (goals, assists, clean sheet, saves, DefCon hit, minutes, position)
to bonus, then apply it to your expected events." This fits ordinary least squares
on every played match (minutes > 0) across all active players THIS SEASON ONLY —
per the project's backtest-scope decision, and because 2025/26 bonus data wouldn't
match anyway (BPS was reworked for 2026/27: no more tackled-player penalty, CBI now
1 BPS per 3 instead of per 2, restructured GK save BPS with a big-chance bonus).

`fit()` is a pure function (entries in, coefficients out) so the accuracy backtest
can call it with history truncated to only what was available before each target
gameweek — avoiding look-ahead bias — without touching the live module-level cache.
`refit()`/`predict_bonus()` are the live-pipeline convenience wrappers: the fit is
refreshed once per fetch_all_data() cache cycle (a few thousand rows, 6 features —
cheap) rather than trained once and frozen, so it improves automatically as more of
this season's own data accumulates. Coefficients are cached at module level so
predictor.py can call predict_bonus() without fetch_all_data() having to thread a
fitted-model object through every call site.
"""
from __future__ import annotations

import numpy as np

from config import DEFCON_THRESHOLD

_FEATURE_NAMES = ["intercept", "goals", "assists", "clean_sheet", "saves", "defcon_hit"]

# Hand-set fallback (roughly BPS-informed) used only until enough of this season's
# own data has accumulated to fit a real regression, or if the fit degenerates.
_DEFAULT_COEFFS = np.array([0.1, 0.9, 0.5, 0.3, 0.05, 0.3])
_MIN_ROWS_TO_FIT = 200

_cached_coeffs: np.ndarray = _DEFAULT_COEFFS
_cached_row_count: int = 0


def _extract_features(h: dict, position: str) -> list[float]:
    threshold = DEFCON_THRESHOLD.get(position)
    defcon_hit = 1.0 if threshold is not None and h.get("defensive_contribution", 0) >= threshold else 0.0
    return [
        1.0,
        float(h.get("goals_scored", 0)),
        float(h.get("assists", 0)),
        float(h.get("clean_sheets", 0)),
        float(h.get("saves", 0)),
        defcon_hit,
    ]


def fit(entries: list[tuple[str, list[dict]]]) -> np.ndarray:
    """
    Pure function: OLS-fit bonus ~ goals + assists + clean_sheet + saves + defcon_hit
    over every played match (minutes > 0) in `entries`, and RETURN the coefficients —
    does not touch the module-level cache. `entries`: (position, history) pairs,
    shape-agnostic like build_team_xg_totals/build_position_priors in fpl_client.py,
    so both the live pipeline (refit(), below) and the accuracy backtest (which needs
    a fresh fit per target gameweek from only prior-to-that-gameweek history) can
    build their own input list and call this directly.
    """
    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    for position, history in entries:
        for h in history:
            if h.get("minutes", 0) <= 0:
                continue
            X_rows.append(_extract_features(h, position))
            y_rows.append(float(h.get("bonus", 0)))

    if len(X_rows) < _MIN_ROWS_TO_FIT:
        return _DEFAULT_COEFFS

    X = np.array(X_rows)
    y = np.array(y_rows)
    try:
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return _DEFAULT_COEFFS
    return coeffs


def refit(results: list[tuple[int, list[dict], list[dict]]], pos_lookup: dict[int, str]) -> None:
    """Refit the module-level cached coefficients from this season's actual
    per-player-match data. Call once per fetch_all_data() cache refresh (live
    pipeline only — the backtest uses fit() directly instead, see above)."""
    global _cached_coeffs, _cached_row_count
    entries = [(pos_lookup.get(pid, "MID"), history) for pid, history, _ in results]
    _cached_row_count = sum(1 for _, history in entries for h in history if h.get("minutes", 0) > 0)
    _cached_coeffs = fit(entries)


def _apply(coeffs: np.ndarray, goals: float, assists: float, cs_prob: float, saves: float, defcon_hit_rate: float) -> float:
    features = np.array([1.0, goals, assists, cs_prob, saves, defcon_hit_rate])
    bonus = float(coeffs @ features)
    return round(max(0.0, min(3.0, bonus)), 3)


def predict_bonus(goals: float, assists: float, cs_prob: float, saves: float, defcon_hit_rate: float) -> float:
    """
    Live pipeline — applies the module-level cached coefficients (see refit()) to
    EXPECTED (not actual) event values. cs_prob (a 0-1 probability) stands in for
    the training data's 0/1 clean_sheet outcome, same idea for the other continuous
    expectations. Bonus is genuinely capped at [0, 3] by FPL's rules, so the output
    is clipped to that range.
    """
    return _apply(_cached_coeffs, goals, assists, cs_prob, saves, defcon_hit_rate)


def predict_bonus_with_coeffs(
    coeffs: np.ndarray, goals: float, assists: float, cs_prob: float, saves: float, defcon_hit_rate: float,
) -> float:
    """Backtest — same as predict_bonus() but with caller-supplied coefficients
    (typically from fit() on history truncated to before the target gameweek, to
    avoid look-ahead bias that the shared live cache would otherwise introduce)."""
    return _apply(coeffs, goals, assists, cs_prob, saves, defcon_hit_rate)
