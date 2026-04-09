from __future__ import annotations

from typing import List, Optional

import numpy as np

from config import STRENGTH_MAX, STRENGTH_MIN

# Signal order used throughout — index matches weight tuple position
ALL_SIGNALS = ["home_away", "season", "xgi", "fixture", "form", "threat", "xgc"]

# Default weights: (w_ha, w_season, w_xgi, w_fixture, w_form, w_threat, w_xgc)
DEFAULT_WEIGHTS = (0.05, 0.20, 0.10, 0.35, 0.10, 0.10, 0.20)


def _weight_combinations_k(k: int) -> list[tuple]:
    """All k-weight tuples summing to 1.0 in 0.1 increments."""
    if k == 1:
        return [(1.0,)]
    result: list[tuple] = []

    def _gen(remaining: int, depth: int, current: list[float]) -> None:
        if depth == k - 1:
            result.append(tuple(current + [round(remaining / 10, 1)]))
            return
        for i in range(remaining + 1):
            _gen(remaining - i, depth + 1, current + [round(i / 10, 1)])

    _gen(10, 0, [])
    return result


def _start_likelihood(prior: list[dict]) -> float:
    recent_mins = [h["minutes"] for h in prior[-5:]]
    if not recent_mins:
        return 0.0
    scores = []
    for m in recent_mins:
        if m >= 60:
            scores.append(1.0)
        elif m > 0:
            scores.append(0.5)
        else:
            scores.append(0.0)
    weights = list(range(1, len(scores) + 1))
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


def compute_backtest(
    raw_histories: dict,
    team_strengths: dict,
    current_gw: int,
    active_signals: Optional[List[str]] = None,
) -> dict:
    """
    For each completed GW, predict points using only data available before that GW,
    then compare to actual. Tests weight combinations using numpy for speed.

    active_signals: subset of ALL_SIGNALS to include. Defaults to all 7.
    Position routing always applies regardless of active_signals:
      GKP/DEF: xgi and threat are always 0
      MID/FWD: xgc is always 0
    """
    if active_signals is None:
        active_signals = list(ALL_SIGNALS)

    # Boolean and float masks for which signals the user wants to include
    signal_active = [s in active_signals for s in ALL_SIGNALS]
    signal_active_arr = np.array([1.0 if a else 0.0 for a in signal_active], dtype=np.float64)
    active_indices = [i for i, a in enumerate(signal_active) if a]
    n_active = len(active_indices)

    if n_active == 0:
        raise ValueError("At least one signal must be selected.")

    str_range = STRENGTH_MAX - STRENGTH_MIN
    mid_str = (STRENGTH_MAX + STRENGTH_MIN) / 2

    gw_col: list[int] = []
    actual_col: list[float] = []
    signals_col: list[list[float]] = []
    mask_col: list[list[float]] = []
    sl_col: list[float] = []

    for player_id, history in raw_histories.items():
        history = sorted(history, key=lambda h: h["round"])

        for i, entry in enumerate(history):
            gw = entry["round"]
            if gw >= current_gw:
                continue

            prior = history[:i]
            played_prior = [h for h in prior if h["minutes"] > 0]
            if len(played_prior) < 3:
                continue

            season_avg = sum(h["total_points"] for h in played_prior) / len(played_prior)

            was_home = entry.get("was_home", False)
            home_away_score = season_avg * (0.85 + (1.0 if was_home else 0.0) * 0.30)

            opp_id = entry["opponent_team"]
            opp_strength = team_strengths.get(opp_id, mid_str)
            ease = max(0.0, min(1.0, (STRENGTH_MAX - opp_strength) / str_range))
            fixture_score = season_avg * (0.5 + ease)

            form_pts = [h["total_points"] for h in played_prior[-4:]]
            form_score = sum(form_pts) / len(form_pts) if form_pts else season_avg

            xgi_prior = sum(h.get("xgi", 0.0) for h in played_prior)
            xg_score = (xgi_prior / len(played_prior)) * 15.0

            threat_prior = sum(float(h.get("threat", 0)) for h in played_prior)
            threat_score = (threat_prior / len(played_prior)) / 10.0

            xgc_prior = sum(float(h.get("xgc", 0)) for h in played_prior)
            xgc_per_game = xgc_prior / len(played_prior)
            xgc_score = max(0.0, (3.0 - xgc_per_game) * 2.0)

            position = entry.get("position", "MID")
            is_defender = position in ("GKP", "DEF")

            signals = [home_away_score, season_avg, xg_score, fixture_score, form_score, threat_score, xgc_score]

            # Position routing mask
            if is_defender:
                signals[2] = 0.0  # xgi
                signals[5] = 0.0  # threat
                pos_mask = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
            else:
                signals[6] = 0.0  # xgc
                pos_mask = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]

            # Combined mask: position routing AND user-selected signals
            combined_mask = [p * a for p, a in zip(pos_mask, signal_active)]

            sl = _start_likelihood(prior)

            gw_col.append(gw)
            actual_col.append(float(entry["total_points"]))
            signals_col.append(signals)
            mask_col.append(combined_mask)
            sl_col.append(sl)

    if not gw_col:
        raise ValueError("No data points for backtest — not enough completed gameweeks.")

    n = len(gw_col)
    gws = sorted(set(gw_col))

    actual_arr = np.array(actual_col, dtype=np.float64)
    signals_arr = np.array(signals_col, dtype=np.float64)
    mask_arr = np.array(mask_col, dtype=np.float64)
    sl_arr = np.array(sl_col, dtype=np.float64)
    gw_arr = np.array(gw_col)

    # Generate weight combos only for active signals, expand to 7-weight tuples
    k_combos = _weight_combinations_k(n_active)
    weight_combos: list[tuple] = []
    for kc in k_combos:
        w = [0.0] * 7
        for pos, idx in enumerate(active_indices):
            w[idx] = kc[pos]
        weight_combos.append(tuple(w))

    combo_maes: list[tuple] = []

    for combo in weight_combos:
        w = np.array(combo, dtype=np.float64)
        eff_w = mask_arr * w[np.newaxis, :]
        total_w = eff_w.sum(axis=1)
        total_w = np.where(total_w == 0, 1.0, total_w)
        numerator = (signals_arr * eff_w).sum(axis=1)
        pred = (numerator / total_w) * sl_arr
        mae = float(np.mean(np.abs(pred - actual_arr)))
        combo_maes.append((round(mae, 4), *combo))

    combo_maes.sort()

    def per_gw_mae(combo: tuple) -> dict[str, float]:
        w = np.array(combo, dtype=np.float64)
        eff_w = mask_arr * w[np.newaxis, :]
        total_w = np.where(eff_w.sum(axis=1) == 0, 1.0, eff_w.sum(axis=1))
        pred = ((signals_arr * eff_w).sum(axis=1) / total_w) * sl_arr
        result = {}
        for g in gws:
            idx = gw_arr == g
            result[str(g)] = round(float(np.mean(np.abs(pred[idx] - actual_arr[idx]))), 3)
        return result

    best_mae = combo_maes[0][0]
    best_combo = combo_maes[0][1:]

    # Normalized default weights for the active signals
    raw_default = [DEFAULT_WEIGHTS[i] if signal_active[i] else 0.0 for i in range(7)]
    total_default = sum(raw_default)
    if total_default > 0:
        normalized_default = tuple(round(w / total_default, 4) for w in raw_default)
    else:
        normalized_default = best_combo

    # Compute default MAE directly (normalized default may not be in combos)
    w_def = np.array(normalized_default, dtype=np.float64)
    eff_w_def = mask_arr * w_def[np.newaxis, :]
    total_w_def = np.where(eff_w_def.sum(axis=1) == 0, 1.0, eff_w_def.sum(axis=1))
    pred_def = ((signals_arr * eff_w_def).sum(axis=1) / total_w_def) * sl_arr
    default_mae = round(float(np.mean(np.abs(pred_def - actual_arr))), 4)

    def sensitivity(weight_idx: int) -> list[dict]:
        # Only return sensitivity for active signals
        if not signal_active[weight_idx]:
            return []
        groups: dict[float, float] = {}
        for entry in combo_maes:
            val = entry[1 + weight_idx]
            mae = entry[0]
            if val not in groups or mae < groups[val]:
                groups[val] = mae
        return [{"value": v, "mae": round(groups[v], 4)} for v in sorted(groups)]

    top_combos = [
        {
            "w_home_away": c[1], "w_season": c[2], "w_xgi": c[3],
            "w_fixture": c[4], "w_form": c[5], "w_threat": c[6], "w_xgc": c[7],
            "mae": c[0],
        }
        for c in combo_maes[:20]
    ]

    w_ha, w_s, w_xgi, w_f, w_form, w_threat, w_xgc = best_combo

    return {
        "top_combinations": top_combos,
        "best": {
            "w_home_away": w_ha, "w_season": w_s, "w_xgi": w_xgi,
            "w_fixture": w_f, "w_form": w_form, "w_threat": w_threat, "w_xgc": w_xgc,
            "mae": best_mae,
        },
        "default_mae": default_mae,
        "best_mae": best_mae,
        "improvement_pct": round((default_mae - best_mae) / default_mae * 100, 2) if default_mae > 0 else 0.0,
        "per_gw_default": per_gw_mae(normalized_default),
        "per_gw_best": per_gw_mae(best_combo),
        "gameweeks": [str(g) for g in gws],
        "total_data_points": n,
        "total_combinations_tested": len(combo_maes),
        "active_signals": active_signals,
        "sensitivity": {
            "w_home_away": sensitivity(0),
            "w_season": sensitivity(1),
            "w_xgi": sensitivity(2),
            "w_fixture": sensitivity(3),
            "w_form": sensitivity(4),
            "w_threat": sensitivity(5),
            "w_xgc": sensitivity(6),
        },
    }
