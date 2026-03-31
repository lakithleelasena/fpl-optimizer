from __future__ import annotations

import numpy as np

from config import STRENGTH_MAX, STRENGTH_MIN

# Default weights: (w_ha, w_season, w_xgi, w_fixture, w_form, w_threat, w_xgc)
DEFAULT_WEIGHTS = (0.05, 0.20, 0.10, 0.35, 0.10, 0.10, 0.20)


def _weight_combinations() -> list[tuple]:
    """All 7-weight tuples summing to 1.0 in 0.1 increments."""
    combos = []
    for a in range(11):
        for b in range(11 - a):
            for c in range(11 - a - b):
                for d in range(11 - a - b - c):
                    for e in range(11 - a - b - c - d):
                        for f in range(11 - a - b - c - d - e):
                            g = 10 - a - b - c - d - e - f
                            combos.append((
                                round(a / 10, 1), round(b / 10, 1), round(c / 10, 1),
                                round(d / 10, 1), round(e / 10, 1), round(f / 10, 1),
                                round(g / 10, 1),
                            ))
    return combos


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


def compute_backtest(raw_histories: dict, team_strengths: dict, current_gw: int) -> dict:
    """
    For each completed GW, predict points using only data available before that GW,
    then compare to actual. Tests weight combinations using numpy for speed.
    """
    str_range = STRENGTH_MAX - STRENGTH_MIN
    mid_str = (STRENGTH_MAX + STRENGTH_MIN) / 2

    # Precompute signal columns per player-GW
    # Signals: [ha, season, xgi, fixture, form, threat, xgc]
    # Position routing: for GKP/DEF, xgi=0 and threat=0; for MID/FWD, xgc=0
    # We also need position-aware total_w per row, stored as a mask matrix

    gw_col: list[int] = []
    actual_col: list[float] = []
    signals_col: list[list[float]] = []  # 7 signals per row
    mask_col: list[list[float]] = []     # 7 booleans (as floats) — which weights apply per row
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

            # Home/Away signal
            was_home = entry.get("was_home", False)
            home_away_score = season_avg * (0.85 + (1.0 if was_home else 0.0) * 0.30)

            # Fixture signal
            opp_id = entry["opponent_team"]
            opp_strength = team_strengths.get(opp_id, mid_str)
            ease = max(0.0, min(1.0, (STRENGTH_MAX - opp_strength) / str_range))
            fixture_score = season_avg * (0.5 + ease)

            # Form signal: avg of last 4 prior played GW points
            form_pts = [h["total_points"] for h in played_prior[-4:]]
            form_score = sum(form_pts) / len(form_pts) if form_pts else season_avg

            # xGI signal (MID/FWD)
            xgi_prior = sum(h.get("xgi", 0.0) for h in played_prior)
            xg_score = (xgi_prior / len(played_prior)) * 15.0

            # ICT Threat signal (MID/FWD)
            threat_prior = sum(float(h.get("threat", 0)) for h in played_prior)
            threat_score = (threat_prior / len(played_prior)) / 10.0

            # xGC signal (GKP/DEF) — inverted defensive quality
            xgc_prior = sum(float(h.get("xgc", 0)) for h in played_prior)
            xgc_per_game = xgc_prior / len(played_prior)
            xgc_score = max(0.0, (3.0 - xgc_per_game) * 2.0)

            position = entry.get("position", "MID")
            is_defender = position in ("GKP", "DEF")

            # Build signal row and mask row
            # Order: [ha, season, xgi, fixture, form, threat, xgc]
            signals = [home_away_score, season_avg, xg_score, fixture_score, form_score, threat_score, xgc_score]

            if is_defender:
                # Zero out xgi and threat; keep xgc
                signals[2] = 0.0  # xgi → 0
                signals[5] = 0.0  # threat → 0
                mask = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]  # ha, season, xgi=off, fixture, form, threat=off, xgc
            else:
                # Zero out xgc; keep xgi and threat
                signals[6] = 0.0  # xgc → 0
                mask = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]  # all except xgc

            sl = _start_likelihood(prior)

            gw_col.append(gw)
            actual_col.append(float(entry["total_points"]))
            signals_col.append(signals)
            mask_col.append(mask)
            sl_col.append(sl)

    if not gw_col:
        raise ValueError("No data points for backtest — not enough completed gameweeks.")

    # ── FPL ep_this baseline ─────────────────────────────────────────────────
    ep_pred_list: list[float] = []
    ep_actual_list: list[float] = []
    ep_gw_list: list[int] = []

    for player_id, history in raw_histories.items():
        history_sorted = sorted(history, key=lambda h: h["round"])
        for entry in history_sorted:
            gw_e = entry["round"]
            if gw_e >= current_gw:
                continue
            ep = entry.get("ep_this", 0.0)
            if ep > 0:
                ep_pred_list.append(float(ep))
                ep_actual_list.append(float(entry["total_points"]))
                ep_gw_list.append(gw_e)

    ep_baseline_mae: float | None = None
    ep_baseline_per_gw: dict[str, float] = {}
    ep_baseline_points = len(ep_pred_list)

    if ep_pred_list:
        _ep_preds = np.array(ep_pred_list, dtype=np.float64)
        _ep_actuals = np.array(ep_actual_list, dtype=np.float64)
        _ep_gws = np.array(ep_gw_list)
        ep_baseline_mae = round(float(np.mean(np.abs(_ep_preds - _ep_actuals))), 4)
        for g in sorted(set(ep_gw_list)):
            idx = _ep_gws == g
            ep_baseline_per_gw[str(g)] = round(
                float(np.mean(np.abs(_ep_preds[idx] - _ep_actuals[idx]))), 3
            )
    # ────────────────────────────────────────────────────────────────────────────

    n = len(gw_col)
    gws = sorted(set(gw_col))

    # Convert to numpy arrays for vectorized computation
    actual_arr = np.array(actual_col, dtype=np.float64)       # shape (N,)
    signals_arr = np.array(signals_col, dtype=np.float64)      # shape (N, 7)
    mask_arr = np.array(mask_col, dtype=np.float64)            # shape (N, 7)
    sl_arr = np.array(sl_col, dtype=np.float64)                # shape (N,)
    gw_arr = np.array(gw_col)

    weight_combos = _weight_combinations()

    # Vectorized evaluation over all combos
    # For each combo w (shape 7,): effective_w = w * mask (per row), total_w = sum(effective_w)
    # pred = (signals @ effective_w) / total_w * sl
    # mae = mean(|pred - actual|)

    combo_maes: list[tuple] = []

    for combo in weight_combos:
        w = np.array(combo, dtype=np.float64)  # shape (7,)
        # effective weight per row: w * mask_arr → shape (N, 7)
        eff_w = mask_arr * w[np.newaxis, :]    # broadcast
        total_w = eff_w.sum(axis=1)            # shape (N,)
        total_w = np.where(total_w == 0, 1.0, total_w)
        numerator = (signals_arr * eff_w).sum(axis=1)  # shape (N,)
        pred = (numerator / total_w) * sl_arr
        mae = float(np.mean(np.abs(pred - actual_arr)))
        combo_maes.append((round(mae, 4), *combo))

    combo_maes.sort()

    # Per-GW MAE for a specific weight tuple
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
    best_combo = combo_maes[0][1:]  # 7 weights

    default_entry = next(
        (c for c in combo_maes if c[1:] == DEFAULT_WEIGHTS),
        combo_maes[-1],
    )
    default_mae = default_entry[0]

    def sensitivity(weight_idx: int) -> list[dict]:
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
        "per_gw_default": per_gw_mae(DEFAULT_WEIGHTS),
        "per_gw_best": per_gw_mae(best_combo),
        "gameweeks": [str(g) for g in gws],
        "total_data_points": n,
        "total_combinations_tested": len(combo_maes),
        "sensitivity": {
            "w_home_away": sensitivity(0),
            "w_season": sensitivity(1),
            "w_xgi": sensitivity(2),
            "w_fixture": sensitivity(3),
            "w_form": sensitivity(4),
            "w_threat": sensitivity(5),
            "w_xgc": sensitivity(6),
        },
        # FPL's own ep_this baseline — how accurate is the FPL model?
        "ep_baseline_mae": ep_baseline_mae,
        "ep_baseline_per_gw": ep_baseline_per_gw,
        "ep_baseline_data_points": ep_baseline_points,
    }
