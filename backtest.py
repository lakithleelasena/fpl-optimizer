from __future__ import annotations

from config import STRENGTH_MAX, STRENGTH_MIN

# Default weights (must be 0.1-multiples summing to 1.0)
DEFAULT_WEIGHTS = (0.30, 0.20, 0.30, 0.20)


def _weight_combinations() -> list[tuple[float, float, float, float]]:
    """All 4-weight tuples summing to 1.0 in 0.1 increments (286 total)."""
    combos = []
    for a in range(11):
        for b in range(11 - a):
            for c in range(11 - a - b):
                d = 10 - a - b - c
                combos.append((round(a / 10, 1), round(b / 10, 1), round(c / 10, 1), round(d / 10, 1)))
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
    then compare to actual. Tests all 286 weight combinations in 0.1 increments.
    """
    str_range = STRENGTH_MAX - STRENGTH_MIN
    mid_str = (STRENGTH_MAX + STRENGTH_MIN) / 2

    # ── Precompute component scores per player-GW ──────────────────────────────
    gw_col: list[int] = []
    actual_col: list[float] = []
    opp_col: list[float] = []
    seas_col: list[float] = []
    mom_col: list[float] = []
    fix_col: list[float] = []
    sl_col: list[float] = []

    for player_id, history in raw_histories.items():
        history = sorted(history, key=lambda h: h["round"])

        for i, entry in enumerate(history):
            gw = entry["round"]
            if gw >= current_gw:
                continue  # only completed GWs

            prior = history[:i]
            played_prior = [h for h in prior if h["minutes"] > 0]
            if len(played_prior) < 3:
                continue  # need enough history for a meaningful prediction

            season_avg = sum(h["total_points"] for h in played_prior) / len(played_prior)
            recent_pts = [h["total_points"] for h in played_prior[-3:]]
            momentum = sum(recent_pts) / len(recent_pts)

            opp_id = entry["opponent_team"]
            opp_hist = [h["total_points"] for h in played_prior if h["opponent_team"] == opp_id]
            opponent_score = sum(opp_hist) / len(opp_hist) if opp_hist else season_avg

            opp_strength = team_strengths.get(opp_id, mid_str)
            ease = max(0.0, min(1.0, (STRENGTH_MAX - opp_strength) / str_range))
            fixture_score = season_avg * (0.5 + ease)

            sl = _start_likelihood(prior)

            gw_col.append(gw)
            actual_col.append(float(entry["total_points"]))
            opp_col.append(opponent_score)
            seas_col.append(season_avg)
            mom_col.append(momentum)
            fix_col.append(fixture_score)
            sl_col.append(sl)

    if not gw_col:
        raise ValueError("No data points for backtest — not enough completed gameweeks.")

    n = len(gw_col)
    gws = sorted(set(gw_col))

    # ── Evaluate all weight combinations (fast linear pass) ────────────────────
    weight_combos = _weight_combinations()
    combo_maes: list[tuple] = []  # (mae, w_o, w_s, w_m, w_f)

    for w_o, w_s, w_m, w_f in weight_combos:
        total_w = w_o + w_s + w_m + w_f
        if total_w == 0:
            continue
        total_err = 0.0
        for idx in range(n):
            pred = (
                (w_o * opp_col[idx] + w_s * seas_col[idx] + w_m * mom_col[idx] + w_f * fix_col[idx])
                / total_w
                * sl_col[idx]
            )
            total_err += abs(pred - actual_col[idx])
        combo_maes.append((round(total_err / n, 4), w_o, w_s, w_m, w_f))

    combo_maes.sort()

    # ── Per-GW MAE for a specific weight tuple ─────────────────────────────────
    def per_gw_mae(w_o: float, w_s: float, w_m: float, w_f: float) -> dict[str, float]:
        total_w = w_o + w_s + w_m + w_f or 1.0
        gw_errs: dict[int, float] = {}
        gw_counts: dict[int, int] = {}
        for idx in range(n):
            pred = (
                (w_o * opp_col[idx] + w_s * seas_col[idx] + w_m * mom_col[idx] + w_f * fix_col[idx])
                / total_w
                * sl_col[idx]
            )
            g = gw_col[idx]
            gw_errs[g] = gw_errs.get(g, 0.0) + abs(pred - actual_col[idx])
            gw_counts[g] = gw_counts.get(g, 0) + 1
        return {str(g): round(gw_errs[g] / gw_counts[g], 3) for g in sorted(gw_errs)}

    best_mae, bw_o, bw_s, bw_m, bw_f = combo_maes[0]

    # Find default in combo list
    default_entry = next(
        ((mae, w_o, w_s, w_m, w_f) for mae, w_o, w_s, w_m, w_f in combo_maes
         if (w_o, w_s, w_m, w_f) == DEFAULT_WEIGHTS),
        combo_maes[-1],
    )
    default_mae = default_entry[0]

    # ── Sensitivity: best MAE at each value of each weight ────────────────────
    def sensitivity(weight_idx: int) -> list[dict]:
        groups: dict[float, float] = {}
        for mae, w_o, w_s, w_m, w_f in combo_maes:
            val = (w_o, w_s, w_m, w_f)[weight_idx]
            if val not in groups or mae < groups[val]:
                groups[val] = mae
        return [{"value": v, "mae": round(groups[v], 4)} for v in sorted(groups)]

    top_combos = [
        {"w_opponent": w_o, "w_season": w_s, "w_momentum": w_m, "w_fixture": w_f, "mae": mae}
        for mae, w_o, w_s, w_m, w_f in combo_maes[:20]
    ]

    return {
        "top_combinations": top_combos,
        "best": {"w_opponent": bw_o, "w_season": bw_s, "w_momentum": bw_m, "w_fixture": bw_f, "mae": best_mae},
        "default_mae": default_mae,
        "best_mae": best_mae,
        "improvement_pct": round((default_mae - best_mae) / default_mae * 100, 2) if default_mae > 0 else 0.0,
        "per_gw_default": per_gw_mae(*DEFAULT_WEIGHTS),
        "per_gw_best": per_gw_mae(bw_o, bw_s, bw_m, bw_f),
        "gameweeks": [str(g) for g in gws],
        "total_data_points": n,
        "total_combinations_tested": len(combo_maes),
        "sensitivity": {
            "w_opponent": sensitivity(0),
            "w_season": sensitivity(1),
            "w_momentum": sensitivity(2),
            "w_fixture": sensitivity(3),
        },
    }
