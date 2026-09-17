from __future__ import annotations

import math

from config import W_ATK_FACTOR, W_CS_FACTOR, W_FORM_FACTOR

# FPL points per goal by position
_PTS_PER_GOAL: dict[str, int] = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
# FPL clean sheet points by position
_CS_PTS: dict[str, int] = {"GKP": 6, "DEF": 4, "MID": 0, "FWD": 0}


def predict_points(
    player: dict,
    form_factor: float = W_FORM_FACTOR,
    cs_factor: float = W_CS_FACTOR,
    atk_factor: float = W_ATK_FACTOR,
) -> dict:
    """
    Participation-based formula.
    form_factor: scales form adjustment (0=ignore, 2=double sensitivity)
    cs_factor:   scales clean-sheet bonus for GKP/DEF
    atk_factor:  scales goal/assist contribution for MID/FWD (and rare DEF/GKP goals)
    """
    stats = player["stats"]
    position = player.get("position", "MID")
    exp_start_pct = float(player.get("exp_start_pct") or 0.0)
    exp_minutes = float(player.get("exp_minutes") or 0.0)
    season_avg = stats["season_avg"]

    match_team_xg = float(player.get("match_team_xg") or 0.0)
    match_opp_xg  = float(player.get("match_opp_xg")  or 0.0)
    goal_share     = float(player.get("goal_share")    or 0.0)
    assist_share   = float(player.get("assist_share")  or 0.0)
    saves_per_game = float(player.get("saves_per_game") or 0.0)
    form           = float(player.get("form")          or 0.0)

    # Form adjustment: base ±0.5 cap, then scaled by form_factor
    form_adj_base = max(-0.5, min(0.5, (form - season_avg) * 0.1)) if season_avg > 0 else 0.0
    form_adj = form_adj_base * form_factor

    fixture_ease = player.get("_gw_ease")
    if fixture_ease is None:
        fixture_ease = 0.5

    is_def = position in ("GKP", "DEF")

    if is_def:
        cs_pts   = _CS_PTS[position]
        cs_prob  = math.exp(-match_opp_xg)
        xgc_pts  = -math.floor(match_opp_xg / 2)
        save_pts = (saves_per_game / 3) if position == "GKP" else 0.0
        atk_pts  = match_team_xg * goal_share  * _PTS_PER_GOAL[position] * atk_factor
        ast_pts  = match_team_xg * assist_share * 3 * atk_factor

        predicted = (exp_start_pct * 2) + exp_minutes * (
            cs_prob * cs_pts * cs_factor + xgc_pts + save_pts + atk_pts + ast_pts
        ) + form_adj
        xgc_score_out = round(cs_prob, 3)
    else:
        goal_pts = match_team_xg * goal_share  * _PTS_PER_GOAL[position] * atk_factor
        asst_pts = match_team_xg * assist_share * 3 * atk_factor

        predicted = (exp_start_pct * 2) + exp_minutes * (goal_pts + asst_pts) + form_adj
        xgc_score_out = 0.0

    player_xg = round(match_team_xg * (goal_share + assist_share), 3)

    return {
        "predicted_points": round(max(0.0, predicted), 2),
        "home_away_score": 0.0,
        "season_avg": round(season_avg, 2),
        "xg_score": player_xg,
        "fixture_ease": fixture_ease,
        "start_likelihood": exp_start_pct,
        "exp_minutes": exp_minutes,
        "form_score": round(form, 2),
        "threat_score": 0.0,
        "xgc_score": xgc_score_out,
    }
