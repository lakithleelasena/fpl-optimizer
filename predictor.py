from __future__ import annotations

import math

from config import STRENGTH_MAX, STRENGTH_MIN, W_ATK_FACTOR, W_CS_FACTOR, W_FIXTURE, W_FORM, W_FORM_FACTOR, W_HOME_AWAY, W_SEASON, W_THREAT, W_XGC, W_XGI

# FPL points per goal by position
_PTS_PER_GOAL: dict[str, int] = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
# FPL clean sheet points by position
_CS_PTS: dict[str, int] = {"GKP": 6, "DEF": 4, "MID": 0, "FWD": 0}


def _compute_start_likelihood(player: dict) -> float:
    stats = player["stats"]
    recent_mins = stats.get("recent_minutes", [])
    if not recent_mins:
        return 0.0
    gw_scores = []
    for mins in recent_mins:
        if mins >= 60:
            gw_scores.append(1.0)
        elif mins > 0:
            gw_scores.append(0.5)
        else:
            gw_scores.append(0.0)
    weights = list(range(1, len(gw_scores) + 1))
    total_weight = sum(weights)
    minutes_likelihood = sum(s * w for s, w in zip(gw_scores, weights)) / total_weight
    chance = player.get("chance_of_playing")
    availability = chance / 100.0 if chance is not None else 1.0
    return round(min(minutes_likelihood * availability, 1.0), 2)


def _compute_fixture_difficulty(opponent_strengths: list[float]) -> tuple[float, float]:
    """Legacy: strength-based ease for backtest path."""
    if not opponent_strengths:
        return 1.0, 1.0
    str_range = STRENGTH_MAX - STRENGTH_MIN
    ease_scores = [max(0.0, min(1.0, (STRENGTH_MAX - s) / str_range)) for s in opponent_strengths]
    avg_ease = sum(ease_scores) / len(ease_scores)
    return round(avg_ease, 2), round(0.5 + avg_ease, 2)


def _predict_points_legacy(
    player: dict,
    w_home_away: float,
    w_season: float,
    w_xgi: float,
    w_fixture: float,
    w_form: float,
    w_threat: float,
    w_xgc: float,
) -> dict:
    """Original 7-signal formula — used by backtest (historical data, no new fields)."""
    stats = player["stats"]
    position = player.get("position", "MID")
    season_avg = stats["season_avg"]
    games_played = stats["games_played"]
    start_likelihood = _compute_start_likelihood(player)

    precomputed_ease = player.get("_gw_ease")
    if precomputed_ease is not None:
        fixture_ease = precomputed_ease
        fixture_multiplier = 0.5 + fixture_ease
    else:
        fixture_ease, fixture_multiplier = _compute_fixture_difficulty(player.get("opponent_strengths", []))

    is_home = player.get("is_home", 0.5)

    if games_played == 0:
        return {
            "predicted_points": 0.0, "home_away_score": 0.0, "season_avg": 0.0,
            "xg_score": 0.0, "fixture_ease": fixture_ease, "start_likelihood": start_likelihood,
            "exp_minutes": start_likelihood,
            "form_score": 0.0, "threat_score": 0.0, "xgc_score": 0.0,
        }

    home_away_score = season_avg * (0.85 + is_home * 0.30)
    fixture_score = season_avg * fixture_multiplier
    form_score = player.get("form", 0.0)
    xgi = player.get("xgi", 0.0)
    xg_score = (xgi / games_played) * 15.0
    threat_raw = player.get("threat", 0.0)
    threat_score = (threat_raw / games_played) / 10.0
    xgc_raw = player.get("xgc", 0.0)
    xgc_per_game = xgc_raw / games_played if games_played > 0 else 1.5
    xgc_score = max(0.0, (3.0 - xgc_per_game) * 2.0)

    is_defender = position in ("GKP", "DEF")
    if is_defender:
        numerator = (w_home_away * home_away_score + w_season * season_avg
                     + w_fixture * fixture_score + w_form * form_score + w_xgc * xgc_score)
        total_w = w_home_away + w_season + w_fixture + w_form + w_xgc
    else:
        numerator = (w_home_away * home_away_score + w_season * season_avg
                     + w_fixture * fixture_score + w_form * form_score
                     + w_xgi * xg_score + w_threat * threat_score)
        total_w = w_home_away + w_season + w_fixture + w_form + w_xgi + w_threat

    if total_w == 0:
        total_w = 1.0
    predicted = (numerator / total_w) * start_likelihood

    return {
        "predicted_points": round(predicted, 2),
        "home_away_score": round(home_away_score, 2),
        "season_avg": round(season_avg, 2),
        "xg_score": round(xg_score, 2),
        "fixture_ease": fixture_ease,
        "start_likelihood": start_likelihood,
        "exp_minutes": start_likelihood,
        "form_score": round(float(form_score), 2),
        "threat_score": round(threat_score, 2),
        "xgc_score": round(xgc_score, 2),
    }


def _predict_points_new(
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


def predict_points(
    player: dict,
    w_home_away: float = W_HOME_AWAY,
    w_season: float = W_SEASON,
    w_xgi: float = W_XGI,
    w_fixture: float = W_FIXTURE,
    w_form: float = W_FORM,
    w_threat: float = W_THREAT,
    w_xgc: float = W_XGC,
    form_factor: float = W_FORM_FACTOR,
    cs_factor: float = W_CS_FACTOR,
    atk_factor: float = W_ATK_FACTOR,
) -> dict:
    """
    Dispatch to new participation formula (live) or legacy formula (backtest).
    New formula uses form_factor / cs_factor / atk_factor.
    Legacy formula uses the 7 w_* weights.
    """
    if player.get("goal_share") is not None:
        return _predict_points_new(player, form_factor, cs_factor, atk_factor)
    return _predict_points_legacy(
        player, w_home_away, w_season, w_xgi, w_fixture, w_form, w_threat, w_xgc
    )
