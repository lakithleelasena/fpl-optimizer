from __future__ import annotations

import math

import bonus_model
from config import W_ATK_FACTOR, W_CS_FACTOR, W_FORM_FACTOR

# FPL points per goal by position — verified against fantasy.premierleague.com/help/rules 2026-09-17
_PTS_PER_GOAL: dict[str, int] = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
# FPL clean sheet points by position — verified against fantasy.premierleague.com/help/rules 2026-09-17
_CS_PTS: dict[str, int] = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}


def _expected_floor_half_poisson(lam: float) -> float:
    """E[floor(K/2)] for K ~ Poisson(lam), closed form.

    floor(k/2) = (k - (k mod 2)) / 2, so E[floor(K/2)] = (E[K] - P(K odd)) / 2.
    P(K odd) = (1 - exp(-2*lam)) / 2 (standard Poisson parity identity), giving:
        E[floor(K/2)] = lam/2 - (1 - exp(-2*lam)) / 4
    This replaces floor(lam/2), which is a biased point-estimate approximation —
    floor(E[K]/2) != E[floor(K/2)] in general (e.g. lam=1.9 floors to 0 every time
    under the old formula, but a real Poisson(1.9) variable is >=2 over 40% of the time).
    """
    return lam / 2 - (1 - math.exp(-2 * lam)) / 4


def predict_points(
    player: dict,
    form_factor: float = W_FORM_FACTOR,
    cs_factor: float = W_CS_FACTOR,
    atk_factor: float = W_ATK_FACTOR,
) -> dict:
    """
    Participation-based formula.
    form_factor: scales form adjustment (0=ignore, 2=double sensitivity)
    cs_factor:   scales clean-sheet bonus (all positions — MID gets 1pt per real
                 FPL rules, not just GKP/DEF's 4pt)
    atk_factor:  scales goal/assist contribution for every position
    """
    stats = player["stats"]
    position = player.get("position", "MID")
    exp_start_pct = float(player.get("exp_start_pct") or 0.0)
    exp_minutes = float(player.get("exp_minutes") or 0.0)
    season_avg = stats["season_avg"]

    match_team_xg   = float(player.get("match_team_xg")   or 0.0)
    match_opp_xg    = float(player.get("match_opp_xg")    or 0.0)
    goal_share      = float(player.get("goal_share")      or 0.0)
    assist_share    = float(player.get("assist_share")    or 0.0)
    saves_per_game  = float(player.get("saves_per_game")  or 0.0)
    defcon_hit_rate = float(player.get("defcon_hit_rate") or 0.0)
    card_rate       = float(player.get("card_rate")       or 0.0)
    form            = float(player.get("form")            or 0.0)

    # Form adjustment: base ±0.5 cap, then scaled by form_factor
    form_adj_base = max(-0.5, min(0.5, (form - season_avg) * 0.1)) if season_avg > 0 else 0.0
    form_adj = form_adj_base * form_factor

    fixture_ease = player.get("_gw_ease")
    if fixture_ease is None:
        fixture_ease = 0.5

    is_def = position in ("GKP", "DEF")

    # Clean sheet — computed for every position now (MID gets 1pt per real rules;
    # FWD's _CS_PTS is 0 so this is naturally a no-op for them).
    cs_pts = _CS_PTS[position]
    cs_prob = math.exp(-match_opp_xg)

    # Goals conceded deduction and saves — GKP/DEF only, per FPL rules.
    xgc_pts = -_expected_floor_half_poisson(match_opp_xg) if is_def else 0.0
    save_pts = (saves_per_game / 3) if position == "GKP" else 0.0

    # Attacking returns — same shape for every position (rare for GKP/DEF, primary
    # scoring source for MID/FWD).
    e_goals = match_team_xg * goal_share
    e_assists = match_team_xg * assist_share
    goal_pts = e_goals * _PTS_PER_GOAL[position] * atk_factor
    asst_pts = e_assists * 3 * atk_factor

    # DefCon and cards — outfield players only (defcon_hit_rate is already 0 for
    # GKP; see fpl_client.compute_defcon_hit_rate).
    defcon_pts = 2 * defcon_hit_rate
    card_pts = -card_rate

    # Bonus — fitted regression on this season's own per-match data (bonus_model.py),
    # applied to these same expected-event values.
    bonus_pts = bonus_model.predict_bonus(e_goals, e_assists, cs_prob, saves_per_game, defcon_hit_rate)

    inner = (
        cs_prob * cs_pts * cs_factor
        + xgc_pts
        + save_pts
        + goal_pts
        + asst_pts
        + defcon_pts
        + card_pts
        + bonus_pts
    )
    predicted = (exp_start_pct * 2) + exp_minutes * inner + form_adj

    player_xg = round(e_goals + e_assists, 3)

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
        # Display field stays GKP/DEF-only, matching existing UI column semantics —
        # cs_prob itself is now used for everyone internally (see above).
        "xgc_score": round(cs_prob, 3) if is_def else 0.0,
    }
