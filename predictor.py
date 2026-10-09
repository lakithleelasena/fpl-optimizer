from __future__ import annotations

import math

import bonus_model
from config import (
    LEAGUE_AVG_GOALS,
    PENALTY_AWARD_RATE_PER_MATCH,
    PENALTY_CONVERSION_RATE,
    PENALTY_MISS_PTS,
    PENALTY_SAVE_PTS,
    PENALTY_SAVE_RATE,
    W_ATK_FACTOR,
    W_CS_FACTOR,
    W_FORM_FACTOR,
)
from team_xg_model import expected_floor_div_poisson

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


def _pen_award_rate(team_xg: float) -> float:
    """Expected penalties AWARDED to a team in a match: the league rate scaled by its attack."""
    return PENALTY_AWARD_RATE_PER_MATCH * max(team_xg, 0.0) / LEAGUE_AVG_GOALS


def goal_expectation(
    match_team_xg: float, goal_share: float, pen_taker_prob: float = 0.0, pen_leftover: float = 1.0,
) -> tuple[float, float]:
    """
    Expected goals for one player per full match on the pitch, and the expected missed-penalty
    points (negative). Shared by predict_points and the backtest so they can't drift.

    match_team_xg includes penalties (bookmaker totals and real xG both do), so the expected
    penalty goals are stripped to get open-play xG; goal_share is a NON-penalty share (see
    fpl_client.compute_xg_share). Penalties come back via pen_taker_prob = P(he takes it | on the
    pitch) from FPL's penalties_order and who is fit (fpl_client.compute_pen_taker_chain); the
    leftover (no listed taker on the pitch, or the ~10% a taker doesn't take) is spread over
    on-pitch players by open-play share. With no taker info (q = 0, leftover = 1) this reduces
    to the old match_team_xg * goal_share.
    """
    pen_rate = _pen_award_rate(match_team_xg)
    pen_goals_team = pen_rate * PENALTY_CONVERSION_RATE
    open_xg = max(0.0, match_team_xg - pen_goals_team)
    e_goals = (open_xg + pen_goals_team * pen_leftover) * goal_share + pen_goals_team * pen_taker_prob
    pen_miss_pts = PENALTY_MISS_PTS * pen_rate * (1 - PENALTY_CONVERSION_RATE) * pen_taker_prob
    return e_goals, pen_miss_pts


def goalkeeper_penalty_save_pts(match_opp_xg: float) -> float:
    """Expected penalty-save points per match for a goalkeeper: penalties the OPPONENT is
    awarded (scaled by its attack) x save rate x 5 points."""
    return _pen_award_rate(match_opp_xg) * PENALTY_SAVE_RATE * PENALTY_SAVE_PTS


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
    p_60_plus = float(player.get("p_60_plus") or 0.0)
    p_1_to_59 = float(player.get("p_1_to_59") or 0.0)
    season_avg = stats["season_avg"]

    match_team_xg    = float(player.get("match_team_xg")    or 0.0)
    match_opp_xg     = float(player.get("match_opp_xg")     or 0.0)
    goal_share       = float(player.get("goal_share")       or 0.0)
    assist_share     = float(player.get("assist_share")     or 0.0)
    saves_per_opp_xg = float(player.get("saves_per_opp_xg") or 0.0)
    defcon_hit_rate  = float(player.get("defcon_hit_rate")  or 0.0)
    card_rate        = float(player.get("card_rate")        or 0.0)
    form             = float(player.get("form")             or 0.0)

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

    # Goals conceded deduction — GKP/DEF only, per FPL rules. Scales with continuous
    # exp_minutes (proportional pitch time), not the discrete p_60_plus gate below —
    # GC deduction has no 60-minute threshold, unlike clean sheets.
    xgc_pts = -_expected_floor_half_poisson(match_opp_xg) if is_def else 0.0

    # Saves (Phase 5): opponent-difficulty-adjusted — saves_per_opp_xg (this
    # goalkeeper's historical saves per unit of real opponent xG faced) x this
    # week's actual match_opp_xg, instead of a flat historical per-game average
    # that didn't vary by fixture difficulty at all. E[floor(saves/3)] via the
    # Poisson closed form/truncated sum, not floor(E[saves]/3).
    e_saves = saves_per_opp_xg * match_opp_xg if position == "GKP" else 0.0
    save_pts = expected_floor_div_poisson(e_saves, 3) if position == "GKP" else 0.0

    # Penalty saves (Phase 5): a small GKP-only term — the league-wide penalty award rate
    # (literature value, no per-team data) scaled by the opponent's attack, times the save rate.
    pen_save_pts = goalkeeper_penalty_save_pts(match_opp_xg) if position == "GKP" else 0.0

    # Attacking returns — same shape for every position (rare for GKP/DEF, primary
    # scoring source for MID/FWD).
    pen_taker_prob = float(player.get("pen_taker_prob") or 0.0)
    pen_leftover = player.get("pen_leftover")
    pen_leftover = 1.0 if pen_leftover is None else float(pen_leftover)
    e_goals, pen_miss_pts = goal_expectation(match_team_xg, goal_share, pen_taker_prob, pen_leftover)
    e_assists = match_team_xg * assist_share
    goal_pts = e_goals * _PTS_PER_GOAL[position] * atk_factor
    asst_pts = e_assists * 3 * atk_factor

    # DefCon and cards — outfield players only (defcon_hit_rate is already 0 for
    # GKP; see fpl_client.compute_defcon_hit_rate).
    # DefCon: defcon_hit_rate is P(threshold | plays 60+ minutes), so it is gated by the
    # discrete P(60+) and sits OUTSIDE the exp_minutes-scaled bundle below — minutes are
    # accounted for once (the old per-appearance rate x exp_minutes double-counted them).
    defcon_term = 2 * defcon_hit_rate * p_60_plus
    card_pts = -card_rate

    # Bonus — fitted regression on this season's own per-match data (bonus_model.py),
    # applied to these same expected-event values.
    bonus_pts = bonus_model.predict_bonus(e_goals, e_assists, cs_prob, e_saves, defcon_hit_rate)

    # Everything except clean sheets scales with continuous exp_minutes
    # (proportional pitch-time exposure this match).
    minutes_scaled = xgc_pts + save_pts + pen_save_pts + goal_pts + asst_pts + card_pts + bonus_pts + pen_miss_pts

    # Appearance points (Phase 3): P(1-59 min)*1 + P(60+ min)*2, instead of assuming
    # every "start" is worth a flat 2 points — a player subbed off early, or one who
    # only ever comes on as a substitute, is credited correctly either way.
    appearance_pts = p_1_to_59 * 1 + p_60_plus * 2

    # Clean sheet points (Phase 5): gated by the discrete P(60+) instead of the
    # continuous exp_minutes fraction — FPL's rule is an explicit 60-minute
    # threshold ("not conceding while on the pitch AND playing at least 60
    # minutes"), not a pro-rated-by-minutes credit.
    cs_term = p_60_plus * cs_prob * cs_pts * cs_factor

    predicted = appearance_pts + cs_term + defcon_term + exp_minutes * minutes_scaled + form_adj

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
