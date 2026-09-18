"""Accuracy backtests for the LIVE participation-based formula (predictor._predict_points_new),
as opposed to backtest.py which grid-searches the old legacy 7-weight formula.

Two independent stages, matching the pipeline's own two steps:
  1. compute_team_xg_backtest  — Tier 1 (odds) / Tier 2 (rolling) / Tier 3 (FDR) team xG,
     each scored separately against actual goals, using only data available before each GW.
  2. compute_player_points_backtest — full predicted_points vs actual total_points per
     player per GW, built on top of stage 1's reconstructed team xG.

Everything here is read-only and backtest-only — it does not touch fpl_client.py's live
prediction path. Tier 2/Tier 3 formulas are intentionally re-implemented (small, ~5 lines
each) rather than imported from fpl_client._model_xg, because that function only returns
the already-taper-blended result — the whole point here is to see the tiers separately.
Keep these in sync with fpl_client._model_xg / _build_team_rolling if those change.
"""
from __future__ import annotations

import math
from collections import defaultdict

from config import (
    AWAY_ADV_MULT,
    HOME_ADV_MULT,
    LEAGUE_AVG_GOALS,
    PENALTY_AWARD_RATE_PER_MATCH,
    PENALTY_SAVE_PTS,
    PENALTY_SAVE_RATE,
    TAPER_GAMES,
    W_ATK_FACTOR,
    W_CS_FACTOR,
    W_FORM_FACTOR,
    W_ODDS_WEIGHT,
)
import bonus_model
from fpl_client import (
    build_minutes_priors,
    build_position_priors,
    build_team_xg_rolling,
    build_team_xg_totals,
    compute_card_rate,
    compute_defcon_hit_rate,
    compute_minutes_model,
    compute_saves_rate,
    compute_xg_share,
)
from odds_client import load_odds_history
from predictor import _CS_PTS, _PTS_PER_GOAL, _expected_floor_half_poisson
from team_xg_model import expected_floor_div_poisson

FORM_WINDOW = 4  # games used to approximate FPL's own "form" stat (see note in compute_player_points_backtest)


# ─── Stage 1: Team xG accuracy ───────────────────────────────────────────────

def _tier3_xg(fdr: int) -> float:
    """FDR-based fallback — identical formula to fpl_client._model_xg's Tier 3."""
    mult = max(0.2, (5 - fdr) / 3)
    return round(LEAGUE_AVG_GOALS * mult, 3)


def _tier2_xg(attack_xg6: float | None, opp_defence_xg6: float | None,
              league_avg_defence: float, is_home: bool) -> float | None:
    """Pure rolling-form estimate, un-tapered — None if either team has no rolling data yet.
    A genuine 0.0 average (team blanked/conceded-none in its games so far) is valid data
    and must NOT be treated the same as "no data" — hence explicit None checks, not truthiness."""
    if attack_xg6 is None or opp_defence_xg6 is None or league_avg_defence <= 0:
        return None
    mult = HOME_ADV_MULT if is_home else AWAY_ADV_MULT
    return round(attack_xg6 * (opp_defence_xg6 / league_avg_defence) * mult, 3)


def _mae_per_gw(rows: list[dict], key: str) -> dict[int, float]:
    per_gw: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        val = r.get(key)
        if val is not None:
            per_gw[r["gw"]].append(abs(val - r["actual_goals"]))
    return {gw: round(sum(errs) / len(errs), 3) for gw, errs in per_gw.items()}


def _sample_count_per_gw(rows: list[dict], key: str) -> dict[int, int]:
    per_gw: dict[int, int] = defaultdict(int)
    for r in rows:
        if r.get(key) is not None:
            per_gw[r["gw"]] += 1
    return dict(per_gw)


def compute_team_xg_backtest(
    fixtures: list[dict],
    teams: dict[int, str],
    raw_histories: dict[int, list[dict]],
    odds_weight: float = W_ODDS_WEIGHT,
) -> dict:
    """
    For each completed gameweek, reconstruct what each tier would have predicted for
    every team using ONLY fixtures finished before that gameweek, then score against
    the actual goals scored. Tier 2's rolling window grows (0..6 games) exactly as
    fpl_client.build_team_xg_rolling already behaves when given a limited fixture
    list. Tier 2 uses real xG (Phase 4, PREDICTION_MODEL_PLAN.md) — summed from
    every player's own FPL-reported xG per fixture — matching the live pipeline
    exactly, not actual goals scored/conceded.
    """
    finished = [f for f in fixtures if f.get("finished") and f.get("team_h_score") is not None]
    completed_gws = sorted({f["event"] for f in finished if f.get("event")})
    odds_history = load_odds_history()

    # Built once globally from the full (not gameweek-truncated) history — safe,
    # see build_team_xg_totals docstring: a lookup only ever hits fixtures already
    # known to be prior to whatever target gameweek is being scored below.
    team_xg_by_fixture, _ = build_team_xg_totals(
        [(hist[0].get("team_id") if hist else None, hist) for hist in raw_histories.values()]
    )

    rows: list[dict] = []

    for target_gw in completed_gws:
        prior_fixtures = [f for f in finished if f["event"] < target_gw]
        team_rolling = build_team_xg_rolling(prior_fixtures, team_xg_by_fixture)
        defences = [v["defence_xg6"] for v in team_rolling.values() if v["defence_xg6"] > 0]
        league_avg_defence = (sum(defences) / len(defences)) if defences else LEAGUE_AVG_GOALS

        gw_odds = odds_history.get(target_gw, {})
        gw_fixtures = [f for f in finished if f["event"] == target_gw]

        for fix in gw_fixtures:
            fid = fix["id"]
            h_id, a_id = fix["team_h"], fix["team_a"]
            h_fdr = fix.get("team_h_difficulty", 3)
            a_fdr = fix.get("team_a_difficulty", 3)
            actual_h, actual_a = fix["team_h_score"], fix["team_a_score"]

            tier3_h, tier3_a = _tier3_xg(h_fdr), _tier3_xg(a_fdr)

            h_roll = team_rolling.get(h_id)
            a_roll = team_rolling.get(a_id)
            tier2_h = _tier2_xg(h_roll["attack_xg6"] if h_roll else None,
                                 a_roll["defence_xg6"] if a_roll else None, league_avg_defence, True)
            tier2_a = _tier2_xg(a_roll["attack_xg6"] if a_roll else None,
                                 h_roll["defence_xg6"] if h_roll else None, league_avg_defence, False)

            # Production-equivalent taper blend of Tier 2/3 (mirrors fpl_client._model_xg)
            if tier2_h is not None and tier2_a is not None:
                w_h = min(1.0, h_roll["games_played"] / TAPER_GAMES)
                w_a = min(1.0, a_roll["games_played"] / TAPER_GAMES)
                model_h = w_h * tier2_h + (1 - w_h) * tier3_h
                model_a = w_a * tier2_a + (1 - w_a) * tier3_a
            else:
                model_h, model_a = tier3_h, tier3_a

            odds_fix = gw_odds.get(fid, {})
            tier1_h = odds_fix.get(h_id, (None, None))[0]
            tier1_a = odds_fix.get(a_id, (None, None))[0]

            prod_h = (odds_weight * tier1_h + (1 - odds_weight) * model_h) if tier1_h is not None else model_h
            prod_a = (odds_weight * tier1_a + (1 - odds_weight) * model_a) if tier1_a is not None else model_a

            rows.append({
                "gw": target_gw, "fixture_id": fid, "team_id": h_id, "team": teams.get(h_id, "?"),
                "opponent": teams.get(a_id, "?"), "is_home": True, "actual_goals": actual_h,
                "tier1": tier1_h, "tier2": tier2_h, "tier3": tier3_h, "production": round(prod_h, 3),
                "error": round(prod_h - actual_h, 3),
            })
            rows.append({
                "gw": target_gw, "fixture_id": fid, "team_id": a_id, "team": teams.get(a_id, "?"),
                "opponent": teams.get(h_id, "?"), "is_home": False, "actual_goals": actual_a,
                "tier1": tier1_a, "tier2": tier2_a, "tier3": tier3_a, "production": round(prod_a, 3),
                "error": round(prod_a - actual_a, 3),
            })

    return {
        "gameweeks": completed_gws,
        "mae_by_tier": {
            "tier1": _mae_per_gw(rows, "tier1"),
            "tier2": _mae_per_gw(rows, "tier2"),
            "tier3": _mae_per_gw(rows, "tier3"),
            "production": _mae_per_gw(rows, "production"),
        },
        "sample_counts": {
            "tier1": _sample_count_per_gw(rows, "tier1"),
            "tier2": _sample_count_per_gw(rows, "tier2"),
            "tier3": _sample_count_per_gw(rows, "tier3"),
        },
        "total_rows": len(rows),
        "rows": rows,
    }


# ─── Stage 2: Player points accuracy ─────────────────────────────────────────

def _team_games_before(fixtures: list[dict], team_id: int, target_gw: int) -> int:
    return sum(
        1 for f in fixtures
        if f.get("finished") and f.get("team_h_score") is not None and f.get("event", 0) < target_gw
        and (f.get("team_h") == team_id or f.get("team_a") == team_id)
    )


def compute_player_points_backtest(
    raw_histories: dict[int, list[dict]],
    fixtures: list[dict],
    player_meta: dict[int, dict],
    player_history_past: dict[int, list[dict]],
    team_backtest_rows: list[dict],
    share_window: int = 6,
    form_factor: float = W_FORM_FACTOR,
    cs_factor: float = W_CS_FACTOR,
    atk_factor: float = W_ATK_FACTOR,
) -> dict:
    """
    For each player, for each completed GW they have a history entry for, reconstruct
    predicted_points using only data available before that GW, with the goal/assist
    share window capped at `share_window` games (1, 3, or 6 — grows if fewer are
    available, same discipline as the team xG backtest). Mirrors the live formula's
    Phase 1 xG-share + shrinkage-toward-prior (see compute_xg_share in fpl_client.py)
    rather than the old actual-goals share, so this backtest stays a faithful test of
    what the live app actually does.

    Uses team_backtest_rows (from compute_team_xg_backtest) as the source of
    match_team_xg / match_opp_xg, so this stage is a true "given our team-xG
    pipeline's actual output, how good is the player-attribution layer" test.

    NOTE (approximation): FPL's own "form" stat isn't retrievable historically (we
    only ever see its current value, same limitation as odds before this session's
    archiving fix). This substitutes a trailing-{FORM_WINDOW}-game points average
    computed from real history as a stand-in for "recent form" in the form_adj term.
    Players with zero prior appearances this season are skipped (no basis to predict —
    this backtest doesn't attempt to reproduce the pre-season last-season fallback).
    """
    team_xg_lookup = {(r["gw"], r["team_id"]): r["production"] for r in team_backtest_rows}

    # Same xG-share aggregation as the live pipeline, built once from the full
    # (not gameweek-truncated) history — safe, see build_team_xg_totals docstring.
    team_xg_by_fixture, team_xa_by_fixture = build_team_xg_totals(
        [(hist[0].get("team_id") if hist else None, hist) for hist in raw_histories.values()]
    )
    position_priors = build_position_priors(
        [(player_meta[pid]["position"], player_history_past.get(pid))
         for pid in raw_histories if pid in player_meta]
    )
    minutes_priors = build_minutes_priors(
        [(player_meta[pid]["position"], player_history_past.get(pid))
         for pid in raw_histories if pid in player_meta]
    )

    # Bonus model coefficients, refit PER TARGET GAMEWEEK from only history strictly
    # before that gameweek — unlike the live pipeline's single shared cache, the
    # backtest must never let a later gameweek's bonus data leak into an earlier
    # gameweek's prediction. Cheap: a handful of features, at most a few thousand rows.
    all_target_gws = sorted({h["round"] for hist in raw_histories.values() for h in hist})
    bonus_coeffs_by_gw: dict[int, object] = {}
    for target_gw in all_target_gws:
        entries = []
        for pid, hist in raw_histories.items():
            meta_p = player_meta.get(pid)
            if not meta_p:
                continue
            prior_hist = [h for h in hist if h["round"] < target_gw]
            if prior_hist:
                entries.append((meta_p["position"], prior_hist))
        bonus_coeffs_by_gw[target_gw] = bonus_model.fit(entries)

    rows: list[dict] = []
    skipped_no_prior = 0

    for player_id, history in raw_histories.items():
        meta = player_meta.get(player_id)
        if not meta:
            continue
        position = meta["position"]
        team_id = meta["team_id"]
        history_past = player_history_past.get(player_id)
        prior_position = position_priors.get(position, (0.0, 0.0))
        minutes_prior = minutes_priors.get(position, (0.0, 0.0))
        history = sorted(history, key=lambda h: h["round"])

        for i, entry in enumerate(history):
            target_gw = entry["round"]
            if entry.get("minutes") is None:
                continue

            prior = history[:i]
            played_prior = [h for h in prior if h["minutes"] > 0]
            if not played_prior:
                skipped_no_prior += 1
                continue

            # Minutes model (Phase 3) — same Beta-prior-blended P(60+)/P(1-59) split
            # as the live pipeline. NOTE (approximation): historical
            # chance_of_playing isn't retrievable (same limitation as "form" below),
            # so availability is fixed at 1.0 here — the backtest can't reproduce a
            # past injury doubt, only the live app's current-moment view of one.
            team_games_before = _team_games_before(fixtures, team_id, target_gw)
            if team_games_before <= 0:
                continue
            exp_start_pct, exp_minutes, p_60_plus, p_1_to_59 = compute_minutes_model(
                prior, history_past, team_games_before, 1.0, minutes_prior,
            )

            # xG-based share, shrunk toward last-season-at-club (or position-average)
            # prior — see compute_xg_share. Pass the FULL prior history (n90 needs
            # every game played this season, not just the share window); share_window
            # only controls the "last N played games" share numerator/denominator.
            goal_share, assist_share = compute_xg_share(
                prior, history_past, team_id, team_xg_by_fixture, team_xa_by_fixture,
                prior_position, window=share_window,
            )

            # Season avg (all prior played games) and form proxy (trailing FORM_WINDOW games)
            season_avg = sum(h["total_points"] for h in played_prior) / len(played_prior)
            form_window = played_prior[-FORM_WINDOW:]
            form_proxy = sum(h["total_points"] for h in form_window) / len(form_window)

            # Team xG from Stage 1's reconstruction
            opponent_team = entry.get("opponent_team")
            match_team_xg = team_xg_lookup.get((target_gw, team_id))
            match_opp_xg = team_xg_lookup.get((target_gw, opponent_team))
            if match_team_xg is None or match_opp_xg is None:
                continue

            form_adj = max(-0.5, min(0.5, (form_proxy - season_avg) * 0.1)) * form_factor
            is_def = position in ("GKP", "DEF")

            # DefCon + cards (Phase 2) — same empirical hit-rate as the live pipeline,
            # computed from prior (not gameweek-truncated-to-share_window) history.
            defcon_hit_rate = compute_defcon_hit_rate(prior, position)
            card_rate = compute_card_rate(prior)

            cs_pts = _CS_PTS[position]
            cs_prob = math.exp(-match_opp_xg)
            xgc_pts = -_expected_floor_half_poisson(match_opp_xg) if is_def else 0.0

            # Saves (Phase 5) — opponent-difficulty-adjusted, same as the live pipeline.
            saves_per_opp_xg = compute_saves_rate(prior, team_xg_by_fixture) if position == "GKP" else 0.0
            e_saves = saves_per_opp_xg * match_opp_xg if position == "GKP" else 0.0
            save_pts = expected_floor_div_poisson(e_saves, 3) if position == "GKP" else 0.0
            pen_save_pts = (
                PENALTY_AWARD_RATE_PER_MATCH * PENALTY_SAVE_RATE * PENALTY_SAVE_PTS
                if position == "GKP" else 0.0
            )

            e_goals = match_team_xg * goal_share
            e_assists = match_team_xg * assist_share
            goal_pts = e_goals * _PTS_PER_GOAL[position] * atk_factor
            asst_pts = e_assists * 3 * atk_factor

            defcon_pts = 2 * defcon_hit_rate
            card_pts = -card_rate
            bonus_pts = bonus_model.predict_bonus_with_coeffs(
                bonus_coeffs_by_gw[target_gw], e_goals, e_assists, cs_prob, e_saves, defcon_hit_rate,
            )

            minutes_scaled = xgc_pts + save_pts + pen_save_pts + goal_pts + asst_pts + defcon_pts + card_pts + bonus_pts
            appearance_pts = p_1_to_59 * 1 + p_60_plus * 2
            # Clean sheet points (Phase 5) gated by discrete P(60+), not continuous
            # exp_minutes — see predictor.py for the FPL-rule rationale.
            cs_term = p_60_plus * cs_prob * cs_pts * cs_factor
            predicted = appearance_pts + cs_term + exp_minutes * minutes_scaled + form_adj

            predicted = round(max(0.0, predicted), 2)
            actual = entry["total_points"]

            rows.append({
                "gw": target_gw, "player_id": player_id, "name": meta["name"],
                "team": meta["team"], "position": position,
                "predicted": predicted, "actual": actual,
                "error": round(predicted - actual, 2),
                "started": entry["minutes"] >= 60,
                "minutes": entry["minutes"],
            })

    gws = sorted({r["gw"] for r in rows})
    positions = ["GKP", "DEF", "MID", "FWD"]

    def mae_for(subset: list[dict]) -> float:
        return round(sum(abs(r["error"]) for r in subset) / len(subset), 3) if subset else 0.0

    mae_by_position_per_gw = {
        pos: {gw: mae_for([r for r in rows if r["gw"] == gw and r["position"] == pos]) for gw in gws}
        for pos in positions
    }
    overall_mae = mae_for(rows)
    starters_only = [r for r in rows if r["started"]]
    starters_mae = mae_for(starters_only)

    return {
        "gameweeks": gws,
        "share_window": share_window,
        "mae_by_position_per_gw": mae_by_position_per_gw,
        "overall_mae": overall_mae,
        "starters_only_mae": starters_mae,
        "total_predictions": len(rows),
        "skipped_no_prior_data": skipped_no_prior,
        "rows": rows,
    }
