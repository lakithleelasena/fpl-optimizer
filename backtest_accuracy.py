"""Accuracy backtests for the LIVE participation-based formula (predictor._predict_points_new),
as opposed to backtest.py which grid-searches the old legacy 7-weight formula.

Two independent stages, matching the pipeline's own two steps:
  1. compute_team_xg_backtest  — Tier 1 (odds) / Tier 2 (rolling) / Tier 3 (FDR) team xG,
     each scored separately against actual goals, using only data available before each GW.
  2. compute_player_points_backtest — full predicted_points vs actual total_points per
     player per GW, built on top of stage 1's reconstructed team xG.

Everything here is read-only and backtest-only — it does not touch fpl_client.py's live
prediction path. Tier 2/Tier 3 use the SAME helper functions as live (fpl_client.tier2_xg /
tier3_xg / tier2_weight), called individually rather than via _model_xg (which only returns
the already-blended result) so each tier can be scored separately — sharing the helpers
means the backtest can't silently drift from what the live app does.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

from config import (
    LEAGUE_AVG_GOALS,
    PENALTY_AWARD_RATE_PER_MATCH,
    PENALTY_SAVE_PTS,
    PENALTY_SAVE_RATE,
    W_ATK_FACTOR,
    W_CS_FACTOR,
    W_FORM_FACTOR,
    W_ODDS_WEIGHT,
    DEFCON_THRESHOLD,
)
import bonus_model
from fpl_client import (
    build_card_priors,
    build_defcon_priors,
    build_minutes_priors,
    build_position_priors,
    build_team_xg_rolling,
    build_team_xg_totals,
    compute_card_rate,
    compute_defcon_hit_rate,
    compute_league_avg_goals,
    compute_mean_fdr,
    compute_minutes_model,
    compute_saves_rate,
    compute_xg_share,
    tier2_weight,
    tier2_xg,
    tier3_xg,
)
from odds_client import load_odds_history
from predictor import _CS_PTS, _PTS_PER_GOAL, _expected_floor_half_poisson
from team_xg_model import expected_floor_div_poisson

FORM_WINDOW = 4  # games used to approximate FPL's own "form" stat (see note in compute_player_points_backtest)


# ─── Stage 1: Team xG accuracy ───────────────────────────────────────────────

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


# ── Scoring + tier-weight fitting ─────────────────────────────────────────────
# MAE rewards predicting a constant (a league-average guess beats every tier on MAE at
# this sample size), so tiers are also scored with Poisson deviance — the natural loss
# for goal counts — plus bias, and compared with the league-average baseline.

def _poisson_deviance(pred: float, y: float) -> float:
    pred = max(pred, 0.05)
    return 2 * ((y * math.log(y / pred) if y > 0 else 0.0) - (y - pred))


def _score(preds: list[float], ys: list[float]) -> dict:
    n = len(ys)
    if n == 0:
        return {"n": 0, "deviance": None, "mae": None, "bias": None}
    return {
        "n": n,
        "deviance": round(sum(_poisson_deviance(p, y) for p, y in zip(preds, ys)) / n, 4),
        "mae": round(sum(abs(p - y) for p, y in zip(preds, ys)) / n, 4),
        "bias": round(sum(p - y for p, y in zip(preds, ys)) / n, 4),
    }


def _tier_summary(rows: list[dict]) -> dict:
    """Each tier scored over the rows where it exists, next to the league-average
    baseline over those SAME rows. skill = 1 - deviance/baseline_deviance (>0 beats
    just guessing the league average)."""
    out = {}
    for key in ("tier1", "tier2", "tier3", "model", "production"):
        sub = [r for r in rows if r.get(key) is not None]
        ys = [r["actual_goals"] for r in sub]
        sc = _score([r[key] for r in sub], ys)
        base = _score([r["league_avg"] for r in sub], ys)
        sc["baseline_deviance"] = base["deviance"]
        sc["skill"] = (round(1 - sc["deviance"] / base["deviance"], 4)
                       if sc["deviance"] is not None and base["deviance"] else None)
        out[key] = sc
    return out


def _dev_for_weights(sub: list[dict], keys: tuple[str, ...], weights: tuple[float, ...]) -> float:
    return sum(
        _poisson_deviance(sum(w * r[k] for k, w in zip(keys, weights)), r["actual_goals"]) for r in sub
    ) / len(sub)


def _bootstrap_best(sub: list[dict], keys: tuple[str, str], grid: list[float], n_boot: int, seed: int) -> dict | None:
    """Resample whole FIXTURES (a fixture's two team-rows share a match, so they're not
    independent) and refit the best weight on the first key each time."""
    by_fix: dict[int, list[dict]] = defaultdict(list)
    for r in sub:
        by_fix[r["fixture_id"]].append(r)
    fids = list(by_fix)
    if len(fids) < 5:
        return None
    rng = random.Random(seed)
    best_ws = []
    for _ in range(n_boot):
        samp = [r for f in rng.choices(fids, k=len(fids)) for r in by_fix[f]]
        best_ws.append(min(grid, key=lambda w: _dev_for_weights(samp, keys, (w, 1 - w))))
    best_ws.sort()
    q = lambda p: best_ws[min(len(best_ws) - 1, int(p * len(best_ws)))]
    return {"median": q(0.5), "p10": q(0.1), "p90": q(0.9), "n_boot": n_boot}


def fit_tier_weights(rows: list[dict], n_boot: int = 300, seed: int = 1) -> dict:
    """
    What blend of the three tiers would have scored best over the backtested gameweeks?
      - tier2_vs_tier3: static Tier 2 weight (rest Tier 3) over rows with both — the
        window with the most data (Tier 2 needs a game of history, so GW2+).
      - three_way: Tier 1/2/3 over rows that have all three (Tier 1 only exists from the
        first gameweek whose odds were archived), plus an odds_weight sweep against the
        production Tier 2/3 model blend — the knob the live slider actually controls.
    Weights are fit in-sample and the samples are small, so each carries a bootstrap range
    and a `reliable` flag; treat low-n results as direction, not truth.
    """
    grid = [i / 20 for i in range(21)]

    # --- Tier 2 vs Tier 3 ---
    a = [r for r in rows if r.get("tier2") is not None]
    t23: dict = {"n": len(a), "n_fixtures": len({r["fixture_id"] for r in a})}
    if a:
        curve = [{"w2": w, "deviance": round(_dev_for_weights(a, ("tier2", "tier3"), (w, 1 - w)), 4)} for w in grid]
        best = min(curve, key=lambda c: c["deviance"])
        ys = [r["actual_goals"] for r in a]
        t23.update({
            "curve": curve,
            "best_w2": best["w2"],
            "best_deviance": best["deviance"],
            "bootstrap": _bootstrap_best(a, ("tier2", "tier3"), grid, n_boot, seed),
            "league_avg_deviance": _score([r["league_avg"] for r in a], ys)["deviance"],
            "production_model_deviance": _score([r["model"] for r in a], ys)["deviance"],
            "reliable": len({r["fixture_id"] for r in a}) >= 30,
        })

    # --- Tier 1 / 2 / 3 ---
    b = [r for r in rows if r.get("tier1") is not None and r.get("tier2") is not None]
    three: dict = {"n": len(b), "n_fixtures": len({r["fixture_id"] for r in b}),
                   "gameweeks": sorted({r["gw"] for r in b})}
    if b:
        ys = [r["actual_goals"] for r in b]
        points = []
        for i in range(21):
            for j in range(21 - i):
                w1, w2 = i / 20, j / 20
                points.append({"w1": w1, "w2": w2, "w3": round(1 - w1 - w2, 2),
                               "deviance": round(_dev_for_weights(b, ("tier1", "tier2", "tier3"), (w1, w2, 1 - w1 - w2)), 4)})
        points.sort(key=lambda p: p["deviance"])
        odds_curve = [{"odds_weight": w, "deviance": round(_dev_for_weights(b, ("tier1", "model"), (w, 1 - w)), 4)} for w in grid]
        best_odds = min(odds_curve, key=lambda c: c["deviance"])
        three.update({
            "top": points[:5],
            "odds_weight_curve": odds_curve,
            "best_odds_weight": best_odds["odds_weight"],
            "bootstrap_odds_weight": _bootstrap_best(b, ("tier1", "model"), grid, n_boot, seed),
            "references": {k: _score([r[k] for r in b], ys) for k in ("tier1", "tier2", "tier3", "model", "league_avg", "production")},
            "reliable": len({r["fixture_id"] for r in b}) >= 30,
        })

    return {"tier2_vs_tier3": t23, "three_way": three}


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

    # Season-long mean FDR (the fixture list is known in advance — no look-ahead).
    mean_fdr = compute_mean_fdr(fixtures)

    rows: list[dict] = []

    for target_gw in completed_gws:
        prior_fixtures = [f for f in finished if f["event"] < target_gw]
        team_rolling = build_team_xg_rolling(prior_fixtures, team_xg_by_fixture)
        defences = [v["defence_xg6"] for v in team_rolling.values() if v["defence_xg6"] > 0]
        league_avg_defence = (sum(defences) / len(defences)) if defences else LEAGUE_AVG_GOALS
        # Live league-average goals as it was known BEFORE this gameweek.
        league_avg = compute_league_avg_goals(prior_fixtures)

        gw_odds = odds_history.get(target_gw, {})
        gw_fixtures = [f for f in finished if f["event"] == target_gw]

        for fix in gw_fixtures:
            fid = fix["id"]
            h_id, a_id = fix["team_h"], fix["team_a"]
            h_fdr = fix.get("team_h_difficulty", 3)
            a_fdr = fix.get("team_a_difficulty", 3)
            actual_h, actual_a = fix["team_h_score"], fix["team_a_score"]

            tier3_h = tier3_xg(h_fdr, league_avg, mean_fdr)
            tier3_a = tier3_xg(a_fdr, league_avg, mean_fdr)

            h_roll = team_rolling.get(h_id)
            a_roll = team_rolling.get(a_id)
            tier2_h = tier2_xg(h_roll["attack_xg6"] if h_roll else None,
                               a_roll["defence_xg6"] if a_roll else None, league_avg_defence, True)
            tier2_a = tier2_xg(a_roll["attack_xg6"] if a_roll else None,
                               h_roll["defence_xg6"] if h_roll else None, league_avg_defence, False)

            # Production-equivalent blend of Tier 2/3 (same helpers as fpl_client._model_xg)
            if tier2_h is not None and tier2_a is not None:
                w_h = tier2_weight(h_roll["games_played"])
                w_a = tier2_weight(a_roll["games_played"])
                model_h = w_h * tier2_h + (1 - w_h) * tier3_h
                model_a = w_a * tier2_a + (1 - w_a) * tier3_a
            else:
                model_h, model_a = tier3_h, tier3_a
            model_h, model_a = max(0.2, model_h), max(0.2, model_a)

            odds_fix = gw_odds.get(fid, {})
            tier1_h = odds_fix.get(h_id, (None, None))[0]
            tier1_a = odds_fix.get(a_id, (None, None))[0]

            prod_h = (odds_weight * tier1_h + (1 - odds_weight) * model_h) if tier1_h is not None else model_h
            prod_a = (odds_weight * tier1_a + (1 - odds_weight) * model_a) if tier1_a is not None else model_a

            def _r(v):
                return round(v, 3) if v is not None else None

            rows.append({
                "gw": target_gw, "fixture_id": fid, "team_id": h_id, "team": teams.get(h_id, "?"),
                "opponent": teams.get(a_id, "?"), "is_home": True, "actual_goals": actual_h,
                "tier1": tier1_h, "tier2": _r(tier2_h), "tier3": _r(tier3_h),
                "model": _r(model_h), "league_avg": _r(league_avg), "production": round(prod_h, 3),
                "error": round(prod_h - actual_h, 3),
            })
            rows.append({
                "gw": target_gw, "fixture_id": fid, "team_id": a_id, "team": teams.get(a_id, "?"),
                "opponent": teams.get(h_id, "?"), "is_home": False, "actual_goals": actual_a,
                "tier1": tier1_a, "tier2": _r(tier2_a), "tier3": _r(tier3_a),
                "model": _r(model_a), "league_avg": _r(league_avg), "production": round(prod_a, 3),
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
        "summary": _tier_summary(rows),
        "weight_fit": fit_tier_weights(rows),
        "rows": rows,
    }


# ─── Component-level scoring ─────────────────────────────────────────────────
# Predicted points split into FPL's scoring components, compared with the same split
# of what the player actually scored. Actuals are rebuilt from per-match stats; "other"
# catches anything we don't predict (red cards, own goals, missed penalties) plus any
# rounding/rule mismatch, so the actual components always sum to total_points.

COMPONENTS = ["appearance", "goals", "assists", "clean_sheet", "goals_conceded",
              "saves", "bonus", "defcon", "cards", "form_adj", "other"]
COMPONENT_LABELS = {
    "appearance": "Appearance (1/2 pts)", "goals": "Goals", "assists": "Assists",
    "clean_sheet": "Clean sheet", "goals_conceded": "Goals conceded (-1 per 2)",
    "saves": "Saves + pen saves", "bonus": "Bonus", "defcon": "DefCon",
    "cards": "Cards", "form_adj": "Form adjustment", "other": "Other / unmodelled",
}


def _actual_components(entry: dict, position: str) -> dict[str, float]:
    mins = entry.get("minutes", 0) or 0
    c = {k: 0.0 for k in COMPONENTS}
    if mins <= 0:
        c["other"] = float(entry.get("total_points", 0))  # e.g. a red card without playing — never happens, kept for safety
        return c
    c["appearance"] = 2.0 if mins >= 60 else 1.0
    c["goals"] = float(entry.get("goals_scored", 0) * _PTS_PER_GOAL[position])
    c["assists"] = float(entry.get("assists", 0) * 3)
    c["clean_sheet"] = float(entry.get("clean_sheets", 0) * _CS_PTS[position])
    if position in ("GKP", "DEF"):
        c["goals_conceded"] = -float((entry.get("goals_conceded", 0) or 0) // 2)
    if position == "GKP":
        c["saves"] = float((entry.get("saves", 0) or 0) // 3 + 5 * (entry.get("penalties_saved", 0) or 0))
    c["bonus"] = float(entry.get("bonus", 0) or 0)
    thr = DEFCON_THRESHOLD.get(position)
    if thr is not None and (entry.get("defensive_contribution", 0) or 0) >= thr:
        c["defcon"] = 2.0
    c["cards"] = -float((entry.get("yellow_cards", 0) or 0) + 3 * (entry.get("red_cards", 0) or 0))
    c["other"] = float(entry.get("total_points", 0)) - sum(c.values())
    return c


def _component_summary(rows: list[dict]) -> dict:
    """Per-component MAE/bias, overall and by position, in two views:
      "played" — only rows where the player had minutes > 0 (what the user asked to see).
         CAUTION: this conditions on an outcome the model is trying to predict. Predictions
         include the chance the player doesn't play, so on this subset appearance, clean
         sheets and everything scaled by exp_minutes look under-predicted by construction.
      "all"    — every player-gameweek the model made a prediction for (incl. 0 minutes);
         the unbiased view for judging calibration.
    baseline_mae = MAE of predicting that component's mean for every row — a component
    whose MAE is not below its baseline isn't adding information."""

    def block(subset: list[dict]) -> dict:
        n = len(subset)
        out = {}
        for comp in COMPONENTS:
            if n == 0:
                continue
            preds = [r["pred_c"][comp] for r in subset]
            acts = [r["act_c"][comp] for r in subset]
            mean_act = sum(acts) / n
            out[comp] = {
                "label": COMPONENT_LABELS[comp],
                "mean_predicted": round(sum(preds) / n, 3),
                "mean_actual": round(mean_act, 3),
                "bias": round(sum(p - a for p, a in zip(preds, acts)) / n, 3),
                "mae": round(sum(abs(p - a) for p, a in zip(preds, acts)) / n, 3),
                "baseline_mae": round(sum(abs(a - mean_act) for a in acts) / n, 3),
            }
        return {"n": n, "components": out}

    def view(subset: list[dict]) -> dict:
        return {
            "ALL": block(subset),
            **{pos: block([r for r in subset if r["position"] == pos]) for pos in ("GKP", "DEF", "MID", "FWD")},
        }

    return {"played": view([r for r in rows if r["minutes"] > 0]), "all": view(rows)}


def _player_components(rows: list[dict]) -> list[dict]:
    """One line per player who played in the window: totals of predicted vs actual
    points by component over the gameweeks they played."""
    by_player: dict[int, dict] = {}
    for r in rows:
        if r["minutes"] <= 0:
            continue
        p = by_player.setdefault(r["player_id"], {
            "player_id": r["player_id"], "name": r["name"], "team": r["team"],
            "position": r["position"], "games": 0, "minutes": 0,
            "predicted": {k: 0.0 for k in COMPONENTS}, "actual": {k: 0.0 for k in COMPONENTS},
        })
        p["games"] += 1
        p["minutes"] += r["minutes"]
        for k in COMPONENTS:
            p["predicted"][k] += r["pred_c"][k]
            p["actual"][k] += r["act_c"][k]
    out = []
    for p in by_player.values():
        p["predicted"] = {k: round(v, 2) for k, v in p["predicted"].items()}
        p["actual"] = {k: round(v, 2) for k, v in p["actual"].items()}
        p["predicted_total"] = round(sum(p["predicted"].values()), 2)
        p["actual_total"] = round(sum(p["actual"].values()), 2)
        out.append(p)
    out.sort(key=lambda p: -p["actual_total"])
    return out


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

    # League DefCon rate by position, per target gameweek from only strictly-earlier games.
    defcon_priors_by_gw = {
        gw: build_defcon_priors([
            (player_meta[pid]["position"], [h for h in hist if h["round"] < gw])
            for pid, hist in raw_histories.items() if pid in player_meta
        ])
        for gw in all_target_gws
    }

    card_priors_by_gw = {
        gw: build_card_priors([
            (player_meta[pid]["position"], [h for h in hist if h["round"] < gw])
            for pid, hist in raw_histories.items() if pid in player_meta
        ])
        for gw in all_target_gws
    }

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
            defcon_hit_rate = compute_defcon_hit_rate(
                prior, position, defcon_priors_by_gw[target_gw].get(position, 0.0),
            )
            card_rate = compute_card_rate(prior, card_priors_by_gw[target_gw].get(position, (0.0, 0.0)))

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

            # DefCon: P(threshold | 60+) gated by P(60+), outside the exp_minutes bundle (see predictor.py).
            defcon_term = 2 * defcon_hit_rate * p_60_plus
            card_pts = -card_rate
            bonus_pts = bonus_model.predict_bonus_with_coeffs(
                bonus_coeffs_by_gw[target_gw], e_goals, e_assists, cs_prob, e_saves, defcon_hit_rate,
            )

            minutes_scaled = xgc_pts + save_pts + pen_save_pts + goal_pts + asst_pts + card_pts + bonus_pts
            appearance_pts = p_1_to_59 * 1 + p_60_plus * 2
            # Clean sheet points (Phase 5) gated by discrete P(60+), not continuous
            # exp_minutes — see predictor.py for the FPL-rule rationale.
            cs_term = p_60_plus * cs_prob * cs_pts * cs_factor
            predicted = appearance_pts + cs_term + defcon_term + exp_minutes * minutes_scaled + form_adj

            predicted = round(max(0.0, predicted), 2)
            actual = entry["total_points"]

            pred_c = {
                "appearance": appearance_pts,
                "goals": exp_minutes * goal_pts,
                "assists": exp_minutes * asst_pts,
                "clean_sheet": cs_term,
                "goals_conceded": exp_minutes * xgc_pts,
                "saves": exp_minutes * (save_pts + pen_save_pts),
                "bonus": exp_minutes * bonus_pts,
                "defcon": defcon_term,
                "cards": exp_minutes * card_pts,
                "form_adj": form_adj,
                "other": 0.0,
            }
            rows.append({
                "gw": target_gw, "player_id": player_id, "name": meta["name"],
                "team": meta["team"], "position": position,
                "predicted": predicted, "actual": actual,
                "error": round(predicted - actual, 2),
                "started": entry["minutes"] >= 60,
                "minutes": entry["minutes"],
                "p_60_plus": round(p_60_plus, 3), "exp_minutes": round(exp_minutes, 3),
                "pred_c": {k: round(v, 3) for k, v in pred_c.items()},
                "act_c": {k: round(v, 3) for k, v in _actual_components(entry, position).items()},
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
        # RMSE / bias reward predictions that are right on average (what the squad optimizer
        # needs); MAE alone favours under-predicting rare events, so read it alongside these.
        "overall_rmse": round(math.sqrt(sum(r["error"] ** 2 for r in rows) / len(rows)), 3) if rows else 0.0,
        "overall_bias": round(sum(r["error"] for r in rows) / len(rows), 3) if rows else 0.0,
        "starters_only_mae": starters_mae,
        "total_predictions": len(rows),
        "skipped_no_prior_data": skipped_no_prior,
        "component_summary": _component_summary(rows),
        "player_components": _player_components(rows),
        "rows": rows,
    }
