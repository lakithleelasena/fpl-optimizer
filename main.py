from __future__ import annotations

import asyncio
import time
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Changes every restart/deploy — forces browser to fetch fresh static files
_BUILD_TS = int(time.time())

from fpl_client import fetch_all_data, invalidate_cache
from odds_client import fetch_odds_xg, load_odds_cache, get_odds_cache_meta
from config import ODDS_API_KEY
from models import (
    ChipRecommendation,
    OptimizeRequest,
    OptimizeResponse,
    PlayerOut,
    SquadPlayer,
    TransferAdviceResponse,
    TransferRequest,
    TransferSuggestion,
)
from backtest_accuracy import compute_player_points_backtest, compute_team_xg_backtest
from config import W_ODDS_WEIGHT
from optimizer import optimize_squad, recommend_transfers
from snapshot import snapshot_status
from predictions import gw1_player as _gw1_player, gwN_player as _gwN_player
from predictor import predict_points

app = FastAPI(title="FPL Squad Optimizer")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    resp = templates.TemplateResponse(
        "index.html", {"request": request, "cache_bust": _BUILD_TS}
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _player_gw_pts(out: dict, upcoming_gws: list) -> list:
    """Compute per-GW predicted points using season_avg and fixture ease."""
    result = []
    for gw_id in upcoming_gws:
        ease = out.get("gw_ease", {}).get(gw_id)
        if ease is None:
            result.append(0.0)
            continue
        n_fix = len(out.get("gw_fixtures", {}).get(gw_id, []))
        if n_fix == 0:
            result.append(0.0)
            continue
        per_match = out.get("season_avg", 0.0) * (0.5 + ease) * out.get("start_likelihood", 0.8)
        result.append(round(per_match * n_fix, 2))
    return result


def _build_player_out(player: dict, prediction: dict) -> dict:
    # Scale per-match prediction by number of fixtures; 0 fixtures → 0 predicted points
    n_fix = player.get("n_fixtures", 0)
    return {
        "id": player["id"],
        "name": player["name"],
        "team": player["team"],
        "team_id": player["team_id"],
        "position": player["position"],
        "cost": player["cost"] / 10,
        "predicted_points": round(prediction["predicted_points"] * n_fix, 2),
        "home_away_score": prediction["home_away_score"],
        "season_avg": prediction["season_avg"],
        "xg_score": prediction["xg_score"],
        "fixture_ease": prediction["fixture_ease"],
        "start_likelihood": prediction["start_likelihood"],
        "exp_minutes": prediction["exp_minutes"],
        "form_score": prediction["form_score"],
        "threat_score": prediction["threat_score"],
        "xgc_score": prediction["xgc_score"],
        "ep_next": round(player.get("ep_next", 0.0), 2),
        "pen_order": player.get("pen_order"),
        "pen_share": round((player.get("exp_minutes") or 0.0) * (player.get("pen_taker_prob") or 0.0), 3)
                     if player.get("pen_order") else None,
        "chance_of_playing": player.get("chance_of_playing"),
        "minutes": player["minutes"],
        "total_points": player["total_points"],
        # Internal fields for chip timing (not in Pydantic model, stripped later)
        "gw_ease": player.get("gw_ease", {}),
        "gw_fixtures": player.get("gw_fixtures", {}),
        "n_fixtures": n_fix,
    }


def _to_player_out(p: dict) -> PlayerOut:
    """Build a PlayerOut from an enriched player dict (cost already in display format)."""
    return PlayerOut(
        id=p["id"],
        name=p["name"],
        team=p["team"],
        team_id=p["team_id"],
        position=p["position"],
        cost=round(p["cost"] / 10, 1),
        predicted_points=p["predicted_points"],
        home_away_score=p["home_away_score"],
        season_avg=p["season_avg"],
        xg_score=p["xg_score"],
        fixture_ease=p["fixture_ease"],
        start_likelihood=p["start_likelihood"],
        exp_minutes=p.get("exp_minutes", 0.0),
        chance_of_playing=p.get("chance_of_playing"),
        minutes=p["minutes"],
        total_points=p["total_points"],
        gw_pts=p.get("gw_pts"),
        form_score=p.get("form_score", 0.0),
        threat_score=p.get("threat_score", 0.0),
        xgc_score=p.get("xgc_score", 0.0),
        ep_next=p.get("ep_next", 0.0),
        pen_order=p.get("pen_order"),
        pen_share=p.get("pen_share"),
    )


def _to_squad_player(p: dict, is_starter: bool) -> SquadPlayer:
    return SquadPlayer(
        id=p["id"],
        name=p["name"],
        team=p["team"],
        team_id=p["team_id"],
        position=p["position"],
        cost=round(p["cost"] / 10, 1),
        predicted_points=p["predicted_points"],
        home_away_score=p["home_away_score"],
        season_avg=p["season_avg"],
        xg_score=p["xg_score"],
        fixture_ease=p["fixture_ease"],
        start_likelihood=p["start_likelihood"],
        exp_minutes=p.get("exp_minutes", 0.0),
        chance_of_playing=p.get("chance_of_playing"),
        minutes=p["minutes"],
        total_points=p["total_points"],
        gw_pts=p.get("gw_pts"),
        form_score=p.get("form_score", 0.0),
        threat_score=p.get("threat_score", 0.0),
        xgc_score=p.get("xgc_score", 0.0),
        ep_next=p.get("ep_next", 0.0),
        pen_order=p.get("pen_order"),
        pen_share=p.get("pen_share"),
        is_starter=is_starter,
    )


@app.get("/api/odds-debug")
async def odds_debug():
    """Diagnostic: show key status and raw Odds API response."""
    import httpx
    from config import ODDS_API_URL
    key_loaded = bool(ODDS_API_KEY)
    key_preview = (ODDS_API_KEY[:6] + "…") if key_loaded else "(empty)"
    if not key_loaded:
        return {"key_loaded": False, "key_preview": key_preview}
    try:
        params = {"apiKey": ODDS_API_KEY, "regions": "uk", "markets": "totals", "oddsFormat": "decimal"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(ODDS_API_URL, params=params)
        return {
            "key_loaded": True,
            "key_preview": key_preview,
            "status_code": resp.status_code,
            "event_count": len(resp.json()) if resp.status_code == 200 else None,
            "error": resp.text if resp.status_code != 200 else None,
            "first_event": resp.json()[0] if resp.status_code == 200 and resp.json() else None,
        }
    except Exception as e:
        return {"key_loaded": True, "key_preview": key_preview, "exception": str(e)}


@app.post("/api/refresh-odds")
async def refresh_odds():
    """Force-fetch fresh odds from The Odds API and save to cache. Invalidates FPL data cache."""
    if not ODDS_API_KEY:
        raise HTTPException(status_code=400, detail="ODDS_API_KEY is not configured")
    data = await fetch_all_data()
    next_gw = data["next_gw"]
    upcoming_fix_list = [f for f in data["fixtures"] if f.get("event") in data["upcoming_gws"]]
    meta_before = get_odds_cache_meta()
    odds = await fetch_odds_xg(
        data["teams"], upcoming_fix_list, current_gw=next_gw, force_refresh=True
    )
    if not odds:
        raise HTTPException(status_code=502, detail="Odds API returned no data — check key or try again later")
    meta = get_odds_cache_meta()
    # If the cache's fetched_at didn't move, the live call failed (e.g. quota
    # exhausted) and fetch_odds_xg served the last saved odds as a fallback —
    # report that honestly instead of claiming a fresh refresh succeeded.
    is_stale = bool(meta_before) and meta_before.get("fetched_at") == (meta or {}).get("fetched_at")
    if not is_stale:
        # Invalidate FPL cache so next prediction fetch uses updated odds
        invalidate_cache()
    return {
        "status": "stale" if is_stale else "ok",
        "gameweek": next_gw,
        "fixtures_found": len(odds),
        "fetched_at": meta["fetched_at"] if meta else None,
    }


@app.get("/api/snapshot/status")
async def get_snapshot_status():
    """Upcoming gameweek, its deadline and the last saved snapshot (see snapshot.py)."""
    data = await fetch_all_data()
    return snapshot_status(data)


@app.post("/api/snapshot")
async def save_snapshot_now():
    """Force a fresh FPL fetch (bypassing the 30-minute cache) and save the upcoming gameweek's
    snapshot, replacing any earlier one for it. Never calls the Odds API — odds are recorded
    as currently cached (only Refresh Odds pulls live odds)."""
    invalidate_cache()
    data = await fetch_all_data(snapshot_trigger="button")
    return snapshot_status(data)


@app.get("/api/next-gw")
async def get_next_gw():
    data = await fetch_all_data()
    return {"next_gw": data["next_gw"], "upcoming_gws": data["upcoming_gws"]}


@app.get("/api/players", response_model=List[PlayerOut])
async def get_players(
    form_factor: float = 1.0,
    cs_factor: float = 1.0,
    atk_factor: float = 1.0,
    odds_weight: float = W_ODDS_WEIGHT,
):
    data = await fetch_all_data()
    result = []
    for p in data["players"]:
        p_gw1 = _gw1_player(p, data["upcoming_gws"], data["team_strengths"], odds_weight)
        pred = predict_points(
            p_gw1, form_factor=form_factor, cs_factor=cs_factor, atk_factor=atk_factor,
        )
        result.append(_build_player_out(p_gw1, pred))
    result.sort(key=lambda x: x["predicted_points"], reverse=True)
    return result


@app.post("/api/optimize", response_model=OptimizeResponse)
async def run_optimize(req: OptimizeRequest):
    data = await fetch_all_data()

    n_gw = max(1, min(req.n_gw, len(data["upcoming_gws"])))  # clamp 1–3

    enriched = []
    for p in data["players"]:
        p_gw1 = _gw1_player(p, data["upcoming_gws"], data["team_strengths"], req.odds_weight)
        pred_gw1 = predict_points(
            p_gw1, form_factor=req.form_factor, cs_factor=req.cs_factor, atk_factor=req.atk_factor,
        )
        out = _build_player_out(p_gw1, pred_gw1)
        out["cost"] = p["cost"]  # keep raw cost for optimizer

        gw_pts = []
        for gw_id in data["upcoming_gws"]:
            p_gwN = _gwN_player(p, gw_id, data["team_strengths"], req.odds_weight)
            pred_gwN = predict_points(
                p_gwN, form_factor=req.form_factor, cs_factor=req.cs_factor, atk_factor=req.atk_factor,
            )
            n_fix = len(p.get("gw_fixtures", {}).get(gw_id, []))
            gw_pts.append(round(pred_gwN["predicted_points"] * n_fix, 2))
        out["gw_pts"] = gw_pts
        # LP objective = sum of first n_gw GWs
        out["predicted_points"] = round(sum(gw_pts[:n_gw]), 2)
        enriched.append(out)

    result = optimize_squad(enriched, budget=req.budget)

    starters = []
    bench = []
    total_cost = 0.0
    total_predicted = 0.0

    for p in result["starters"]:
        p["cost"] = p["cost"] / 10
        total_cost += p["cost"]
        total_predicted += p["predicted_points"]
        starters.append(SquadPlayer(**p))

    for p in result["bench"]:
        p["cost"] = p["cost"] / 10
        total_cost += p["cost"]
        bench.append(SquadPlayer(**p))

    pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    starters.sort(key=lambda s: pos_order.get(s.position, 99))
    bench.sort(key=lambda s: pos_order.get(s.position, 99))

    # Captain = highest predicted points among starters (exclude GKP)
    eligible = sorted(
        [s for s in starters if s.position != "GKP"],
        key=lambda s: s.predicted_points,
        reverse=True,
    )
    captain_id = eligible[0].id if len(eligible) > 0 else None
    vice_captain_id = eligible[1].id if len(eligible) > 1 else None

    # Per-GW XI totals for the summary banner
    gw_totals = [
        round(sum((s.gw_pts[i] if s.gw_pts and i < len(s.gw_pts) else 0) for s in starters), 2)
        for i in range(n_gw)
    ]

    return OptimizeResponse(
        starters=starters,
        bench=bench,
        total_cost=round(total_cost, 1),
        total_predicted_points=round(total_predicted, 2),
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        gw_totals=gw_totals,
        upcoming_gws=data["upcoming_gws"][:n_gw],
        n_gw=n_gw,
    )


@app.post("/api/transfer-advice", response_model=TransferAdviceResponse)
async def get_transfer_advice(req: TransferRequest):
    if len(req.current_team) != 15:
        raise HTTPException(status_code=400, detail=f"Squad must have exactly 15 players, got {len(req.current_team)}")

    data = await fetch_all_data()

    # Build enriched players with per-GW fixture context for accurate scores.
    # LP objective = total across all upcoming GWs.
    enriched = []
    for p in data["players"]:
        p_gw1 = _gw1_player(p, data["upcoming_gws"], data["team_strengths"], req.odds_weight)
        pred_gw1 = predict_points(
            p_gw1, form_factor=req.form_factor, cs_factor=req.cs_factor, atk_factor=req.atk_factor,
        )
        out = _build_player_out(p_gw1, pred_gw1)
        out["cost"] = p["cost"]

        gw_pts = []
        for gw_id in data["upcoming_gws"]:
            p_gwN = _gwN_player(p, gw_id, data["team_strengths"], req.odds_weight)
            pred_gwN = predict_points(
                p_gwN, form_factor=req.form_factor, cs_factor=req.cs_factor, atk_factor=req.atk_factor,
            )
            n_fix = len(p.get("gw_fixtures", {}).get(gw_id, []))
            gw_pts.append(round(pred_gwN["predicted_points"] * n_fix, 2))
        out["gw_pts"] = gw_pts
        out["predicted_points"] = round(sum(gw_pts[:req.n_gw]), 2)  # LP objective = n_gw total
        enriched.append(out)

    result = recommend_transfers(
        current_team_ids=req.current_team,
        all_players=enriched,
        free_transfers=req.free_transfers,
        budget_in_bank=req.budget_in_bank,
        chips_available=req.chips_available,
        upcoming_gws=data["upcoming_gws"],
        n_gw=req.n_gw,
    )

    # Convert to response models (cost → display format)
    starters = [_to_squad_player(p, True) for p in result["current_xi"]["starters"]]
    bench = [_to_squad_player(p, False) for p in result["current_xi"]["bench"]]

    transfers = []
    for t in result["transfers"]:
        transfers.append(TransferSuggestion(
            transfer_out=_to_player_out(t["transfer_out"]),
            transfer_in=_to_player_out(t["transfer_in"]),
            points_gain=round(t["points_gain"], 2),
            is_hit=t["is_hit"],
        ))

    captain_id = result["captain"]["id"] if result["captain"] else None
    vice_captain_id = result["vice_captain"]["id"] if result["vice_captain"] else None

    total_predicted_3gw = round(
        sum(p["predicted_points"] for p in result["current_xi"]["starters"]), 2
    )

    return TransferAdviceResponse(
        starters=starters,
        bench=bench,
        transfers=transfers,
        chip_recommendation=ChipRecommendation(**result["chip_recommendation"]),
        hits_required=result["hits_required"],
        net_points_gain=result["net_points_gain"],
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        total_predicted_3gw=total_predicted_3gw,
        n_gw=req.n_gw,
        upcoming_gws=data["upcoming_gws"][:req.n_gw],
    )


@app.get("/api/backtest/team-xg")
async def run_team_xg_backtest():
    """Tier 1 (odds) / Tier 2 (rolling) / Tier 3 (FDR) team xG, scored separately
    against actual goals for every completed gameweek. Tier 1 will be sparse/empty
    for gameweeks before odds history archiving started."""
    data = await fetch_all_data()
    result = await asyncio.to_thread(
        compute_team_xg_backtest, data["fixtures"], data["teams"], data["raw_histories"],
    )
    return result


@app.get("/api/backtest/player-points")
async def run_player_points_backtest(share_window: int = 6):
    """Full predicted_points vs actual total_points per player per completed GW,
    built on top of the team-xG backtest above. share_window: 1, 3, or 6 games."""
    if share_window not in (1, 3, 6):
        raise HTTPException(status_code=400, detail="share_window must be 1, 3, or 6")
    data = await fetch_all_data()
    player_meta = {p["id"]: p for p in data["players"]}

    def _run():
        team_result = compute_team_xg_backtest(data["fixtures"], data["teams"], data["raw_histories"])
        return compute_player_points_backtest(
            data["raw_histories"], data["fixtures"], player_meta,
            data["player_history_past"], team_result["rows"], share_window=share_window,
        )

    result = await asyncio.to_thread(_run)
    return result


@app.get("/api/teams")
async def get_teams():
    """Team overview + fixture tracker data for the next 8 GWs."""
    data = await fetch_all_data()
    bootstrap_teams = data["bootstrap_teams"]
    fixtures = data["fixtures"]
    next_gw = data["next_gw"]
    teams_short = data.get("teams_short", {})
    team_rolling = data.get("team_rolling", {})

    # Next 8 upcoming (non-finished) GWs
    all_events = sorted(set(f["event"] for f in fixtures if f.get("event")))
    tracker_gws = [gw for gw in all_events if gw >= next_gw][:8]

    team_map = {t["id"]: t for t in bootstrap_teams}

    # Goals for/against from finished fixtures
    goals_for: dict[int, int] = {}
    goals_against: dict[int, int] = {}
    for fix in fixtures:
        if fix.get("finished") and fix.get("team_h_score") is not None:
            h, a = fix["team_h"], fix["team_a"]
            hs, as_ = fix["team_h_score"], fix["team_a_score"]
            goals_for[h] = goals_for.get(h, 0) + hs
            goals_for[a] = goals_for.get(a, 0) + as_
            goals_against[h] = goals_against.get(h, 0) + as_
            goals_against[a] = goals_against.get(a, 0) + hs

    # Per-team upcoming fixtures for the tracker GWs
    # {team_id: {gw_id: [{opp_id, opp_short, is_home, fdr}]}}
    team_fixtures: dict[int, dict[int, list[dict]]] = {t["id"]: {} for t in bootstrap_teams}
    for fix in fixtures:
        gw = fix.get("event")
        if gw not in tracker_gws:
            continue
        h, a = fix["team_h"], fix["team_a"]
        h_fdr = fix.get("team_h_difficulty", 3)
        a_fdr = fix.get("team_a_difficulty", 3)
        team_fixtures[h].setdefault(gw, []).append({
            "opp_id": a,
            "opp_short": teams_short.get(a, "?"),
            "is_home": True,
            "fdr": h_fdr,
        })
        team_fixtures[a].setdefault(gw, []).append({
            "opp_id": h,
            "opp_short": teams_short.get(h, "?"),
            "is_home": False,
            "fdr": a_fdr,
        })

    # Odds API xG for the next GW (from file cache)
    odds_xg_cache = load_odds_cache(next_gw)
    next_gw_team_odds: dict[int, dict] = {}
    for fix in fixtures:
        fid = fix.get("id")
        if fix.get("event") != next_gw or fid not in odds_xg_cache:
            continue
        h_id, a_id = fix["team_h"], fix["team_a"]
        fix_odds = odds_xg_cache[fid]
        if h_id in fix_odds:
            vals = fix_odds[h_id]
            next_gw_team_odds[h_id] = {"team_xg": vals[0], "opp_xg": vals[1]}
        if a_id in fix_odds:
            vals = fix_odds[a_id]
            next_gw_team_odds[a_id] = {"team_xg": vals[0], "opp_xg": vals[1]}

    odds_meta = get_odds_cache_meta()
    # Odds are only valid for the gameweek they were pulled for — if the cache is last
    # gameweek's, report "none yet for this gameweek" (blank), not last gameweek's odds.
    odds_current = bool(odds_meta) and odds_meta["gameweek"] == next_gw
    odds_status = {
        "has_key": bool(ODDS_API_KEY),
        "gameweek": odds_meta["gameweek"] if odds_current else None,
        "fetched_at": odds_meta["fetched_at"] if odds_current else None,
        "fixture_count": odds_meta["fixture_count"] if odds_current else 0,
        "next_gw": next_gw,
        "last_cached_gameweek": odds_meta["gameweek"] if odds_meta and not odds_current else None,
    }

    result = []
    for t in bootstrap_teams:
        tid = t["id"]
        gf = goals_for.get(tid, 0)
        ga = goals_against.get(tid, 0)
        upcoming = []
        for gw in tracker_gws:
            matches = team_fixtures[tid].get(gw, [])
            if matches:
                upcoming.append({"gw": gw, "matches": matches})
            else:
                upcoming.append({"gw": gw, "matches": []})  # blank GW
        t_roll = team_rolling.get(tid, {})
        team_odds = next_gw_team_odds.get(tid, {})
        result.append({
            "id": tid,
            "name": t["name"],
            "short_name": t["short_name"],
            "position": t.get("position", 0),
            "played": t.get("played", 0),
            "won": t.get("win", 0),
            "drawn": t.get("draw", 0),
            "lost": t.get("loss", 0),
            "points": t.get("points", 0),
            "goals_for": gf,
            "goals_against": ga,
            "goal_diff": gf - ga,
            "strength_home": t.get("strength_overall_home", 3),
            "strength_away": t.get("strength_overall_away", 3),
            "attack_xg6": t_roll.get("attack_xg6", 0.0),
            "defence_xg6": t_roll.get("defence_xg6", 0.0),
            "odds_team_xg": team_odds.get("team_xg"),
            "odds_opp_xg": team_odds.get("opp_xg"),
            "upcoming": upcoming,
        })

    # Sort by league position (0 = not set yet, put at end), then name
    result.sort(key=lambda t: (t["position"] or 99, t["name"]))

    return {"teams": result, "gws": tracker_gws, "odds_status": odds_status}
