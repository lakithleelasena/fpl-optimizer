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

from fpl_client import fetch_all_data
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
from backtest import compute_backtest
from config import W_FIXTURE, W_FORM, W_HOME_AWAY, W_SEASON, W_THREAT, W_XGC, W_XGI
from optimizer import optimize_squad, recommend_transfers
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


def _gw1_player(player: dict, upcoming_gws: list, team_strengths: dict) -> dict:
    """Return a copy of player with opponents/strengths/n_fixtures restricted to GW1 only."""
    return _gwN_player(player, upcoming_gws[0] if upcoming_gws else None, team_strengths)


def _gwN_player(player: dict, gw_id, team_strengths: dict) -> dict:
    """Return a copy of player with fixture context set for a specific GW."""
    if gw_id is None:
        return player
    opps = player["gw_fixtures"].get(gw_id, [])
    strengths = [team_strengths.get(o, 0) for o in opps]
    is_home = player.get("gw_home", {}).get(gw_id, 0.5)
    return {**player, "opponents": opps, "opponent_strengths": strengths, "n_fixtures": len(opps), "is_home": is_home}


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
        "form_score": prediction["form_score"],
        "threat_score": prediction["threat_score"],
        "xgc_score": prediction["xgc_score"],
        "ep_next": round(player.get("ep_next", 0.0), 2),
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
        chance_of_playing=p.get("chance_of_playing"),
        minutes=p["minutes"],
        total_points=p["total_points"],
        gw_pts=p.get("gw_pts"),
        form_score=p.get("form_score", 0.0),
        threat_score=p.get("threat_score", 0.0),
        xgc_score=p.get("xgc_score", 0.0),
        ep_next=p.get("ep_next", 0.0),
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
        chance_of_playing=p.get("chance_of_playing"),
        minutes=p["minutes"],
        total_points=p["total_points"],
        gw_pts=p.get("gw_pts"),
        form_score=p.get("form_score", 0.0),
        threat_score=p.get("threat_score", 0.0),
        xgc_score=p.get("xgc_score", 0.0),
        ep_next=p.get("ep_next", 0.0),
        is_starter=is_starter,
    )


@app.get("/api/next-gw")
async def get_next_gw():
    data = await fetch_all_data()
    return {"next_gw": data["next_gw"], "upcoming_gws": data["upcoming_gws"]}


@app.get("/api/players", response_model=List[PlayerOut])
async def get_players(
    w_home_away: float = W_HOME_AWAY,
    w_season: float = W_SEASON,
    w_xgi: float = W_XGI,
    w_fixture: float = W_FIXTURE,
    w_form: float = W_FORM,
    w_threat: float = W_THREAT,
    w_xgc: float = W_XGC,
):
    data = await fetch_all_data()
    result = []
    for p in data["players"]:
        p_gw1 = _gw1_player(p, data["upcoming_gws"], data["team_strengths"])
        pred = predict_points(p_gw1, w_home_away, w_season, w_xgi, w_fixture, w_form, w_threat, w_xgc)
        result.append(_build_player_out(p_gw1, pred))
    result.sort(key=lambda x: x["predicted_points"], reverse=True)
    return result


@app.post("/api/optimize", response_model=OptimizeResponse)
async def run_optimize(req: OptimizeRequest):
    data = await fetch_all_data()

    n_gw = max(1, min(req.n_gw, len(data["upcoming_gws"])))  # clamp 1–3

    enriched = []
    for p in data["players"]:
        p_gw1 = _gw1_player(p, data["upcoming_gws"], data["team_strengths"])
        pred_gw1 = predict_points(p_gw1, req.w_home_away, req.w_season, req.w_xgi, req.w_fixture, req.w_form, req.w_threat, req.w_xgc)
        out = _build_player_out(p_gw1, pred_gw1)
        out["cost"] = p["cost"]  # keep raw cost for optimizer

        # Per-GW breakdown: run predict_points with each GW's own fixture context
        # so home/away and fixture difficulty reflect the actual upcoming opponent
        gw_pts = []
        for gw_id in data["upcoming_gws"]:
            p_gwN = _gwN_player(p, gw_id, data["team_strengths"])
            pred_gwN = predict_points(p_gwN, req.w_home_away, req.w_season, req.w_xgi, req.w_fixture, req.w_form, req.w_threat, req.w_xgc)
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
        p_gw1 = _gw1_player(p, data["upcoming_gws"], data["team_strengths"])
        pred_gw1 = predict_points(p_gw1, req.w_home_away, req.w_season, req.w_xgi, req.w_fixture, req.w_form, req.w_threat, req.w_xgc)
        out = _build_player_out(p_gw1, pred_gw1)
        out["cost"] = p["cost"]  # raw tenths for optimizer budget calculations

        # Per-GW breakdown using each GW's own fixture context
        gw_pts = []
        for gw_id in data["upcoming_gws"]:
            p_gwN = _gwN_player(p, gw_id, data["team_strengths"])
            pred_gwN = predict_points(p_gwN, req.w_home_away, req.w_season, req.w_xgi, req.w_fixture, req.w_form, req.w_threat, req.w_xgc)
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


@app.get("/api/backtest")
async def run_backtest(request: Request):
    signals = request.query_params.get("signals")
    data = await fetch_all_data()
    active_signals = [s.strip() for s in signals.split(",")] if signals else None
    try:
        result = await asyncio.to_thread(
            compute_backtest,
            data["raw_histories"],
            data["team_strengths"],
            data["next_gw"],
            active_signals,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result
