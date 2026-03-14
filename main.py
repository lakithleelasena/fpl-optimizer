from __future__ import annotations

import asyncio
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
from optimizer import optimize_squad, recommend_transfers
from predictor import predict_points

app = FastAPI(title="FPL Squad Optimizer")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def _build_player_out(player: dict, prediction: dict) -> dict:
    # Scale per-match prediction by total fixtures across next 3 GWs
    n_fix = max(1, player.get("n_fixtures", 1))
    return {
        "id": player["id"],
        "name": player["name"],
        "team": player["team"],
        "team_id": player["team_id"],
        "position": player["position"],
        "cost": player["cost"] / 10,
        "predicted_points": round(prediction["predicted_points"] * n_fix, 2),
        "opponent_score": prediction["opponent_score"],
        "season_avg": prediction["season_avg"],
        "momentum": prediction["momentum"],
        "fixture_ease": prediction["fixture_ease"],
        "start_likelihood": prediction["start_likelihood"],
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
        opponent_score=p["opponent_score"],
        season_avg=p["season_avg"],
        momentum=p["momentum"],
        fixture_ease=p["fixture_ease"],
        start_likelihood=p["start_likelihood"],
        chance_of_playing=p.get("chance_of_playing"),
        minutes=p["minutes"],
        total_points=p["total_points"],
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
        opponent_score=p["opponent_score"],
        season_avg=p["season_avg"],
        momentum=p["momentum"],
        fixture_ease=p["fixture_ease"],
        start_likelihood=p["start_likelihood"],
        chance_of_playing=p.get("chance_of_playing"),
        minutes=p["minutes"],
        total_points=p["total_points"],
        is_starter=is_starter,
    )


@app.get("/api/players", response_model=List[PlayerOut])
async def get_players():
    data = await fetch_all_data()
    result = []
    for p in data["players"]:
        pred = predict_points(p)
        result.append(_build_player_out(p, pred))
    result.sort(key=lambda x: x["predicted_points"], reverse=True)
    return result


@app.post("/api/optimize", response_model=OptimizeResponse)
async def run_optimize(req: OptimizeRequest):
    data = await fetch_all_data()

    enriched = []
    for p in data["players"]:
        pred = predict_points(p, req.w_opponent, req.w_season, req.w_momentum, req.w_fixture)
        out = _build_player_out(p, pred)
        out["cost"] = p["cost"]  # keep raw cost for optimizer
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

    return OptimizeResponse(
        starters=starters,
        bench=bench,
        total_cost=round(total_cost, 1),
        total_predicted_points=round(total_predicted, 2),
    )


@app.post("/api/transfer-advice", response_model=TransferAdviceResponse)
async def get_transfer_advice(req: TransferRequest):
    if len(req.current_team) != 15:
        raise HTTPException(status_code=400, detail=f"Squad must have exactly 15 players, got {len(req.current_team)}")

    data = await fetch_all_data()

    # Build enriched players with raw cost for optimizer
    enriched = []
    for p in data["players"]:
        pred = predict_points(p, req.w_opponent, req.w_season, req.w_momentum, req.w_fixture)
        out = _build_player_out(p, pred)
        out["cost"] = p["cost"]  # raw tenths for optimizer budget calculations
        enriched.append(out)

    result = recommend_transfers(
        current_team_ids=req.current_team,
        all_players=enriched,
        free_transfers=req.free_transfers,
        budget_in_bank=req.budget_in_bank,
        chips_available=req.chips_available,
        upcoming_gws=data["upcoming_gws"],
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

    return TransferAdviceResponse(
        starters=starters,
        bench=bench,
        transfers=transfers,
        chip_recommendation=ChipRecommendation(**result["chip_recommendation"]),
        hits_required=result["hits_required"],
        net_points_gain=result["net_points_gain"],
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
    )


@app.get("/api/backtest")
async def run_backtest():
    data = await fetch_all_data()
    try:
        result = await asyncio.to_thread(
            compute_backtest,
            data["raw_histories"],
            data["team_strengths"],
            data["next_gw"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result
