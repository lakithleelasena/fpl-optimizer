"""Per-gameweek prediction helpers shared by the API (main.py) and the snapshot archive.

Kept free of FastAPI/pydantic and of fpl_client so both main.py and snapshot.py (which
fpl_client calls) can import it without a cycle.
"""
from __future__ import annotations

from config import W_ODDS_WEIGHT
from predictor import predict_points


def gwN_player(player: dict, gw_id, team_strengths: dict, odds_weight: float = W_ODDS_WEIGHT) -> dict:
    """Return a copy of player with fixture context set for a specific GW."""
    if gw_id is None:
        return player
    opps = player["gw_fixtures"].get(gw_id, [])
    strengths = [team_strengths.get(o, 0) for o in opps]
    is_home = player.get("gw_home", {}).get(gw_id, 0.5)
    fdr_ease = player.get("gw_ease", {}).get(gw_id)
    gw_xg = player.get("gw_match_xg", {}).get(gw_id, {})
    # Blend Tier 1 (odds) with the model (Tier 2/3) when odds are available for this
    # fixture; otherwise fall back to the model alone. Done per-request (not cached)
    # so the odds_weight slider takes effect without needing a fresh data fetch.
    if gw_xg.get("odds_team_xg") is not None:
        match_team_xg = odds_weight * gw_xg["odds_team_xg"] + (1 - odds_weight) * gw_xg.get("model_team_xg", 0.0)
        match_opp_xg = odds_weight * gw_xg["odds_opp_xg"] + (1 - odds_weight) * gw_xg.get("model_opp_xg", 0.0)
    else:
        match_team_xg = gw_xg.get("model_team_xg", 0.0)
        match_opp_xg = gw_xg.get("model_opp_xg", 0.0)
    return {
        **player,
        "opponents": opps,
        "opponent_strengths": strengths,
        "n_fixtures": len(opps),
        "is_home": is_home,
        "_gw_ease": fdr_ease,
        "match_team_xg": round(match_team_xg, 3),
        "match_opp_xg": round(match_opp_xg, 3),
    }


def gw1_player(player: dict, upcoming_gws: list, team_strengths: dict, odds_weight: float = W_ODDS_WEIGHT) -> dict:
    """Return a copy of player with opponents/strengths/n_fixtures restricted to GW1 only."""
    return gwN_player(player, upcoming_gws[0] if upcoming_gws else None, team_strengths, odds_weight)


def predict_gameweeks(data: dict, odds_weight: float = W_ODDS_WEIGHT) -> dict[int, dict]:
    """
    Default-parameter predictions for every player over every upcoming gameweek — what the
    app shows with all sliders at their defaults. Returns {player_id: {"gw_pts": [...],
    "match_team_xg": x, "match_opp_xg": y, "n_fixtures": n}} where gw_pts[i] is the predicted
    TOTAL points for upcoming_gws[i] (per-match points x fixtures that gameweek, 0 for a blank)
    and the xG/fixture fields describe the next gameweek.
    """
    out: dict[int, dict] = {}
    upcoming = data["upcoming_gws"]
    for p in data["players"]:
        gw_pts = []
        first = None
        for i, gw_id in enumerate(upcoming):
            p_gw = gwN_player(p, gw_id, data["team_strengths"], odds_weight)
            n_fix = len(p.get("gw_fixtures", {}).get(gw_id, []))
            pred = predict_points(p_gw)
            gw_pts.append(round(pred["predicted_points"] * n_fix, 2))
            if i == 0:
                first = p_gw
        out[p["id"]] = {
            "gw_pts": gw_pts,
            "n_fixtures": first.get("n_fixtures", 0) if first else 0,
            "match_team_xg": first.get("match_team_xg") if first else None,
            "match_opp_xg": first.get("match_opp_xg") if first else None,
        }
    return out
