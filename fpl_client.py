from __future__ import annotations

import asyncio
import time

import httpx

from config import (
    BOOTSTRAP_URL,
    CACHE_TTL_SECONDS,
    ELEMENT_SUMMARY_URL,
    FIXTURES_URL,
    POSITION_MAP,
    SEMAPHORE_LIMIT,
)

_cache: dict = {}
_cache_time: float = 0.0


async def _fetch_json(client: httpx.AsyncClient, url: str) -> dict | list:
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.json()


async def _fetch_player_history(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    player_id: int,
) -> tuple[int, list[dict]]:
    async with sem:
        url = ELEMENT_SUMMARY_URL.format(player_id=player_id)
        data = await _fetch_json(client, url)
        return player_id, data.get("history", [])


def _build_player_stats(history: list[dict]) -> dict:
    opponent_points: dict[int, list[int]] = {}
    recent_points: list[int] = []
    total_points = 0
    games_played = 0

    for gw in history:
        pts = gw["total_points"]
        opp = gw["opponent_team"]
        mins = gw["minutes"]

        if mins > 0:
            opponent_points.setdefault(opp, []).append(pts)
            total_points += pts
            games_played += 1

    # Last 3 GWs with minutes
    played_gws = [gw for gw in history if gw["minutes"] > 0]
    recent_points = [gw["total_points"] for gw in played_gws[-3:]]

    # Last 5 GW entries (regardless of minutes) for starting likelihood
    recent_minutes = [gw["minutes"] for gw in history[-5:]]

    season_avg = total_points / games_played if games_played > 0 else 0.0

    return {
        "opponent_points": opponent_points,
        "recent_points": recent_points,
        "recent_minutes": recent_minutes,
        "season_avg": season_avg,
        "total_points": total_points,
        "games_played": games_played,
    }


async def fetch_all_data() -> dict:
    global _cache, _cache_time

    now = time.time()
    if _cache and (now - _cache_time) < CACHE_TTL_SECONDS:
        return _cache

    async with httpx.AsyncClient(timeout=30.0) as client:
        bootstrap, fixtures = await asyncio.gather(
            _fetch_json(client, BOOTSTRAP_URL),
            _fetch_json(client, FIXTURES_URL),
        )

        # Find next gameweek
        next_gw = None
        for event in bootstrap["events"]:
            if event.get("is_next"):
                next_gw = event["id"]
                break

        if next_gw is None:
            raise ValueError("Could not determine next gameweek")

        # Build team lookup
        teams = {t["id"]: t["name"] for t in bootstrap["teams"]}

        # Build team strength lookup (average of home + away overall strength)
        team_strengths = {
            t["id"]: (t["strength_overall_home"] + t["strength_overall_away"]) / 2
            for t in bootstrap["teams"]
        }

        # Next GW fixtures: map team_id -> list of opponent team_ids
        next_fixtures: dict[int, list[int]] = {}
        for fix in fixtures:
            if fix["event"] == next_gw:
                h, a = fix["team_h"], fix["team_a"]
                next_fixtures.setdefault(h, []).append(a)
                next_fixtures.setdefault(a, []).append(h)

        # Filter active players (minutes > 0)
        elements = bootstrap["elements"]
        active_players = [p for p in elements if p["minutes"] > 0]

        # Fetch histories concurrently
        sem = asyncio.Semaphore(SEMAPHORE_LIMIT)
        tasks = [
            _fetch_player_history(client, sem, p["id"])
            for p in active_players
        ]
        results = await asyncio.gather(*tasks)

        player_stats = {}
        raw_histories = {}
        for player_id, history in results:
            player_stats[player_id] = _build_player_stats(history)
            raw_histories[player_id] = [
                {
                    "round": h["round"],
                    "total_points": h["total_points"],
                    "minutes": h["minutes"],
                    "opponent_team": h["opponent_team"],
                }
                for h in history
            ]

        # Build player list
        players = []
        for p in active_players:
            pid = p["id"]
            stats = player_stats.get(pid)
            if not stats:
                continue

            pos = POSITION_MAP.get(p["element_type"], "UNK")
            team_id = p["team"]
            opponents = next_fixtures.get(team_id, [])

            players.append({
                "id": pid,
                "name": p["web_name"],
                "team": teams.get(team_id, "Unknown"),
                "team_id": team_id,
                "position": pos,
                "cost": p["now_cost"],
                "chance_of_playing": p.get("chance_of_playing_next_round"),
                "minutes": p["minutes"],
                "total_points": p["total_points"],
                "opponents": opponents,
                "opponent_strengths": [team_strengths[opp] for opp in opponents if opp in team_strengths],
                "stats": stats,
            })

        data = {
            "players": players,
            "next_gw": next_gw,
            "teams": teams,
            "team_strengths": team_strengths,
            "raw_histories": raw_histories,
        }

        _cache = data
        _cache_time = time.time()
        return data
