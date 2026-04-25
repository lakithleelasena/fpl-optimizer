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
    STRENGTH_MAX,
    STRENGTH_MIN,
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

        # Collect fixtures for next 4 gameweeks
        sorted_events = sorted(bootstrap["events"], key=lambda e: e["id"])
        upcoming_gws: list[int] = []
        for event in sorted_events:
            if event["id"] >= next_gw and not event.get("finished", False):
                upcoming_gws.append(event["id"])
            if len(upcoming_gws) == 4:
                break

        # Per-GW fixture map: {gw_id: {team_id: [opp_ids]}}
        gw_fixture_map: dict[int, dict[int, list[int]]] = {gw: {} for gw in upcoming_gws}
        for fix in fixtures:
            if fix["event"] in gw_fixture_map:
                h, a = fix["team_h"], fix["team_a"]
                gw_fixture_map[fix["event"]].setdefault(h, []).append(a)
                gw_fixture_map[fix["event"]].setdefault(a, []).append(h)

        # Per-GW home/away map: {gw_id: {team_id: [bool]}} — True if team is home
        gw_home_map: dict[int, dict[int, list[bool]]] = {gw: {} for gw in upcoming_gws}
        for fix in fixtures:
            if fix["event"] in gw_home_map:
                h, a = fix["team_h"], fix["team_a"]
                gw_home_map[fix["event"]].setdefault(h, []).append(True)
                gw_home_map[fix["event"]].setdefault(a, []).append(False)

        str_range = STRENGTH_MAX - STRENGTH_MIN
        mid_str = (STRENGTH_MAX + STRENGTH_MIN) / 2

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

        pos_lookup = {p["id"]: POSITION_MAP.get(p["element_type"], "MID") for p in active_players}

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
                    "was_home": h.get("was_home", False),
                    "xgi": float(h.get("expected_goal_involvements") or 0),
                    "threat": float(h.get("threat") or 0),
                    "xgc": float(h.get("expected_goals_conceded") or 0),
                    "position": pos_lookup.get(player_id, "MID"),
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

            # Build per-GW fixture and ease data across next 3 GWs
            gw_fixtures: dict[int, list[int]] = {}
            gw_ease: dict[int, float | None] = {}
            for gw_id in upcoming_gws:
                opps = gw_fixture_map[gw_id].get(team_id, [])
                gw_fixtures[gw_id] = opps
                if opps:
                    avg_str = sum(team_strengths.get(o, mid_str) for o in opps) / len(opps)
                    gw_ease[gw_id] = round(max(0.0, min(1.0, (STRENGTH_MAX - avg_str) / str_range)), 2)
                else:
                    gw_ease[gw_id] = None  # blank GW

            # Build per-GW home fraction (1.0 = all home, 0.0 = all away, 0.5 = mixed/unknown)
            gw_home: dict[int, float] = {}
            for gw_id in upcoming_gws:
                flags = gw_home_map[gw_id].get(team_id, [])
                gw_home[gw_id] = sum(flags) / len(flags) if flags else 0.5

            # Flatten all opponents and strengths across 3 GWs
            all_opponents = [o for opps in gw_fixtures.values() for o in opps]
            all_opp_strengths = [team_strengths[o] for o in all_opponents if o in team_strengths]
            n_fixtures = len(all_opponents)

            # xG involvement from bootstrap element
            xgi = float(p.get("expected_goal_involvements") or 0)
            form = float(p.get("form") or 0)
            threat = float(p.get("threat") or 0)
            xgc = float(p.get("expected_goals_conceded") or 0)
            ep_next = float(p.get("ep_next") or 0)

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
                "opponents": all_opponents,
                "opponent_strengths": all_opp_strengths,
                "n_fixtures": n_fixtures,
                "gw_fixtures": gw_fixtures,
                "gw_ease": gw_ease,
                "gw_home": gw_home,
                "xgi": xgi,
                "form": form,
                "threat": threat,
                "xgc": xgc,
                "ep_next": ep_next,
                "stats": stats,
            })

        data = {
            "players": players,
            "next_gw": next_gw,
            "upcoming_gws": upcoming_gws,
            "teams": teams,
            "team_strengths": team_strengths,
            "raw_histories": raw_histories,
        }

        _cache = data
        _cache_time = time.time()
        return data
