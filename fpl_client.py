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
) -> tuple[int, list[dict], list[dict]]:
    async with sem:
        url = ELEMENT_SUMMARY_URL.format(player_id=player_id)
        data = await _fetch_json(client, url)
        return player_id, data.get("history", []), data.get("history_past", [])


def _build_player_stats(history: list[dict], history_past: list[dict] | None = None) -> dict:
    opponent_points: dict[int, list[int]] = {}
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

    played_gws = [gw for gw in history if gw["minutes"] > 0]
    recent_points = [gw["total_points"] for gw in played_gws[-3:]]
    recent_minutes = [gw["minutes"] for gw in history[-5:]]
    season_avg = total_points / games_played if games_played > 0 else 0.0

    # ── Pre-season / new-player fallback: seed from last available season ───────
    # If this season's history is empty (season not started or player is new),
    # use history_past to estimate season_avg and start_likelihood inputs.
    if games_played == 0 and history_past:
        # history_past is ordered oldest→newest; take the most recent season
        past = sorted(history_past, key=lambda s: s.get("season_name", ""))[-1]
        past_mins = past.get("minutes", 0)
        past_pts = past.get("total_points", 0)
        past_starts = past.get("starts", 0)
        if past_mins > 0:
            est_games = past_mins / 90
            season_avg = past_pts / est_games
            games_played = round(est_games)
            # Approximate recent_minutes: if player started regularly, assume 90s
            start_rate = past_starts / max(est_games, 1)
            approx_mins = 90 if start_rate >= 0.6 else (45 if start_rate >= 0.2 else 0)
            recent_minutes = [approx_mins] * 5
            recent_points = [round(season_avg)] * 3
            total_points = past_pts

    return {
        "opponent_points": opponent_points,
        "recent_points": recent_points,
        "recent_minutes": recent_minutes,
        "season_avg": round(season_avg, 2),
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

        # Build team lookups
        teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
        teams_short = {t["id"]: t["short_name"] for t in bootstrap["teams"]}

        # Team strength lookup (still used by backtest; 1-5 scale for 2026-27)
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

        # ── Per-GW fixture map: {gw_id: {team_id: [opp_ids]}} ──────────────────
        gw_fixture_map: dict[int, dict[int, list[int]]] = {gw: {} for gw in upcoming_gws}
        for fix in fixtures:
            if fix["event"] in gw_fixture_map:
                h, a = fix["team_h"], fix["team_a"]
                gw_fixture_map[fix["event"]].setdefault(h, []).append(a)
                gw_fixture_map[fix["event"]].setdefault(a, []).append(h)

        # ── Per-GW FDR ease map: {gw_id: {team_id: [ease_values]}} ─────────────
        # FDR is the FPL fixture difficulty rating (1=easiest, 5=hardest).
        # We convert to ease: ease = (5 - fdr) / 4  → FDR1=1.0, FDR5=0.0
        gw_fdr_map: dict[int, dict[int, list[float]]] = {gw: {} for gw in upcoming_gws}
        for fix in fixtures:
            if fix["event"] in gw_fdr_map:
                h, a = fix["team_h"], fix["team_a"]
                h_ease = round((5 - fix["team_h_difficulty"]) / 4, 2)
                a_ease = round((5 - fix["team_a_difficulty"]) / 4, 2)
                gw_fdr_map[fix["event"]].setdefault(h, []).append(h_ease)
                gw_fdr_map[fix["event"]].setdefault(a, []).append(a_ease)

        # ── Per-GW home/away fraction: {gw_id: {team_id: float}} ───────────────
        gw_home_map: dict[int, dict[int, list[bool]]] = {gw: {} for gw in upcoming_gws}
        for fix in fixtures:
            if fix["event"] in gw_home_map:
                h, a = fix["team_h"], fix["team_a"]
                gw_home_map[fix["event"]].setdefault(h, []).append(True)
                gw_home_map[fix["event"]].setdefault(a, []).append(False)

        # ── Active players filter ────────────────────────────────────────────────
        # Include players with prior minutes OR new/promoted players who are
        # available and expected to score (ep_next > 0) — covers season start.
        elements = bootstrap["elements"]
        active_players = [
            p for p in elements
            if p["minutes"] > 0
            or (p.get("status") == "a" and float(p.get("ep_next") or 0) > 0)
        ]

        # Fetch histories concurrently
        sem = asyncio.Semaphore(SEMAPHORE_LIMIT)
        tasks = [_fetch_player_history(client, sem, p["id"]) for p in active_players]
        results = await asyncio.gather(*tasks)

        pos_lookup = {p["id"]: POSITION_MAP.get(p["element_type"], "MID") for p in active_players}

        player_stats = {}
        raw_histories = {}
        for player_id, history, history_past in results:
            player_stats[player_id] = _build_player_stats(history, history_past)
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

            # Per-GW fixture and FDR-based ease data
            gw_fixtures: dict[int, list[int]] = {}
            gw_ease: dict[int, float | None] = {}
            for gw_id in upcoming_gws:
                opps = gw_fixture_map[gw_id].get(team_id, [])
                gw_fixtures[gw_id] = opps
                fdr_eases = gw_fdr_map[gw_id].get(team_id, [])
                if fdr_eases:
                    gw_ease[gw_id] = round(sum(fdr_eases) / len(fdr_eases), 2)
                else:
                    gw_ease[gw_id] = None  # blank GW

            # Per-GW home fraction
            gw_home: dict[int, float] = {}
            for gw_id in upcoming_gws:
                flags = gw_home_map[gw_id].get(team_id, [])
                gw_home[gw_id] = sum(flags) / len(flags) if flags else 0.5

            # Flatten opponents and strengths across all upcoming GWs
            all_opponents = [o for opps in gw_fixtures.values() for o in opps]
            all_opp_strengths = [team_strengths[o] for o in all_opponents if o in team_strengths]
            n_fixtures = len(all_opponents)

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
            "teams_short": teams_short,
            "team_strengths": team_strengths,
            "raw_histories": raw_histories,
            "bootstrap_teams": bootstrap["teams"],   # full team objects for team overview tab
            "fixtures": fixtures,                    # all fixtures for fixture tracker tab
        }

        _cache = data
        _cache_time = time.time()
        return data
