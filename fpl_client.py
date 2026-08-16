from __future__ import annotations

import asyncio
import time
from collections import defaultdict

import httpx

from config import (
    BOOTSTRAP_URL,
    CACHE_TTL_SECONDS,
    ELEMENT_SUMMARY_URL,
    FIXTURES_URL,
    LEAGUE_AVG_GOALS,
    POSITION_MAP,
    SEMAPHORE_LIMIT,
)
from odds_client import fetch_odds_xg

_cache: dict = {}
_cache_time: float = 0.0


def invalidate_cache() -> None:
    global _cache_time
    _cache_time = 0.0


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


def _build_team_rolling(fixtures: list[dict]) -> dict[int, dict]:
    """
    Compute per-team rolling stats from up to the last 6 finished fixtures.
    Returns {team_id: {attack_xg6, defence_xg6, games_played}}.
    """
    finished = [
        f for f in fixtures
        if f.get("finished") and f.get("team_h_score") is not None
    ]
    finished.sort(key=lambda f: (f.get("event") or 0, f.get("kickoff_time") or ""))

    team_history: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for f in finished:
        h, a = f["team_h"], f["team_a"]
        hs, as_ = f["team_h_score"], f["team_a_score"]
        team_history[h].append((hs, as_))
        team_history[a].append((as_, hs))

    rolling: dict[int, dict] = {}
    for tid, matches in team_history.items():
        last6 = matches[-6:]
        attack = sum(m[0] for m in last6) / len(last6)
        defence = sum(m[1] for m in last6) / len(last6)
        rolling[tid] = {
            "attack_xg6": round(attack, 2),
            "defence_xg6": round(defence, 2),
            "games_played": len(team_history[tid]),
        }
    return rolling


def _build_fixture_results(fixtures: list[dict]) -> dict[int, dict]:
    """Map finished fixture IDs to {h_score, a_score} for team goal lookups."""
    return {
        f["id"]: {"h_score": f["team_h_score"], "a_score": f["team_a_score"]}
        for f in fixtures
        if f.get("finished") and f.get("team_h_score") is not None
    }


def _build_player_stats(
    history: list[dict],
    history_past: list[dict] | None = None,
    fixture_results: dict | None = None,
) -> dict:
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

    # ── Participation rates: last 6 played matches ───────────────────────────
    last6 = played_gws[-6:]
    p_goals = sum(h.get("goals_scored", 0) for h in last6)
    p_assists = sum(h.get("assists", 0) for h in last6)
    p_saves = sum(h.get("saves", 0) for h in last6)
    saves_per_game = round(p_saves / len(last6), 2) if last6 else 0.0

    team_goals_last6 = 0
    if fixture_results:
        for h in last6:
            fid = h.get("fixture")
            fr = fixture_results.get(fid) if fid else None
            if fr:
                team_goals_last6 += fr["h_score"] if h.get("was_home") else fr["a_score"]

    goal_share = round(p_goals / team_goals_last6, 4) if team_goals_last6 > 0 else 0.0
    assist_share = round(p_assists / team_goals_last6, 4) if team_goals_last6 > 0 else 0.0

    # ── Pre-season / new-player fallback: seed from last available season ────
    if games_played == 0 and history_past:
        past = sorted(history_past, key=lambda s: s.get("season_name", ""))[-1]
        past_mins = past.get("minutes", 0)
        past_pts = past.get("total_points", 0)
        past_starts = past.get("starts", 0)
        past_goals = past.get("goals_scored", 0)
        past_assists = past.get("assists", 0)
        if past_mins > 0:
            est_games = past_mins / 90
            season_avg = past_pts / est_games
            games_played = round(est_games)
            start_rate = past_starts / max(est_games, 1)
            approx_mins = 90 if start_rate >= 0.6 else (45 if start_rate >= 0.2 else 0)
            recent_minutes = [approx_mins] * 5
            recent_points = [round(season_avg)] * 3
            total_points = past_pts
            # Estimate participation from prior season
            est_team_goals = LEAGUE_AVG_GOALS * est_games
            goal_share = round(past_goals / est_team_goals, 4) if est_team_goals > 0 else 0.0
            assist_share = round(past_assists / est_team_goals, 4) if est_team_goals > 0 else 0.0

    return {
        "opponent_points": opponent_points,
        "recent_points": recent_points,
        "recent_minutes": recent_minutes,
        "season_avg": round(season_avg, 2),
        "total_points": total_points,
        "games_played": games_played,
        "goal_share": goal_share,
        "assist_share": assist_share,
        "saves_per_game": saves_per_game,
    }


def _build_gw_match_xg(
    fixtures: list[dict],
    upcoming_gws: list[int],
    team_rolling: dict[int, dict],
    league_avg_defence: float,
    odds_xg: dict[int, dict[int, tuple[float, float]]],
) -> dict[int, dict[int, dict]]:
    """
    Compute per-GW per-team (team_xg, opp_xg) using three-tier resolution:
      Tier 1: Odds API  (if available)
      Tier 2: rolling 6-game averages adjusted by opponent defensive record
      Tier 3: FDR-based fallback (pre-season)

    Returns {gw_id: {team_id: {team_xg, opp_xg}}} with per-match averages.
    DGW teams: accumulate then average so n_fixtures multiplication still works.
    """
    accum: dict[int, dict[int, dict]] = {gw: {} for gw in upcoming_gws}

    for fix in fixtures:
        gw = fix.get("event")
        if gw not in accum:
            continue
        fid = fix["id"]
        h_id, a_id = fix["team_h"], fix["team_a"]
        h_fdr = fix.get("team_h_difficulty", 3)
        a_fdr = fix.get("team_a_difficulty", 3)

        # Tier 1: Odds API
        odds_fix = odds_xg.get(fid, {})
        if h_id in odds_fix:
            h_xg, a_xg = odds_fix[h_id]
        else:
            h_roll = team_rolling.get(h_id)
            a_roll = team_rolling.get(a_id)
            if h_roll and a_roll and h_roll["attack_xg6"] > 0 and a_roll["attack_xg6"] > 0 and league_avg_defence > 0:
                # Tier 2: rolling averages × opponent defensive factor
                h_xg = h_roll["attack_xg6"] * (a_roll["defence_xg6"] / league_avg_defence)
                a_xg = a_roll["attack_xg6"] * (h_roll["defence_xg6"] / league_avg_defence)
                h_xg = round(max(0.2, h_xg), 3)
                a_xg = round(max(0.2, a_xg), 3)
            else:
                # Tier 3: FDR fallback (pre-season)
                h_mult = max(0.2, (5 - h_fdr) / 3)
                a_mult = max(0.2, (5 - a_fdr) / 3)
                h_xg = round(LEAGUE_AVG_GOALS * h_mult, 3)
                a_xg = round(LEAGUE_AVG_GOALS * a_mult, 3)

        for tid, t_xg, o_xg in ((h_id, h_xg, a_xg), (a_id, a_xg, h_xg)):
            if tid not in accum[gw]:
                accum[gw][tid] = {"t_sum": 0.0, "o_sum": 0.0, "n": 0}
            accum[gw][tid]["t_sum"] += t_xg
            accum[gw][tid]["o_sum"] += o_xg
            accum[gw][tid]["n"] += 1

    # Convert sums to per-match averages
    result: dict[int, dict[int, dict]] = {}
    for gw, teams in accum.items():
        result[gw] = {}
        for tid, acc in teams.items():
            n = acc["n"]
            result[gw][tid] = {
                "team_xg": round(acc["t_sum"] / n, 3) if n else 0.0,
                "opp_xg":  round(acc["o_sum"] / n, 3) if n else 0.0,
            }
    return result


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

        team_strengths = {
            t["id"]: (t["strength_overall_home"] + t["strength_overall_away"]) / 2
            for t in bootstrap["teams"]
        }

        # Upcoming GWs (next 4)
        sorted_events = sorted(bootstrap["events"], key=lambda e: e["id"])
        upcoming_gws: list[int] = []
        for event in sorted_events:
            if event["id"] >= next_gw and not event.get("finished", False):
                upcoming_gws.append(event["id"])
            if len(upcoming_gws) == 4:
                break

        # ── Per-GW fixture maps ──────────────────────────────────────────────
        gw_fixture_map: dict[int, dict[int, list[int]]] = {gw: {} for gw in upcoming_gws}
        gw_fdr_map: dict[int, dict[int, list[float]]] = {gw: {} for gw in upcoming_gws}
        gw_home_map: dict[int, dict[int, list[bool]]] = {gw: {} for gw in upcoming_gws}

        for fix in fixtures:
            gw = fix.get("event")
            if gw not in gw_fixture_map:
                continue
            h, a = fix["team_h"], fix["team_a"]
            h_fdr = fix.get("team_h_difficulty", 3)
            a_fdr = fix.get("team_a_difficulty", 3)
            gw_fixture_map[gw].setdefault(h, []).append(a)
            gw_fixture_map[gw].setdefault(a, []).append(h)
            gw_fdr_map[gw].setdefault(h, []).append(round((5 - h_fdr) / 4, 2))
            gw_fdr_map[gw].setdefault(a, []).append(round((5 - a_fdr) / 4, 2))
            gw_home_map[gw].setdefault(h, []).append(True)
            gw_home_map[gw].setdefault(a, []).append(False)

        # ── Team rolling stats (last ≤6 finished matches) ───────────────────
        team_rolling = _build_team_rolling(fixtures)
        attacks = [v["attack_xg6"] for v in team_rolling.values() if v["attack_xg6"] > 0]
        defences = [v["defence_xg6"] for v in team_rolling.values() if v["defence_xg6"] > 0]
        league_avg_attack = round(sum(attacks) / len(attacks), 3) if attacks else LEAGUE_AVG_GOALS
        league_avg_defence = round(sum(defences) / len(defences), 3) if defences else LEAGUE_AVG_GOALS

        # ── Fixture results map for player goal-share lookup ─────────────────
        fixture_results = _build_fixture_results(fixtures)

        # ── Odds API match xG (Tier 1) ───────────────────────────────────────
        upcoming_fix_list = [f for f in fixtures if f.get("event") in upcoming_gws]
        odds_xg = await fetch_odds_xg(teams, upcoming_fix_list, current_gw=next_gw)

        # ── Per-GW match xG for each team ────────────────────────────────────
        gw_match_xg = _build_gw_match_xg(
            fixtures, upcoming_gws, team_rolling, league_avg_defence, odds_xg
        )

        # ── Active players ───────────────────────────────────────────────────
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

        player_stats: dict[int, dict] = {}
        raw_histories: dict[int, list] = {}
        for player_id, history, history_past in results:
            player_stats[player_id] = _build_player_stats(history, history_past, fixture_results)
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

        # ── Build player list ─────────────────────────────────────────────────
        players = []
        for p in active_players:
            pid = p["id"]
            stats = player_stats.get(pid)
            if not stats:
                continue

            pos = POSITION_MAP.get(p["element_type"], "UNK")
            team_id = p["team"]

            gw_fixtures: dict[int, list[int]] = {}
            gw_ease: dict[int, float | None] = {}
            gw_home: dict[int, float] = {}
            for gw_id in upcoming_gws:
                opps = gw_fixture_map[gw_id].get(team_id, [])
                gw_fixtures[gw_id] = opps
                fdr_eases = gw_fdr_map[gw_id].get(team_id, [])
                gw_ease[gw_id] = round(sum(fdr_eases) / len(fdr_eases), 2) if fdr_eases else None
                flags = gw_home_map[gw_id].get(team_id, [])
                gw_home[gw_id] = sum(flags) / len(flags) if flags else 0.5

            all_opponents = [o for opps in gw_fixtures.values() for o in opps]
            all_opp_strengths = [team_strengths[o] for o in all_opponents if o in team_strengths]

            t_rolling = team_rolling.get(team_id, {})

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
                "n_fixtures": len(all_opponents),
                "gw_fixtures": gw_fixtures,
                "gw_ease": gw_ease,
                "gw_home": gw_home,
                "xgi": float(p.get("expected_goal_involvements") or 0),
                "form": float(p.get("form") or 0),
                "threat": float(p.get("threat") or 0),
                "xgc": float(p.get("expected_goals_conceded") or 0),
                "ep_next": float(p.get("ep_next") or 0),
                "stats": stats,
                # New participation fields
                "goal_share": stats["goal_share"],
                "assist_share": stats["assist_share"],
                "saves_per_game": stats["saves_per_game"],
                # Per-GW match xG (team and opponent)
                "gw_match_xg": {
                    gw_id: gw_match_xg.get(gw_id, {}).get(team_id, {"team_xg": 0.0, "opp_xg": 0.0})
                    for gw_id in upcoming_gws
                },
                # Team-level rolling stats (for Teams tab display)
                "team_attack_xg6": t_rolling.get("attack_xg6", 0.0),
                "team_defence_xg6": t_rolling.get("defence_xg6", 0.0),
            })

    data = {
        "players": players,
        "next_gw": next_gw,
        "upcoming_gws": upcoming_gws,
        "teams": teams,
        "teams_short": teams_short,
        "team_strengths": team_strengths,
        "raw_histories": raw_histories,
        "bootstrap_teams": bootstrap["teams"],
        "fixtures": fixtures,
        "team_rolling": team_rolling,
        "league_avg_attack": league_avg_attack,
        "league_avg_defence": league_avg_defence,
    }

    _cache = data
    _cache_time = time.time()
    return data
