"""Fetch match-level xG from The Odds API for upcoming EPL fixtures.

Results are cached to odds_cache.json keyed by gameweek — only one live
API call is made per round. Returns empty dict when ODDS_API_KEY is not
configured so callers fall back to Tier 2/3 automatically. If a live call
fails (e.g. free-tier quota exhausted mid-query) or returns no usable data,
the last successfully saved odds for the current gameweek are served instead
— the cache file is only ever overwritten by a successful fetch.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import httpx

from config import ODDS_API_KEY, ODDS_API_URL

ODDS_CACHE_FILE = "odds_cache.json"
# Historical archive: {gw_str: {fetched_at, data}} — one entry per gameweek,
# kept forever (never overwritten by a later GW) so future backtests can see
# what the market actually said at the time. Updated in place if a gameweek's
# odds are re-fetched before kickoff; the live single-GW cache above is
# unaffected by this and keeps working exactly as before.
ODDS_HISTORY_FILE = "odds_history.json"

# Common FPL name → list of Odds API name variants
_FPL_ALIASES: dict[str, list[str]] = {
    "Man Utd":        ["Manchester United"],
    "Man City":       ["Manchester City"],
    "Newcastle Utd":  ["Newcastle United"],
    "Nott'm Forest":  ["Nottingham Forest"],
    "Spurs":          ["Tottenham Hotspur"],
    "Brighton":       ["Brighton and Hove Albion", "Brighton & Hove Albion"],
    "Wolves":         ["Wolverhampton Wanderers"],
    "West Ham":       ["West Ham United"],
    "Leicester":      ["Leicester City"],
    "Ipswich":        ["Ipswich Town"],
    "Sheffield Utd":  ["Sheffield United"],
    "Leeds":          ["Leeds United"],
    "Luton":          ["Luton Town"],
    "West Brom":      ["West Bromwich Albion"],
    "Stoke":          ["Stoke City"],
    "Watford":        ["Watford"],
    "Swansea":        ["Swansea City"],
    "Cardiff":        ["Cardiff City"],
    "Norwich":        ["Norwich City"],
    "Derby":          ["Derby County"],
    "Plymouth":       ["Plymouth Argyle"],
    "Sunderland":     ["Sunderland"],
    "Coventry":       ["Coventry City"],
    "Hull":           ["Hull City"],
    "Sheffield Wed":  ["Sheffield Wednesday"],
    "Preston":        ["Preston North End"],
    "Middlesbrough":  ["Middlesbrough"],
    "QPR":            ["Queens Park Rangers"],
}


# ── Cache helpers ────────────────────────────────────────────────────────────

def get_odds_cache_meta() -> dict | None:
    """Return {gameweek, fetched_at, fixture_count} from cache, or None if missing."""
    if not os.path.exists(ODDS_CACHE_FILE):
        return None
    try:
        with open(ODDS_CACHE_FILE) as f:
            cache = json.load(f)
        return {
            "gameweek": cache.get("gameweek"),
            "fetched_at": cache.get("fetched_at"),
            "fixture_count": len(cache.get("data", {})),
        }
    except Exception:
        return None


def load_odds_cache(current_gw: int) -> dict:
    """Return cached odds dict if it matches current_gw, else {}."""
    if not os.path.exists(ODDS_CACHE_FILE):
        return {}
    try:
        with open(ODDS_CACHE_FILE) as f:
            cache = json.load(f)
        if cache.get("gameweek") == current_gw:
            # JSON stringifies int keys — convert back
            return {
                int(fid): {int(tid): vals for tid, vals in teams.items()}
                for fid, teams in cache.get("data", {}).items()
            }
    except Exception:
        pass
    return {}


def _save_odds_cache(gw_id: int, data: dict) -> None:
    serializable = {
        str(fid): {str(tid): list(vals) for tid, vals in teams.items()}
        for fid, teams in data.items()
    }
    payload = {
        "gameweek": gw_id,
        "fetched_at": datetime.utcnow().isoformat(),
        "data": serializable,
    }
    with open(ODDS_CACHE_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def _archive_odds_snapshot(gw_id: int, data: dict) -> None:
    """Record this gameweek's odds in the permanent historical archive, keyed
    by GW so past gameweeks are never lost when later ones are fetched. Safe
    to call repeatedly for the same GW (e.g. re-refreshed before kickoff) —
    that GW's entry is just replaced with the newest snapshot each time."""
    history: dict = {}
    if os.path.exists(ODDS_HISTORY_FILE):
        try:
            with open(ODDS_HISTORY_FILE) as f:
                history = json.load(f)
        except Exception:
            history = {}

    serializable = {
        str(fid): {str(tid): list(vals) for tid, vals in teams.items()}
        for fid, teams in data.items()
    }
    history[str(gw_id)] = {
        "fetched_at": datetime.utcnow().isoformat(),
        "data": serializable,
    }

    with open(ODDS_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def load_odds_history() -> dict[int, dict]:
    """Return the full historical odds archive as {gw: {fixture_id: {team_id: (team_xg, opp_xg)}}}.
    Used by the backtest — GWs never fetched (e.g. before this feature existed) are simply absent."""
    if not os.path.exists(ODDS_HISTORY_FILE):
        return {}
    try:
        with open(ODDS_HISTORY_FILE) as f:
            history = json.load(f)
        return {
            int(gw): {
                int(fid): {int(tid): tuple(vals) for tid, vals in teams.items()}
                for fid, teams in entry.get("data", {}).items()
            }
            for gw, entry in history.items()
        }
    except Exception:
        return {}


# ── Name matching ────────────────────────────────────────────────────────────

def _match_name(fpl_name: str, odds_names: list[str]) -> str | None:
    """Return the odds API team name that best matches an FPL team name."""
    fl = fpl_name.lower().strip()
    for name in odds_names:
        if name.lower() == fl:
            return name
    for alias, variants in _FPL_ALIASES.items():
        if fl == alias.lower():
            for name in odds_names:
                if name in variants:
                    return name
    for name in odds_names:
        if fl in name.lower() or name.lower() in fl:
            return name
    return None


def _implied_total_goals(outcomes: list[dict]) -> float:
    over = next((o for o in outcomes if o["name"] == "Over"), None)
    under = next((o for o in outcomes if o["name"] == "Under"), None)
    if not over or not under:
        return 2.65
    line = float(over.get("point", 2.5))
    total_inv = 1.0 / over["price"] + 1.0 / under["price"]
    over_prob = (1.0 / over["price"]) / total_inv
    return round(line + (over_prob - 0.5) * 1.0, 2)


def _home_share_from_h2h(outcomes: list[dict], home_name: str) -> float:
    """Devig h2h odds and return home expected goal share (home_win + 0.5*draw)."""
    home_price = draw_price = away_price = None
    for o in outcomes:
        n = o["name"]
        p = float(o["price"])
        if n == "Draw":
            draw_price = p
        elif n.lower() == home_name.lower() or home_name.lower() in n.lower():
            home_price = p
        else:
            away_price = p
    if not home_price or not draw_price or not away_price:
        return 0.5
    inv = 1/home_price + 1/draw_price + 1/away_price
    home_prob = (1/home_price) / inv
    draw_prob = (1/draw_price) / inv
    return home_prob + 0.5 * draw_prob


def _extract_xg(bookmaker: dict, home_name: str) -> tuple[float, float] | None:
    markets = {m["key"]: m["outcomes"] for m in bookmaker.get("markets", [])}
    if "totals" not in markets:
        return None
    total_xg = _implied_total_goals(markets["totals"])
    home_share = _home_share_from_h2h(markets["h2h"], home_name) if "h2h" in markets else 0.5
    home_xg = max(0.2, round(total_xg * home_share, 2))
    away_xg = max(0.2, round(total_xg * (1 - home_share), 2))
    return home_xg, away_xg


# ── Main fetch ───────────────────────────────────────────────────────────────

async def fetch_odds_xg(
    fpl_teams: dict[int, str],
    upcoming_fixtures: list[dict],
    current_gw: int | None = None,
    force_refresh: bool = False,
) -> dict[int, dict[int, tuple[float, float]]]:
    """
    Return {fixture_id: {team_id: (team_xg, opp_xg)}} for upcoming fixtures.

    Uses file cache when current_gw is supplied and cache is valid for that GW.
    Pass force_refresh=True to bypass the cache and fetch fresh data.
    Returns {} if ODDS_API_KEY is absent or the request fails.
    """
    if not ODDS_API_KEY:
        return {}

    # Serve from cache unless caller requested a refresh
    if current_gw and not force_refresh:
        cached = load_odds_cache(current_gw)
        if cached:
            return cached

    try:
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "uk",
            "markets": "totals,h2h",
            "oddsFormat": "decimal",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(ODDS_API_URL, params=params)
            resp.raise_for_status()
            events: list[dict] = resp.json()
    except Exception:
        # Live call failed (e.g. free-tier quota exhausted mid-query) — keep
        # using the last successfully saved odds for this gameweek instead of
        # losing them. Falls back to {} only if nothing was ever cached.
        return load_odds_cache(current_gw) if current_gw else {}

    odds_map: dict[tuple[str, str], tuple[float, float]] = {}
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        for bm in event.get("bookmakers", []):
            xg = _extract_xg(bm, home)
            if xg:
                odds_map[(home, away)] = xg
                break

    if not odds_map:
        return load_odds_cache(current_gw) if current_gw else {}

    odds_names = list({n for pair in odds_map for n in pair})
    id_to_odds_name: dict[int, str] = {}
    for tid, tname in fpl_teams.items():
        matched = _match_name(tname, odds_names)
        if matched:
            id_to_odds_name[tid] = matched

    result: dict[int, dict[int, tuple[float, float]]] = {}
    for fix in upcoming_fixtures:
        fid = fix["id"]
        h_id, a_id = fix["team_h"], fix["team_a"]
        h_name = id_to_odds_name.get(h_id)
        a_name = id_to_odds_name.get(a_id)
        if h_name and a_name:
            xg = odds_map.get((h_name, a_name))
            if xg:
                result.setdefault(fid, {})[h_id] = (xg[0], xg[1])
                result.setdefault(fid, {})[a_id] = (xg[1], xg[0])

    if not result:
        return load_odds_cache(current_gw) if current_gw else {}

    if current_gw:
        _save_odds_cache(current_gw, result)
        _archive_odds_snapshot(current_gw, result)

    return result
