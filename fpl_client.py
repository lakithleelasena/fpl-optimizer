from __future__ import annotations

import asyncio
import time
from collections import defaultdict

import httpx

from config import (
    AWAY_ADV_MULT,
    BOOTSTRAP_URL,
    CACHE_TTL_SECONDS,
    DEFCON_THRESHOLD,
    ELEMENT_SUMMARY_URL,
    FIXTURES_URL,
    HOME_ADV_MULT,
    LAST_SEASON_GAMES,
    LEAGUE_AVG_GOALS,
    MINUTES_SHRINKAGE_GAMES,
    POSITION_MAP,
    SEMAPHORE_LIMIT,
    SHARE_SHRINKAGE_K,
    TAPER_GAMES,
)
import bonus_model
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


def _build_player_stats(
    history: list[dict],
    history_past: list[dict] | None = None,
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

    # ── Saves: last 6 played matches (goal/assist shares now come from
    #    _compute_xg_share — an xG-based, shrunk-toward-prior replacement for the
    #    actual-goals share this function used to compute here; see Phase 1 in
    #    PREDICTION_MODEL_PLAN.md) ───────────────────────────────────────────
    last6 = played_gws[-6:]
    p_saves = sum(h.get("saves", 0) for h in last6)
    saves_per_game = round(p_saves / len(last6), 2) if last6 else 0.0

    # ── Pre-season / new-player fallback: seed from last available season ────
    if games_played == 0 and history_past:
        past = sorted(history_past, key=lambda s: s.get("season_name", ""))[-1]
        past_mins = past.get("minutes", 0)
        past_pts = past.get("total_points", 0)
        past_starts = past.get("starts", 0)
        if past_mins > 0:
            est_games = past_mins / 90
            season_avg = past_pts / est_games
            games_played = round(est_games)
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
        "saves_per_game": saves_per_game,
    }


def build_team_xg_totals(
    entries: list[tuple[int | None, list[dict]]],
) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, float]]]:
    """
    Sum every player's own `expected_goals`/`expected_assists` by (team_id, fixture_id)
    — reconstructs each team's real total xG/xA per match directly from the FPL API,
    with no separate data source (every active player's own history already carries
    their own xG for every match they played). This is the denominator individual
    players' shares are computed against — see compute_xg_share.

    `entries`: (team_id, history) pairs — shape-agnostic so both the live pipeline
    (fetch_all_data) and the accuracy backtest (backtest_accuracy.py) can build their
    own list from whatever data shape they already have and share this aggregation.
    Safe to build once globally (rather than per backtest target-gameweek) because a
    lookup only ever hits fixtures already known to be prior — the caller only ever
    looks up fixtures appearing in a played-before-target-gw window.
    """
    team_xg: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    team_xa: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for team_id, history in entries:
        if team_id is None:
            continue
        for h in history:
            if h.get("minutes", 0) <= 0:
                continue
            fid = h.get("fixture")
            if fid is None:
                continue
            team_xg[team_id][fid] += float(h.get("expected_goals") or 0)
            team_xa[team_id][fid] += float(h.get("expected_assists") or 0)
    return team_xg, team_xa


def build_position_priors(
    entries: list[tuple[str, list[dict] | None]],
) -> dict[str, tuple[float, float]]:
    """
    League-average last-season xG/xA share by position — the shrinkage prior for
    players with no last-season data at all (a genuine new arrival: promoted-team
    signing, first pro season, etc.), per Phase 1 of PREDICTION_MODEL_PLAN.md.

    `entries`: (position, history_past) pairs — shape-agnostic, see build_team_xg_totals.
    """
    sums: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for pos, history_past in entries:
        if not history_past:
            continue
        past = sorted(history_past, key=lambda s: s.get("season_name", ""))[-1]
        past_mins = past.get("minutes", 0)
        if past_mins <= 0:
            continue
        est_team_goals = LEAGUE_AVG_GOALS * (past_mins / 90)
        if est_team_goals <= 0:
            continue
        past_xg = float(past.get("expected_goals") or 0)
        past_xa = float(past.get("expected_assists") or 0)
        sums[pos].append((past_xg / est_team_goals, past_xa / est_team_goals))

    return {
        pos: (sum(v[0] for v in vals) / len(vals), sum(v[1] for v in vals) / len(vals))
        for pos, vals in sums.items()
    }


def compute_xg_share(
    history: list[dict],
    history_past: list[dict] | None,
    team_id: int | None,
    team_xg_by_fixture: dict[int, dict[int, float]],
    team_xa_by_fixture: dict[int, dict[int, float]],
    position_prior: tuple[float, float] = (0.0, 0.0),
    k: float = SHARE_SHRINKAGE_K,
    window: int = 6,
) -> tuple[float, float]:
    """
    xG-based share of team output, shrunk toward a prior — replaces a raw
    actual-goals share, which is nearly pure noise over only a handful of games
    (see Phase 1, PREDICTION_MODEL_PLAN.md).

        share_this_season = player's own xG (or xA) over the last `window` games
                             THEY played this season / team's total xG (or xA) over
                             those SAME fixtures (from build_team_xg_totals)
        share_prior        = last season's xG/xA at the same club, over an estimated
                             team-goals figure (LEAGUE_AVG_GOALS x games) — or the
                             position-average prior with no last-season data at all
        n90                = 90-minute-equivalents played THIS season (ALL games,
                             not just the share window) — the shrinkage weight

        share = (n90*share_this_season + k*share_prior) / (n90+k)

    A player with 0 minutes this season relies entirely on the prior; by n90=k
    (e.g. k=6 full games), the prior and this season's own signal are weighted
    equally, fading out as the season goes on. `history` should always be the full
    season-to-date history (live pipeline) or full prior-to-target-gameweek history
    (backtest) — never pre-truncated to `window`, since n90 needs every game, not
    just the share window; `window` controls only the share numerator/denominator.
    """
    played = [h for h in history if h.get("minutes", 0) > 0]
    recent = played[-window:]

    player_xg = sum(float(h.get("expected_goals") or 0) for h in recent)
    player_xa = sum(float(h.get("expected_assists") or 0) for h in recent)

    team_xg_total = 0.0
    team_xa_total = 0.0
    if team_id is not None:
        for h in recent:
            fid = h.get("fixture")
            if fid is None:
                continue
            team_xg_total += team_xg_by_fixture.get(team_id, {}).get(fid, 0.0)
            team_xa_total += team_xa_by_fixture.get(team_id, {}).get(fid, 0.0)

    share_goal_now = (player_xg / team_xg_total) if team_xg_total > 0 else 0.0
    share_assist_now = (player_xa / team_xa_total) if team_xa_total > 0 else 0.0

    n90 = sum(h.get("minutes", 0) for h in history) / 90.0

    prior_goal, prior_assist = position_prior
    if history_past:
        past = sorted(history_past, key=lambda s: s.get("season_name", ""))[-1]
        past_mins = past.get("minutes", 0)
        if past_mins > 0:
            est_team_goals = LEAGUE_AVG_GOALS * (past_mins / 90)
            if est_team_goals > 0:
                prior_goal = float(past.get("expected_goals") or 0) / est_team_goals
                prior_assist = float(past.get("expected_assists") or 0) / est_team_goals

    denom = n90 + k
    if denom <= 0:
        return 0.0, 0.0
    goal_share = round((n90 * share_goal_now + k * prior_goal) / denom, 4)
    assist_share = round((n90 * share_assist_now + k * prior_assist) / denom, 4)
    return goal_share, assist_share


def compute_defcon_hit_rate(history: list[dict], position: str, window: int = 6) -> float:
    """
    Empirical estimate of P(defensive_contribution >= threshold this match), from
    the player's own history over their last `window` played games — the "start
    simple" DefCon v1 from Phase 2 (PREDICTION_MODEL_PLAN.md), before a
    negative-binomial/game-state-adjusted version. FPL's own `defensive_contribution`
    field already sums exactly the right stats per position (CBIT for defenders,
    CBIRT for mid/forwards) — no need to combine the individual clearances/blocks/
    interceptions/tackles/recoveries fields ourselves.
    """
    threshold = DEFCON_THRESHOLD.get(position)
    if threshold is None:
        return 0.0
    played = [h for h in history if h.get("minutes", 0) > 0]
    recent = played[-window:]
    if not recent:
        return 0.0
    hits = sum(1 for h in recent if h.get("defensive_contribution", 0) >= threshold)
    return round(hits / len(recent), 4)


def compute_card_rate(history: list[dict], window: int = 6) -> float:
    """P(yellow card this match), from the last `window` played games — used as a
    small per-match points deduction. Red cards are rare enough, and already
    dominate the match outcome so heavily via lost minutes, to skip for this v1."""
    played = [h for h in history if h.get("minutes", 0) > 0]
    recent = played[-window:]
    if not recent:
        return 0.0
    return round(sum(1 for h in recent if h.get("yellow_cards", 0) >= 1) / len(recent), 4)


def _completion_rate_from_avg_mins(avg_mins_per_start: float) -> float:
    """Heuristic: given a player's average minutes-per-start, what fraction of
    those starts likely reached 60+? history_past only has season TOTALS (no
    per-game minutes breakdown), so this can't be derived exactly — a smooth,
    bounded proxy: ~90 min/start -> ~1.0, ~60 -> ~0.5, <=30 -> 0.0."""
    return min(1.0, max(0.0, (avg_mins_per_start - 30) / 60))


def build_minutes_priors(
    entries: list[tuple[str, list[dict] | None]],
) -> dict[str, tuple[float, float]]:
    """
    League-average last-season (start_rate, completion_rate) by position — the
    shrinkage prior for players with no last-season data at all, mirroring
    build_position_priors (Phase 1) for the minutes model (Phase 3).
    `entries`: (position, history_past) pairs.
    """
    sums: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for pos, history_past in entries:
        if not history_past:
            continue
        past = sorted(history_past, key=lambda s: s.get("season_name", ""))[-1]
        past_starts = past.get("starts", 0)
        past_mins = past.get("minutes", 0)
        if past_starts <= 0:
            continue
        prior_start = past_starts / LAST_SEASON_GAMES
        prior_completion = _completion_rate_from_avg_mins(past_mins / past_starts)
        sums[pos].append((prior_start, prior_completion))

    return {
        pos: (sum(v[0] for v in vals) / len(vals), sum(v[1] for v in vals) / len(vals))
        for pos, vals in sums.items()
    }


def compute_minutes_model(
    history: list[dict],
    history_past: list[dict] | None,
    team_games_so_far: int,
    availability: float,
    position_prior: tuple[float, float] = (0.0, 0.0),
    k: float = MINUTES_SHRINKAGE_GAMES,
) -> tuple[float, float, float, float]:
    """
    Returns (exp_start_pct, exp_minutes, p_60_plus, p_1_to_59) — a Beta-prior-blended
    minutes model, replacing the old flat this-season-only ratio and its hard
    pre-season-fallback cutover (see Phase 3, PREDICTION_MODEL_PLAN.md). Same
    Beta-Binomial-posterior-mean shrinkage as Phase 1's compute_xg_share:
        rate = (n*rate_this_season + k*rate_prior) / (n+k)
    where n = team_games_so_far (0 pre-season, so the blend correctly reduces to
    the prior alone — no separate pre-season branch needed, unlike the old code).

    p_60_plus / p_1_to_59 are computed directly from real per-game minutes THIS
    SEASON — no start/appearance conditioning needed, so this naturally captures
    genuine substitute cameos as well as starts hooked early. The prior can't do
    the same (history_past has no per-game breakdown, only season totals) —
    approximated via last season's start rate and a completion-rate proxy from
    average minutes-per-start (see _completion_rate_from_avg_mins).

    Appearance points then become P(1-59)*1 + P(60+)*2 in predictor.py, instead of
    assuming every "start" is worth a flat 2 points.
    """
    n = team_games_so_far
    count_appeared = sum(1 for h in history if h.get("minutes", 0) > 0)
    count_60plus = sum(1 for h in history if h.get("minutes", 0) >= 60)
    count_1to59 = count_appeared - count_60plus
    if history and "starts" in history[0]:
        total_starts = sum(h.get("starts", 0) for h in history)
    else:
        total_starts = count_60plus
    total_minutes = sum(h.get("minutes", 0) for h in history)

    start_rate_now = (total_starts / n) if n > 0 else 0.0
    minutes_rate_now = (total_minutes / (n * 90)) if n > 0 else 0.0
    p60_now = (count_60plus / n) if n > 0 else 0.0
    p1to59_now = (count_1to59 / n) if n > 0 else 0.0

    prior_start, prior_completion = position_prior
    if history_past:
        past = sorted(history_past, key=lambda s: s.get("season_name", ""))[-1]
        past_starts = past.get("starts", 0)
        past_mins = past.get("minutes", 0)
        if past_starts > 0:
            prior_start = past_starts / LAST_SEASON_GAMES
            prior_completion = _completion_rate_from_avg_mins(past_mins / past_starts)

    prior_minutes_rate = prior_start * (0.3 + 0.7 * prior_completion)
    prior_p60 = prior_start * prior_completion
    prior_p1to59 = max(0.0, prior_start - prior_p60)

    denom = n + k
    if denom <= 0:
        exp_start_pct = exp_minutes = p_60_plus = p_1_to_59 = 0.0
    else:
        exp_start_pct = (n * start_rate_now + k * prior_start) / denom
        exp_minutes = (n * minutes_rate_now + k * prior_minutes_rate) / denom
        p_60_plus = (n * p60_now + k * prior_p60) / denom
        p_1_to_59 = (n * p1to59_now + k * prior_p1to59) / denom

    exp_start_pct = round(min(1.0, exp_start_pct * availability), 3)
    exp_minutes = round(min(1.0, exp_minutes * availability), 3)
    p_60_plus = round(min(1.0, p_60_plus * availability), 4)
    p_1_to_59 = round(min(1.0, p_1_to_59 * availability), 4)

    return exp_start_pct, exp_minutes, p_60_plus, p_1_to_59


def _model_xg(
    h_id: int, a_id: int, h_fdr: int, a_fdr: int, team_rolling: dict[int, dict], league_avg_defence: float,
) -> tuple[float, float]:
    """
    Model-based (non-market) expected goals for one fixture, blending:
      Tier 2: rolling 6-game averages × opponent defensive factor × home/away factor
      Tier 3: FDR-based fallback
    Tier 2 is phased in per team via a linear taper over its first TAPER_GAMES
    played this season (0 games = pure Tier 3, TAPER_GAMES+ = pure Tier 2),
    rather than switching all-or-nothing the moment a team has any data.
    """
    # Tier 3: FDR fallback — always available, used as the pre-season prior
    h_mult = max(0.2, (5 - h_fdr) / 3)
    a_mult = max(0.2, (5 - a_fdr) / 3)
    tier3_h = LEAGUE_AVG_GOALS * h_mult
    tier3_a = LEAGUE_AVG_GOALS * a_mult

    h_roll = team_rolling.get(h_id)
    a_roll = team_rolling.get(a_id)
    if h_roll and a_roll and h_roll["attack_xg6"] > 0 and a_roll["attack_xg6"] > 0 and league_avg_defence > 0:
        # Tier 2: rolling averages × opponent defensive factor × home/away factor
        tier2_h = h_roll["attack_xg6"] * (a_roll["defence_xg6"] / league_avg_defence) * HOME_ADV_MULT
        tier2_a = a_roll["attack_xg6"] * (h_roll["defence_xg6"] / league_avg_defence) * AWAY_ADV_MULT
        w_h = min(1.0, h_roll["games_played"] / TAPER_GAMES)
        w_a = min(1.0, a_roll["games_played"] / TAPER_GAMES)
        model_h = w_h * tier2_h + (1 - w_h) * tier3_h
        model_a = w_a * tier2_a + (1 - w_a) * tier3_a
    else:
        model_h, model_a = tier3_h, tier3_a

    return round(max(0.2, model_h), 3), round(max(0.2, model_a), 3)


def _build_gw_match_xg(
    fixtures: list[dict],
    upcoming_gws: list[int],
    team_rolling: dict[int, dict],
    league_avg_defence: float,
    odds_xg: dict[int, dict[int, tuple[float, float]]],
) -> dict[int, dict[int, dict]]:
    """
    Compute per-GW per-team model xG and odds xG (kept separate — the final
    blend between them is applied later, per-request, using the user's
    odds_weight slider rather than baked into this cached fetch):
      model_team_xg/model_opp_xg: Tier 2 (rolling averages) tapered against
        Tier 3 (FDR fallback) by each team's games played this season — see _model_xg.
      odds_team_xg/odds_opp_xg: Tier 1 (Odds API), or None if unavailable for the fixture.

    Returns {gw_id: {team_id: {model_team_xg, model_opp_xg, odds_team_xg, odds_opp_xg}}}
    with per-match averages. DGW teams: accumulate then average so n_fixtures
    multiplication still works; odds are averaged only over fixtures that had odds.
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

        model_h_xg, model_a_xg = _model_xg(h_id, a_id, h_fdr, a_fdr, team_rolling, league_avg_defence)

        odds_fix = odds_xg.get(fid, {})
        if h_id in odds_fix:
            odds_h_xg, odds_a_xg = odds_fix[h_id]
        else:
            odds_h_xg = odds_a_xg = None

        for tid, model_t, model_o, odds_t, odds_o in (
            (h_id, model_h_xg, model_a_xg, odds_h_xg, odds_a_xg),
            (a_id, model_a_xg, model_h_xg, odds_a_xg, odds_h_xg),
        ):
            if tid not in accum[gw]:
                accum[gw][tid] = {
                    "model_t_sum": 0.0, "model_o_sum": 0.0, "model_n": 0,
                    "odds_t_sum": 0.0, "odds_o_sum": 0.0, "odds_n": 0,
                }
            acc = accum[gw][tid]
            acc["model_t_sum"] += model_t
            acc["model_o_sum"] += model_o
            acc["model_n"] += 1
            if odds_t is not None:
                acc["odds_t_sum"] += odds_t
                acc["odds_o_sum"] += odds_o
                acc["odds_n"] += 1

    # Convert sums to per-match averages
    result: dict[int, dict[int, dict]] = {}
    for gw, teams in accum.items():
        result[gw] = {}
        for tid, acc in teams.items():
            model_n = acc["model_n"]
            odds_n = acc["odds_n"]
            result[gw][tid] = {
                "model_team_xg": round(acc["model_t_sum"] / model_n, 3) if model_n else 0.0,
                "model_opp_xg":  round(acc["model_o_sum"] / model_n, 3) if model_n else 0.0,
                "odds_team_xg":  round(acc["odds_t_sum"] / odds_n, 3) if odds_n else None,
                "odds_opp_xg":   round(acc["odds_o_sum"] / odds_n, 3) if odds_n else None,
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

        team_lookup = {p["id"]: p["team"] for p in active_players}
        pos_lookup = {p["id"]: POSITION_MAP.get(p["element_type"], "MID") for p in active_players}
        elem_lookup = {p["id"]: p for p in active_players}

        # ── xG-based goal/assist shares (Phase 1) ─────────────────────────────
        team_xg_by_fixture, team_xa_by_fixture = build_team_xg_totals(
            [(team_lookup.get(pid), history) for pid, history, _ in results]
        )
        position_priors = build_position_priors(
            [(pos_lookup.get(pid, "MID"), history_past) for pid, _, history_past in results]
        )

        # ── Minutes model priors (Phase 3) ────────────────────────────────────
        minutes_priors = build_minutes_priors(
            [(pos_lookup.get(pid, "MID"), history_past) for pid, _, history_past in results]
        )

        # ── Bonus model (Phase 2) — refit from this season's own data ────────
        bonus_model.refit(results, pos_lookup)

        player_stats: dict[int, dict] = {}
        raw_histories: dict[int, list] = {}
        player_history_past: dict[int, list] = {}
        for player_id, history, history_past in results:
            player_stats[player_id] = _build_player_stats(history, history_past)
            tid = team_lookup.get(player_id)
            position = pos_lookup.get(player_id, "MID")
            prior = position_priors.get(position, (0.0, 0.0))
            goal_share, assist_share = compute_xg_share(
                history, history_past, tid, team_xg_by_fixture, team_xa_by_fixture, prior,
            )
            player_stats[player_id]["goal_share"] = goal_share
            player_stats[player_id]["assist_share"] = assist_share
            # DefCon + cards (Phase 2) — player-intrinsic, fixture-independent rates
            player_stats[player_id]["defcon_hit_rate"] = compute_defcon_hit_rate(history, position)
            player_stats[player_id]["card_rate"] = compute_card_rate(history)

            # Minutes model (Phase 3) — Beta-prior-blended start/minutes rates and
            # a proper P(60+)/P(1-59) split, replacing the old flat this-season-only
            # ratio and its hard pre-season-fallback cutover.
            team_games_so_far = team_rolling.get(tid, {}).get("games_played", 0)
            chance = elem_lookup.get(player_id, {}).get("chance_of_playing_next_round")
            availability = chance / 100.0 if chance is not None else 1.0
            minutes_prior = minutes_priors.get(position, (0.0, 0.0))
            exp_start_pct, exp_minutes, p_60_plus, p_1_to_59 = compute_minutes_model(
                history, history_past, team_games_so_far, availability, minutes_prior,
            )
            player_stats[player_id]["exp_start_pct"] = exp_start_pct
            player_stats[player_id]["exp_minutes"] = exp_minutes
            player_stats[player_id]["p_60_plus"] = p_60_plus
            player_stats[player_id]["p_1_to_59"] = p_1_to_59

            player_history_past[player_id] = history_past
            raw_histories[player_id] = [
                {
                    "round": h["round"],
                    "fixture": h.get("fixture"),
                    "total_points": h["total_points"],
                    "minutes": h["minutes"],
                    "opponent_team": h["opponent_team"],
                    "was_home": h.get("was_home", False),
                    "goals_scored": h.get("goals_scored", 0),
                    "assists": h.get("assists", 0),
                    "expected_goals": float(h.get("expected_goals") or 0),
                    "expected_assists": float(h.get("expected_assists") or 0),
                    "saves": h.get("saves", 0),
                    "starts": h.get("starts", 0),
                    "clean_sheets": h.get("clean_sheets", 0),
                    "yellow_cards": h.get("yellow_cards", 0),
                    "defensive_contribution": h.get("defensive_contribution", 0),
                    "bonus": h.get("bonus", 0),
                    "team_id": team_lookup.get(player_id),
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
                "defcon_hit_rate": stats["defcon_hit_rate"],
                "card_rate": stats["card_rate"],
                "exp_minutes": stats["exp_minutes"],
                "exp_start_pct": stats["exp_start_pct"],
                "p_60_plus": stats["p_60_plus"],
                "p_1_to_59": stats["p_1_to_59"],
                # Per-GW match xG (team and opponent) — model (Tier 2/3) and odds (Tier 1)
                # kept separate; final blend applied per-request in main.py using odds_weight
                "gw_match_xg": {
                    gw_id: gw_match_xg.get(gw_id, {}).get(team_id, {
                        "model_team_xg": 0.0, "model_opp_xg": 0.0,
                        "odds_team_xg": None, "odds_opp_xg": None,
                    })
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
        "player_history_past": player_history_past,
        "bootstrap_teams": bootstrap["teams"],
        "fixtures": fixtures,
        "team_rolling": team_rolling,
        "league_avg_attack": league_avg_attack,
        "league_avg_defence": league_avg_defence,
    }

    _cache = data
    _cache_time = time.time()
    return data
