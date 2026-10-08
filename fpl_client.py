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
    FDR_SENSITIVITY,
    FIXTURES_URL,
    HOME_ADV_MULT,
    LAST_SEASON_GAMES,
    LEAGUE_AVG_GOALS,
    LEAGUE_AVG_SHRINK_MATCHES,
    MINUTES_SHRINKAGE_GAMES,
    POSITION_MAP,
    SEMAPHORE_LIMIT,
    SHARE_PRIOR_RELIABILITY_N90,
    SHARE_SHRINKAGE_K,
    TIER2_SHRINKAGE_GAMES,
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


def build_team_xg_rolling(
    fixtures: list[dict], team_xg_by_fixture: dict[int, dict[int, float]],
) -> dict[int, dict]:
    """
    Same shape and purpose as _build_team_rolling (last-6-finished-matches rolling
    average, used as Tier 2's underlying signal), but built from real xG — summed
    from every player's own FPL-reported xG per fixture, via build_team_xg_totals —
    instead of actual goals scored/conceded. Phase 4, PREDICTION_MODEL_PLAN.md:
    "use xG rather than goals as the target" for the rolling-form component.
    _build_team_rolling (actual goals) is kept separately and unchanged — it still
    feeds the Team Overview tab's GF6/GA6 display, which is explicitly meant to
    show real goals, not a model estimate.
    """
    finished = [f for f in fixtures if f.get("finished") and f.get("team_h_score") is not None]
    finished.sort(key=lambda f: (f.get("event") or 0, f.get("kickoff_time") or ""))

    team_history: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for f in finished:
        fid = f["id"]
        h, a = f["team_h"], f["team_a"]
        h_xg = team_xg_by_fixture.get(h, {}).get(fid, 0.0)
        a_xg = team_xg_by_fixture.get(a, {}).get(fid, 0.0)
        team_history[h].append((h_xg, a_xg))
        team_history[a].append((a_xg, h_xg))

    rolling: dict[int, dict] = {}
    for tid, matches in team_history.items():
        last6 = matches[-6:]
        attack = sum(m[0] for m in last6) / len(last6)
        defence = sum(m[1] for m in last6) / len(last6)
        rolling[tid] = {
            "attack_xg6": round(attack, 3),
            "defence_xg6": round(defence, 3),
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

    # (Goal/assist shares, saves rate, DefCon, and cards are all computed outside
    # this function now — see compute_xg_share/compute_saves_rate/
    # compute_defcon_hit_rate/compute_card_rate — since they each need data this
    # function doesn't have access to: team-xG totals, position, etc. See Phase 1
    # and Phase 5 in PREDICTION_MODEL_PLAN.md.)

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
                # Reliability-weight last season's share toward the position average so a
                # few minutes of last-season data can't produce an absurd share.
                past_n90 = past_mins / 90.0
                rel = SHARE_PRIOR_RELIABILITY_N90
                w = past_n90 / (past_n90 + rel) if rel > 0 else 1.0
                prior_goal = w * (float(past.get("expected_goals") or 0) / est_team_goals) + (1 - w) * prior_goal
                prior_assist = w * (float(past.get("expected_assists") or 0) / est_team_goals) + (1 - w) * prior_assist

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


def compute_saves_rate(
    history: list[dict], team_xg_by_fixture: dict[int, dict[int, float]], window: int = 6,
) -> float:
    """
    Saves per unit of opponent attacking xG actually faced — an opponent-difficulty
    -adjusted save rate (Phase 5, PREDICTION_MODEL_PLAN.md), replacing a flat
    saves-per-game average that didn't vary by fixture difficulty at all. From the
    goalkeeper's last `window` played games: total actual saves made / total real
    opponent xG they faced in those SAME matches (looked up via
    team_xg_by_fixture — the real per-fixture xG built in Phase 1, keyed by the
    opponent they actually played, not their own team).

    Multiply this rate by THIS WEEK's match_opp_xg at prediction time to project
    expected saves for the upcoming fixture — the same
    rate-times-this-weeks-opponent-xG pattern as compute_xg_share's goal/assist
    shares (share x match_team_xg).
    """
    played = [h for h in history if h.get("minutes", 0) > 0]
    recent = played[-window:]
    if not recent:
        return 0.0
    total_saves = sum(h.get("saves", 0) for h in recent)
    total_opp_xg_faced = 0.0
    for h in recent:
        opp = h.get("opponent_team")
        fid = h.get("fixture")
        if opp is None or fid is None:
            continue
        total_opp_xg_faced += team_xg_by_fixture.get(opp, {}).get(fid, 0.0)
    if total_opp_xg_faced <= 0:
        return 0.0
    return round(total_saves / total_opp_xg_faced, 4)


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


def compute_league_avg_goals(fixtures: list[dict]) -> float:
    """
    Live league-average goals per team per match from finished fixtures, shrunk
    toward the LEAGUE_AVG_GOALS prior (weight LEAGUE_AVG_SHRINK_MATCHES team-matches)
    so one high- or low-scoring gameweek doesn't swing it. Tier 3's level anchors on
    this — it was previously a fixed 1.35 combined with an FDR multiplier that
    averaged ~0.67, making Tier 3 predict ~0.86 goals against ~1.41 actual.
    """
    finished = [f for f in fixtures if f.get("finished") and f.get("team_h_score") is not None]
    n = 2 * len(finished)
    if n == 0:
        return LEAGUE_AVG_GOALS
    mean = sum(f["team_h_score"] + f["team_a_score"] for f in finished) / n
    return (n * mean + LEAGUE_AVG_SHRINK_MATCHES * LEAGUE_AVG_GOALS) / (n + LEAGUE_AVG_SHRINK_MATCHES)


def compute_mean_fdr(fixtures: list[dict]) -> float:
    """Mean FDR over every side of every fixture in the season list (known in advance,
    so no look-ahead) — Tier 3 centres on this so an average-difficulty fixture predicts
    the league-average goals. It's ~3.1 this season, not exactly 3."""
    vals = [v for f in fixtures for v in (f.get("team_h_difficulty"), f.get("team_a_difficulty")) if v]
    return sum(vals) / len(vals) if vals else 3.0


def tier3_xg(fdr: float, league_avg: float, mean_fdr: float = 3.0) -> float:
    """
    Tier 3 (FDR fallback): the live league-average goals, scaled up or down by
    FDR_SENSITIVITY per FDR point away from the mean-difficulty fixture. Centred, so it
    averages the league mean (the old `LEAGUE_AVG_GOALS * (5-FDR)/3` averaged ~0.67x).
    """
    return league_avg * max(0.3, 1 + FDR_SENSITIVITY * (mean_fdr - fdr))


def tier2_xg(
    attack_xg6: float | None, opp_defence_xg6: float | None, league_avg_defence: float, is_home: bool,
) -> float | None:
    """Tier 2 (rolling real xG): attack x (opponent defence / league-average defence) x
    home/away multiplier. None if either team has no rolling data yet. A genuine 0.0
    average is valid data — hence explicit None checks, not truthiness."""
    if attack_xg6 is None or opp_defence_xg6 is None or league_avg_defence <= 0:
        return None
    mult = HOME_ADV_MULT if is_home else AWAY_ADV_MULT
    return attack_xg6 * (opp_defence_xg6 / league_avg_defence) * mult


def tier2_weight(games_played: int) -> float:
    """Weight on Tier 2 in the Tier 2/3 model blend: n / (n + TIER2_SHRINKAGE_GAMES) —
    the same shrinkage form as Phases 1 and 3. Replaces a linear 10-game taper."""
    return games_played / (games_played + TIER2_SHRINKAGE_GAMES) if games_played > 0 else 0.0


def _model_xg(
    h_id: int, a_id: int, h_fdr: int, a_fdr: int, team_rolling: dict[int, dict],
    league_avg_defence: float, league_avg: float, mean_fdr: float,
) -> tuple[float, float]:
    """
    Model-based (non-market) expected goals for one fixture, blending:
      Tier 2: rolling real-xG averages x opponent defensive factor x home/away factor
      Tier 3: centred FDR fallback (live league-average goals scaled by FDR)
    Tier 2's weight grows with each team's games played, n/(n+TIER2_SHRINKAGE_GAMES),
    rather than switching all-or-nothing the moment a team has any data.
    """
    tier3_h = tier3_xg(h_fdr, league_avg, mean_fdr)
    tier3_a = tier3_xg(a_fdr, league_avg, mean_fdr)

    h_roll = team_rolling.get(h_id)
    a_roll = team_rolling.get(a_id)
    tier2_h = tier2_xg(h_roll["attack_xg6"] if h_roll else None,
                       a_roll["defence_xg6"] if a_roll else None, league_avg_defence, True)
    tier2_a = tier2_xg(a_roll["attack_xg6"] if a_roll else None,
                       h_roll["defence_xg6"] if h_roll else None, league_avg_defence, False)
    if tier2_h is not None and tier2_a is not None:
        w_h = tier2_weight(h_roll["games_played"])
        w_a = tier2_weight(a_roll["games_played"])
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
    league_avg: float,
    mean_fdr: float,
) -> dict[int, dict[int, dict]]:
    """
    Compute per-GW per-team model xG and odds xG (kept separate — the final
    blend between them is applied later, per-request, using the user's
    odds_weight slider rather than baked into this cached fetch):
      model_team_xg/model_opp_xg: Tier 2 (rolling real xG) blended with centred
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

        model_h_xg, model_a_xg = _model_xg(
            h_id, a_id, h_fdr, a_fdr, team_rolling, league_avg_defence, league_avg, mean_fdr,
        )

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
        # Computed here (before Team rolling stats, below) because Tier 2's model
        # now runs on real xG too (Phase 4) — team_xg_by_fixture needs to exist
        # before build_team_xg_rolling can use it.
        team_xg_by_fixture, team_xa_by_fixture = build_team_xg_totals(
            [(team_lookup.get(pid), history) for pid, history, _ in results]
        )

        # ── Team rolling stats (last ≤6 finished matches) ───────────────────
        # _build_team_rolling (actual goals) still feeds the Team Overview tab's
        # GF6/GA6 display, which is meant to show real goals. Tier 2's own model
        # input (Phase 4) is build_team_xg_rolling — real xG summed from every
        # player's own FPL-reported xG per fixture, per the review's "use xG
        # rather than goals as the target" recommendation.
        team_rolling = _build_team_rolling(fixtures)
        team_xg_rolling = build_team_xg_rolling(fixtures, team_xg_by_fixture)
        attacks = [v["attack_xg6"] for v in team_xg_rolling.values() if v["attack_xg6"] > 0]
        defences = [v["defence_xg6"] for v in team_xg_rolling.values() if v["defence_xg6"] > 0]
        league_avg_attack = round(sum(attacks) / len(attacks), 3) if attacks else LEAGUE_AVG_GOALS
        league_avg_defence = round(sum(defences) / len(defences), 3) if defences else LEAGUE_AVG_GOALS

        # ── Odds API match xG (Tier 1) ───────────────────────────────────────
        upcoming_fix_list = [f for f in fixtures if f.get("event") in upcoming_gws]
        odds_xg = await fetch_odds_xg(teams, upcoming_fix_list, current_gw=next_gw)

        # ── Per-GW match xG for each team ────────────────────────────────────
        # Tier 3 anchors on the live league-average goals per team and the fixture
        # list's actual mean FDR (see tier3_xg), not a fixed 1.35 and an assumed FDR of 3.
        league_avg_goals_now = compute_league_avg_goals(fixtures)
        mean_fdr = compute_mean_fdr(fixtures)
        gw_match_xg = _build_gw_match_xg(
            fixtures, upcoming_gws, team_xg_rolling, league_avg_defence, odds_xg,
            league_avg_goals_now, mean_fdr,
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
            # Opponent-adjusted save rate (Phase 5) — multiplied by this week's
            # match_opp_xg at prediction time, same pattern as goal/assist shares.
            player_stats[player_id]["saves_per_opp_xg"] = compute_saves_rate(history, team_xg_by_fixture)

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
                    # Extra scoring events — only used by the component-level backtest
                    # (backtest_accuracy) to rebuild each player's actual points by component.
                    "goals_conceded": h.get("goals_conceded", 0),
                    "red_cards": h.get("red_cards", 0),
                    "own_goals": h.get("own_goals", 0),
                    "penalties_saved": h.get("penalties_saved", 0),
                    "penalties_missed": h.get("penalties_missed", 0),
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
                "saves_per_opp_xg": stats["saves_per_opp_xg"],
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
