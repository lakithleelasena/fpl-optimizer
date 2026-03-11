from config import STRENGTH_MAX, STRENGTH_MIN, W_FIXTURE, W_MOMENTUM, W_OPPONENT, W_SEASON


def _compute_start_likelihood(player: dict) -> float:
    """Estimate probability (0.0-1.0) that a player starts the next match.

    Based on:
    - Recent minutes pattern (last 5 GWs): 60+ mins = started, <60 = sub/benched, 0 = not used
    - FPL's chance_of_playing_next_round flag (injury/suspension doubts)
    """
    stats = player["stats"]
    recent_mins = stats.get("recent_minutes", [])

    if not recent_mins:
        return 0.0

    # Score each recent GW: 1.0 if started (60+ mins), 0.5 if subbed on, 0.0 if unused
    gw_scores = []
    for mins in recent_mins:
        if mins >= 60:
            gw_scores.append(1.0)
        elif mins > 0:
            gw_scores.append(0.5)
        else:
            gw_scores.append(0.0)

    # Weight recent GWs more heavily (most recent = highest weight)
    weights = list(range(1, len(gw_scores) + 1))
    total_weight = sum(weights)
    minutes_likelihood = sum(s * w for s, w in zip(gw_scores, weights)) / total_weight

    # Factor in FPL's chance_of_playing flag
    chance = player.get("chance_of_playing")
    if chance is not None:
        availability = chance / 100.0
    else:
        # None means no doubt — player is available
        availability = 1.0

    return round(min(minutes_likelihood * availability, 1.0), 2)


def _compute_fixture_difficulty(opponent_strengths: list[float]) -> tuple[float, float]:
    """Return (fixture_ease, fixture_score_multiplier) from a list of opponent strength ratings.

    Strength ratings are FPL's strength_overall values (home+away averaged).
    Higher strength = tougher opponent = lower ease.

    ease ranges 0.0 (hardest possible) → 1.0 (easiest possible).
    The returned multiplier scales season_avg: range 0.5 (hardest) → 1.5 (easiest).
    """
    if not opponent_strengths:
        return 1.0, 1.0  # neutral when no fixture data

    str_range = STRENGTH_MAX - STRENGTH_MIN
    ease_scores = [
        max(0.0, min(1.0, (STRENGTH_MAX - s) / str_range))
        for s in opponent_strengths
    ]
    avg_ease = sum(ease_scores) / len(ease_scores)
    multiplier = 0.5 + avg_ease  # 0.5 (hardest) → 1.5 (easiest)
    return round(avg_ease, 2), round(multiplier, 2)


def predict_points(
    player: dict,
    w_opponent: float = W_OPPONENT,
    w_season: float = W_SEASON,
    w_momentum: float = W_MOMENTUM,
    w_fixture: float = W_FIXTURE,
) -> dict:
    stats = player["stats"]
    opponents = player["opponents"]
    opponent_strengths = player.get("opponent_strengths", [])

    season_avg = stats["season_avg"]
    games_played = stats["games_played"]

    start_likelihood = _compute_start_likelihood(player)
    fixture_ease, fixture_multiplier = _compute_fixture_difficulty(opponent_strengths)

    if games_played == 0:
        return {
            "predicted_points": 0.0,
            "opponent_score": 0.0,
            "season_avg": 0.0,
            "momentum": 0.0,
            "fixture_ease": fixture_ease,
            "start_likelihood": start_likelihood,
        }

    # Signal 1: Opponent-specific historical score
    if opponents:
        opp_scores = []
        for opp_id in opponents:
            opp_history = stats["opponent_points"].get(opp_id)
            if opp_history:
                opp_scores.append(sum(opp_history) / len(opp_history))
            else:
                opp_scores.append(season_avg)
        opponent_score = sum(opp_scores) / len(opp_scores)
    else:
        opponent_score = season_avg

    # Signal 2: Season average

    # Signal 3: Momentum (last 3 GWs with minutes)
    recent = stats["recent_points"]
    momentum = sum(recent) / len(recent) if recent else season_avg

    # Signal 4: Fixture difficulty — season_avg scaled by opponent strength ease
    fixture_score = season_avg * fixture_multiplier

    # Normalize weights so they always sum to 1.0 regardless of values
    total_w = w_opponent + w_season + w_momentum + w_fixture
    if total_w == 0:
        total_w = 1.0

    predicted = (
        w_opponent * opponent_score
        + w_season * season_avg
        + w_momentum * momentum
        + w_fixture * fixture_score
    ) / total_w

    # Scale by starting likelihood — rotation/bench players get reduced predictions
    predicted *= start_likelihood

    return {
        "predicted_points": round(predicted, 2),
        "opponent_score": round(opponent_score, 2),
        "season_avg": round(season_avg, 2),
        "momentum": round(momentum, 2),
        "fixture_ease": fixture_ease,
        "start_likelihood": start_likelihood,
    }
