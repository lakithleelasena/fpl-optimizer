from config import STRENGTH_MAX, STRENGTH_MIN, W_FIXTURE, W_HOME_AWAY, W_SEASON, W_XG


def _compute_start_likelihood(player: dict) -> float:
    """Estimate probability (0.0-1.0) that a player starts the next match."""
    stats = player["stats"]
    recent_mins = stats.get("recent_minutes", [])

    if not recent_mins:
        return 0.0

    gw_scores = []
    for mins in recent_mins:
        if mins >= 60:
            gw_scores.append(1.0)
        elif mins > 0:
            gw_scores.append(0.5)
        else:
            gw_scores.append(0.0)

    weights = list(range(1, len(gw_scores) + 1))
    total_weight = sum(weights)
    minutes_likelihood = sum(s * w for s, w in zip(gw_scores, weights)) / total_weight

    chance = player.get("chance_of_playing")
    if chance is not None:
        availability = chance / 100.0
    else:
        availability = 1.0

    return round(min(minutes_likelihood * availability, 1.0), 2)


def _compute_fixture_difficulty(opponent_strengths: list[float]) -> tuple[float, float]:
    """Return (fixture_ease, fixture_score_multiplier) from opponent strength ratings."""
    if not opponent_strengths:
        return 1.0, 1.0

    str_range = STRENGTH_MAX - STRENGTH_MIN
    ease_scores = [
        max(0.0, min(1.0, (STRENGTH_MAX - s) / str_range))
        for s in opponent_strengths
    ]
    avg_ease = sum(ease_scores) / len(ease_scores)
    multiplier = 0.5 + avg_ease
    return round(avg_ease, 2), round(multiplier, 2)


def predict_points(
    player: dict,
    w_home_away: float = W_HOME_AWAY,
    w_season: float = W_SEASON,
    w_xg: float = W_XG,
    w_fixture: float = W_FIXTURE,
) -> dict:
    stats = player["stats"]
    opponent_strengths = player.get("opponent_strengths", [])

    season_avg = stats["season_avg"]
    games_played = stats["games_played"]

    start_likelihood = _compute_start_likelihood(player)
    fixture_ease, fixture_multiplier = _compute_fixture_difficulty(opponent_strengths)

    # Home/away fraction: 1.0 = fully home, 0.0 = fully away, 0.5 = mixed/unknown
    is_home = player.get("is_home", 0.5)

    if games_played == 0:
        return {
            "predicted_points": 0.0,
            "home_away_score": 0.0,
            "season_avg": 0.0,
            "xg_score": 0.0,
            "fixture_ease": fixture_ease,
            "start_likelihood": start_likelihood,
        }

    # Signal 1: Home/Away advantage — season_avg scaled by venue
    # Away = 0.85×, Home = 1.15× (30% spread)
    home_away_score = season_avg * (0.85 + is_home * 0.30)

    # Signal 2: Season average

    # Signal 3: xG Involvement — expected_goal_involvements per game × 15
    xgi = player.get("xgi", 0.0)
    xgi_per_game = xgi / games_played if games_played > 0 else 0.0
    xg_score = xgi_per_game * 15.0

    # Signal 4: Fixture difficulty — season_avg scaled by opponent ease
    fixture_score = season_avg * fixture_multiplier

    # Normalize weights so they always sum to 1.0
    total_w = w_home_away + w_season + w_xg + w_fixture
    if total_w == 0:
        total_w = 1.0

    predicted = (
        w_home_away * home_away_score
        + w_season * season_avg
        + w_xg * xg_score
        + w_fixture * fixture_score
    ) / total_w

    predicted *= start_likelihood

    return {
        "predicted_points": round(predicted, 2),
        "home_away_score": round(home_away_score, 2),
        "season_avg": round(season_avg, 2),
        "xg_score": round(xg_score, 2),
        "fixture_ease": fixture_ease,
        "start_likelihood": start_likelihood,
    }
