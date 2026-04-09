from config import STRENGTH_MAX, STRENGTH_MIN, W_FIXTURE, W_FORM, W_HOME_AWAY, W_SEASON, W_THREAT, W_XGC, W_XGI


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
    w_xgi: float = W_XGI,
    w_fixture: float = W_FIXTURE,
    w_form: float = W_FORM,
    w_threat: float = W_THREAT,
    w_xgc: float = W_XGC,
) -> dict:
    stats = player["stats"]
    opponent_strengths = player.get("opponent_strengths", [])
    position = player.get("position", "MID")

    season_avg = stats["season_avg"]
    games_played = stats["games_played"]

    start_likelihood = _compute_start_likelihood(player)
    fixture_ease, fixture_multiplier = _compute_fixture_difficulty(opponent_strengths)

    is_home = player.get("is_home", 0.5)

    if games_played == 0:
        return {
            "predicted_points": 0.0,
            "home_away_score": 0.0,
            "season_avg": 0.0,
            "xg_score": 0.0,
            "fixture_ease": fixture_ease,
            "start_likelihood": start_likelihood,
            "form_score": 0.0,
            "threat_score": 0.0,
            "xgc_score": 0.0,
        }

    # Signal 1: Home/Away advantage
    home_away_score = season_avg * (0.85 + is_home * 0.30)

    # Signal 2: Season average (used as-is)

    # Signal 3: Fixture difficulty
    fixture_score = season_avg * fixture_multiplier

    # Signal 4: FPL Form (rolling 30-day avg — already in points units)
    form_score = player.get("form", 0.0)

    # Signal 5: xG Involvement (MID/FWD only)
    xgi = player.get("xgi", 0.0)
    xg_score = (xgi / games_played) * 15.0 if games_played > 0 else 0.0

    # Signal 6: ICT Threat (MID/FWD only)
    threat_raw = player.get("threat", 0.0)
    threat_score = (threat_raw / games_played) / 10.0 if games_played > 0 else 0.0

    # Signal 7: xGC — inverted defensive quality (GKP/DEF only)
    xgc_raw = player.get("xgc", 0.0)
    xgc_per_game = xgc_raw / games_played if games_played > 0 else 1.5
    xgc_score = max(0.0, (3.0 - xgc_per_game) * 2.0)

    # Position-aware weighted sum — only include relevant signals in numerator AND denominator
    is_defender = position in ("GKP", "DEF")

    if is_defender:
        # GKP/DEF: use season, home_away, fixture, form, xgc — exclude xgi and threat
        numerator = (
            w_home_away * home_away_score
            + w_season * season_avg
            + w_fixture * fixture_score
            + w_form * form_score
            + w_xgc * xgc_score
        )
        total_w = w_home_away + w_season + w_fixture + w_form + w_xgc
    else:
        # MID/FWD: use season, home_away, fixture, form, xgi, threat — exclude xgc
        numerator = (
            w_home_away * home_away_score
            + w_season * season_avg
            + w_fixture * fixture_score
            + w_form * form_score
            + w_xgi * xg_score
            + w_threat * threat_score
        )
        total_w = w_home_away + w_season + w_fixture + w_form + w_xgi + w_threat

    if total_w == 0:
        total_w = 1.0

    predicted = (numerator / total_w) * start_likelihood

    return {
        "predicted_points": round(predicted, 2),
        "home_away_score": round(home_away_score, 2),
        "season_avg": round(season_avg, 2),
        "xg_score": round(xg_score, 2),
        "fixture_ease": fixture_ease,
        "start_likelihood": start_likelihood,
        "form_score": round(float(form_score), 2),
        "threat_score": round(threat_score, 2),
        "xgc_score": round(xgc_score, 2),
    }
