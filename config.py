from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root

BASE_URL = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{BASE_URL}/bootstrap-static/"
FIXTURES_URL = f"{BASE_URL}/fixtures/"
ELEMENT_SUMMARY_URL = f"{BASE_URL}/element-summary/{{player_id}}/"

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
LEAGUE_AVG_GOALS = 1.35  # EPL avg goals per team per match — pre-season prior; replaced by the live season average as games are played (see LEAGUE_AVG_SHRINK_MATCHES)
LAST_SEASON_GAMES = 38  # full EPL season length, used for pre-season exp_minutes/exp_start_pct fallback

# Team-level match xG model (fpl_client._model_xg / _build_gw_match_xg)
HOME_ADV_MULT = 1.10   # home teams score ~10% more than a neutral venue would suggest
AWAY_ADV_MULT = 0.90   # away teams score ~10% less
# Tier 2 (rolling xG) vs Tier 3 (FDR) model blend, per team: w_tier2 = n / (n + TIER2_SHRINKAGE_GAMES),
# n = games the team has played. Replaces a linear 10-game taper that gave Tier 2 full weight
# at game 10 and (with a low-biased Tier 3) dragged early-season predictions ~0.4 goals low.
# Backtest GW2-5 (Tier 3 centred): best static Tier 2 weight ~0.3 at 1-4 games played, which
# n/(n+6) reproduces (0.14-0.40) — and it's the same shrinkage family as Phases 1 and 3.
TIER2_SHRINKAGE_GAMES = 6.0
# Tier 3 sensitivity: each FDR point away from the fixture-list mean FDR moves the prediction
# by +/- this fraction of the league-average goals (centred, so Tier 3 averages the league mean).
FDR_SENSITIVITY = 0.25
# Live league-average goals per team: blend this season's actual mean with LEAGUE_AVG_GOALS,
# weighting the prior as this many team-matches (40 = two gameweeks), so one wild gameweek
# doesn't swing Tier 3's level.
LEAGUE_AVG_SHRINK_MATCHES = 40
W_ODDS_WEIGHT = 0.6     # default blend weight for Tier 1 (odds) vs model (Tier 2/3) xG, when odds are available

BUDGET = 1000  # £100.0m stored as tenths
SQUAD_SIZE = 15
STARTING_XI = 11

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

SQUAD_COMPOSITION = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
MIN_STARTING = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}

MAX_PER_TEAM = 3

# Objective weight on bench (non-starting) squad members' predicted points.
# Starters are weighted 1.0; this must stay well below that so bench quality
# never outbids a genuine starting-XI improvement. But it must be large enough
# to stop the LP treating bench slots as pure budget filler — bench players
# can be auto-subbed in if a starter blanks, so a 0%-chance player is worse
# bench cover than a cheap fringe player who might actually get minutes.
BENCH_WEIGHT = 0.2

# New formula multipliers (1.0 = neutral)
W_FORM_FACTOR = 1.0   # scales form adjustment (0 = ignore form, 2 = double sensitivity)
W_CS_FACTOR   = 1.0   # scales clean-sheet bonus for GKP/DEF
W_ATK_FACTOR  = 1.0   # scales goal/assist contribution for MID/FWD (and rare DEF/GKP goals)

# Goal/assist share shrinkage (fpl_client._compute_xg_share): weight given to the
# last-season-at-club prior versus this-season's own xG-per-90s-played sample.
# share = (n90*share_this_season + k*share_prior) / (n90+k) — k=6 means the prior
# still gets 1/3 the weight of a full 90-minute season of current-season xG data by
# GW6-ish, fading out as more of this season accumulates. Tune via backtest.
SHARE_SHRINKAGE_K = 6.0
# Reliability of the last-season share prior: that prior is blended toward the position
# average with weight w = past_n90 / (past_n90 + this), past_n90 = last season's minutes/90.
# Without it a player with a tiny last-season sample gets an absurd share (e.g. 6 minutes and
# 0.18 xG -> "200% of team xG"), which blew up a defender's predicted goals to ~1.5/match.
# 6 matches the k above (a 9-minute cameo ~ 0 weight, a 2,700-minute season ~ 83%).
SHARE_PRIOR_RELIABILITY_N90 = 6.0

# Minutes-model (fpl_client.compute_minutes_model): this season's per-game minutes are
# recency-weighted (a game `a` games ago counts DECAY**a as much as the latest) and then
# shrunk toward last season's start rate / the position average with a Beta-Binomial-style
# prior worth MINUTES_SHRINKAGE_GAMES effective games, same idea as SHARE_SHRINKAGE_K.
# Role and minutes change in steps (rotation, injury, a manager's call), so recent games are
# far more informative than the whole-season average, and the prior should fade fast.
# Backtest GW2-5 (1,668 player-gameweeks): flat average with k=6 -> P(60+) Brier 0.182;
# k=1.5 flat 0.151; decay 0.5 + k=0.75 -> 0.134 (MSE of player points 7.74 -> 7.51).
# Leave-one-gameweek-out picks decay 0.3 / k 0.5-0.75 every time; improvement is flat across
# decay 0.3-0.5, so the less extreme 0.5 is used.
MINUTES_SHRINKAGE_GAMES = 0.75
MINUTES_RECENCY_DECAY = 0.5

# DefCon thresholds (single-match CBIT/CBIRT count needed for the flat 2-point award) —
# verified against fantasy.premierleague.com/help/rules 2026-09-17. GKPs aren't eligible
# (outfield players only). Lives here (not predictor.py) so fpl_client.py, bonus_model.py,
# and predictor.py can all import it without a circular dependency.
DEFCON_THRESHOLD: dict[str, int | None] = {"GKP": None, "DEF": 10, "MID": 12, "FWD": 12}

# DefCon hit rate (fpl_client.compute_defcon_hit_rate): P(reaching the threshold | plays 60+ minutes),
# a player's own rate over this season's 60+ minute games shrunk toward the position's league rate
# with weight DEFCON_SHRINKAGE_GAMES such games. Over ~4 games a raw per-player rate is noisier than
# just using the position average (backtest GW2-5 Brier 0.088 raw vs 0.081 position mean vs 0.075
# shrunk), and conditioning on 60+ minutes stops cameo appearances diluting the rate while
# exp_minutes then scales it a second time. The league rate itself is shrunk toward the fallback
# below by DEFCON_PRIOR_PSEUDO_GAMES pseudo-appearances, so the first gameweeks don't use a
# tiny-sample rate; fallbacks are this season's GW1-5 league rates among 60+ minute appearances
# (FWD threshold is essentially never reached).
DEFCON_SHRINKAGE_GAMES = 4.0
DEFCON_PRIOR_PSEUDO_GAMES = 20.0
DEFCON_PRIOR_FALLBACK: dict[str, float] = {"GKP": 0.0, "DEF": 0.25, "MID": 0.12, "FWD": 0.0}

# Card rate (fpl_client.compute_card_rate): expected card points per 90 minutes = a player's yellows
# per 90, shrunk toward the position's league yellow rate with weight CARD_SHRINKAGE_N90 90-minute
# periods, plus the league red-card rate x 3 (reds are too rare to say anything per player). Over a
# handful of games a player-specific yellow rate is mostly noise (backtest GW2-5: MSE 0.140 for the
# old per-appearance rate, 0.117 shrunk, 0.116 pure league average), and per-90 scales with
# exp_minutes properly. League rates are shrunk toward the fallbacks below (this season's GW1-5
# league rates) by CARD_PRIOR_PSEUDO_N90 pseudo-periods early on.
CARD_SHRINKAGE_N90 = 8.0
CARD_PRIOR_PSEUDO_N90 = 100.0
CARD_YELLOW_FALLBACK_PER90: dict[str, float] = {"GKP": 0.03, "DEF": 0.17, "MID": 0.19, "FWD": 0.23}
CARD_RED_FALLBACK_PER90 = 0.0055

# Penalties (explicit model — fpl_client.compute_pen_taker_chain / predictor.predict_points).
# Rates are league-wide literature values, not locally fit (we have no per-team penalty data):
# The Analyst counted ~0.23 awarded per match (0.115 per team) over the first 13 matchdays of
# 2024/25 and ~107 awards (0.28 per match, 0.14 per team) in 2023/24; conversion was 82.6% /
# 89.7% in those samples against a long-run ~78-80%. 0.12 / 0.80 sit in the middle — calibrate
# against takers' goal residuals once more gameweeks exist (PENALTY_AWARD_RATE is the key knob).
# A team's award rate scales with its attacking strength: rate * match_team_xg / LEAGUE_AVG_GOALS.
PENALTY_AWARD_RATE_PER_MATCH = 0.12  # penalties awarded to ONE team per match, at league-average attack
PENALTY_CONVERSION_RATE = 0.80       # share of penalties scored
PENALTY_XG = 0.76                    # Opta xG of a penalty — FPL's expected_goals includes penalties
# P(the first AVAILABLE listed taker actually takes it) — the rest goes to whoever is on the pitch.
PENALTY_TAKER_RELIABILITY = 0.90
# Nominal share of a team's penalties by FPL penalties_order when everyone is fit; used only to
# strip the EXPECTED penalty xG out of a player's historical xG so open-play shares aren't
# double-counted (we can't tell which past games had a penalty).
PENALTY_NOMINAL_WEIGHTS: dict[int, float] = {1: 0.90, 2: 0.06, 3: 0.02, 4: 0.01, 5: 0.01}
PENALTY_MISS_PTS = -2
# Goalkeepers: penalties faced scale with the OPPONENT's attack, same award rate as above.
PENALTY_SAVE_RATE = 0.21             # ~1 in 5 penalties saved by keepers
PENALTY_SAVE_PTS = 5

SEMAPHORE_LIMIT = 20
CACHE_TTL_SECONDS = 1800  # 30 minutes
