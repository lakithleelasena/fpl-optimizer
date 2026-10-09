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

# Penalty saves (Phase 5, PREDICTION_MODEL_PLAN.md): a small flat GKP-only term,
# not fixture-specific — we don't have team-level penalty-award data (see Phase
# 4's scoped-out penalty split), so this uses commonly-cited league-wide rates
# rather than a per-team estimate. Literature defaults, not locally fit.
PENALTY_AWARD_RATE_PER_MATCH = 0.09  # ~1 penalty per team roughly every 11 games
PENALTY_SAVE_RATE = 0.21             # ~1 in 5 penalties saved by keepers
PENALTY_SAVE_PTS = 5

SEMAPHORE_LIMIT = 20
CACHE_TTL_SECONDS = 1800  # 30 minutes
