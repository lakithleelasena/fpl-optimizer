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
LEAGUE_AVG_GOALS = 1.35  # EPL avg goals per team per match (Tier 3 pre-season fallback)
LAST_SEASON_GAMES = 38  # full EPL season length, used for pre-season exp_minutes/exp_start_pct fallback

# Team-level match xG model (fpl_client._model_xg / _build_gw_match_xg)
HOME_ADV_MULT = 1.10   # home teams score ~10% more than a neutral venue would suggest
AWAY_ADV_MULT = 0.90   # away teams score ~10% less
TAPER_GAMES = 10        # games until a team's Tier 2 rolling average fully replaces the Tier 3 prior
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

# Minutes-model shrinkage (fpl_client.compute_minutes_model): same Beta-Binomial
# posterior-mean idea as SHARE_SHRINKAGE_K, applied to start rate / minutes rate /
# P(60+) / P(1-59) instead of goal share. k=6 games.
MINUTES_SHRINKAGE_GAMES = 6.0

# DefCon thresholds (single-match CBIT/CBIRT count needed for the flat 2-point award) —
# verified against fantasy.premierleague.com/help/rules 2026-09-17. GKPs aren't eligible
# (outfield players only). Lives here (not predictor.py) so fpl_client.py, bonus_model.py,
# and predictor.py can all import it without a circular dependency.
DEFCON_THRESHOLD: dict[str, int | None] = {"GKP": None, "DEF": 10, "MID": 12, "FWD": 12}

SEMAPHORE_LIMIT = 20
CACHE_TTL_SECONDS = 1800  # 30 minutes
