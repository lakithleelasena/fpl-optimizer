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

BUDGET = 1000  # £100.0m stored as tenths
SQUAD_SIZE = 15
STARTING_XI = 11

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

SQUAD_COMPOSITION = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
MIN_STARTING = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}

MAX_PER_TEAM = 3

W_HOME_AWAY = 0.05
W_SEASON = 0.20
W_XGI = 0.10
W_FIXTURE = 0.35
W_FORM = 0.10
W_THREAT = 0.10
W_XGC = 0.20

# New formula multipliers (1.0 = neutral)
W_FORM_FACTOR = 1.0   # scales form adjustment (0 = ignore form, 2 = double sensitivity)
W_CS_FACTOR   = 1.0   # scales clean-sheet bonus for GKP/DEF
W_ATK_FACTOR  = 1.0   # scales goal/assist contribution for MID/FWD (and rare DEF/GKP goals)

# FPL team strength rating bounds — 2026-27 API uses 1-5 integer scale
STRENGTH_MIN = 2
STRENGTH_MAX = 5

SEMAPHORE_LIMIT = 20
CACHE_TTL_SECONDS = 1800  # 30 minutes
