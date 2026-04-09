BASE_URL = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{BASE_URL}/bootstrap-static/"
FIXTURES_URL = f"{BASE_URL}/fixtures/"
ELEMENT_SUMMARY_URL = f"{BASE_URL}/element-summary/{{player_id}}/"

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

# FPL team strength rating bounds (from bootstrap-static strength_overall fields)
STRENGTH_MIN = 975
STRENGTH_MAX = 1365

SEMAPHORE_LIMIT = 20
CACHE_TTL_SECONDS = 1800  # 30 minutes
