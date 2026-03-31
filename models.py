from typing import List, Optional

from pydantic import BaseModel


class OptimizeRequest(BaseModel):
    budget: int = 1000
    w_home_away: float = 0.05
    w_season: float = 0.20
    w_xgi: float = 0.10
    w_fixture: float = 0.35
    w_form: float = 0.10
    w_threat: float = 0.10
    w_xgc: float = 0.20


class PlayerOut(BaseModel):
    id: int
    name: str
    team: str
    team_id: int
    position: str
    cost: float
    predicted_points: float
    home_away_score: float
    season_avg: float
    xg_score: float
    fixture_ease: float
    start_likelihood: float
    chance_of_playing: Optional[int] = None
    minutes: int
    total_points: int
    gw_pts: Optional[List[float]] = None  # per-GW predicted pts [gw1, gw2, gw3]
    form_score: float = 0.0
    threat_score: float = 0.0
    xgc_score: float = 0.0


class SquadPlayer(PlayerOut):
    is_starter: bool


class OptimizeResponse(BaseModel):
    starters: List[SquadPlayer]
    bench: List[SquadPlayer]
    total_cost: float
    total_predicted_points: float


# --- Transfer advice models ---

class TransferRequest(BaseModel):
    current_team: List[int]          # list of 15 player IDs
    free_transfers: int = 1
    budget_in_bank: int = 0          # tenths of £ (e.g. 5 = £0.5m)
    chips_available: List[str] = []  # "wildcard", "free_hit", "bench_boost", "triple_captain"
    w_home_away: float = 0.05
    w_season: float = 0.20
    w_xgi: float = 0.10
    w_fixture: float = 0.35
    w_form: float = 0.10
    w_threat: float = 0.10
    w_xgc: float = 0.20


class TransferSuggestion(BaseModel):
    transfer_out: PlayerOut
    transfer_in: PlayerOut
    points_gain: float   # gross gain before hit penalty
    is_hit: bool


class ChipRecommendation(BaseModel):
    chip: Optional[str] = None
    reason: str


class TransferAdviceResponse(BaseModel):
    starters: List[SquadPlayer]
    bench: List[SquadPlayer]
    transfers: List[TransferSuggestion]
    chip_recommendation: ChipRecommendation
    hits_required: int
    net_points_gain: float
    captain_id: Optional[int] = None
    vice_captain_id: Optional[int] = None
    total_predicted_3gw: float = 0.0  # sum of starters' 3GW predicted points
