# Prediction Model Improvement Plan

Status as of 2026-09-17. Tracks the multi-phase rebuild of the player/team prediction
formula, agreed after comparing the current model against a detailed external review.
Each phase is meant to be implemented as its own separate step/session — check off
phases as they land so future sessions can pick up from here without re-deriving context.

**Backtest scope (confirmed):** this season only. We do not want cross-season
backtesting (no `vaastav/Fantasy-Premier-League` historical archive, no
`football-data.co.uk` historical odds import). Historical *odds* for this season are
valuable and already being collected — see `odds_history.json` (archiving started
GW5; GW1-4 odds are permanently unrecoverable, already established and accepted).

---

## Phase 0 — DONE (2026-09-17)

Verified the full scoring table live against `fantasy.premierleague.com/help/rules`
(rendered via browser, not memory — the static page shell has no content, the table
values only appear after expanding the accordion). Confirmed table:

| Action | Points | Our code before | Fixed? |
|---|---|---|---|
| Playing up to 60 min | 1 | — | already correct |
| Playing 60+ min | 2 | — | already correct |
| Goal — GKP | **10** | 6 | ✅ fixed |
| Goal — DEF | 6 | 6 | already correct |
| Goal — MID | 5 | 5 | already correct |
| Goal — FWD | 4 | 4 | already correct |
| Assist | 3 | 3 | already correct |
| Clean sheet — GKP | **4** | 6 | ✅ fixed |
| Clean sheet — DEF | 4 | 4 | already correct |
| Clean sheet — MID | **1** | 0 | ✅ fixed |
| Clean sheet — FWD | 0 | 0 | already correct |
| Save (per 3) | 1 | 1 | already correct |
| DefCon — DEF (10+ CBIT, single match, capped at 2) | 2 | n/a | still unmodeled (Phase 2) |
| DefCon — MID/FWD (12+ CBIRT, single match, capped at 2) | 2 | n/a | still unmodeled (Phase 2) |
| Penalty save | 5 | n/a | still unmodeled (Phase 5) |
| Penalty miss | -2 | n/a | still unmodeled |
| Bonus | 1-3 | n/a | still unmodeled (Phase 2) |
| GC per 2 (GKP/DEF) | -1 | approximated as `floor(λ/2)` | ✅ fixed (see below) |
| Yellow card | -1 | n/a | still unmodeled (Phase 2) |
| Red card | -3 | n/a | still unmodeled |
| Own goal | -2 | n/a | still unmodeled |

Two real bugs fixed in `predictor.py`:
1. `_PTS_PER_GOAL["GKP"]`: 6 → **10**.
2. `_CS_PTS`: `{"GKP": 6, "MID": 0}` → `{"GKP": 4, "MID": 1}`.
3. GC-deduction math replaced: was `-floor(λ_opp / 2)` (floor of the point estimate —
   e.g. λ=1.9 always floored to 0, even though a real Poisson(1.9) variable is ≥2 over
   40% of the time). Now uses the exact closed form
   `E[floor(K/2)] = λ/2 − (1 − e^(−2λ))/4` for K ~ Poisson(λ), via a new
   `_expected_floor_half_poisson()` helper.

Verified live: GKP predictions dropped as expected (Donnarumma/Raya/Sánchez all lower
clean-sheet credit), overall backtest MAE improved slightly (2.015 → 1.995), all
endpoints (`/api/players`, `/api/optimize`, backtest tabs) still return 200 with no
errors. Not yet committed.

Still deliberately unmodeled at this stage (tracked for Phase 2/5 below): DefCon,
bonus, cards, penalty saves/misses, own goals.

---

## Gap analysis summary (current model vs. target)

| Area | Current state | Target | Priority |
|---|---|---|---|
| Scoring table | ✅ DONE — verified against live FPL rules page, 3 bugs fixed | — | ~~P0~~ |
| GC deduction | ✅ DONE — exact Poisson closed form via `_expected_floor_half_poisson()` | — | ~~P0~~ |
| Goal/assist share | ✅ DONE — xG/xA share, shrunk toward last-season-at-club (or position-average) prior via `n90/(n90+k)` | Still uses total xG (not non-penalty-split) — carried into Phase 4 | ~~P1~~ |
| Penalty goals | Not separated at all — still true after Phase 1 | Split team λ into open-play/penalty/OG; explicit `pen_share`/`team_pens`/`conversion` term | P4 (moved from P1, bundled with team-λ rework) |
| DefCon (CBIT/CBIRT) | **Absent entirely** | Modeled from FPL's own `clearances_blocks_interceptions`/`recoveries`/`tackles`/`defensive_contribution` fields (already fetched by the API, unused by us) | P2 |
| Bonus points | **Absent entirely** | v1: regression on expected events → bonus. v2 (later): Monte Carlo match simulation | P2 |
| Cards | **Absent entirely** | Small per-90 yellow card rate term | P2 (low value, bundle with bonus work) |
| Minutes model | Flat ratios (`total_starts/team_games`), no smoothing, no P(60+) split, appearance term assumes every start = 60+ mins | Beta-prior blend of last-season rate + this-season starts; separate P(start)/P(60+)/P(sub); European-fixture rotation adjustment | P3 |
| Team λ (Tier 1 odds) | Simple proportional devig (`1/price ÷ Σ`); total-goals × h2h-implied home-share split | Shin's/power-method devig; joint Poisson-grid fit against both 1X2 and totals simultaneously; Dixon-Coles low-score correction | P4 |
| Team λ (Tier 2/3 model) | Raw-goals rolling average (≤6 games) × opponent factor × home/away mult | Proper ratings model: `log λ = μ + attack_i − defence_j` with time decay, last-season base + this-season data, market λ as heavily-weighted pseudo-observations | P4 |
| Save volume | Flat historical per-90 average, no opponent adjustment | Modeled from opponent's expected shots-on-target against; Poisson `E[floor(saves/3)]`; penalty-save term | P5 |
| Clean sheet minutes gate | Scaled by continuous `exp_minutes` | Gated by discrete `P(60+)` | P5 |
| Player props cross-check | Not used | Blend `player_goal_scorer_anytime` odds-implied λ with the share model | P5 |

---

## Phase plan

### Phase 0 — Scoring table fixes ✅ DONE (2026-09-17) — see full detail above

### Phase 1 — Attacking returns via xG-share + shrinkage ✅ DONE (2026-09-17)

Implemented in `fpl_client.py` (new shared functions, also used by the backtest):
- `build_team_xg_totals()` — sums every player's own `expected_goals`/`expected_assists`
  by (team_id, fixture_id), reconstructing each team's real total xG/xA per match
  directly from the FPL API (no separate data source, as the review suggested).
- `build_position_priors()` — league-average last-season xG/xA share by position, for
  genuine new arrivals with zero last-season data at all.
- `compute_xg_share()` — the core formula:
  `share = (n90·share_this_season + k·share_prior) / (n90+k)`, where
  `share_this_season` = player's own xG/xA over the last `window` games they played
  ÷ their team's total xG/xA (from `build_team_xg_totals`) over those same fixtures,
  and `share_prior` = last season's xG/xA at the same club ÷ an estimated team-goals
  figure, falling back to the position-average prior. `k=6` (`SHARE_SHRINKAGE_K` in
  config.py, tunable). Replaces `_build_player_stats`'s old actual-goals share
  entirely — `_build_player_stats` no longer computes `goal_share`/`assist_share` at
  all; `fetch_all_data` calls `compute_xg_share` per player and sets it directly.
- `backtest_accuracy.py` updated to use the exact same three functions (via a new
  `player_history_past` data flow threaded through `fetch_all_data` → the backtest
  endpoint), so the backtest stays a faithful test of the live formula. Also picked
  up the Phase 0 GC-deduction fix (`_expected_floor_half_poisson`), which the
  backtest had its own un-fixed copy of.

**Scoped out of this pass** (documented gaps, not yet done):
- No penalty/open-play/own-goal split of team λ or player xG — `expected_goals` is
  used as-is (includes penalty xG). A player who is their team's penalty taker will
  still be somewhat overrated relative to a teammate with similar open-play xG who
  doesn't take penalties. Revisit alongside Phase 4's team-λ work.
- No separate `assists_per_team_goal` calibration constant — assist_share is defined
  directly as "player's xA ÷ team's xA" (not "÷ team's goals" with a conversion
  step), which sidesteps the need for that constant while achieving the same effect;
  documenting the deviation from the source review's exact formula shape.
- New-arrival prior is position-average only, not position+role+price-tier
  segmented (e.g. Ndiaye/Enzo-style big-money attacking arrivals get the same prior
  as a squad-depth signing at the same position) — a coarser prior than ideal.

Verified live: Haaland/Fernandes-style explosive-performance predictions moved up
significantly (Haaland GW2 2.00→4.27, Fernandes GW2 2.00→4.16) — both were being
underrated by the old noisy actual-goals share; overall backtest MAE improved
1.995→1.981. All endpoints (`/api/players`, `/api/optimize`, `/api/transfer-advice`,
both backtest tabs) verified 200 with a real 15-player squad. Not yet committed.

### Phase 2 — DefCon + bonus
- Add a DefCon term using already-fetched `clearances_blocks_interceptions`/
  `recoveries`/`tackles`/`defensive_contribution` fields — start with a simple per-90
  rate × minutes / threshold check (2 pts if count ≥ threshold), before the
  negative-binomial/game-state-adjusted version.
- Add a bonus regression (v1: simple regression mapping expected events → bonus points).
- Add a small cards term while in this area.

### Phase 3 — Minutes model overhaul
- Beta-prior blend of last-season start rate with this season's accumulating starts.
- Split into P(start) / P(60+) / P(sub appearance) instead of one flat ratio.
- Appearance points become `P(1-59)·1 + P(60+)·2` instead of `exp_start_pct·2`.
- Add a European-fixture rotation-risk adjustment.

### Phase 4 — Team λ upgrade
- Replace proportional devig with Shin's or power method.
- Joint Poisson-grid fit against 1X2 + totals simultaneously (not a simple total ×
  home-share split).
- Add Dixon-Coles low-score correlation correction.
- Replace the Tier 2/3 raw-goals rolling average with a proper ratings model
  (`log λ = μ + attack − defence`, time-decayed, last-season base + this-season data,
  market λ as pseudo-observations for re-anchoring).
- Promoted-team prior from historical promoted-side performance.

### Phase 5 — Polish
- Opponent-adjusted save volume (expected shots on target against → Poisson saves).
- Penalty-save term.
- Clean-sheet minutes gate switched from continuous `exp_minutes` to discrete `P(60+)`.
- Player-props cross-check (`player_goal_scorer_anytime` → implied λ, blended with the
  share model; watch for void-if-not-played conditioning, multiply by P(plays)).
- Monte Carlo match simulation for bonus (v2) and captaincy variance/correlation.

---

## Backtest follow-up (parallel track, not blocking the phases above)

The existing Prediction Accuracy tab (team xG + player points backtest) should be
re-run after each phase lands to measure real improvement — that's the whole point of
having built it. Each phase's "done" criterion should include a fresh backtest MAE
comparison, not just a code review.

No cross-season historical data import planned (explicitly out of scope per this
season-only backtesting decision). `odds_history.json` continues accumulating one
snapshot per gameweek going forward as the only historical-odds source we use.
