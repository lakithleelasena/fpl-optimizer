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
| DefCon (CBIT/CBIRT) | ✅ DONE — empirical hit-rate v1 | Negative-binomial + game-state (win-prob) adjustment | ~~P2~~ (v2 → P5) |
| Bonus points | ✅ DONE — real OLS regression, this-season data, refit each cache cycle | v2: Monte Carlo match simulation | ~~P2~~ (v2 → P5) |
| Cards | ✅ DONE — empirical yellow-card rate v1 | — | ~~P2~~ |
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
both backtest tabs) verified 200 with a real 15-player squad.

### Phase 2 — DefCon + bonus ✅ DONE (2026-09-17)

Also fixed a bug found while restructuring the formula: Phase 0 corrected
`_CS_PTS["MID"]` from 0 to 1, but `predict_points()`'s `is_def` branch meant MID
never actually received any clean-sheet credit at all — the dict value was right,
the formula never read it for that position. Unified the formula so `cs_prob`/
`cs_pts` are computed for every position (naturally a no-op for FWD, whose
`_CS_PTS` is 0).

Implemented:
- **DefCon** (`fpl_client.compute_defcon_hit_rate`): empirical
  `P(defensive_contribution >= threshold)` from the player's last 6 played games —
  the "start simple" v1, before a negative-binomial/game-state-adjusted version.
  FPL's own `defensive_contribution` field already sums exactly the right stats per
  position (CBIT for defenders, CBIRT for mid/forwards), so no need to combine the
  individual clearances/blocks/interceptions/tackles/recoveries fields ourselves.
  `E[DefCon] = 2 * hit_rate`, added inside the `exp_minutes * (...)` bracket for
  every position (0 for GKP, per FPL rules — outfield players only).
- **Cards**: `fpl_client.compute_card_rate` — empirical P(yellow) from the last 6
  played games, `-1 * rate` as a small deduction. Red cards skipped for v1 (rare,
  and already dominate the match outcome via lost minutes).
- **Bonus** (new `bonus_model.py`): a genuine OLS regression —
  `bonus ~ goals + assists + clean_sheet + saves + defcon_hit` — fit on every played
  match across all active players THIS SEASON ONLY (not last season's: BPS was
  reworked for 2026/27 — no more tackled-player penalty, CBI now 1 BPS per 3 instead
  of per 2, restructured GK save BPS with a big-chance bonus — so old bonus data
  wouldn't even be measuring the same thing). Refit once per `fetch_all_data()`
  cache cycle (~1200+ rows already at GW5, cheap). Applied to each player's
  *expected* event values (`cs_prob` standing in for the training data's 0/1
  clean-sheet outcome, etc.), clipped to bonus's real `[0,3]` range. Falls back to a
  hand-set default until 200+ rows exist to fit against.
- `backtest_accuracy.py` mirrors all three, with one correctness point that needed
  care: the bonus model's live cache is fit on *all* current data, which would leak
  future gameweeks' bonus outcomes into an earlier backtest target gameweek.
  `bonus_model.fit()` is a pure function for exactly this reason — the backtest
  refits it separately **per target gameweek**, from only history strictly before
  that gameweek, instead of reusing the shared live cache.

**Scoped out of this pass**: no negative-binomial/game-state adjustment for DefCon
(regress on market-implied win probability — underdogs defend more) — still the
flat empirical hit-rate. No Monte Carlo bonus simulation (v2) — this is the
regression-only v1. No `defcon_factor`/bonus-weight UI slider — kept as fixed
formula terms for now, consistent with how `SHARE_SHRINKAGE_K` was also kept
backend-only in Phase 1, to avoid the UI sprawling with a slider per new term.

Verified live: GW5 optimized XI total rose 65.2→66.6 points (previously-uncredited
bonus/DefCon points now counted). Backtest MAE moved 1.981→2.082 — a real but small
increase, expected: three brand-new additive terms (bonus, DefCon, cards) replace
what was previously a hard `0` for every single prediction, so some model risk is
now present where there used to be none. Since bonus and DefCon points are
genuinely part of every actual score, leaving them at zero wasn't "safer" — it was
a different, larger, systematic bias (undercounting real points every game). Worth
re-checking this MAE trend as more of this season's data accumulates and the
bonus regression firms up. All endpoints re-verified 200 with a real 15-player
squad; no console errors in the browser.

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
