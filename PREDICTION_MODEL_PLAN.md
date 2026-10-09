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
| Minutes model | ✅ DONE — Beta-prior-blended start/minutes rates, P(60+)/P(1-59) split from real per-game data | European-fixture rotation adjustment — no data source available, deliberately scoped out | ~~P3~~ |
| Team λ (Tier 1 odds) | ✅ DONE — Shin's devig + joint Poisson/Dixon-Coles fit for the home/away split | Full 2D joint least-squares (both markets move together, not just the split) — marginal gain, skipped | ~~P4~~ |
| Team λ (Tier 2/3 model) | ✅ DONE — Tier 2 now real xG (summed from player xG), not actual goals | Full log-linear ratings model via MLE with time decay + last-season base — kept the existing taper-blend architecture, only upgraded its data source | ~~P4~~ (full MLE version → P5+) |
| Save volume | ✅ DONE — opponent-adjusted rate (saves per unit real opponent xG faced) × Poisson `E[floor(saves/3)]`; penalty-save term added | Full shots-on-target modeling (we approximated via realized opponent xG instead) | ~~P5~~ |
| Clean sheet minutes gate | ✅ DONE — gated by discrete `P(60+)` | — | ~~P5~~ |
| Player props cross-check | Not used | Blend `player_goal_scorer_anytime` odds-implied λ with the share model | Deferred — needs an explicit go-ahead to spend Odds API quota on a new endpoint |

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

### Phase 3 — Minutes model overhaul ✅ DONE (2026-09-17)

Implemented in `fpl_client.py`:
- **`compute_minutes_model()`**: replaces the old flat this-season-only ratio and
  its hard pre-season-fallback cutover (`team_games>0 ? real ratio : 0`, or a
  separate `LAST_SEASON_GAMES` branch) with the same Beta-Binomial-posterior-mean
  shrinkage as Phase 1: `rate = (n*rate_this_season + k*rate_prior) / (n+k)`,
  `k=6` (`MINUTES_SHRINKAGE_GAMES`). Because `n` (team games played so far) is 0
  pre-season, the formula *naturally* reduces to the prior alone with no separate
  branch needed — a cleaner unification than the old code had.
- **`p_60_plus` / `p_1_to_59`**: computed directly from real per-game minutes this
  season (no start/appearance conditioning — captures genuine substitute cameos as
  well as starts hooked early), each independently Beta-shrunk toward a prior.
  Appearance points in `predictor.py` are now `P(1-59)*1 + P(60+)*2` instead of
  `exp_start_pct*2`, so a player who's a nailed starter but a rotation risk for
  60+ minutes (returning from injury, etc.) is no longer credited the full 2
  points just for being penciled in to start.
- **Prior approximation**: `history_past` only has season *totals* (no per-game
  minutes breakdown), so the prior can't be computed as precisely as the
  this-season side. Approximated via last season's start rate
  (`past_starts / LAST_SEASON_GAMES`) and a completion-rate proxy from average
  minutes-per-start (`_completion_rate_from_avg_mins`: ~90 min/start → ~1.0
  completion, ~60 → ~0.5, ≤30 → 0.0) — a smooth, bounded, but genuinely
  approximate mapping, since we don't have last season's real per-game
  distribution to fit against.
- **`build_minutes_priors()`**: position-average fallback for genuine new
  arrivals with no last-season data at all, mirroring Phase 1's
  `build_position_priors`.
- `backtest_accuracy.py` mirrors this exactly. NOTE (approximation, same pattern
  as the existing "form" note): historical `chance_of_playing` isn't retrievable,
  so the backtest fixes availability at 1.0 — it can reproduce a past minutes
  *pattern* but not a past injury doubt.

**Scoped out of this pass** (explicitly, not silently dropped): **no
European-fixture rotation-risk adjustment.** FPL's own API (bootstrap-static,
fixtures) is Premier-League-only — it has no Champions/Europa/Conference League
fixture data at all, so detecting "this team played in Europe midweek" would
require an entirely new external data source we don't currently have access to
or a fetch pipeline for. This is a real, acknowledged gap, not an oversight —
worth revisiting if a free European-fixtures source is found, but out of scope
for "start simple."

Verified live: Kinsky (the pre-season Kinsky/Dubravka mispricing case from much
earlier in this project) now shows `exp_start_pct=0.511` after 4 real gameweeks —
the model has self-corrected from the pure-prior estimate toward reality, exactly
as expected once real current-season data accumulates. Backtest MAE improved
2.082→2.046 overall, and more notably on starters specifically (2.586→2.463) —
consistent with this phase targeting minutes-prediction accuracy, which the
original review flagged as likely the single biggest error source. All endpoints
(`/api/players`, `/api/optimize` at n_gw 1 and 3, `/api/transfer-advice`, both
backtest tabs) verified 200 with a real 15-player squad; no console errors.

### Phase 4 — Team λ upgrade ✅ DONE (2026-09-18)

New `team_xg_model.py`:
- **`devig_shin()`**: Shin's method devigging for the h2h (1X2) market, replacing
  simple proportional normalisation (`p_i = π_i/Σπ_j`). Solves for the "informed
  money" fraction z such that `p_i(z) = (√(z²+4(1-z)π_i²/B) - z)/(2(1-z))` sums to
  1 (bisection; B = raw implied-prob sum, the overround). Verified against two
  synthetic test cases before wiring in: (1) sums to exactly 1.0, (2) correctly
  shifts probability *toward* the favorite and *away* from the longshot relative
  to proportional normalisation (0.7242→0.7423 favorite, 0.1046→0.0942 longshot in
  a big-favorite test case) — the documented direction of the favorite-longshot
  bias correction, confirming the implementation isn't just plausible-looking but
  behaving correctly. Falls back to proportional normalisation if the input is
  degenerate or the result fails a sanity check.
- **`poisson_match_probs()`** + **`fit_team_lambdas()`**: solves for `(λ_home, λ_away)`
  whose Poisson scoreline grid (Dixon-Coles-adjusted — see below) reproduces the
  market's implied `P(home win)`, holding the total fixed at the totals market's
  implied total goals. Replaces the old `home_xg = total_xg × home_share` linear
  split, which ignored the shape of the Poisson distribution entirely. This is a
  1D bisection over the home/away split (not a full 2D joint least-squares over
  both markets moving together) — a deliberate simplification to avoid a
  numerical-solver dependency for a small expected additional gain; documented,
  not hidden.
- **Dixon-Coles correction** (`_dixon_coles_tau`, ρ=-0.13, a literature default
  from Dixon & Coles 1997, not locally fit): applied to the four low-score cells
  (0-0, 1-0, 0-1, 1-1) inside `poisson_match_probs`, so the λ *recovered* from
  market odds is corrected for pure-Poisson's known under-prediction of draws.
  Scoped decision: **not** additionally re-applied to `predictor.py`'s own
  clean-sheet marginal (`cs_prob = exp(-λ)`) — doing that correctly needs the
  home/away-identity-aware asymmetric tau terms threaded through predict_points,
  adding real complexity for a small further gain once the λ feeding into it is
  already DC-corrected upstream. `odds_client.py` wired to use both new functions
  in place of the old devig + linear split.
- **`fpl_client.build_team_xg_rolling()`**: Tier 2's rolling signal switched from
  actual goals scored/conceded to real xG — summed from every player's own
  FPL-reported xG per fixture via Phase 1's `build_team_xg_totals`, per the
  review's "use xG rather than goals as the target" recommendation. Required
  reordering `fetch_all_data()` so player histories (and thus `team_xg_by_fixture`)
  are fetched before Tier 2/3 runs, not after. `_build_team_rolling` (actual
  goals) is kept **unchanged** and separate — it still feeds the Team Overview
  tab's GF6/GA6 display, which is explicitly meant to show real goals, not a
  model estimate.
- `backtest_accuracy.py`'s `compute_team_xg_backtest` mirrors the Tier 2 xG switch
  (new `raw_histories` parameter) so the backtest's "Tier 2" line keeps meaning
  what the live app actually does; Tier 1 automatically benefits since it just
  replays whatever `odds_history.json` already has archived (built by the updated
  `odds_client.py`).

**Scoped out of this pass** (documented, not silently dropped — these are the
hardest, most data-hungry pieces of the original review's Phase 4):
- **No full log-linear ratings model** (`log λ = μ + attack − defence` via MLE,
  time-decayed, last-season base blended with this season). This is a genuine
  statistical model-fitting exercise (an iterative solver, effectively a
  simplified Dixon-Coles/Bradley-Terry-style fit across the whole league at once)
  — a materially bigger undertaking than the achievable pieces above. The
  existing taper-blend architecture (Tier 2 rolling xG × opponent factor ×
  home/away mult, tapering into Tier 3 FDR) is kept as the underlying structure;
  only its *data source* was upgraded this phase.
- **No promoted-team-specific prior.** Needs historical data on how previously
  promoted sides performed — an external data source we don't have access to.
- **No penalty/open-play/own-goal split** of team λ. FPL doesn't cleanly expose
  team-level penalty-award data (only `penalties_missed`/`penalties_saved` per
  player), and reconstructing it accurately would need real research into what's
  derivable from available fields — not attempted here to avoid a half-right
  penalty model quietly corrupting every attacker's `E[goals]`.
- **No last-season team-xG base** for Tier 2/3 (this season's rolling xG only).
  A last-season base would need every player who was on a team's roster last
  season, including those who've since transferred away — a wider data-fetch
  than we currently do (only this season's active roster gets `history_past`
  pulled). Consistent with the project's this-season backtest-scope decision.

Verified live: Shin's/joint-fit tested against two independent synthetic cases
before wiring in (both passed, see above); a live `/api/refresh-odds` call
produced sane, non-degenerate xG values across all 20 GW5 fixtures. Team xG
backtest Tier 2 MAE moved measurably (e.g. GW2 1.398→1.181, GW3 0.972→0.992,
GW4 1.188→1.054 — improved in 2 of 3 gameweeks) now that it's genuinely xG-based.
Player-points backtest overall MAE barely moved (2.046→2.044) — expected, since
Tier 1 has no completed-gameweek data yet to show up in this backtest (GW5
hasn't finished), and Tier 2's xG-swap is a second-order refinement several
steps removed from final player points, whose accuracy is now dominated by other
noise sources (bonus regression variance, minutes model, etc.). All endpoints
(`/api/players`, `/api/optimize` at n_gw 1 and 3, `/api/transfer-advice`, both
backtest tabs) verified 200 with a real 15-player squad; Team Overview's GF6/GA6
display confirmed still actual-goals-based and unaffected; no console errors.

### Phase 5 — Polish ✅ DONE (2026-09-18), partial

Implemented:
- **Opponent-adjusted saves** (`fpl_client.compute_saves_rate`): replaces the old
  flat saves-per-game average (which never varied by fixture difficulty at all)
  with `saves_per_opp_xg` — a goalkeeper's historical saves per unit of *real*
  opponent xG actually faced (looked up via `team_xg_by_fixture`, same real
  per-fixture data used throughout this pipeline), multiplied by THIS WEEK's
  `match_opp_xg` at prediction time — the same rate-times-this-weeks-xG pattern
  as Phase 1's goal/assist shares. `save_pts` now uses `E[floor(saves/3)]`
  (`team_xg_model.expected_floor_div_poisson`, a truncated-sum Poisson
  expectation) instead of `floor(E[saves]/3)` — the same point-estimate bias
  fixed for the GC deduction in Phase 0, now fixed here too. New function
  cross-validated two ways before use: against the n=2 case's existing closed
  form (`_expected_floor_half_poisson`) — exact agreement to 6 decimal places —
  and against a 200k-trial Monte Carlo simulation across several λ/n combos.
- **Penalty saves**: a small flat GKP-only term
  (`PENALTY_AWARD_RATE_PER_MATCH × PENALTY_SAVE_RATE × PENALTY_SAVE_PTS`, config
  constants — literature defaults ~1 penalty per team per 11 games, ~1-in-5 save
  rate, not locally fit). Not fixture-specific, since we don't have team-level
  penalty-award data (same gap flagged in Phase 4's scoped-out penalty split).
- **Clean-sheet minutes gate**: clean-sheet points now gated by the discrete
  `p_60_plus` (from Phase 3) instead of the continuous `exp_minutes` fraction,
  matching FPL's actual rule ("not conceding while on the pitch AND playing at
  least 60 minutes" — an explicit threshold, not a pro-rated credit). The GC
  deduction stays scaled by continuous `exp_minutes`, since that rule has no
  60-minute threshold.
- `backtest_accuracy.py` mirrors all three exactly.

**Deliberately not implemented, for reasons distinct from "too hard" (unlike
earlier phases' scope-outs):**
- **Player-props cross-check** — this needs a *new, additional* live Odds API
  call to the per-event player-props endpoint, spending real quota/cost on the
  user's Odds API subscription (the user's own original brief estimated ~20
  credits per slate for this specifically, "budget roughly two prop pulls per
  gameweek"). Unlike everything else in Phases 0-5, which only ever used data
  already being fetched, this is an ongoing operational cost decision — not
  something to spend on unilaterally. Skipped pending an explicit go-ahead.
- **Monte Carlo match simulation** (bonus v2, captaincy variance/correlation) —
  the original source review itself frames this as a separate "v2" tier beyond
  the v1 regression already shipped in Phase 2, not a "polish" item. It's a
  substantial standalone feature (simulate scorelines, assign goals/assists by
  share, approximate BPS, rank within each match, track captain covariance
  across simulations) — more like its own phase than a finishing touch on this one.

Verified live: `expected_floor_div_poisson` cross-validated against both the
existing closed-form (n=2, exact match) and Monte Carlo (n=3, matched to 3
decimal places) before wiring in. Backtest MAE improved 2.044→2.027 overall;
GKP MAE continued its downward trend across every phase that's touched it
(3.099 pre-Phase-0 → 2.93 → 2.82 → 2.934 → ~2.9 → **2.785** now). All endpoints
(`/api/players`, `/api/optimize` at n_gw 1 and 3, `/api/transfer-advice`, both
backtest tabs) verified 200 with a real 15-player squad; no console errors.

---

## Backtest follow-up (parallel track, not blocking the phases above)

The existing Prediction Accuracy tab (team xG + player points backtest) should be
re-run after each phase lands to measure real improvement — that's the whole point of
having built it. Each phase's "done" criterion should include a fresh backtest MAE
comparison, not just a code review.

No cross-season historical data import planned (explicitly out of scope per this
season-only backtesting decision). `odds_history.json` continues accumulating one
snapshot per gameweek going forward as the only historical-odds source we use.

---

## Post-plan tuning (2026-10-08, after GW5)

First backtest with a meaningful sample (GW1-5, 100 team-matches) found the team-goal
tiers weakly discriminating, and a large *level* bias in the blend.

**Finding.** Tier 3 (FDR fallback) averaged ~0.86 goals/team against 1.41 actual: the old
`1.35 × (5-FDR)/3` treated the average fixture as FDR 2, but the season's mean FDR is 3.1.
With the old 10-game linear taper handing Tier 3 most of the weight early, the production
blend's bias was −0.386 goals/team (Poisson deviance 1.561).

**Changes.**
- Tier 3 is now centred: `league_avg × max(0.3, 1 + FDR_SENSITIVITY × (mean_fdr − fdr))`, with
  `league_avg` = live season mean goals/team shrunk toward the 1.35 prior (weight
  `LEAGUE_AVG_SHRINK_MATCHES` = 40 team-matches) and `mean_fdr` from the full fixture list.
- Tier 2 vs Tier 3 weight is `n/(n+TIER2_SHRINKAGE_GAMES)` (6) instead of the 10-game taper —
  same shrinkage family as Phases 1 and 3; reproduces the data-optimal ~0.25-0.3 at 1-4 games.
- Helpers (`compute_league_avg_goals`, `compute_mean_fdr`, `tier2_xg`, `tier3_xg`, `tier2_weight`)
  are shared between `fpl_client.py` and `backtest_accuracy.py` so live and backtest can't drift.
- The team-xG backtest endpoint now returns `summary` (per-tier Poisson deviance, bias, skill vs a
  league-average baseline) and `weight_fit` (Tier 2-vs-3 deviance curve, three-way mix, `odds_weight`
  sweep, fixture-level bootstrap ranges), shown on the Prediction Accuracy tab.

**Result (GW1-5).** Production deviance 1.561 → 1.352, bias −0.386 → −0.001. Skill vs the
league-average baseline is only ≈ +1.4%, so the models barely beat "predict the league mean" —
goal counts are very noisy. Tier 2 alone is worse than baseline (−14%). Player-points backtest is
essentially neutral (MAE 1.986 → 1.992; starters 2.449 unchanged), as expected: team-xG level is
second-order for player points; the per-position signed-error shift is the visible effect.

**Caveats.** Only 100 team-matches; fits are in-sample; `FDR_SENSITIVITY` = 0.25 was chosen after
looking at the same data. Tier 1 has only 10 fixtures (GW5), so `odds_weight` (default 0.6) is not
determinable yet — best 0.3, bootstrap 0.0-1.0 — and was deliberately left alone. Re-run after GW6+.

**Ideas not yet done.** Shrink Tier 2 toward the league average; re-tune `odds_weight` once ≥30
fixtures have odds; add `p_60_plus`/`exp_minutes` to player backtest rows for minutes diagnostics;
check top-end player over-prediction; DefCon/bonus tuning.

---

## Player-points tuning from the component backtest (2026-10-08, after GW5)

**Method.** The Prediction Accuracy tab now splits predicted and actual points into FPL
scoring components (appearance, goals, assists, clean sheet, goals conceded, saves, bonus,
DefCon, cards, other) so error can be located instead of just measured. Actual components
reconcile exactly to `total_points`. Use the "All predictions" view for calibration — the
"players who played" view conditions on an outcome the model predicts (its predictions
include the chance of not playing), so minutes-driven components look under-predicted there
by construction.

**Gaps found (GW2-5, 1,467 player-gameweeks, bias = predicted − actual, points):**
appearance −0.154, clean sheet −0.101 (GKP −0.28, DEF −0.17), defender goals +0.126 (MAE
worse than guessing the average), assists −0.043 (FWD −0.12), DefCon −0.051 (DEF −0.09),
cards +0.046, saves −0.005, bonus ≈ 0 overall; net −0.23/player-gameweek. Form adjustment is
zero for GW1-5 by construction (needs >4 prior games).

**Done.**
1. *Minutes* (`MINUTES_SHRINKAGE_GAMES` 6 → 1.5): last season's start rate held ~2/3 of the
   weight after 3 games, under-predicting regulars (60% predicted P(60+) happened 73% of the
   time). P(60+) Brier 0.182 → 0.152; appearance bias −0.154 → −0.049; player MAE 1.956 → 1.911.
   Flat for k = 1-2 (k=1 has the best Brier, 0.145); 1.5 chosen conservatively.
2. *Share-prior reliability* (`SHARE_PRIOR_RELIABILITY_N90` = 6): last season's xG share was
   computed per minute played, so tiny samples exploded (Tomiyasu: 6 min, 0.18 xG → share
   2.0 → ~1.5 predicted goals/match; 10 players had a prior share > 0.30). Now blended
   toward the position average with weight n90/(n90+6). DEF goals MAE 0.411 → 0.368, goals
   MAE 0.625 → 0.607, player MAE 1.992 → 1.956. The remaining DEF over-prediction (+0.095)
   is mostly finishing luck — defenders scored 13 goals from ~20 xG against ~24 predicted.
3. *Stale last season* (`drop_stale_past_seasons`): only the latest completed season counts
   as "last season"; players whose latest record is older (29 players) use the position
   average. MAE 1.911 → 1.909, Brier 0.1521 → 0.1513.

Combined: player-points MAE 1.992 → 1.909 (starters 2.449 → 2.396), P(60+) Brier 0.1835 → 0.1513.

**Considered and not changed.**
- *Clean sheets.* Team-level predicted CS rate 25.9% vs 30% actual (26 vs 30 of 100 team-
  matches) is ~1 standard error — noise at this sample. Part of the player-level gap was the
  minutes under-prediction (now fixed); the remainder is not significant. Revisit around
  GW12-15; if still ~25% low, replace `exp(-λ)` with the P(0 goals) from Phase 4's joint
  Dixon-Coles fit rather than adding an ad-hoc multiplier.
- *Dropping last-season priors.* Removing the minutes prior changes almost nothing at k=1.5
  (MAE +0.006, Brier equal) but it still helps players with no appearances yet; removing the
  share prior hurts goals MAE (0.607 → 0.638, mostly GW2-3, converging by GW4-5) and it fades
  automatically with n90. Minutes from this season alone is clearly worse (MAE 1.971). Re-test
  around GW10; the share prior should matter little by then.
- *Share k (6 → 2-3).* Slightly better goals MAE but worse assists/early GWs, total MAE within
  0.003 — noise.

**Next candidates (not done).**
- Assists under-predicted in every position (FWD worst: 0.087 vs 0.206) — check assist share
  vs xA, and whether xA understates real assists.
- DefCon under-predicted for defenders (0.235 vs 0.326) — hit-rate is empirical; check
  recency weighting and whether the exp_minutes scaling is double-penalising.
- Cards under-penalised by 0.03-0.09 per player-gameweek (also missing red cards).
- GKP saves slightly worse than guessing the mean (MAE 0.697 vs 0.659).
- Bonus for MID/FWD is no better than the mean — consider the Monte Carlo v2.
- Minutes: add per-player expected-minutes diagnostics; a "never appeared" cohort (201
  player-gameweeks) is where the prior matters most and is only partly covered by the backtest.
- Re-run this whole table after GW10: all conclusions above rest on 4 target gameweeks,
  in-sample, with one or two tuned constants each.

---

## Follow-up: minutes recency, DefCon, cards (2026-10-08, after GW5)

Planned from read-only backtests of the component gaps above; implemented as one commit each.
The comparison metric is MSE of player points (rewards predictions that are right on average,
which is what the squad optimizer needs) plus the component's own bias — MAE alone favours
under-predicting rare events, so it can rise when a prediction gets better (e.g. DefCon).
RMSE and average bias were added to the player-points summary on the Accuracy tab.

| Step | Change | Before → after (GW2-5, 1,467 player-gameweeks) |
|---|---|---|
| Minutes | Recency-weight this season's games (decay 0.5; game `a` ago counts 0.5^a) and shrink toward the last-season/position prior with n_eff-based weight k = 0.75 (was a flat average, k = 1.5) | MSE 7.740 → 7.506; P(60+) Brier 0.1513 → 0.1342; appearance bias −0.044 → +0.002, MAE 0.549 → 0.478 |
| DefCon | P(threshold \| plays 60+) shrunk to the position's league rate (k = 4 games), points = 2 × rate × P(60+) outside the exp_minutes bundle | MSE 7.506 → 7.382; defcon bias −0.029 → −0.009 |
| Cards | Yellows per 90 shrunk to the position's league yellow rate (k = 8 × 90 min) + 3 × pooled league red rate, scaled by exp_minutes | MSE 7.382 → 7.347; cards bias +0.035 → +0.003 |

Overall MAE 1.909 → 1.860, starters-only 2.396 → 2.361, MSE 7.740 → 7.347, RMSE 2.782 → 2.711.

**Why these.** The existing minutes model treated every game this season equally, but role changes
come in steps (a "went 60+ last game" feature alone took the out-of-sample Brier from 0.147 to
0.128). Raw per-player DefCon and card rates over ~4 games were noisier than the position average;
DefCon was also diluted by cameo appearances and then scaled by minutes a second time.

**Validation.** Minutes parameters were scanned on a (decay, k) grid; leave-one-gameweek-out picks
decay 0.3 / k 0.5-0.75 for every held-out week (CV MSE 7.494), and the result is flat for decay
0.3-0.5, so the less extreme 0.5 / 0.75 is used. League priors (DefCon, cards) are built per target
gameweek from strictly earlier games in the backtest, so nothing leaks. All of this rests on four
target gameweeks with 2-3 tuned numbers per change — re-check around GW10.

**Not done, with reasons.**
- *Assists:* every variant (global scale, consistent prior basis, assists-per-goal ratio) left the
  assist Poisson deviance flat (0.341-0.343); the forward gap is 12 actual assists vs 4.4 xA, i.e.
  mostly luck. There is a real structural inconsistency worth fixing eventually — the last-season
  prior share is measured against team goals (est. 1.35 × games) while this season's share is
  against team xA, and FPL xA sums to only ~0.61 of xG while actual assists run ~0.9 per goal — but
  it brings bias to zero without improving accuracy, so it is parked.
- *Saves:* shrinking the goalkeeper rate to the league rate gave MSE −0.01 on 87 goalkeeper rows,
  and the league rate under-predicts total saves (2.76 vs 2.93 per match; xG undercounts shots
  faced). Skipped as too weak to justify a change.
- *Clean sheets:* the recency change leaves the −0.08 bias unchanged, so it is not a minutes issue
  and the earlier "mostly noise at 100 team-matches" reading stands (revisit ~GW12-15).

**Still open.** Bonus for MID/FWD (no better than the mean), goalkeeper saves (needs shots-faced data,
not xG), FWD assist share, penalty/set-piece takers (`penalties_order` etc. in the bootstrap data;
current-only, so forward-test only), archiving our predictions and FPL's `ep_next` each gameweek for
a true out-of-sample benchmark, and ranking metrics (top-N overlap, captain pick, predicted-best XI).

---

## Penalty takers (2026-10-08, GW6 next)

**Data.** The bootstrap `elements` carry `penalties_order`, `direct_freekicks_order` and
`corners_and_indirect_freekicks_order` (1 = first choice; 20 teams each have exactly one #1;
~56 active players have a penalty rank). They are current-only, so `fetch_all_data` now snapshots
them per gameweek into `set_piece_history.json` (last fetch before the deadline wins), and the
backtest uses the archived order for a gameweek when it exists, otherwise today's (mild look-ahead
— orders are sticky). At GW6, 6 of the 20 first-choice takers were doubtful or out (Kroupi Jr,
Wright, Mateta, Diarra at 0%, Osula 50%, Palmer 75%), so who takes the next penalty is a live question.

**Why explicit.** Backtest GW2-5: the model was already roughly unbiased for takers' goals
(−0.004 ± 0.044 goals/gameweek for #1 takers) but their xG exceeded the prediction by +0.066/gw
(t = 2.3), consistent with a missing ~0.07 goals ≈ 0.3-0.4 points/gw. FPL's xG includes penalties (0.76
each) so shares already carry them, but lumpily and without knowing who is fit. A cheap prior bump
on takers' share would double-count as observed xG arrives and ignore availability.

**Model** (`predictor.goal_expectation`, `fpl_client.compute_pen_taker_chain`):
- Penalties awarded to a team: `PENALTY_AWARD_RATE_PER_MATCH` (0.12, scaled by match_team_xg / league avg
  goals) × `PENALTY_CONVERSION_RATE` (0.80) = expected penalty goals. Team xG includes them, so they are
  stripped to get open-play xG.
- `goal_share` is now a NON-penalty share: the expected penalty share of team xG (0.12 × 0.76 / 1.35 ≈
  6.8%) is removed from the team total and, by nominal `penalties_order` weight (0.90 / 0.06 / 0.02 / 0.01 /
  0.01), from the taker's own share. Done at the share level, not by subtracting an absolute 0.09 xG per
  game — the first version did that and a MID in a low-xG game was predicted at 15 points.
- Who takes it: walk the listed order; the first taker on the pitch (a = exp_minutes) takes it with
  probability 0.90 (`PENALTY_TAKER_RELIABILITY`), otherwise it falls through. `q` = P(takes it | on the
  pitch) scales with exp_minutes like the open-play terms; the leftover (no listed taker on the pitch, or the
  other 10%) is spread over on-pitch players by open-play share, so team expected goals are unchanged.
- Expected missed penalties (−2 × (1 − 0.80)) are charged to the taker; the goalkeeper penalty-save term now
  uses the same award rate scaled by the opponent's attack (was a flat 0.09).
- Player tables show a P1/P2… badge, dimmed when the chance he takes the next penalty is under 10%.

**Result (backtest GW2-5, 1,467 rows).** Overall MSE 7.347 → 7.346, MAE 1.860 → 1.862 — neutral, as expected:
only ~5% of rows change. #1 takers' MSE 18.48 → 18.30, #2-3 9.48 → 9.61 (n = 102, noise), non-takers unchanged.
The case for this is structural and for live use (e.g. Palace has no fit taker; Palmer at 75%), not the
backtest score. Live: Fernandes 6.47 → 6.78, Saka 5.98 → 6.23, Haaland 5.96 → 6.05.

**Not done / to watch.**
- Corners and direct free kicks (skipped by decision): corner #2-3 takers' assists were under-predicted by
  +0.064/gw (t = 2.6, one of ~12 tests); direct-FK #1 goals +0.055 (n.s.). Re-test after GW10 with the
  archived snapshots.
- `PENALTY_AWARD_RATE_PER_MATCH` is the key knob (literature value; no per-team data). Calibrate against
  #1 takers' goal residuals after GW10 and check Σ(team players' expected goals) vs team xG.
- Last season's share prior (`history_past`) still includes that season's penalties for former takers — not stripped.
- Taker-specific conversion rates are tiny samples (Haaland 57% cited), so the league rate is used for everyone.

---

## Snapshot archive and live accuracy (2026-10-08, GW6 next)

**Why.** Several inputs exist only "right now" in the FPL API — injury/availability flags, set-piece taker
orders, prices and ownership, FPL's own `ep_next` — and so does the prediction the app would have made.
Archiving them each gameweek gives leak-free backtests (availability, taker orders) and a true out-of-sample
"live accuracy" test (our archived predictions vs real points vs FPL's `ep_next`). Archiving began at GW6, so
nothing earlier can be recovered.

**What is saved** (`snapshot.py`; one gzipped file per gameweek, `archive/<season>/gwNN.json.gz`, ~30 KB):
every player's availability (status, news, chance of playing this/next round), set-piece orders (pen / free kick /
corner), price and ownership, FPL's `ep_next`/`ep_this`/form; our default-parameter predictions for each upcoming
gameweek with `exp_minutes`, P(60+), penalty share and blended match xG; the upcoming fixtures; and provenance
(git commit + dirty flag, odds weight, whether odds exist for THIS gameweek and when they were pulled).

**Rules.**
- Keyed by the UPCOMING gameweek (the one FPL marks as next). Every save replaces that gameweek's file, so the last
  one before the deadline wins; once the deadline passes, saves go to the next gameweek's file and the earlier one is
  never touched again.
- Saved automatically on every fresh FPL fetch and by the **Save snapshot now** button (forces a fresh fetch).
  Manual by design — no scheduled job (no LaunchAgent, no GitHub Action). The header bar shows the deadline in the
  browser's time zone with a countdown, the last save, how many times it was replaced, and turns amber when the deadline
  is within 6 hours and the last save is over an hour old. Snapshot files are deliberately left uncommitted.
- **Odds API quota:** only the Refresh Odds button pulls live odds. A normal fetch, the snapshot button and the snapshot
  job use the cache only (`fetch_odds_xg(cache_only=True)`); previously the first load of each new gameweek called the API
  by itself (cache keyed by gameweek). With no odds for the current gameweek the model runs on Tier 2/3 alone, odds columns
  are blank (never last gameweek's), the Team Overview status says "no odds for GWn yet", and the snapshot records
  `odds.present = false`. Verified with the API patched to fail and a previous-gameweek cache: zero calls.

**Uses.**
- Backtest: `availability_at(pid, gw)` feeds the archived chance-of-playing into the minutes model (and the penalty-taker
  chain) for archived gameweeks, 1.0 otherwise; set-piece orders read the archive too.
- Live accuracy (`live_accuracy.py`, `/api/backtest/live-accuracy`, Accuracy tab): once all of an archived gameweek's
  fixtures finish, our archived predicted points and FPL's `ep_next` are scored on the same players (selected on predictions
  only) — MAE, RMSE, bias, Spearman rank correlation, top-10/20 hits, highest-predicted ("captain") player's points.

**To do / watch.**
- Re-run **Save snapshot now** shortly before each deadline (after the Friday injury news); press Refresh Odds first if
  you want odds in it (uses quota — a normal run never does).
- After GW6 finishes, the first live-accuracy row appears; read it as a single noisy gameweek until ~5+ are archived.
- Candidates once snapshots accumulate: use archived market data (price/ownership) as features, and test the
  corner/free-kick order effects with leak-free orders (assists residual for corner #2-3 takers was +0.064, t = 2.6).
