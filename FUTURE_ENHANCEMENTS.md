# Future enhancements (parked list)

Everything discussed but **not built yet**, with the evidence we have and when to revisit. Nothing here is a commitment —
each item should be re-measured on more gameweeks before any change. Backtests so far rest on GW2-6 (about 1,900 player-gameweeks),
so most conclusions are "indicative, not conclusive". Companion docs: `PREDICTION_MODEL_PLAN.md` (what was built and why, plus the
post-GW10 re-check list at its end).

Status key: **[GW10]** re-test around gameweek 10 · **[GW12-15]** later · **[data]** blocked on data we don't have ·
**[decision]** needs a call from you · **[parked]** low value on current evidence.

---

## 0. Standing rules (how we work on this)
- Analysis first, plan, then change; commit/push only when asked; snapshot files (`archive/`) stay uncommitted.
- Judge changes by MSE/bias of player points (MAE misleads for rare events). Brier/deviance are parked in discussion.
- Only the explicit **Refresh Odds** button spends Odds API quota (about 2 calls per gameweek); everything else is cache-only.
- Archive is manual: press **Save snapshot now** after Friday injury news, before the deadline (last save before it wins).

## 1. Player-points model

### Minutes and participation
1. **First-start adjustment [GW10]** — players whose latest game was their first 60+ of the season play 60+ again 69% of the time
   (n = 49) vs model 54%; strict "all earlier games 0 minutes" was 7/8. The opposite pattern (regular who missed last game) is
   over-predicted: 24% actual vs 40% model (n = 50). Net Brier gain only 0.1377 -> 0.1354, so only worth it if it holds with more data.
   Test leave-one-GW-out; add terms to `compute_minutes_model`.
2a. **Last-season bench record drags new starters [GW10]** — Kostoulas (Brighton, started the last 6 games, 60+ minutes in 5 of 7)
   had P(60+) 0.60 / 56 expected minutes for GW8 because his last-season record (2 starts in 351 minutes) is used as the prior;
   without it 0.68 / 63 minutes (2.40 -> 2.67 pts), on his own record alone ~0.83 / 76 minutes (3.07 pts). Test shrinking or dropping
   the last-season minutes prior once a player has >= 4 starts this season (same family as the first-start adjustment, item 1).
2. **Re-validate minutes parameters [GW10]** — decay 0.5, k 0.75 (chosen on 4 gameweeks; leave-one-out preferred decay 0.3).
3. **Use archived chance-of-playing and news** — snapshots now give real availability flags per gameweek (from GW6). Test whether the
   flag should scale minutes differently, and whether "news added" timing matters.
4. **Rotation / fixture congestion** [data] — no source for European and cup fixtures.
5. **Early-substitution and game-state effects** — blowouts and tight games change when players come off; no data in hand.
6. **Minutes uncertainty for fringe players** — rotation players are lottery tickets; GW6's hindsight best XI had 5 of 11 players we
   expected to play under 54 minutes (0-1 in earlier weeks) although overall calibration for that group is fine (20.9% vs 21.2%).

### Goals, assists, share model
7. **Cap a single game's xG in the goal/assist share [GW10]** — one huge game inflates shares (Fernandes: 2.02 of his 2.56 xG came
   in GW2; GW6 prediction 6.78 -> ~6.1-6.2 with a 0.8-1.0 cap). 13 outlier rows in GW2-5, over-predicted ~0.08 goals/game (n = 13, about 1 SE).
8. **Top-end optimism [GW10]** — predicted goal points 0.8-1.5: 1.06 predicted vs 0.77 actual (n = 140); bonus >= 0.6: 0.80 vs 0.62
   (n = 56); predicted >= 6 points: 6.73 vs 4.07 (n = 14). The top-15 weekly average swings 3.7-6.6 actual vs ~5.4 predicted, so GW6's
   shortfall is within normal variance, but the top end is the real model issue. Candidate: shrink the top of goal and bonus
   predictions; check how much of the Wildcard gain is optimism.
9. **Assists [parked]** — under-predicted in every position (FWD worst: 0.087 vs 0.206, mostly luck: 12 actual vs 4.4 xA). Structural
   inconsistency: last-season prior share is measured against team goals while this season's share is against team xA; FPL xA sums to
   ~0.61 of xG while actual assists run ~0.9 per goal. Fixing it zeroes the bias without improving accuracy. Revisit with more data. (Kostoulas after GW7: 4 assists from 0.12 xA, 3 goals from 1.83 xG — the model deliberately ignores that run.)
10. **Last-season priors [GW10]** — share prior still helps early (goals MAE 0.607 vs 0.638 without) but fades; minutes prior already
    tiny. Re-test whether it still matters. Penalties from last season are still inside former takers' prior share.
11. **xG-based recent form** — the form adjustment is points-based, capped at +/-0.5 and zero through GW5 by construction (needs more
    than 4 prior games), so the Form Sensitivity slider is untestable until ~GW10. Consider replacing it with recent xGI per 90.
12. **Promoted-team prior and full MLE ratings model** [parked] — deferred from the original plan.

### Penalties and set pieces
13. **Calibrate `PENALTY_AWARD_RATE_PER_MATCH` [GW10]** (0.12) against #1 takers' goal residuals; check team expected goals sum to
    team xG; consider taker-specific conversion once samples exist (league rate used for everyone now).
14. **Corners and direct free kicks [GW10]** — skipped by decision. Corner #2-3 takers' assists were +0.064/gw under-predicted
    (t = 2.6, one of ~12 tests); direct-FK #1 goals +0.055 (n.s.). Re-test with leak-free archived orders.

### Other components
15. **Clean sheets [GW12-15]** — bias -0.08 (team CS rate 26% predicted vs 30% actual at GW5; GW6 league-wide 13% vs 29% predicted,
    a one-off). If still ~25% low, use the joint Dixon-Coles P(0 goals) from the Phase 4 fit instead of exp(-lambda).
16. **DefCon** — calibrated; consider opponent possession/game-state effects.
17. **Goalkeeper saves [data]** — needs shots faced (xG undercounts: 2.76 vs 2.93 saves per match); small (87 rows) and low value.
18. **Bonus for MID/FWD** — no better than the mean. Monte Carlo/BPS simulation ("bonus v2") would also give captaincy variance and
    correlation; a substantial standalone feature.
19. **Cards** — fine after the per-90 change; red cards pooled at the league rate.

## 2. Team goals (xG) model
20. **`odds_weight` [GW10]** — default 0.6; best 0.3 on 10 fixtures (bootstrap 0-1). Re-tune once >= 30 fixtures have odds. Odds exist only
    for gameweeks where Refresh Odds was pressed, so this fills in slowly.
21. **Shrink Tier 2 toward the league average** — Tier 2 alone is worse than "predict the league mean" (skill -14%); `FDR_SENSITIVITY`
    0.25 was chosen on the same data.
22. **Player-props cross-check** [decision] — uses Odds API quota; skipped pending an explicit go-ahead.

## 3. Evaluation and measurement
23. **Live accuracy [first read after GW6/GW7]** — archived predictions vs real points vs FPL `ep_next` (built). One gameweek is noise;
    read after ~5+. Add: our-best-XI vs hindsight-best-XI and "optimal squad vs ours" benchmarks to the tab.
24. **GW6 post-mortem facts to compare against next time** — hindsight best XI 132 (GW2-6: 143, 128, 146, 150, 132) vs our 30.0 for
    those 11 players; our model's own best XI predicted 57.6 and scored 40.0; the best attainable squad costs £84.7m and also scores 132.
    The gap decomposes into team scores -1.1, participation -1.4, player-level -6.8 points per player (about 73% unforecastable).
    Your team: XI 41 + captain 2 = 43 (predicted 57.5; bench 6 vs 15.1 predicted).
25. **Ranking metrics in the player backtest** — top-N overlap and rank correlation, captain pick, predicted-best XI vs real best XI
    (the live-accuracy tab has these for archived gameweeks only).
26. **Archived market data as features [GW10+]** — price, ownership and transfers are in every snapshot from GW6; so far they carry no
    useful lift for "who will start" (AUC ~0.5-0.6).

## 4. Optimizer and app
27. **FPL selling prices [data]** — Wildcard/Free Hit budgets use current prices, which overstates what you can spend (FPL pays purchase
    price + half the profit). We don't have your purchase prices; would need a manual input.
28. **Free Hit takes precedence in Transfer Advice** — if Free Hit is ticked it returns a GW1-only Free Hit squad before the Wildcard
    logic runs, regardless of the horizon. Consider making chip handling explicit.
29. **Wildcard threshold** — recommend only if the LP rebuild beats the current XI by max(8, 5 x gameweeks); untested.
30. **Risk-aware optimisation** — the optimizer maximises expected points; hauls are skewed (median 2 for a premium midfielder), so
    captaincy/differential choices may deserve variance-aware logic (ties in with bonus v2 and top-end shrinkage).
31. **Live accuracy preview** — show per-gameweek fixture progress and pending predictions before a gameweek finishes.
32. **Single horizon default** — currently 3 gameweeks for both tabs (changed from 1 vs 3); revisit if that feels wrong.

## 5. Data and infrastructure (decisions already made)
- Scheduled/automatic snapshots: **declined** (LaunchAgent and cron judged too risky); archive stays manual.
- GitHub Actions as redundancy: **not wanted** for now.
- Backtests only use this season's data (no cross-season historical import).

## 6. Checklist for the next review (after GW7)
1. Press **Save snapshot now** before the GW7 deadline (Refresh Odds first only if you want odds in it).
2. After GW7: compare predicted vs actual for your team again; re-run the "best XI we missed" check.
3. Open **Load Live Accuracy** — GW6 and GW7 rows (ours vs FPL `ep_next`).
4. Re-measure the first-start / absent-last-game minutes patterns with GW7 added.
5. Re-check top-end calibration (predicted >= 5 and top-15 per gameweek) and single-game-xG outliers.

## 7. Small UI / clarity ideas (from the GW8 review)
33. **Team Overview column labels** — "GF6/GA6" are ACTUAL goals per game over the last 6 (Brighton 3.17), while the model's Tier 2 uses
    real xG (Brighton 1.92) and "Odds Att" is the bookmaker view for the next gameweek (Brighton 1.10 at Liverpool, 18th of 20). The
    mix-up led to a "Brighton looks strong" misread; add tooltips/labels ("goals, not xG") or an xG-for column.
34. **Player table fixture context** — show the next opponent and the team's blended xG for/against next to the prediction, so a low
    number can be traced to the fixture without opening the API.

## 8. GW8 decision log (2026-10-23, deadline 17:30 UTC)
- Your squad entering GW8: Leno, Raya; Calafiori, Guéhi, Hall, Bogle, Schuster; Groß, E.Le Fée, Schade, Saka, Enzo; Haaland, Kostoulas, Thiago.
  Saka was out (thigh injury, 0%) and Schuster a 75% doubt; 2 free transfers, £0.0m bank; Haaland the model's clear captain (7.0).
- Model's pick for 2 free transfers: Saka + Schuster -> Mbeumo + Konsa (GW8 XI+captain 58.5 vs 53.5 doing nothing; four gameweeks
  198.7 vs 182.7). Chelsea route: Saka + Schade -> Mbeumo + Neto (58.4 / 198.6). Palmer-in best: Saka + Guéhi -> Konsa + Palmer (57.8 / 196.3).
  Rogers + João Pedro together does not fit at £0.0m unless Thiago is sold. **What you actually played is not recorded here — tell me
  next time so the post-mortem can use it.**
- The last GW8 snapshot was saved 15:19 UTC (131 minutes before the deadline), after which the app rolled to GW9.

## 9. Checklist for the next review (after GW8)
1. Compare predicted vs actual for the squad you actually played; re-run "best XI we missed" and the minutes-surprise check.
2. **Load Live Accuracy**: GW6, GW7 and GW8 rows (ours vs FPL `ep_next`).
3. Re-measure first-start / absent-last-game minutes patterns and the new-starter prior drag (2a) with GW8 added.
4. Re-check top-end calibration (predicted >= 5, top-15 per gameweek) and single-game xG outliers.
5. Re-run the section-11 signal study with GW9 (and GW10 for the proper re-test) before deciding on any tweak.
6. Before the GW9 deadline (2026-10-31 11:00 UTC): optionally Refresh Odds (uses quota), then **Save snapshot now** after injury news.

## 10. GW8 post-mortem (2026-10-31)
- **Your GW8:** XI Raya; Calafiori, Guéhi, Bogle; Groß, Schade, Enzo, Palmer; Haaland (c), Kostoulas, Thiago; bench Leno, E.Le Fée, Hall, Thomas.
  Predicted 52.4 (final snapshot; the app showed 55+ when you set the team) vs actual **39** (31 XI + 8 captain). Bench scored 17; the best
  XI from your own 15 would have scored 50, and the model's own XI (Raya; Calafiori, Thomas, Hall, Bogle, Guéhi; Palmer, E.Le Fée, Enzo; Haaland,
  Thiago) predicted 57.0 and scored 44.
- **Why:** the six teams in the XI scored 4 goals vs 9.7 predicted (league-wide 0.95 vs 1.42 goals per team, 8 clean sheets vs ~5 predicted);
  XI goal points 13.8 predicted vs 4.0; discipline cost ~7 points (Palmer red card — the only one in the league, Groß own goal, two yellows);
  Calafiori played 22 minutes (hip, now a 75% doubt), Guéhi 45, Thomas 59 (one minute short of the appearance points). Of the XI's 15.9-point
  error: team scores ~5, participation ~3, player-level ~7.
- **What we missed:** Gabriel 17 (our #2) and Konsa 14 (our #8) at Arsenal — rated by the model, not owned; N.Jackson 13 (rank 43); four
  Bournemouth players 10-11 (they beat prediction by 1.76 goals); Dedić (expected 23 mins, played 90) and Tel (12 -> 88) 11 each; Trafford 10.
  Hindsight best XI 128 vs our 34.2 for those players; GW2-8 best XI ranges 128-150.
- League-wide GW8 the model was fine in total (1.97 predicted vs 2.07 actual per player) but goals were 42% below prediction while clean-sheet
  points were 61% above — an attack-heavy squad had the worst of it.

## 11. Outsize-scorer signal study (2026-10-31, GW2-8, 2,773 player-gameweeks, nothing changed)
Question: can pre-deadline signals (ownership and transfers, price changes, opponent strength, form, minutes) pick the top scorers we miss?
Pre-deadline market data per gameweek came from FPL's own per-gameweek history (`selected`, `value`, `transfers_in/out`), cross-checked against
the GW8 snapshot (snapshot counts ~80% of the final at 131 minutes before the deadline). Top-5 scorers per gameweek = 35 events.
- **Who the top-5 scorers are:** 31% DEF, 40% MID, 26% FWD (forwards are 12% of all rows), 1 GKP; 89% started the previous game (52% overall);
  54% play for a team with xG >= 1.5 (39% overall); median ownership 5.2% vs 0.6% for all players (49% under 5% — they are NOT the deep
  differentials); only 20% price-risers. Our model has them at: top-15 20%, top-30 29%, top-60 46%, top-100 71% (base rates 4% / 8% / 15% / 25%),
  so it already enriches heavily, but 105 of the top-20 scorers per gameweek (75%) were ranked outside our top-30.
- **Single signals (AUC for "top-20 scorer"):** our prediction 0.76, transfers-in 0.75, ownership 0.74, expected minutes 0.72, BPS (last 3) 0.69,
  minutes last game 0.69, price 0.66. The market signals mostly proxy quality our prediction already carries.
- **Do they add anything beyond our prediction?** Leave-one-gameweek-out ridge on actual points: MSE 6.90 -> 6.79-6.86 with market (ownership,
  transfers-in), fixture, participation or form signals added (-0.6% to -1.6%); top-15 picks' mean points 4.88 -> up to 5.1-5.2 (+0.27-0.35, SE
  0.14-0.17) for some combinations, but individual signals and some combinations are NEGATIVE (transfers-in only -0.13, ownership only -0.18,
  crowd + last mins + goals-xG -0.07) — not robust, and the winning subsets were chosen on the same 7 weeks.
- **Is the model biased anywhere?** Overall +0.03 (calibrated). Largest gaps: players who started last game on a team with xG >= 1.5 are
  under-predicted by +0.35 (SE 0.15, 5 of 7 weeks); goals-minus-xG over-performers (+1.5) are over-predicted by -0.89 (n = 30, SE 0.46); our
  top-15 each week over-predicted by -0.48 (SE 0.43); price risers, last-week haulers, ownership and transfers-in groups are within noise.
- **Rule-style "dart throws" (players outside our top-30):** started last game + transfers-in top quartile + team xG >= 1.5 reaches a top-20
  finish 13.8% of the time (n = 174, 3.4x the 4.1% base) — useful as a differential shortlist (~25 players a week, ~3-4 hits), not as a way to
  fill a team. DEF with CS chance >= 40% and low-ownership cheap defenders showed no lift.
- **Conclusion:** most outsize scoring is luck (finishing, bonus, red cards, clean sheets); no single reliable signal. Candidate tweaks to
  re-test at GW10-12 with more weeks and the real snapshot market data: (a) lift regulars who started last game on high-xG teams (+0.35),
  (b) shrink goals-minus-xG over-performers, (c) shrink the top of the ranking (see item 8), (d) add transfers-in/ownership only if a
  held-out test beats the plain prediction by a wide margin (>= +0.4 points per top-15 pick, positive in >= 6 of 8 weeks).

## 12. Squad-selection observations from GW8
35. **Lineup choice** — in GW8 the XI you picked scored 39 vs 44 for the XI by predicted points (E.Le Fée, Hall and Thomas were benched for
    Schade, Kostoulas and Groß). Worth checking whether the app should output the recommended XI and captain explicitly next to the squad.
36. **Attack-heavy exposure** — GW8 punished an attack-heavy squad in a clean-sheet week; ties to risk-aware optimisation (item 30): consider
    balancing defender/clean-sheet and attacker exposure instead of maximising expected points alone.
