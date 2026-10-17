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
   ~0.61 of xG while actual assists run ~0.9 per goal. Fixing it zeroes the bias without improving accuracy. Revisit with more data.
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
