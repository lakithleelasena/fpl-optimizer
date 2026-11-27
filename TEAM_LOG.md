# My team log (what I actually played, for post-mortems)

Each entry: squad, XI, bench, captain, transfers/chips, predicted vs actual. "Predicted" = the final archived snapshot before the deadline
(`archive/2026-27/gwNN.json.gz`). Update after each gameweek; the post-mortem scripts use this file instead of guessing the XI.

## GW12 (deadline Fri 2026-11-27 18:30 UTC) — result pending
- **Transfer:** Thiago (Brentford, suspended until 5 Dec; sold £7.7m assumed) -> Calvert-Lewin (Leeds, £6.0m, 19.6% owned, first-choice penalty taker) — 1 free transfer,
  no hit. **Bank after: ~£2.5m** (assumes Thiago's sale at £7.7m and the £0.8m GW11 bank — confirm). Model's best single move for GW12 (+3.1 GW12, +8.6 over four GWs);
  alternative was Thiago -> N.Jackson (+2.7 / +9.5, £1.2m left).
- **Squad:** Leno, Raya; Konsa, Van Hecke, Hall, Bogle, Thomas; Groß, E.Le Fée, Schade, Rogers, Palmer; Haaland, Kostoulas, Calvert-Lewin.
- **XI (3-5-2, the model's best formation; followed all recommendations):** Leno (BOU h); Van Hecke (SUN a), Bogle (COV h), Konsa (MCI h); Groß (NEW h),
  E.Le Fée (TOT h), Palmer (NFO a), Rogers (NFO a), Schade (MUN a); Calvert-Lewin (COV h), Haaland (ARS a).
- **Bench (order):** Raya (75% shoulder); Kostoulas, Thomas, Hall (75% unspecified).
- **Captain / vice:** Calvert-Lewin (c) / Van Hecke (vc). Differential: Haaland (75% owned) predicted only 3.6 at Arsenal.
- **Predicted (snapshot saved 2026-11-27 16:39 UTC, 110 minutes before the deadline; odds refreshed 16:30 UTC, 20 fixtures):** XI 47.0 + captain 5.3 = **52.3**
  (no-transfer best was 49.2). Leno 3.32, Van Hecke 5.20, Bogle 4.78, Konsa 4.40, Groß 4.81, E.Le Fée 4.76, Palmer 3.99, Rogers 3.96, Schade 2.90 (56 exp min),
  Calvert-Lewin 5.26, Haaland 3.63; bench Raya 2.73, Kostoulas 2.21, Thomas 1.85, Hall 1.82. Team xG: Leeds 1.77-0.79 v COV, Spurs 1.45-0.98, Arsenal 1.38-0.97 v MCI,
  Brighton 2.09-1.07, Sunderland 0.98-1.45, Chelsea 1.24-1.47, Fulham 1.21-1.38.
- **Correlation notes:** Konsa vs Haaland in the same match (opposite sides — mild hedge); Bogle + Calvert-Lewin both Leeds v Coventry (Thomas on the bench is Coventry).
- **Post-mortem to do:** predicted 52.3 vs actual; Calvert-Lewin (c) vs Haaland; Leno vs Raya; Schade's minutes (rotation-risk lesson from GW11).
- **Plan for GW13 (deadline Tue 2026-12-01 18:30 UTC, quick turnaround):** Schade -> Tavernier (Bournemouth £6.8m, best over four GWs) is affordable with ~£2.5m bank;
  re-check Raya and Hall fitness.

## GW11 (deadline Sat 2026-11-21 11:00 UTC) — **29 points** (GW average 35)
- **Transfer:** Guéhi (Man City, 75% hip doubt) -> Van Hecke (Spurs, £5.1m) — 1 free transfer, no hit. **Bank after: £0.8m (confirmed from the FPL screenshot)**,
  so Guéhi sold for £5.9m. Overall rank going in: **2,122,957**. Model's pick was the same transfer (best of 526 single moves: +4.4 GW11, +11.6 over four GWs).
- **Squad:** Leno, Raya; Konsa, Van Hecke, Hall, Bogle, Thomas; Groß, E.Le Fée, Schade, Rogers, Palmer; Haaland, Kostoulas, Thiago.
- **XI (3-5-2, deliberate risk pick to climb):** Raya (NEW a); Konsa (NEW a), Hall (ARS h), Van Hecke (IPS h); Palmer (LEE h), Rogers (LEE h), Groß (HUL a),
  Schade (EVE h), E.Le Fée (AVL a); Haaland (FUL h), Kostoulas (HUL a).
- **Bench (order):** Leno; Bogle, Thiago (suspended until 5 Dec), Thomas (75% calf).
- **Captain / vice:** Haaland (c) / Palmer (vc).
- **Predicted (snapshot saved 2026-11-21 03:18 UTC, 462 minutes before the deadline; odds refreshed 03:03 UTC, 20 fixtures; odds_weight 0.6; code ade5943):**
  XI 49.7 + Haaland captain 5.9 = **55.5**. Raya 4.15, Konsa 5.34, Hall 3.18, Van Hecke 6.61, Palmer 4.45, Rogers 4.33, Groß 4.15, Schade 4.09, E.Le Fée 4.61,
  Haaland 5.86, Kostoulas 2.91 (p60 0.78); bench Leno 3.71, Bogle 3.07, Thiago 0.10, Thomas 0.51. FPL `ep_next` for the XI: 55.4 + captain 4.0 = 59.4.
- **Choices vs the model's line-up:** model preferred 4-5-1 (Bogle for Kostoulas, -0.16 before autosubs, ~0.1 after) and leaned Van Hecke (c) (6.6, 17% owned)
  over Haaland (5.9, 79% owned). Together these cost ~0.9 predicted points (55.5 vs ~56.4) in exchange for the 3-5-2 goal upside.
- **Actual: 29 points** = XI 27 + Haaland captain extra 2 (no autosubs; everyone played). Raya 1 (69 min, shoulder), Konsa 1, Hall 1, Van Hecke 6, Palmer 2, Rogers 5,
  Groß 4, Schade 1 (24 min), E.Le Fée 3, Haaland 2, Kostoulas 1 (23 min); bench Leno 2, Bogle 4, Thiago 0, Thomas 3.
- **Post-mortem (2026-11-27): -26.5 vs predicted 55.5.** Breakdown: Haaland captain blank -7.7 (1.03 xG, no goal in a 1-1 with Fulham); Newcastle 2-2 Arsenal
  hit three starters (Raya, Konsa, Hall) -9.7 (model had Arsenal 1.63-0.85; clean sheet lost); minutes -5.0 (Schade 24 min in Brentford's 4-0 win, Kostoulas 23 min);
  Chelsea lost 1-4 at home to Leeds (Palmer/Rogers -1.8); E.Le Fée -1.6. Finishing luck: the XI had 1.9 xG + 0.8 xA but 0 goals + 1 assist (~11 points of the gap).
  Line-up choices cost 7: Bogle 4 vs Kostoulas 1 (-3) and Van Hecke (c) 6 vs Haaland (c) 2 (-4); the model's line-up would have scored ~36.
  The model itself was calibrated this week: 197 players predicted >=2 averaged 3.26 predicted vs 3.24 actual; its top 15 averaged 5.2 actual vs 5.04 predicted
  (FPL `ep_next` top 15: 5.53). GW11 top scorers (Lewis-Potter 16, Okafor 15, Hill 14, Wirtz 14) — Hill and Wirtz were in the model's top 11.

## GW10 (deadline Fri 2026-11-06 18:30 UTC; kickoffs Fri 20:00 UTC - Sun 2026-11-08 16:30 UTC) — **53 points**
- **Transfer:** Enzo (Man City) -> Morgan Rogers (Chelsea, £8.2m now) — 1 free transfer. **Bank:** not stated (my £1.2m estimate from GW9 would be £0.1m short
  of the £8.2m price, so the actual bank or sale prices differ) — **confirm**.
- **Squad:** Leno, Raya; Konsa, Guéhi, Hall, Bogle, Thomas; Groß, E.Le Fée, Schade, Rogers, Palmer; Haaland, Kostoulas, Thiago.
- **XI (3-5-2):** Raya; Konsa, Hall, Bogle; Groß, E.Le Fée, Schade, Rogers, Palmer; Haaland, Thiago.
- **Bench:** Leno, Guéhi, Kostoulas, Thomas.
- **Captain / vice:** Haaland (c) / Palmer (vc) (confirmed).
- **Predictions for the post-mortem:** there is NO pre-deadline GW10 snapshot (the first save was ~39 minutes after the deadline, so it was filed under GW11 and
  carries the placeholder-row minutes bug). Copy kept at `archive/2026-27/post_deadline_gw10_saved_20261106T1909Z.json.gz` (FPL availability, `ep_this`, ownership,
  prices are usable as a proxy for the deadline state). Our GW10 predictions must be reconstructed from history up to GW9 (the backtest path; unaffected by the bug).
- **Actual (computed from FPL's history; assumes no autosubs — all 11 starters played; bank still unconfirmed): 53 points** = XI 51 + Haaland captain extra 2. Raya 6, Konsa 8, Hall 4, Bogle 7,
  Groß 10, E.Le Fée 11, Schade 1, Rogers 2, Palmer 2, Haaland 2, Thiago -2 (red card, 22 minutes); bench 5 (Leno 3, Guéhi 0, Kostoulas 2, Thomas 0).
- **Predicted (RECONSTRUCTED from history up to GW9, no odds pulled for GW10, availability = post-deadline FPL flags):** XI 45.4 + captain 5.3 = **50.7** (FPL's `ep_this` sum 41.2 + 5.8 captain).
  Result 53 vs 50.7 predicted: within 2.3. Team-level: your 8 teams scored 7 goals vs 10.9 predicted but kept all 4 clean sheets in the league (Arsenal, Leeds, Brighton, Sunderland), so clean-sheet points
  were 14 vs 6.7 predicted; XI assists 0 vs 4.2 predicted; Thiago's red card (only one in the league) and a Schade yellow cost ~4. Hindsight best XI 132; the model's own best XI would have scored 72 (predicted 56.9).
- **Model changes agreed for GW11 (not made yet):** start with the placeholder-row bug (the history gets a 0-minute row for every player's unplayed fixture right after
  the deadline, which caps start probabilities at ~0.6 until the games are played); then the re-test list in `FUTURE_ENHANCEMENTS.md` (sections 9 and 11).

## GW9 (deadline Sat 2026-10-31 11:00 UTC) — finished 2026-11-02
- **Transfer:** Calafiori -> Konsa (1 free transfer used; assumed sale at the listed £5.6m, so ~£1.2m left in the bank — confirm).
- **XI (3-5-2):** Raya; Konsa, Hall, Bogle; Groß, E.Le Fée, Schade, Enzo, Palmer; Haaland, Thiago.
- **Bench:** Leno, Kostoulas, Guéhi, Thomas.
- **Captain:** Haaland (confirmed). Vice not stated.
- **Predicted (snapshot saved 2026-10-31 00:50 UTC, 610 minutes before the deadline, odds included, not re-saved afterwards):**
  XI 45.6 + captain 6.6 = **52.2**. Raya 4.1, Konsa 4.2, Hall 3.7, Bogle 3.0, Groß 3.0, E.Le Fée 5.2, Schade 2.8, Enzo 4.0, Palmer 4.1,
  Haaland 6.6, Thiago 4.8; bench Leno 3.7, Kostoulas 2.6, Guéhi 2.4, Thomas 1.3 (50% doubt at the time).
- **Actual: 54 points (confirmed)** — XI 52 (Raya 5, Konsa 10, Hall 2, Bogle 1, Groß 2, E.Le Fée 8, Schade 3, Enzo 2, Palmer 8, Haaland 2, Thiago 9) + Haaland captain extra 2;
  bench 4 (Leno 2, Kostoulas 2, Guéhi 0, Thomas 0). **Predicted 52.2 (snapshot 610 minutes before the deadline) -> within 2 points.** The XI you played matches the one logged.
  No component post-mortem done yet.
- Context: model's GW9 recommendation was Calafiori -> Van Hecke (best for GW9) or Konsa/Silva (best over four gameweeks); Palmer's one-match
  red-card ban was served in the EFL Cup at Liverpool on Oct 28, so he was available for Chelsea v Man Utd.

## GW8 (deadline 2026-10-23 17:30 UTC) — **39 points**
- **Transfers:** Saka -> Palmer, Schuster -> Thomas (2 free transfers; model suggested Mbeumo + Konsa).
- **XI (3-4-3):** Raya; Calafiori, Guéhi, Bogle; Groß, Schade, Enzo, Palmer; Haaland (c), Kostoulas, Thiago. **Bench:** Leno, E.Le Fée, Hall, Thomas.
- **Predicted** 52.4 (final snapshot; the app showed 55+ when the team was set) vs **actual 39** (XI 31 + captain 8; bench 17). Palmer red card, Groß own goal,
  Calafiori 22 minutes. Full analysis: `FUTURE_ENHANCEMENTS.md` section 10.

## GW7 (deadline 2026-10-17) — score not recorded
- Squad as in GW6 as far as known (no transfers recorded). Haaland scored 13 (our captain pick in live accuracy).

## GW6 (deadline 2026-10-10 10:00 UTC) — Wildcard played — **43 points**
- **XI (3-4-3):** Raya; Calafiori, Guéhi, Hall; Groß, Schade, Saka (c), Enzo; Haaland (vc), Kostoulas, Thiago. **Bench:** Leno, E.Le Fée, Bogle, Schuster.
- **Predicted** 57.5 (XI 51.3 + captain 6.2) vs **actual 43** (XI 41 + Saka captain 2). Hindsight best XI 132. Analysis: `FUTURE_ENHANCEMENTS.md` section 6 / plan doc.

## After GW10 finishes (Sun 2026-11-08) — analysis checklist
0. GW9 post-mortem not done (54 vs 52.2 predicted — close); optional, can be combined with GW10.
1. Fill in the GW10 actual above (and confirm the GW10 bank). GW9 is confirmed at 54; GW10 captain/vice confirmed.
2. Predicted vs actual for the 15 (components, team scores vs participation vs player-level), best XI we missed, and who the model would have picked.
3. **Load Live Accuracy** — GW6-GW9 rows (ours vs FPL `ep_next`); GW10 has no pre-deadline snapshot, so score it from the history-based reconstruction.
4. Re-run the section-11 signal study with GW9 added; re-measure the first-start / new-starter prior-drag minutes patterns.
5. Re-check the lineup question: XI picked vs XI by predicted points (GW8 cost ~5 points).
6. Before the GW11 deadline (2026-11-21 11:00 UTC): optionally Refresh Odds (uses quota), then **Save snapshot now** BEFORE the deadline after the injury news.
