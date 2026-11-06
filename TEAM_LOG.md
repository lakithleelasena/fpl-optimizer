# My team log (what I actually played, for post-mortems)

Each entry: squad, XI, bench, captain, transfers/chips, predicted vs actual. "Predicted" = the final archived snapshot before the deadline
(`archive/2026-27/gwNN.json.gz`). Update after each gameweek; the post-mortem scripts use this file instead of guessing the XI.

## GW10 (deadline Fri 2026-11-06 18:30 UTC; kickoffs Fri 20:00 UTC - Sun 2026-11-08 16:30 UTC) — result pending
- **Transfer:** Enzo (Man City) -> Morgan Rogers (Chelsea, £8.2m now) — 1 free transfer. **Bank:** not stated (my £1.2m estimate from GW9 would be £0.1m short
  of the £8.2m price, so the actual bank or sale prices differ) — **confirm**.
- **Squad:** Leno, Raya; Konsa, Guéhi, Hall, Bogle, Thomas; Groß, E.Le Fée, Schade, Rogers, Palmer; Haaland, Kostoulas, Thiago.
- **XI (3-5-2):** Raya; Konsa, Hall, Bogle; Groß, E.Le Fée, Schade, Rogers, Palmer; Haaland, Thiago.
- **Bench:** Leno, Guéhi, Kostoulas, Thomas.
- **Captain / vice:** Haaland (c) / Palmer (vc) (confirmed).
- **Predictions for the post-mortem:** there is NO pre-deadline GW10 snapshot (the first save was ~39 minutes after the deadline, so it was filed under GW11 and
  carries the placeholder-row minutes bug). Copy kept at `archive/2026-27/post_deadline_gw10_saved_20261106T1909Z.json.gz` (FPL availability, `ep_this`, ownership,
  prices are usable as a proxy for the deadline state). Our GW10 predictions must be reconstructed from history up to GW9 (the backtest path; unaffected by the bug).
- **Actual:** _to fill in after GW10 (finishes Sun 2026-11-08)_.
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
