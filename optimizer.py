from __future__ import annotations

import pulp

from config import MAX_PER_TEAM, MIN_STARTING, SQUAD_COMPOSITION, SQUAD_SIZE, STARTING_XI


def optimize_squad(players: list[dict], budget: int = 1000) -> dict:
    n = len(players)
    prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)

    # Binary variables
    x = [pulp.LpVariable(f"squad_{i}", cat="Binary") for i in range(n)]
    y = [pulp.LpVariable(f"start_{i}", cat="Binary") for i in range(n)]

    # Objective: maximize predicted points of starters (primary),
    # with a tiny secondary term to prefer higher-quality bench and
    # encourage spending up to the budget limit.
    epsilon = 0.001
    prob += (
        pulp.lpSum(y[i] * players[i]["predicted_points"] for i in range(n))
        + epsilon * pulp.lpSum(x[i] * players[i]["predicted_points"] for i in range(n))
    )

    # Squad size = 15
    prob += pulp.lpSum(x[i] for i in range(n)) == SQUAD_SIZE

    # Starting XI = 11
    prob += pulp.lpSum(y[i] for i in range(n)) == STARTING_XI

    # Starter must be in squad
    for i in range(n):
        prob += y[i] <= x[i]

    # Budget
    prob += pulp.lpSum(x[i] * players[i]["cost"] for i in range(n)) <= budget

    # Squad composition by position
    for pos, count in SQUAD_COMPOSITION.items():
        indices = [i for i in range(n) if players[i]["position"] == pos]
        prob += pulp.lpSum(x[i] for i in indices) == count

    # Minimum starters per position
    for pos, min_count in MIN_STARTING.items():
        indices = [i for i in range(n) if players[i]["position"] == pos]
        prob += pulp.lpSum(y[i] for i in indices) >= min_count

    # Max GK starters = 1
    gk_indices = [i for i in range(n) if players[i]["position"] == "GKP"]
    prob += pulp.lpSum(y[i] for i in gk_indices) <= 1

    # Max 3 players per team
    team_ids = set(p["team_id"] for p in players)
    for tid in team_ids:
        indices = [i for i in range(n) if players[i]["team_id"] == tid]
        prob += pulp.lpSum(x[i] for i in indices) <= MAX_PER_TEAM

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != 1:
        raise ValueError("Optimization failed — no feasible solution found")

    starters = []
    bench = []
    for i in range(n):
        if x[i].varValue and x[i].varValue > 0.5:
            p = players[i].copy()
            if y[i].varValue and y[i].varValue > 0.5:
                p["is_starter"] = True
                starters.append(p)
            else:
                p["is_starter"] = False
                bench.append(p)

    return {"starters": starters, "bench": bench}


def pick_starting_xi(squad: list[dict]) -> dict:
    """Pick optimal starting XI from a 15-player squad using LP."""
    n = len(squad)
    prob = pulp.LpProblem("FPL_XI", pulp.LpMaximize)
    y = [pulp.LpVariable(f"xi_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(y[i] * squad[i]["predicted_points"] for i in range(n))
    prob += pulp.lpSum(y[i] for i in range(n)) == STARTING_XI

    for pos, min_count in MIN_STARTING.items():
        idx = [i for i in range(n) if squad[i]["position"] == pos]
        prob += pulp.lpSum(y[i] for i in idx) >= min_count

    gk_idx = [i for i in range(n) if squad[i]["position"] == "GKP"]
    prob += pulp.lpSum(y[i] for i in gk_idx) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    starters, bench = [], []
    for i in range(n):
        p = squad[i].copy()
        if y[i].varValue and y[i].varValue > 0.5:
            p["is_starter"] = True
            starters.append(p)
        else:
            p["is_starter"] = False
            bench.append(p)

    starters.sort(key=lambda p: pos_order.get(p["position"], 99))
    bench.sort(key=lambda p: p["predicted_points"], reverse=True)
    return {"starters": starters, "bench": bench}


# ── Scoring helpers (use pre-computed 7-signal gw_pts, not simplified formula) ──

def _player_gw_pts(player: dict, gw_index: int) -> float:
    """Predicted points for a player in a specific GW (0-based index into gw_pts)."""
    gw_pts = player.get("gw_pts", [])
    return float(gw_pts[gw_index]) if gw_index < len(gw_pts) else 0.0


def _squad_gw_pts(squad: list[dict], gw_index: int) -> float:
    """Total predicted points for all players in a squad for a specific GW."""
    return round(sum(_player_gw_pts(p, gw_index) for p in squad), 2)


def recommend_transfers(
    current_team_ids: list[int],
    all_players: list[dict],
    free_transfers: int,
    budget_in_bank: int,
    chips_available: list[str],
    upcoming_gws: list[int] | None = None,
    n_gw: int = 3,
) -> dict:
    id_to_player = {p["id"]: p for p in all_players}
    current_team = [id_to_player[pid] for pid in current_team_ids if pid in id_to_player]

    if len(current_team) != 15:
        raise ValueError(f"Expected 15 players in squad, got {len(current_team)}")

    # Budget available for FH/WC: sell entire current squad + money in bank
    fh_budget = sum(p["cost"] for p in current_team) + budget_in_bank

    # ── Free Hit: LP-optimise for GW1 only ──────────────────────────────────────
    # FH is a single-GW chip — always optimise for the immediate next gameweek.
    if "free_hit" in chips_available and upcoming_gws:
        fh_players = [{**p, "predicted_points": _player_gw_pts(p, 0)} for p in all_players]
        fh_result = optimize_squad(fh_players, budget=fh_budget)
        fh_squad = fh_result["starters"] + fh_result["bench"]

        current_id_set = {p["id"] for p in current_team}
        fh_id_set = {p["id"] for p in fh_squad}
        id_to_current = {p["id"]: p for p in current_team}

        transfers = []
        for pos in ("GKP", "DEF", "MID", "FWD"):
            going_out = sorted(
                [id_to_current[pid] for pid in (current_id_set - fh_id_set)
                 if id_to_current.get(pid, {}).get("position") == pos],
                key=lambda p: _player_gw_pts(p, 0),
            )
            coming_in = sorted(
                [p for p in fh_squad if p["id"] in (fh_id_set - current_id_set) and p["position"] == pos],
                key=lambda p: _player_gw_pts(p, 0), reverse=True,
            )
            for out_p, in_p in zip(going_out, coming_in):
                transfers.append({
                    "transfer_out": out_p,
                    "transfer_in": in_p,
                    "points_gain": round(_player_gw_pts(in_p, 0) - _player_gw_pts(out_p, 0), 2),
                    "is_hit": False,
                })

        starters_sorted = sorted(fh_result["starters"], key=lambda p: p["predicted_points"], reverse=True)
        captain = starters_sorted[0] if starters_sorted else None
        vice_captain = starters_sorted[1] if len(starters_sorted) > 1 else None

        starter_gw1_pts = _squad_gw_pts(fh_result["starters"], 0)
        chip_rec = {
            "chip": "free_hit",
            "reason": (
                f"Free Hit squad optimised for GW{upcoming_gws[0]} — your starters are predicted "
                f"{starter_gw1_pts:.0f} pts this gameweek using the globally optimal 15."
            ),
        }

        return {
            "current_xi": fh_result,
            "transfers": transfers,
            "chip_recommendation": chip_rec,
            "hits_required": 0,
            "net_points_gain": round(sum(t["points_gain"] for t in transfers), 2),
            "captain": captain,
            "vice_captain": vice_captain,
        }

    # ── Wildcard: run LP with full budget and compare against current squad ──────
    # If the LP rebuild gains enough over the selected horizon, recommend wildcard
    # and return the LP-optimal squad (no greedy transfers needed).
    wc_held = False
    wc_gain_pts = 0.0
    if "wildcard" in chips_available:
        wc_result = optimize_squad(all_players, budget=fh_budget)
        wc_starters_pts = sum(p["predicted_points"] for p in wc_result["starters"])

        current_xi_check = pick_starting_xi(current_team)
        current_pts = sum(p["predicted_points"] for p in current_xi_check["starters"])
        wc_gain_pts = round(wc_starters_pts - current_pts, 2)

        # Threshold scales with the number of GWs being optimised
        wc_threshold = max(8.0, n_gw * 5.0)

        if wc_gain_pts >= wc_threshold:
            wc_squad = wc_result["starters"] + wc_result["bench"]
            current_id_set = {p["id"] for p in current_team}
            wc_id_set = {p["id"] for p in wc_squad}
            id_to_current = {p["id"]: p for p in current_team}

            transfers = []
            for pos in ("GKP", "DEF", "MID", "FWD"):
                going_out = sorted(
                    [id_to_current[pid] for pid in (current_id_set - wc_id_set)
                     if id_to_current.get(pid, {}).get("position") == pos],
                    key=lambda p: p["predicted_points"],
                )
                coming_in = sorted(
                    [p for p in wc_squad if p["id"] in (wc_id_set - current_id_set) and p["position"] == pos],
                    key=lambda p: p["predicted_points"], reverse=True,
                )
                for out_p, in_p in zip(going_out, coming_in):
                    transfers.append({
                        "transfer_out": out_p,
                        "transfer_in": in_p,
                        "points_gain": round(in_p["predicted_points"] - out_p["predicted_points"], 2),
                        "is_hit": False,
                    })

            starters_sorted = sorted(wc_result["starters"], key=lambda p: p["predicted_points"], reverse=True)
            captain = starters_sorted[0] if starters_sorted else None
            vice_captain = starters_sorted[1] if len(starters_sorted) > 1 else None

            n_gw_label = f"{n_gw} GW{'s' if n_gw > 1 else ''}"
            chip_rec = {
                "chip": "wildcard",
                "reason": (
                    f"LP rebuild predicts {wc_starters_pts:.0f} pts over the next {n_gw_label} "
                    f"vs {current_pts:.0f} pts with your current squad "
                    f"(gain: +{wc_gain_pts:.0f} pts). Use your Wildcard to rebuild for free."
                ),
            }
            return {
                "current_xi": wc_result,
                "transfers": transfers,
                "chip_recommendation": chip_rec,
                "hits_required": 0,
                "net_points_gain": wc_gain_pts,
                "captain": captain,
                "vice_captain": vice_captain,
            }
        else:
            wc_held = True  # wildcard available but gain not sufficient — fall through

    # ── Greedy transfer loop ─────────────────────────────────────────────────────
    current_ids = set(p["id"] for p in current_team)

    team_counts: dict[int, int] = {}
    for p in current_team:
        team_counts[p["team_id"]] = team_counts.get(p["team_id"], 0) + 1

    working_ids = set(current_ids)
    working_counts = dict(team_counts)
    working_bank = budget_in_bank
    remaining_free = free_transfers

    available = {p["id"]: p for p in all_players if p["id"] not in current_ids}
    sorted_by_pts = sorted(current_team, key=lambda p: p["predicted_points"])

    transfers = []
    max_transfers = free_transfers + 2

    for candidate_out in sorted_by_pts:
        if len(transfers) >= max_transfers:
            break
        if candidate_out["id"] not in working_ids:
            continue

        pos = candidate_out["position"]
        budget_for_in = candidate_out["cost"] + working_bank
        hit_cost = 0 if remaining_free > 0 else 4

        best_in = None
        best_net = 0.0

        for p_in in available.values():
            if p_in["position"] != pos:
                continue
            if p_in["cost"] > budget_for_in:
                continue
            if working_counts.get(p_in["team_id"], 0) >= MAX_PER_TEAM:
                continue
            net = p_in["predicted_points"] - candidate_out["predicted_points"] - hit_cost
            if net > best_net:
                best_net = net
                best_in = p_in

        if best_in is None:
            continue

        transfers.append({
            "transfer_out": candidate_out,
            "transfer_in": best_in,
            "points_gain": best_in["predicted_points"] - candidate_out["predicted_points"],
            "is_hit": remaining_free <= 0,
        })

        working_bank = budget_for_in - best_in["cost"]
        working_ids.discard(candidate_out["id"])
        working_ids.add(best_in["id"])
        working_counts[candidate_out["team_id"]] = working_counts.get(candidate_out["team_id"], 0) - 1
        working_counts[best_in["team_id"]] = working_counts.get(best_in["team_id"], 0) + 1
        del available[best_in["id"]]
        available[candidate_out["id"]] = candidate_out

        if remaining_free > 0:
            remaining_free -= 1

    final_squad = [id_to_player[pid] for pid in working_ids if pid in id_to_player]
    xi_result = pick_starting_xi(final_squad)

    starters_sorted = sorted(xi_result["starters"], key=lambda p: p["predicted_points"], reverse=True)
    captain = starters_sorted[0] if starters_sorted else None
    vice_captain = starters_sorted[1] if len(starters_sorted) > 1 else None

    chip_rec = _recommend_chip(
        current_team=current_team,
        xi_result=xi_result,
        transfers=transfers,
        chips_available=chips_available,
        all_players=all_players,
        free_transfers=free_transfers,
        upcoming_gws=upcoming_gws or [],
        n_gw=n_gw,
        fh_budget=fh_budget,
        wc_held=wc_held,
        wc_gain=wc_gain_pts,
    )

    hits_required = max(0, len(transfers) - free_transfers)
    gross_gain = sum(t["points_gain"] for t in transfers)
    net_points_gain = gross_gain - hits_required * 4

    return {
        "current_xi": xi_result,
        "transfers": transfers,
        "chip_recommendation": chip_rec,
        "hits_required": hits_required,
        "net_points_gain": round(net_points_gain, 2),
        "captain": captain,
        "vice_captain": vice_captain,
    }


def _recommend_chip(
    current_team: list[dict],
    xi_result: dict,
    transfers: list[dict],
    chips_available: list[str],
    all_players: list[dict],
    free_transfers: int,
    upcoming_gws: list[int],
    n_gw: int = 3,
    fh_budget: int = 1000,
    wc_held: bool = False,
    wc_gain: float = 0.0,
) -> dict:
    if not chips_available:
        return {"chip": None, "reason": "No chips available. Hold for a future gameweek."}

    starters = xi_result["starters"]
    bench = xi_result["bench"]
    n_compare = min(n_gw, len(upcoming_gws))

    if n_compare == 0:
        return {"chip": None, "reason": "No upcoming gameweeks available for chip analysis."}

    # Per-GW totals using the 7-signal gw_pts array (not simplified formula)
    starter_gw = [_squad_gw_pts(starters, i) for i in range(n_compare)]
    bench_gw = [_squad_gw_pts(bench, i) for i in range(n_compare)]

    starters_sorted = sorted(starters, key=lambda p: p["predicted_points"], reverse=True)
    top = starters_sorted[0] if starters_sorted else None
    cap_gw = [_player_gw_pts(top, i) for i in range(n_compare)] if top else [0.0] * n_compare

    gw_labels = [
        f"GW{upcoming_gws[i]}" if i < len(upcoming_gws) else f"GW{i + 1}"
        for i in range(n_compare)
    ]

    chip_use: dict[str, str] = {}
    chip_hold: dict[str, str] = {}

    # ── Wildcard hold note (evaluated upstream — gain was below threshold) ───────
    if wc_held and "wildcard" in chips_available:
        chip_hold["wildcard"] = (
            f"Wildcard: LP rebuild gains only +{wc_gain:.0f} pts over {n_gw} GW(s) — "
            "hold for a double gameweek or a larger squad overhaul."
        )

    # ── Bench Boost ───────────────────────────────────────────────────────────────
    if "bench_boost" in chips_available:
        best_i = max(range(n_compare), key=lambda i: bench_gw[i])
        if bench_gw[0] >= 10 and best_i == 0:
            breakdown = ", ".join(f"{gw_labels[i]}: {bench_gw[i]:.1f}" for i in range(n_compare))
            chip_use["bench_boost"] = (
                f"Your bench is predicted {bench_gw[0]:.1f} pts this GW — the strongest "
                f"of the next {n_compare} GW(s) ({breakdown}). "
                "Activate Bench Boost now for maximum impact."
            )
        else:
            chip_hold["bench_boost"] = (
                f"Bench Boost: bench scores {bench_gw[0]:.1f} pts this GW "
                f"but peaks in {gw_labels[best_i]} ({bench_gw[best_i]:.1f} pts) — hold."
            )

    # ── Triple Captain ────────────────────────────────────────────────────────────
    if "triple_captain" in chips_available and top:
        best_i = max(range(n_compare), key=lambda i: cap_gw[i])
        if cap_gw[0] >= 6 and best_i == 0:
            breakdown = ", ".join(f"{gw_labels[i]}: {cap_gw[i]:.1f}" for i in range(n_compare))
            chip_use["triple_captain"] = (
                f"{top['name']} has his best fixture this GW ({breakdown}). "
                "Use Triple Captain now for maximum return."
            )
        else:
            chip_hold["triple_captain"] = (
                f"Triple Captain: {top['name']} peaks in {gw_labels[best_i]} "
                f"({cap_gw[best_i]:.1f} vs {cap_gw[0]:.1f} pts this GW) — hold."
            )

    # ── Free Hit timing evaluation ────────────────────────────────────────────────
    # Run LP for each GW to compute a valid constrained optimal squad benchmark
    # (replaces the old unconstrained top-15 which ignored squad rules).
    if "free_hit" in chips_available:
        deficits = []
        current_totals = []
        optimal_totals = []
        for i in range(n_compare):
            # Score players by gw_pts[i] for the LP
            scored = [{**p, "predicted_points": _player_gw_pts(p, i)} for p in all_players]
            try:
                opt_result = optimize_squad(scored, budget=fh_budget)
                opt_total = _squad_gw_pts(opt_result["starters"], i)
            except Exception:
                opt_total = 0.0
            current_total = starter_gw[i]
            deficits.append(opt_total - current_total)
            current_totals.append(current_total)
            optimal_totals.append(opt_total)

        best_i = max(range(n_compare), key=lambda i: deficits[i])
        gw1_weak = optimal_totals[0] > 0 and current_totals[0] < optimal_totals[0] * 0.80
        if deficits[0] > 0 and best_i == 0 and gw1_weak:
            chip_use["free_hit"] = (
                f"Your starters predict {current_totals[0]:.0f} pts this GW vs {optimal_totals[0]:.0f} pts "
                f"with an optimal Free Hit squad (gap: {deficits[0]:.0f} pts). "
                f"This is your weakest GW of the next {n_compare} — use Free Hit now."
            )
        else:
            chip_hold["free_hit"] = (
                f"Free Hit: your squad is relatively strong this GW — "
                f"biggest deficit is in {gw_labels[best_i]} ({deficits[best_i]:.0f} pts gap). "
                "Consider saving for that week."
            )

    # ── Return best recommendation ────────────────────────────────────────────────
    # Priority: free_hit > bench_boost > triple_captain
    for chip in ("free_hit", "bench_boost", "triple_captain"):
        if chip in chip_use:
            return {"chip": chip, "reason": chip_use[chip]}

    if chip_hold:
        hold_summary = " | ".join(chip_hold.values())
        return {"chip": None, "reason": hold_summary}

    return {
        "chip": None,
        "reason": (
            f"No chip recommended this gameweek. Your squad is well set over the next {n_gw} GW(s) — "
            "hold chips for a double gameweek or a standout captain fixture."
        ),
    }
