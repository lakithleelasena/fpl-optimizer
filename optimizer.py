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


def recommend_transfers(
    current_team_ids: list[int],
    all_players: list[dict],
    free_transfers: int,
    budget_in_bank: int,
    chips_available: list[str],
    upcoming_gws: list[int] | None = None,
) -> dict:
    id_to_player = {p["id"]: p for p in all_players}
    current_team = [id_to_player[pid] for pid in current_team_ids if pid in id_to_player]

    if len(current_team) != 15:
        raise ValueError(f"Expected 15 players in squad, got {len(current_team)}")

    # ── Free Hit: LP-optimise with GW1-only predictions ──────────────────────
    # When Free Hit is available use the same LP solver as Tab 1 so both tabs
    # show the identical optimal squad.  Total budget = sell value of current
    # squad + money in bank (mirrors how FPL calculates the Free Hit budget).
    if "free_hit" in chips_available and upcoming_gws:
        gw1 = upcoming_gws[0]
        total_budget = sum(p["cost"] for p in current_team) + budget_in_bank

        # Score every player on GW1 only (same as Tab 1 logic)
        fh_players = [{**p, "predicted_points": _gw_player_pts(p, gw1)} for p in all_players]

        fh_result = optimize_squad(fh_players, budget=total_budget)
        fh_squad = fh_result["starters"] + fh_result["bench"]

        # Build transfer list by diffing current squad vs LP squad (paired by position)
        current_id_set = {p["id"] for p in current_team}
        fh_id_set = {p["id"] for p in fh_squad}
        id_to_current = {p["id"]: p for p in current_team}

        transfers = []
        for pos in ("GKP", "DEF", "MID", "FWD"):
            going_out = sorted(
                [id_to_current[pid] for pid in (current_id_set - fh_id_set) if id_to_current.get(pid, {}).get("position") == pos],
                key=lambda p: _gw_player_pts(p, gw1),
            )
            coming_in = sorted(
                [p for p in fh_squad if p["id"] in (fh_id_set - current_id_set) and p["position"] == pos],
                key=lambda p: _gw_player_pts(p, gw1), reverse=True,
            )
            for out_p, in_p in zip(going_out, coming_in):
                transfers.append({
                    "transfer_out": out_p,
                    "transfer_in": in_p,
                    "points_gain": round(_gw_player_pts(in_p, gw1) - _gw_player_pts(out_p, gw1), 2),
                    "is_hit": False,
                })

        starters_sorted = sorted(fh_result["starters"], key=lambda p: p["predicted_points"], reverse=True)
        captain = starters_sorted[0] if starters_sorted else None
        vice_captain = starters_sorted[1] if len(starters_sorted) > 1 else None

        starter_gw1 = _gw_squad_pts(fh_result["starters"], gw1)
        chip_rec = {
            "chip": "free_hit",
            "reason": (
                f"Free Hit squad optimised for GW{gw1} — your starters are predicted "
                f"{starter_gw1:.0f} pts this gameweek using the globally optimal 15."
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

    current_ids = set(p["id"] for p in current_team)

    # Per-team player counts
    team_counts: dict[int, int] = {}
    for p in current_team:
        team_counts[p["team_id"]] = team_counts.get(p["team_id"], 0) + 1

    # Mutable working state
    working_ids = set(current_ids)
    working_counts = dict(team_counts)
    working_bank = budget_in_bank
    remaining_free = free_transfers

    # Available pool: all players NOT currently in squad
    available = {p["id"]: p for p in all_players if p["id"] not in current_ids}

    # Sort current team worst → best (candidates for transfer out)
    sorted_by_pts = sorted(current_team, key=lambda p: p["predicted_points"])

    transfers = []
    # Allow up to free_transfers + 2 hits if strongly worthwhile
    max_transfers = free_transfers + 2

    for candidate_out in sorted_by_pts:
        if len(transfers) >= max_transfers:
            break
        # Skip if already swapped out in a previous iteration
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

        # Apply transfer to working state
        working_bank = budget_for_in - best_in["cost"]
        working_ids.discard(candidate_out["id"])
        working_ids.add(best_in["id"])
        working_counts[candidate_out["team_id"]] = working_counts.get(candidate_out["team_id"], 0) - 1
        working_counts[best_in["team_id"]] = working_counts.get(best_in["team_id"], 0) + 1
        del available[best_in["id"]]
        available[candidate_out["id"]] = candidate_out

        if remaining_free > 0:
            remaining_free -= 1

    # Build final squad and pick optimal XI
    final_squad = [id_to_player[pid] for pid in working_ids if pid in id_to_player]
    xi_result = pick_starting_xi(final_squad)

    # Captain = highest predicted starter
    starters_sorted = sorted(xi_result["starters"], key=lambda p: p["predicted_points"], reverse=True)
    captain = starters_sorted[0] if starters_sorted else None
    vice_captain = starters_sorted[1] if len(starters_sorted) > 1 else None

    # Chip recommendation
    chip_rec = _recommend_chip(
        current_team=current_team,
        xi_result=xi_result,
        transfers=transfers,
        chips_available=chips_available,
        all_players=all_players,
        free_transfers=free_transfers,
        upcoming_gws=upcoming_gws or [],
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


def _gw_squad_pts(squad: list[dict], gw_id: int) -> float:
    """Estimate a squad's total predicted points for one specific gameweek."""
    total = 0.0
    for p in squad:
        ease = p.get("gw_ease", {}).get(gw_id)
        if ease is None:
            continue  # blank GW for this player
        n_fix = len(p.get("gw_fixtures", {}).get(gw_id, []))
        if n_fix == 0:
            continue
        per_match = p.get("season_avg", 0.0) * (0.5 + ease) * p.get("start_likelihood", 0.8)
        total += per_match * n_fix
    return round(total, 2)


def _gw_player_pts(player: dict, gw_id: int) -> float:
    """Predicted points for a single player in a specific GW."""
    ease = player.get("gw_ease", {}).get(gw_id)
    if ease is None:
        return 0.0
    n_fix = len(player.get("gw_fixtures", {}).get(gw_id, []))
    if n_fix == 0:
        return 0.0
    per_match = player.get("season_avg", 0.0) * (0.5 + ease) * player.get("start_likelihood", 0.8)
    return per_match * n_fix


def _recommend_chip(
    current_team: list[dict],
    xi_result: dict,
    transfers: list[dict],
    chips_available: list[str],
    all_players: list[dict],
    free_transfers: int,
    upcoming_gws: list[int],
) -> dict:
    if not chips_available:
        return {"chip": None, "reason": "No chips available. Hold for a future gameweek."}

    starters = xi_result["starters"]
    bench = xi_result["bench"]
    gw1 = upcoming_gws[0] if len(upcoming_gws) > 0 else None
    gw2 = upcoming_gws[1] if len(upcoming_gws) > 1 else None
    gw3 = upcoming_gws[2] if len(upcoming_gws) > 2 else None

    # ── Per-GW squad totals ──────────────────────────────────────────────────
    starter_gw1 = _gw_squad_pts(starters, gw1) if gw1 else 0
    starter_gw2 = _gw_squad_pts(starters, gw2) if gw2 else 0
    starter_gw3 = _gw_squad_pts(starters, gw3) if gw3 else 0

    bench_gw1 = _gw_squad_pts(bench, gw1) if gw1 else 0
    bench_gw2 = _gw_squad_pts(bench, gw2) if gw2 else 0
    bench_gw3 = _gw_squad_pts(bench, gw3) if gw3 else 0

    # ── Per-GW captain value ─────────────────────────────────────────────────
    starters_sorted = sorted(starters, key=lambda p: p["predicted_points"], reverse=True)
    top = starters_sorted[0] if starters_sorted else None
    cap_gw1 = _gw_player_pts(top, gw1) if (top and gw1) else 0
    cap_gw2 = _gw_player_pts(top, gw2) if (top and gw2) else 0
    cap_gw3 = _gw_player_pts(top, gw3) if (top and gw3) else 0

    n_beneficial = len(transfers)

    # ── Wildcard ─────────────────────────────────────────────────────────────
    # Always check first — overhaul need trumps timing of other chips.
    if "wildcard" in chips_available and n_beneficial >= 5:
        return {
            "chip": "wildcard",
            "reason": (
                f"{n_beneficial} beneficial transfers found across the next 3 gameweeks. "
                "Use your Wildcard now to rebuild for free — the squad needs a significant overhaul."
            ),
        }

    # ── Evaluate all remaining chips independently (no early returns on "hold") ──
    # Each chip produces either a positive recommendation or a hold note.
    # At the end the highest-priority positive recommendation wins.

    chip_use: dict[str, str] = {}   # chip -> reason to USE it
    chip_hold: dict[str, str] = {}  # chip -> reason to HOLD it

    # ── Bench Boost ──────────────────────────────────────────────────────────
    if "bench_boost" in chips_available:
        if bench_gw1 >= 10 and bench_gw1 >= bench_gw2 and bench_gw1 >= bench_gw3:
            chip_use["bench_boost"] = (
                f"Your bench is predicted {bench_gw1:.1f} pts this GW — the strongest over the next "
                f"3 gameweeks (GW2: {bench_gw2:.1f}, GW3: {bench_gw3:.1f}). "
                "This is the optimal week to activate Bench Boost."
            )
        else:
            best_gw = "GW2" if bench_gw2 >= bench_gw3 else "GW3"
            best_val = max(bench_gw2, bench_gw3)
            chip_hold["bench_boost"] = (
                f"Bench Boost: your bench scores {bench_gw1:.1f} pts this GW "
                f"but is stronger in {best_gw} ({best_val:.1f} pts) — hold."
            )

    # ── Triple Captain ───────────────────────────────────────────────────────
    if "triple_captain" in chips_available and top:
        if cap_gw1 >= 6 and cap_gw1 >= cap_gw2 and cap_gw1 >= cap_gw3:
            chip_use["triple_captain"] = (
                f"{top['name']} has his best fixture of the next 3 GWs this week "
                f"(GW1: {cap_gw1:.1f}, GW2: {cap_gw2:.1f}, GW3: {cap_gw3:.1f} pts). "
                "Triple Captain now to maximise the return."
            )
        else:
            best_gw = "GW2" if cap_gw2 >= cap_gw3 else "GW3"
            chip_hold["triple_captain"] = (
                f"Triple Captain: {top['name']} has a better fixture in {best_gw} "
                f"({max(cap_gw2, cap_gw3):.1f} vs {cap_gw1:.1f} pts this GW) — hold."
            )

    # ── Free Hit ─────────────────────────────────────────────────────────────
    # Recommend if this GW has the largest gap between current starters and the
    # optimal free-pick 15. Uses per-GW scoring so blanks are correctly penalised.
    if "free_hit" in chips_available and gw1:
        # Sort all players by their GW1 expected pts and take the best 15
        players_by_gw1 = sorted(all_players, key=lambda p: _gw_player_pts(p, gw1), reverse=True)
        top15_gw1 = sum(_gw_player_pts(p, gw1) for p in players_by_gw1[:15])

        players_by_gw2 = sorted(all_players, key=lambda p: _gw_player_pts(p, gw2), reverse=True) if gw2 else []
        top15_gw2 = sum(_gw_player_pts(p, gw2) for p in players_by_gw2[:15]) if gw2 else 0

        players_by_gw3 = sorted(all_players, key=lambda p: _gw_player_pts(p, gw3), reverse=True) if gw3 else []
        top15_gw3 = sum(_gw_player_pts(p, gw3) for p in players_by_gw3[:15]) if gw3 else 0

        deficit_gw1 = top15_gw1 - starter_gw1
        deficit_gw2 = top15_gw2 - starter_gw2
        deficit_gw3 = top15_gw3 - starter_gw3

        if deficit_gw1 > 0 and deficit_gw1 >= deficit_gw2 and deficit_gw1 >= deficit_gw3 and (top15_gw1 == 0 or starter_gw1 < top15_gw1 * 0.80):
            chip_use["free_hit"] = (
                f"Your starters are predicted just {starter_gw1:.0f} pts this GW vs a possible "
                f"{top15_gw1:.0f} pts with an optimal Free Hit squad "
                f"(gap: {deficit_gw1:.0f} pts). This is your weakest GW of the next 3 — "
                "use Free Hit now for maximum impact."
            )
        else:
            worst_gw = max(["GW2", "GW3"], key=lambda g: deficit_gw2 if g == "GW2" else deficit_gw3)
            chip_hold["free_hit"] = (
                f"Free Hit: your squad looks stronger this GW than in coming weeks — "
                f"consider saving for {worst_gw} when the deficit may be larger."
            )

    # ── Return best recommendation ────────────────────────────────────────────
    # Priority: free_hit > bench_boost > triple_captain (most squad-altering first)
    for chip in ("free_hit", "bench_boost", "triple_captain"):
        if chip in chip_use:
            return {"chip": chip, "reason": chip_use[chip]}

    # No chip recommended — compile all hold notes
    if chip_hold:
        hold_summary = " | ".join(chip_hold.values())
        return {"chip": None, "reason": hold_summary}

    return {
        "chip": None,
        "reason": (
            "No chip recommended this gameweek. Your squad is well set across the next 3 GWs — "
            "hold chips for double gameweeks, a blank for your rivals, or a standout captain fixture."
        ),
    }
