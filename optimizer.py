from __future__ import annotations

import pulp

from config import MAX_PER_TEAM, MIN_STARTING, SQUAD_COMPOSITION, SQUAD_SIZE, STARTING_XI


def optimize_squad(players: list[dict], budget: int = 1000) -> dict:
    n = len(players)
    prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)

    # Binary variables
    x = [pulp.LpVariable(f"squad_{i}", cat="Binary") for i in range(n)]
    y = [pulp.LpVariable(f"start_{i}", cat="Binary") for i in range(n)]

    # Objective: maximize predicted points of starters
    prob += pulp.lpSum(y[i] * players[i]["predicted_points"] for i in range(n))

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
) -> dict:
    id_to_player = {p["id"]: p for p in all_players}
    current_team = [id_to_player[pid] for pid in current_team_ids if pid in id_to_player]

    if len(current_team) != 15:
        raise ValueError(f"Expected 15 players in squad, got {len(current_team)}")

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
) -> dict:
    if not chips_available:
        return {"chip": None, "reason": "No chips available. Hold for a future gameweek."}

    n_beneficial = len(transfers)

    # Wildcard: 5+ beneficial transfers suggest a full rebuild
    if "wildcard" in chips_available and n_beneficial >= 5:
        return {
            "chip": "wildcard",
            "reason": f"{n_beneficial} beneficial transfers found. Use your Wildcard to overhaul your squad for free with no points hits.",
        }

    # Bench Boost: strong bench
    bench_pts = sum(p["predicted_points"] for p in xi_result["bench"])
    if "bench_boost" in chips_available and bench_pts >= 18:
        return {
            "chip": "bench_boost",
            "reason": f"Your bench is predicted to score {bench_pts:.1f} pts. Bench Boost would count all 15 players this gameweek.",
        }

    # Triple Captain: standout performer
    starters = xi_result["starters"]
    if starters:
        starters_sorted = sorted(starters, key=lambda p: p["predicted_points"], reverse=True)
        top = starters_sorted[0]
        avg = sum(p["predicted_points"] for p in starters) / len(starters)
        if "triple_captain" in chips_available and top["predicted_points"] >= 9 and top["predicted_points"] > avg * 1.7:
            return {
                "chip": "triple_captain",
                "reason": f"{top['name']} is predicted {top['predicted_points']:.1f} pts — well above team average. Triple Captain would triple his score.",
            }

    # Free Hit: team badly below theoretical best
    if "free_hit" in chips_available:
        current_pts = sum(p["predicted_points"] for p in current_team)
        top15_pts = sum(
            p["predicted_points"]
            for p in sorted(all_players, key=lambda x: x["predicted_points"], reverse=True)[:15]
        )
        if top15_pts > 0 and current_pts < top15_pts * 0.72:
            return {
                "chip": "free_hit",
                "reason": (
                    f"Your squad is predicted {current_pts:.0f} pts vs a theoretical best of {top15_pts:.0f} pts. "
                    "Free Hit lets you field any team this week with no long-term consequences."
                ),
            }

    return {
        "chip": None,
        "reason": "No chip recommended this gameweek — save them for double gameweeks, a strong bench, or a standout captain pick.",
    }
