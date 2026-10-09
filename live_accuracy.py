"""Live accuracy: the predictions the app ACTUALLY made (archived in the per-gameweek snapshots,
before each deadline) scored against what players really scored — plus FPL's own ep_next from the
same snapshot as a benchmark. Unlike the backtests this is a true out-of-sample test: nothing is
reconstructed, so there is no look-ahead.

A gameweek is scored once every one of its fixtures has finished. Players are the ones the app
predicted for whose predicted points OR FPL's ep_next are positive — selected on predictions only,
never on what they went on to score.
"""
from __future__ import annotations

import math
from collections import defaultdict

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
TOP_NS = (10, 20)


def _ranks(values: list[float]) -> list[float]:
    """Average ranks (1 = smallest), ties share the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 3) if den else None


def _errors(pred: list[float], actual: list[float]) -> dict:
    n = len(pred)
    errs = [p - a for p, a in zip(pred, actual)]
    return {
        "mae": round(sum(abs(e) for e in errs) / n, 3),
        "rmse": round(math.sqrt(sum(e * e for e in errs) / n), 3),
        "bias": round(sum(errs) / n, 3),
    }


def _top_overlap(pred: list[float], actual: list[float], n: int) -> int:
    top_p = set(sorted(range(len(pred)), key=lambda i: -pred[i])[:n])
    top_a = set(sorted(range(len(actual)), key=lambda i: -actual[i])[:n])
    return len(top_p & top_a)


def _score_gameweek(rows: list[dict]) -> dict:
    ours = [r["ours"] for r in rows]
    fpl = [r["fpl"] for r in rows]
    act = [r["actual"] for r in rows]
    best_actual = max(act)

    def captain(key: str) -> dict:
        r = max(rows, key=lambda r: r[key])
        return {"name": r["name"], "predicted": round(r[key], 2), "actual": r["actual"]}

    by_pos = {}
    for pos in ("GKP", "DEF", "MID", "FWD"):
        sub = [r for r in rows if r["pos"] == pos]
        if sub:
            by_pos[pos] = {
                "n": len(sub),
                "ours": _errors([r["ours"] for r in sub], [r["actual"] for r in sub]),
                "fpl": _errors([r["fpl"] for r in sub], [r["actual"] for r in sub]),
            }
    return {
        "n": len(rows),
        "ours": _errors(ours, act), "fpl": _errors(fpl, act),
        "spearman": {"ours": _spearman(ours, act), "fpl": _spearman(fpl, act)},
        "top_overlap": {str(n): {"ours": _top_overlap(ours, act, n), "fpl": _top_overlap(fpl, act, n)} for n in TOP_NS},
        "captain": {"ours": captain("ours"), "fpl": captain("fpl"), "best_actual": best_actual},
        "by_position": by_pos,
    }


def compute_live_accuracy(
    raw_histories: dict[int, list[dict]], fixtures: list[dict], snapshots: dict[int, dict],
) -> dict:
    """Score every archived gameweek that has finished. `snapshots` = snapshot.list_snapshots()."""
    actual_by_gw: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for pid, hist in raw_histories.items():
        for h in hist:
            actual_by_gw[h["round"]][pid] += h["total_points"]

    results = []
    pooled: list[dict] = []
    for gw in sorted(snapshots):
        snap = snapshots[gw]
        gw_fix = [f for f in fixtures if f.get("event") == gw]
        entry = {
            "gameweek": gw, "saved_at": snap.get("saved_at"), "trigger": snap.get("trigger"),
            "minutes_to_deadline_at_save": snap.get("minutes_to_deadline"),
            "odds_present": snap["provenance"]["odds"]["present"],
            "git": snap["provenance"]["git"],
        }
        if not gw_fix or not all(f.get("finished") for f in gw_fix):
            done = sum(1 for f in gw_fix if f.get("finished"))
            entry.update({"status": "pending", "fixtures_finished": done, "fixtures_total": len(gw_fix)})
            results.append(entry)
            continue
        rows = []
        for pid_s, p in snap["players"].items():
            pred = p.get("pred")
            ep = p.get("fpl", {}).get("ep_next")
            if not pred or not pred.get("gw_pts") or ep is None:
                continue
            ours = pred["gw_pts"][0]
            if ours <= 0 and ep <= 0:
                continue
            rows.append({
                "name": p.get("name"), "pos": POSITIONS.get(p.get("pos"), "?"),
                "ours": ours, "fpl": ep, "actual": actual_by_gw[gw].get(int(pid_s), 0.0),
            })
        if len(rows) < 5:
            entry.update({"status": "pending", "note": "too few players to score"})
            results.append(entry)
            continue
        entry.update({"status": "scored", **_score_gameweek(rows)})
        results.append(entry)
        pooled.extend(rows)

    out = {"gameweeks": results, "scored_gameweeks": [r["gameweek"] for r in results if r["status"] == "scored"]}
    if pooled and len(out["scored_gameweeks"]) > 1:
        out["pooled"] = {"n": len(pooled), "ours": _errors([r["ours"] for r in pooled], [r["actual"] for r in pooled]),
                         "fpl": _errors([r["fpl"] for r in pooled], [r["actual"] for r in pooled])}
    return out
