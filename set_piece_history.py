"""Permanent per-gameweek archive of FPL's set-piece taker orders.

The bootstrap `elements` carry penalties_order, direct_freekicks_order and
corners_and_indirect_freekicks_order (1 = first-choice taker; None = not listed) — the
CURRENT values only, with no history. This archive records them each gameweek, the same
way odds_history.json does for bookmaker odds, so a backtest of gameweek G can later use
the orders that were actually in force before G instead of today's (look-ahead).

Format: {gw_str: {"fetched_at": iso, "players": {player_id_str: {"pen": n, "fk": n, "corner": n}}}}
— only players with at least one order are stored. A gameweek's entry is replaced by the
newest snapshot each time it's called for that same upcoming gameweek, so the last fetch
before the deadline wins.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

SET_PIECE_HISTORY_FILE = "set_piece_history.json"


def extract_orders(element: dict) -> dict[str, int | None]:
    """The three taker orders for one bootstrap element (None = not listed)."""
    return {
        "pen": element.get("penalties_order"),
        "fk": element.get("direct_freekicks_order"),
        "corner": element.get("corners_and_indirect_freekicks_order"),
    }


def archive_set_piece_snapshot(gw_id: int, elements: list[dict]) -> None:
    """Record `gw_id`'s taker orders in the permanent archive (replaces that gameweek's entry)."""
    history = _read()
    players = {}
    for el in elements:
        orders = extract_orders(el)
        if any(v is not None for v in orders.values()):
            players[str(el["id"])] = {k: v for k, v in orders.items() if v is not None}
    history[str(gw_id)] = {"fetched_at": datetime.utcnow().isoformat(), "players": players}
    tmp = SET_PIECE_HISTORY_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f, indent=1, sort_keys=True)
    os.replace(tmp, SET_PIECE_HISTORY_FILE)


def load_set_piece_history() -> dict[int, dict[int, dict[str, int]]]:
    """{gw: {player_id: {"pen": n, "fk": n, "corner": n}}} — gameweeks never archived are absent."""
    return {
        int(gw): {int(pid): orders for pid, orders in entry.get("players", {}).items()}
        for gw, entry in _read().items()
    }


def _read() -> dict:
    if not os.path.exists(SET_PIECE_HISTORY_FILE):
        return {}
    try:
        with open(SET_PIECE_HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return {}
