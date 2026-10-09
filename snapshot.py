"""Per-gameweek archive of the live-only FPL data and of our own predictions.

Several inputs exist only "right now" in the FPL API — injury/availability flags, set-piece taker
orders, prices and ownership, FPL's own expected points (ep_next) — and so does the prediction
the app would have made. Archiving them each gameweek makes leak-free backtests and a true
out-of-sample "live accuracy" check possible later.

One gzipped JSON file per gameweek: archive/<season>/gwNN.json.gz, keyed by the UPCOMING
gameweek (the one FPL marks as next). Every save for that gameweek REPLACES the file, so the last
snapshot before its deadline wins; once the deadline passes FPL's "next" moves on and a save goes
to the following gameweek's file — an earlier gameweek's file is never touched again.

Saves happen automatically on every fresh FPL fetch (fpl_client.fetch_all_data) and on the
"Save snapshot now" button (which forces a fresh fetch). Snapshot files are left uncommitted
on purpose. Odds are NEVER pulled here — the snapshot records the odds cache state as-is
(only the Refresh Odds action spends Odds API quota).
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
from datetime import datetime, timezone

from config import W_ODDS_WEIGHT
from odds_client import get_odds_cache_meta
from predictions import predict_gameweeks
from set_piece_history import extract_orders

# Overridable (tests point it at a temp dir so they never touch the real archive).
ARCHIVE_DIR = os.environ.get("FPL_ARCHIVE_DIR", "archive")
SNAPSHOT_VERSION = 1


# ── Paths ────────────────────────────────────────────────────────────────────

def season_label(events: list[dict]) -> str:
    """'2026-27' from the first event's deadline (a season starts in Jul-Sep)."""
    first = min(events, key=lambda e: e["id"])
    d = datetime.fromisoformat(first["deadline_time"].replace("Z", "+00:00"))
    start = d.year if d.month >= 6 else d.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def snapshot_path(season: str, gw: int) -> str:
    return os.path.join(ARCHIVE_DIR, season, f"gw{gw:02d}.json.gz")


# ── Reading ──────────────────────────────────────────────────────────────────

def _read(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_snapshot(gw: int, season: str | None = None) -> dict | None:
    """The saved snapshot for `gw` (any season dir if `season` is None), or None."""
    if season:
        return _read(snapshot_path(season, gw))
    if not os.path.isdir(ARCHIVE_DIR):
        return None
    for s in sorted(os.listdir(ARCHIVE_DIR), reverse=True):
        snap = _read(snapshot_path(s, gw))
        if snap:
            return snap
    return None


def list_snapshots(season: str | None = None) -> dict[int, dict]:
    """{gameweek: snapshot} for every saved snapshot (newest season wins on a clash)."""
    out: dict[int, dict] = {}
    if not os.path.isdir(ARCHIVE_DIR):
        return out
    seasons = [season] if season else sorted(os.listdir(ARCHIVE_DIR))
    for s in seasons:
        d = os.path.join(ARCHIVE_DIR, s)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.startswith("gw") and fn.endswith(".json.gz"):
                snap = _read(os.path.join(d, fn))
                if snap:
                    out[int(snap["gameweek"])] = snap
    return out


# ── Building ─────────────────────────────────────────────────────────────────

def _git_info() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5)
        return {"commit": commit.stdout.strip() or None, "dirty": bool(dirty.stdout.strip())}
    except Exception:
        return {"commit": None, "dirty": None}


def _f(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _deadline(events: list[dict], gw: int) -> datetime | None:
    for e in events:
        if e["id"] == gw and e.get("deadline_time"):
            return datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00"))
    return None


def build_snapshot(data: dict, trigger: str = "auto", now: datetime | None = None) -> dict:
    """The snapshot dict for the upcoming gameweek in `data` (fetch_all_data's result)."""
    now = now or datetime.now(timezone.utc)
    gw = data["next_gw"]
    events = data["events"]
    deadline = _deadline(events, gw)
    upcoming = data["upcoming_gws"]
    preds = predict_gameweeks(data, W_ODDS_WEIGHT)
    pen_info = {p["id"]: p for p in data["players"]}

    odds_meta = get_odds_cache_meta()
    odds_present = bool(odds_meta) and odds_meta.get("gameweek") == gw

    players: dict[str, dict] = {}
    for el in data["elements"]:
        pid = el["id"]
        entry = {
            "name": el.get("web_name"), "team": el.get("team"), "pos": el.get("element_type"),
            "availability": {
                "status": el.get("status"), "news": el.get("news"), "news_added": el.get("news_added"),
                "chance_this_round": el.get("chance_of_playing_this_round"),
                "chance_next_round": el.get("chance_of_playing_next_round"),
            },
            "market": {
                "now_cost": el.get("now_cost"), "selected_by_percent": _f(el.get("selected_by_percent")),
                "transfers_in_event": el.get("transfers_in_event"), "transfers_out_event": el.get("transfers_out_event"),
                "cost_change_event": el.get("cost_change_event"),
            },
            "fpl": {"ep_next": _f(el.get("ep_next")), "ep_this": _f(el.get("ep_this")), "form": _f(el.get("form"))},
            "orders": {k: v for k, v in extract_orders(el).items() if v is not None},
        }
        if pid in preds:
            pl = pen_info.get(pid, {})
            entry["pred"] = {
                **preds[pid],
                "exp_minutes": pl.get("exp_minutes"), "p_60_plus": pl.get("p_60_plus"),
                "pen_share": round((pl.get("exp_minutes") or 0.0) * (pl.get("pen_taker_prob") or 0.0), 3)
                             if pl.get("pen_order") else None,
            }
        players[str(pid)] = entry

    fixtures = [
        {k: f.get(k) for k in ("id", "event", "kickoff_time", "team_h", "team_a",
                               "team_h_difficulty", "team_a_difficulty")}
        for f in data["fixtures"] if f.get("event") in upcoming
    ]
    minutes_to_deadline = round((deadline - now).total_seconds() / 60) if deadline else None
    return {
        "version": SNAPSHOT_VERSION,
        "season": season_label(events),
        "gameweek": gw,
        "upcoming_gws": upcoming,
        "saved_at": now.isoformat(),
        "trigger": trigger,
        "deadline_time": deadline.isoformat() if deadline else None,
        "minutes_to_deadline": minutes_to_deadline,
        "provenance": {
            "git": _git_info(),
            "odds_weight": W_ODDS_WEIGHT,
            # Odds are as-cached, never pulled here. present=False => predictions are model-only
            # (Tier 2/3) for this gameweek; a previous gameweek's odds are never used.
            "odds": {
                "present": odds_present,
                "gameweek": odds_meta.get("gameweek") if odds_meta else None,
                "fetched_at": odds_meta.get("fetched_at") if odds_meta and odds_present else None,
                "fixture_count": odds_meta.get("fixture_count") if odds_meta and odds_present else 0,
            },
        },
        "fixtures": fixtures,
        "players": players,
    }


# ── Saving ───────────────────────────────────────────────────────────────────

def save_snapshot(data: dict, trigger: str = "auto") -> dict:
    """Build and (atomically) write the upcoming gameweek's snapshot, replacing any earlier one
    for that gameweek. Returns the snapshot's metadata (everything except the bulky fields)."""
    snap = build_snapshot(data, trigger)
    path = snapshot_path(snap["season"], snap["gameweek"])
    previous = _read(path)
    snap["save_count"] = (previous.get("save_count", 1) + 1) if previous else 1
    snap["first_saved_at"] = previous.get("first_saved_at", previous["saved_at"]) if previous else snap["saved_at"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(snap, f, separators=(",", ":"))
    os.replace(tmp, path)
    return _meta(snap, path)


def _meta(snap: dict, path: str | None = None) -> dict:
    return {
        "gameweek": snap["gameweek"], "season": snap["season"], "saved_at": snap["saved_at"],
        "trigger": snap["trigger"], "deadline_time": snap["deadline_time"],
        "minutes_to_deadline_at_save": snap["minutes_to_deadline"],
        "save_count": snap.get("save_count", 1), "first_saved_at": snap.get("first_saved_at"),
        "odds_present": snap["provenance"]["odds"]["present"],
        "git": snap["provenance"]["git"], "path": path,
    }


def snapshot_status(data: dict, now: datetime | None = None) -> dict:
    """What the UI shows: the upcoming gameweek, its deadline, and the last saved snapshot."""
    now = now or datetime.now(timezone.utc)
    gw = data["next_gw"]
    deadline = _deadline(data["events"], gw)
    season = season_label(data["events"])
    path = snapshot_path(season, gw)
    snap = _read(path)
    return {
        "gameweek": gw, "season": season,
        "deadline_time": deadline.isoformat() if deadline else None,
        "minutes_to_deadline": round((deadline - now).total_seconds() / 60) if deadline else None,
        "saved": snap is not None,
        "snapshot": _meta(snap, path) if snap else None,
    }
