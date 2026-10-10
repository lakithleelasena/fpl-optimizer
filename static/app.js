const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let allPlayers = [];
let currentSort = { key: "predicted_points", asc: false };
let posFilter = "ALL";

// My Squad state
const slotLimits = { GKP: 2, DEF: 5, MID: 5, FWD: 3 };
let mySquad = { GKP: [], DEF: [], MID: [], FWD: [] };

// ─── Init ────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    // Weight slider sync
    ["form-factor", "cs-factor", "atk-factor"].forEach((id) => {
        const slider = $(`#${id}`);
        const display = $(`#${id}-val`);
        slider.addEventListener("input", () => { display.textContent = parseFloat(slider.value).toFixed(1); });
    });
    const oddsWeightSlider = $("#odds-weight");
    const oddsWeightVal = $("#odds-weight-val");
    oddsWeightSlider.addEventListener("input", () => {
        oddsWeightVal.textContent = `${Math.round(parseFloat(oddsWeightSlider.value) * 100)}%`;
    });

    // One optimisation horizon shared by the Squad Optimizer and Transfer Advice (persisted).
    const horizonSelects = [$("#n-gw"), $("#n-gw-transfer")];
    let savedHorizon = null;
    try { savedHorizon = localStorage.getItem("fpl_horizon"); } catch (e) { /* ignore */ }
    horizonSelects.forEach((sel) => { if (savedHorizon) sel.value = savedHorizon; });
    horizonSelects.forEach((sel) => sel.addEventListener("change", () => {
        horizonSelects.forEach((other) => { other.value = sel.value; });
        try { localStorage.setItem("fpl_horizon", sel.value); } catch (e) { /* ignore */ }
    }));
    horizonSelects[1].value = horizonSelects[0].value;  // the two always start in step

    // Budget auto-fills with squad value + bank until the user types their own figure.
    $("#budget").addEventListener("input", (ev) => {
        if (ev.isTrusted) { _budgetManual = true; updateBudgetFromSquad(); }
    });
    $("#bank").addEventListener("input", updateBudgetFromSquad);
    $("#budget-hint").addEventListener("click", (ev) => {
        if (ev.target.dataset.reset) { _budgetManual = false; updateBudgetFromSquad(); }
    });

    $("#btn-optimize").addEventListener("click", runOptimize);
    $("#search").addEventListener("input", renderTable);
    $("#btn-transfer-advice").addEventListener("click", runTransferAdvice);
    $("#btn-team-xg-backtest").addEventListener("click", runTeamXgBacktest);
    $("#btn-player-points-backtest").addEventListener("click", runPlayerPointsBacktest);
    $("#btn-live-accuracy").addEventListener("click", runLiveAccuracy);

    // Position filter buttons
    $$(".filter-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            $$(".filter-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            posFilter = btn.dataset.pos;
            renderTable();
        });
    });

    // Tabs
    $$(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            $$(".tab-btn").forEach((b) => b.classList.remove("active"));
            $$(".tab-content").forEach((c) => c.classList.add("hidden"));
            btn.classList.add("active");
            $(`#tab-${btn.dataset.tab}`).classList.remove("hidden");
            if (btn.dataset.tab === "teams" || btn.dataset.tab === "fixtures") {
                loadTeams();
            }
        });
    });

    // Squad search
    const squadSearch = $("#squad-search");
    squadSearch.addEventListener("input", onSquadSearch);
    squadSearch.addEventListener("focus", onSquadSearch);
    document.addEventListener("click", (e) => {
        if (!e.target.closest(".squad-search-wrap")) {
            $("#squad-search-results").innerHTML = "";
        }
    });

    loadSavedSquad();
    loadPlayers();
    loadNextGw();
    loadArchiveStatus();
    setInterval(renderArchiveBar, 60000);  // keep the countdown fresh without hitting the server
});

async function loadNextGw() {
    try {
        const resp = await fetch("/api/next-gw");
        const data = await resp.json();
        $("#next-gw-badge").textContent = `Next Gameweek: GW${data.next_gw}`;
    } catch (e) {
        $("#next-gw-badge").textContent = "";
    }
}

// ─── Snapshot archive (header bar) ───────────────────────────────────────────

let _archiveStatus = null;

async function loadArchiveStatus() {
    try {
        const resp = await fetch("/api/snapshot/status");
        _archiveStatus = await resp.json();
    } catch (e) {
        _archiveStatus = null;
    }
    renderArchiveBar();
}

async function saveSnapshotNow() {
    const btn = document.getElementById("btn-snapshot");
    if (btn) { btn.disabled = true; btn.textContent = "Saving…"; }
    try {
        const resp = await fetch("/api/snapshot", { method: "POST" });
        if (!resp.ok) throw new Error(`Error ${resp.status}`);
        _archiveStatus = await resp.json();
    } catch (e) {
        alert("Snapshot failed: " + e.message);
    }
    renderArchiveBar();
}

function _fmtDuration(mins) {
    const m = Math.max(0, Math.round(mins));
    const d = Math.floor(m / 1440), h = Math.floor((m % 1440) / 60), mm = m % 60;
    return (d ? `${d}d ` : "") + (h || d ? `${h}h ` : "") + `${mm}m`;
}

function _fmtLocal(iso) {
    return new Date(iso).toLocaleString([], {
        weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short",
    });
}

function renderArchiveBar() {
    const el = document.getElementById("archive-bar");
    if (!el) return;
    const st = _archiveStatus;
    if (!st) { el.innerHTML = ""; return; }
    const now = Date.now();
    const toDeadline = st.deadline_time ? (new Date(st.deadline_time).getTime() - now) / 60000 : null;
    const snap = st.snapshot;
    let html = `<span>Archive GW${st.gameweek}`;
    if (st.deadline_time) html += ` · deadline ${_fmtLocal(st.deadline_time)}` + (toDeadline > 0 ? ` (in ${_fmtDuration(toDeadline)})` : " (passed — refresh the page)");
    html += `</span>`;
    if (!snap) {
        html += `<span class="warn">no snapshot saved yet</span>`;
    } else {
        const age = (now - new Date(snap.saved_at).getTime()) / 60000;
        const before = st.deadline_time ? (new Date(st.deadline_time).getTime() - new Date(snap.saved_at).getTime()) / 60000 : null;
        const stale = toDeadline != null && toDeadline > 0 && toDeadline <= 360 && age > 60;
        html += `<span class="${stale ? "warn" : "ok"}">snapshot saved ${_fmtLocal(snap.saved_at)}` +
            (before != null ? ` (${_fmtDuration(before)} before deadline)` : "") +
            ` · ${snap.save_count} save${snap.save_count === 1 ? "" : "s"} · ` +
            (snap.odds_present ? "odds included" : "NO odds this GW — model only") + `</span>`;
        if (stale) html += `<span class="warn">deadline soon — save again</span>`;
    }
    html += `<button id="btn-snapshot" class="btn-snapshot" onclick="saveSnapshotNow()" title="Fresh FPL pull + replace this gameweek's snapshot. Never calls the Odds API.">Save snapshot now</button>`;
    el.innerHTML = html;
}

// ─── Players ─────────────────────────────────────────────────────────────────

async function loadPlayers() {
    $("#table-body").innerHTML = '<tr><td colspan="15" class="loading"><span class="spinner"></span>Loading players...</td></tr>';
    try {
        const resp = await fetch("/api/players");
        allPlayers = await resp.json();
        renderTable();
        renderSquadBuilder();  // also refreshes the auto budget with current prices
    } catch (e) {
        $("#table-body").innerHTML = `<tr><td colspan="15" class="loading">Failed to load: ${e.message}</td></tr>`;
    }
}

// ─── Squad Optimizer tab ──────────────────────────────────────────────────────

async function runOptimize() {
    const btn = $("#btn-optimize");
    btn.disabled = true;
    btn.textContent = "Optimizing...";
    $("#pitch-starters").innerHTML = '<div class="loading"><span class="spinner"></span>Finding optimal squad...</div>';
    $("#pitch-bench").innerHTML = "";

    const body = {
        budget: parseInt($("#budget").value) || 1000,
        n_gw: parseInt($("#n-gw").value) || 3,
        form_factor: parseFloat($("#form-factor").value),
        cs_factor: parseFloat($("#cs-factor").value),
        atk_factor: parseFloat($("#atk-factor").value),
        odds_weight: parseFloat($("#odds-weight").value),
    };

    try {
        const resp = await fetch("/api/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Server error");
        }
        const data = await resp.json();
        renderSquad(data, data.captain_id, data.vice_captain_id);

        // Refresh player table with the same factors so scores match pitch cards
        try {
            const playersResp = await fetch(
                `/api/players?form_factor=${body.form_factor}&cs_factor=${body.cs_factor}&atk_factor=${body.atk_factor}&odds_weight=${body.odds_weight}`
            );
            if (!playersResp.ok) throw new Error(`HTTP ${playersResp.status}`);
            allPlayers = await playersResp.json();
        } catch (tableErr) {
            console.warn("Could not refresh player table weights:", tableErr);
        }
        renderTable();
    } catch (e) {
        $("#pitch-starters").innerHTML = `<div class="loading">Optimization failed: ${e.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = "Optimize Squad";
    }
}

function renderSquad(data, captainId = null, viceCaptainId = null) {
    const nGw = data.n_gw || 1;
    const upcomingGws = data.upcoming_gws || [];

    // Rebuild summary cards dynamically based on n_gw
    const summaryEl = $("#optimizer-summary");
    // Remove any previously added GW cards (keep Cost and Squad Size)
    summaryEl.querySelectorAll(".summary-card-gw").forEach((el) => el.remove());

    // Insert per-GW cards after Cost card (track insertion point to preserve order)
    const costCard = summaryEl.querySelector(".summary-card");
    let insertAfter = costCard;
    upcomingGws.forEach((gw, i) => {
        const pts = data.gw_totals && data.gw_totals[i] != null ? data.gw_totals[i].toFixed(1) : "-";
        const card = document.createElement("div");
        card.className = "summary-card summary-card-gw";
        card.innerHTML = `<div class="value">${pts}</div><div class="label">GW${gw} XI Pts</div>`;
        insertAfter.insertAdjacentElement("afterend", card);
        insertAfter = card;
    });

    $("#summary-cost").textContent = `£${data.total_cost.toFixed(1)}m`;
    $("#summary-count").textContent = `${data.starters.length + data.bench.length}`;

    const groups = { GKP: [], DEF: [], MID: [], FWD: [] };
    data.starters.forEach((p) => groups[p.position].push(p));

    let html = "";
    for (const [pos, players] of Object.entries(groups)) {
        if (players.length === 0) continue;
        html += `<div class="position-row"><div class="position-row-label">${pos}</div>`;
        html += players.map((p) => cardHTML(p, p.id === captainId, p.id === viceCaptainId, false, nGw, upcomingGws)).join("");
        html += `</div>`;
    }
    $("#pitch-starters").innerHTML = html;
    $("#pitch-bench").innerHTML = data.bench.map((p) => cardHTML(p, false, false, false, nGw, upcomingGws)).join("");
}

// ─── My Team & Transfers tab ──────────────────────────────────────────────────

function squadCount() {
    return Object.values(mySquad).reduce((s, arr) => s + arr.length, 0);
}

function isInSquad(playerId) {
    return Object.values(mySquad).some((arr) => arr.some((p) => p.id === playerId));
}

function saveSquad() {
    localStorage.setItem("fpl_my_squad", JSON.stringify(mySquad));
}

function loadSavedSquad() {
    try {
        const saved = localStorage.getItem("fpl_my_squad");
        if (saved) mySquad = JSON.parse(saved);
    } catch (e) {
        // ignore corrupt data
    }
}

function clearSavedSquad() {
    localStorage.removeItem("fpl_my_squad");
    mySquad = { GKP: [], DEF: [], MID: [], FWD: [] };
    renderSquadBuilder();
}

function addPlayerToSquad(player) {
    const pos = player.position;
    if (mySquad[pos].length >= slotLimits[pos]) {
        alert(`You already have ${slotLimits[pos]} ${pos} players in your squad.`);
        return;
    }
    if (isInSquad(player.id)) {
        alert(`${player.name} is already in your squad.`);
        return;
    }
    mySquad[pos].push(player);
    saveSquad();
    renderSquadBuilder();
    $("#squad-search").value = "";
    $("#squad-search-results").innerHTML = "";
}

function removePlayerFromSquad(playerId) {
    for (const pos of Object.keys(mySquad)) {
        mySquad[pos] = mySquad[pos].filter((p) => p.id !== playerId);
    }
    saveSquad();
    renderSquadBuilder();
}

// Squad value at CURRENT prices (not FPL selling prices — we don't have purchase prices) + bank,
// in tenths — what a Wildcard/Free Hit can actually spend, and what Transfer Advice assumes.
// Pre-fills the Squad Optimizer budget once 15 players are picked, unless the user typed their own.
let _budgetManual = false;

function squadValueTenths() {
    const current = new Map(allPlayers.map((p) => [p.id, p.cost]));
    const squad = Object.values(mySquad).flat();
    const players = squad.reduce((sum, p) => sum + Math.round((current.get(p.id) ?? p.cost) * 10), 0);
    const bank = Math.round(parseFloat($("#bank").value || "0") * 10);
    return players + bank;
}

function updateBudgetFromSquad() {
    const input = $("#budget"), hint = $("#budget-hint");
    if (!input || !hint) return;
    if (squadCount() !== 15) {
        hint.textContent = "Add 15 players on My Team to fill this with your squad value + bank.";
        return;
    }
    const value = squadValueTenths();
    const label = `£${(value / 10).toFixed(1)}m`;
    if (_budgetManual) {
        hint.innerHTML = `Manual budget. Squad value + bank is ${label} — <a data-reset="1">use that</a>`;
    } else {
        input.value = value;
        hint.textContent = `Auto: squad value at current prices + bank = ${label}`;
    }
}

function renderSquadBuilder() {
    updateBudgetFromSquad();
    const total = squadCount();
    $("#squad-total-count").textContent = total;

    for (const [pos, limit] of Object.entries(slotLimits)) {
        const players = mySquad[pos];
        $(`#count-${pos}`).textContent = `${players.length}/${limit}`;

        const slotsEl = $(`#slots-${pos}`);
        let html = players.map((p) => `
            <div class="squad-slot filled">
                <span class="slot-name">${p.name}${penBadge(p)}</span>
                <span class="slot-team">${p.team} · £${p.cost.toFixed(1)}m</span>
                <button class="slot-remove" onclick="removePlayerFromSquad(${p.id})" title="Remove">✕</button>
            </div>
        `).join("");

        // Empty slots
        for (let i = players.length; i < limit; i++) {
            html += `<div class="squad-slot empty"><span class="slot-empty-label">Empty slot</span></div>`;
        }
        slotsEl.innerHTML = html;
    }
}

function onSquadSearch() {
    const query = $("#squad-search").value.trim().toLowerCase();
    const resultsEl = $("#squad-search-results");

    if (query.length < 2) {
        resultsEl.innerHTML = "";
        return;
    }

    const matches = allPlayers
        .filter((p) => p.name.toLowerCase().includes(query) || p.team.toLowerCase().includes(query))
        .slice(0, 12);

    if (matches.length === 0) {
        resultsEl.innerHTML = `<div class="search-no-results">No players found</div>`;
        return;
    }

    resultsEl.innerHTML = matches.map((p) => {
        const inSquad = isInSquad(p.id);
        const full = mySquad[p.position].length >= slotLimits[p.position];
        const disabled = inSquad || full;
        const note = inSquad ? " (in squad)" : full ? " (position full)" : "";
        return `
            <div class="search-result-item ${disabled ? "disabled" : ""}"
                 onclick="${disabled ? "" : `addPlayerToSquad(${JSON.stringify(p).replace(/"/g, "&quot;")})`}">
                <span class="sr-name">${p.name}${penBadge(p)}${note}</span>
                <span class="sr-meta">${p.team} · ${p.position} · £${p.cost.toFixed(1)}m · ${p.predicted_points.toFixed(1)}pts</span>
            </div>`;
    }).join("");
}

// ─── Transfer Advice ─────────────────────────────────────────────────────────

async function runTransferAdvice() {
    if (squadCount() !== 15) {
        alert(`Please add exactly 15 players to your squad. You currently have ${squadCount()}.`);
        return;
    }

    const btn = $("#btn-transfer-advice");
    btn.disabled = true;
    btn.textContent = "Analysing...";
    $("#transfer-results").classList.add("hidden");

    const currentTeamIds = Object.values(mySquad).flatMap((arr) => arr.map((p) => p.id));
    const chips = Array.from($$(".chip-check input:checked")).map((el) => el.value);
    const bankValue = Math.round(parseFloat($("#bank").value || "0") * 10);

    const body = {
        current_team: currentTeamIds,
        free_transfers: parseInt($("#free-transfers").value),
        budget_in_bank: bankValue,
        chips_available: chips,
        n_gw: parseInt($("#n-gw-transfer").value) || 3,
        form_factor: parseFloat($("#form-factor").value),
        cs_factor: parseFloat($("#cs-factor").value),
        atk_factor: parseFloat($("#atk-factor").value),
        odds_weight: parseFloat($("#odds-weight").value),
    };

    try {
        const resp = await fetch("/api/transfer-advice", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Server error");
        }
        const data = await resp.json();
        renderTransferAdvice(data);
    } catch (e) {
        alert(`Failed to get transfer advice: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = "Get Transfer Advice";
    }
}

function renderTransferAdvice(data) {
    const nGw = data.n_gw || 3;

    // Summary
    $("#ts-transfers").textContent = data.transfers.length;
    const hitsEl = $("#ts-hits");
    hitsEl.textContent = data.hits_required;
    hitsEl.style.color = data.hits_required > 0 ? "#ff6b6b" : "#00ff87";

    const gainEl = $("#ts-net-gain");
    gainEl.textContent = `+${data.net_points_gain.toFixed(1)}`;
    gainEl.style.color = data.net_points_gain >= 0 ? "#00ff87" : "#ff6b6b";

    $("#ts-total-3gw").textContent = data.total_predicted_3gw.toFixed(1);
    const summaryLabel = $("#ts-3gw-label");
    if (summaryLabel) summaryLabel.textContent = `${nGw}GW Predicted Pts (XI)`;

    // Chip recommendation
    renderChipRec(data.chip_recommendation);

    // Transfer suggestions
    renderTransfers(data.transfers, data.free_transfers, nGw);

    // Recommended XI (pitch)
    renderAdvicePitch(data);

    $("#transfer-results").classList.remove("hidden");
    $("#transfer-results").scrollIntoView({ behavior: "smooth" });
}

function renderChipRec(rec) {
    const chipNames = {
        wildcard: "Wildcard",
        free_hit: "Free Hit",
        bench_boost: "Bench Boost",
        triple_captain: "Triple Captain",
    };
    const chipIcons = {
        wildcard: "🃏",
        free_hit: "🔄",
        bench_boost: "📈",
        triple_captain: "3️⃣",
    };

    let html;
    if (rec.chip) {
        html = `
            <div class="chip-rec active-chip">
                <div class="chip-badge">${chipNames[rec.chip] || rec.chip}</div>
                <p class="chip-reason">${rec.reason}</p>
            </div>`;
    } else {
        html = `<div class="chip-rec no-chip"><p class="chip-reason">${rec.reason}</p></div>`;
    }
    $("#chip-rec-content").innerHTML = html;
}

function renderTransfers(transfers, freeTransfers, nGw = 3) {
    if (transfers.length === 0) {
        $("#transfers-content").innerHTML = `<p class="no-transfers">No beneficial transfers found this gameweek. Hold your transfers.</p>`;
        return;
    }

    const gwLabel = `${nGw}GW pts`;
    let freeLeft = freeTransfers;
    const html = transfers.map((t) => {
        const isHit = freeLeft <= 0;
        freeLeft = Math.max(0, freeLeft - 1);
        const hitLabel = isHit ? `<span class="hit-badge">-4 pts hit</span>` : `<span class="free-badge">Free</span>`;
        return `
            <div class="transfer-card">
                ${hitLabel}
                <div class="transfer-row">
                    <div class="transfer-out">
                        <div class="t-label">OUT</div>
                        <div class="t-name">${t.transfer_out.name}</div>
                        <div class="t-meta">${t.transfer_out.team} · £${t.transfer_out.cost.toFixed(1)}m · ${t.transfer_out.predicted_points.toFixed(1)} ${gwLabel}</div>
                    </div>
                    <div class="transfer-arrow">→</div>
                    <div class="transfer-in">
                        <div class="t-label">IN</div>
                        <div class="t-name">${t.transfer_in.name}</div>
                        <div class="t-meta">${t.transfer_in.team} · £${t.transfer_in.cost.toFixed(1)}m · ${t.transfer_in.predicted_points.toFixed(1)} ${gwLabel}</div>
                    </div>
                    <div class="transfer-gain">
                        <div class="t-gain-val">+${t.points_gain.toFixed(1)}</div>
                        <div class="t-gain-label">pts gain</div>
                    </div>
                </div>
            </div>`;
    }).join("");

    $("#transfers-content").innerHTML = html;
}

function renderAdvicePitch(data) {
    const nGw = data.n_gw || 1;
    const gwNums = data.upcoming_gws || [];

    const groups = { GKP: [], DEF: [], MID: [], FWD: [] };
    data.starters.forEach((p) => groups[p.position].push(p));

    let html = "";
    for (const [pos, players] of Object.entries(groups)) {
        if (players.length === 0) continue;
        html += `<div class="position-row"><div class="position-row-label">${pos}</div>`;
        html += players.map((p) => cardHTML(p, p.id === data.captain_id, p.id === data.vice_captain_id, false, nGw, gwNums)).join("");
        html += `</div>`;
    }
    $("#advice-pitch-starters").innerHTML = html;
    $("#advice-pitch-bench").innerHTML = data.bench.map((p) => cardHTML(p, false, false, false, nGw, gwNums)).join("");

    // Compute and display XI totals
    const starters = data.starters || [];
    // Next GW total: always uses gw_pts[0] (first upcoming GW)
    const totalNextGw = starters.reduce((sum, p) => sum + ((p.gw_pts && p.gw_pts[0]) || 0), 0);
    // nGW total: sum of predicted_points (which equals sum(gw_pts[:n_gw]))
    const totalNGw = starters.reduce((sum, p) => sum + (p.predicted_points || 0), 0);
    // Captain doubles their next GW points
    const captain = starters.find((p) => p.id === data.captain_id);
    const captainBonus = captain ? ((captain.gw_pts && captain.gw_pts[0]) || 0) : 0;

    const totalsEl = $("#advice-xi-totals");
    if (totalsEl) {
        totalsEl.style.display = "flex";
        $("#advice-total-next-gw").textContent = `${(totalNextGw + captainBonus).toFixed(1)} pts`;
        $("#advice-total-3gw").textContent = `${totalNGw.toFixed(1)} pts`;
        const nGwLabel = $("#advice-ngw-label");
        if (nGwLabel) nGwLabel.textContent = `${nGw} GW Total`;
    }
}

// ─── Shared rendering helpers ─────────────────────────────────────────────────

function startColor(likelihood) {
    if (likelihood >= 0.8) return "#00ff87";
    if (likelihood >= 0.5) return "#f5a623";
    return "#ff6b6b";
}

// fixture_ease: 0.0 = hardest, 1.0 = easiest
function fixtureColor(ease) {
    if (ease >= 0.7) return "#00ff87";   // easy
    if (ease >= 0.4) return "#f5a623";   // medium
    return "#ff6b6b";                    // hard
}

function fixtureLabel(ease) {
    if (ease >= 0.7) return "Easy";
    if (ease >= 0.4) return "Med";
    return "Hard";
}

function cardHTML(p, isCaptain = false, isViceCaptain = false, show3gw = false, numGw = 1, gwNums = []) {
    const slPct = Math.round(p.start_likelihood * 100);
    const slColor = startColor(p.start_likelihood);
    const emPct = Math.round((p.exp_minutes ?? 0) * 100);
    const emColor = startColor(p.exp_minutes ?? 0);
    const badge = isCaptain
        ? `<div class="captain-badge">C</div>`
        : isViceCaptain
        ? `<div class="captain-badge vc-badge">V</div>`
        : "";

    // Determine display pts and label
    let displayPts, ptsLabel;
    if (show3gw) {
        displayPts = p.predicted_points;
        ptsLabel = "3GW";
    } else if (numGw > 1 && p.gw_pts) {
        displayPts = p.gw_pts.slice(0, numGw).reduce((s, v) => s + v, 0);
        ptsLabel = `${numGw}GW`;
    } else {
        displayPts = (p.gw_pts && p.gw_pts.length > 0) ? p.gw_pts[0] : p.predicted_points;
        ptsLabel = "GW";
    }

    // Per-GW breakdown row (shown when numGw > 1 or show3gw)
    let gwBreakdown = "";
    if ((numGw > 1 || show3gw) && p.gw_pts && p.gw_pts.length > 0) {
        const count = show3gw ? p.gw_pts.length : numGw;
        const parts = Array.from({ length: count }, (_, i) => {
            const label = gwNums[i] ? `GW${gwNums[i]}` : `GW${i + 1}`;
            const val = p.gw_pts[i] != null ? p.gw_pts[i].toFixed(1) : "0.0";
            return `<span>${label}:${val}</span>`;
        });
        gwBreakdown = `<div class="gw-breakdown">${parts.join("")}</div>`;
    }
    return `
        <div class="player-card pos-${p.position}" style="position:relative">
            ${badge}
            <div class="player-name">${p.name}${penBadge(p)}</div>
            <div class="player-team">${p.team} · ${p.position}</div>
            <div class="player-pts">${displayPts.toFixed(1)}<span class="pts-label">${ptsLabel}</span></div>
            ${gwBreakdown}
            <div class="player-cost">£${p.cost.toFixed(1)}m</div>
            <div class="start-likelihood">
                <span style="color:${slColor}">${slPct}% start</span>
                <span style="color:${emColor}">${emPct}% min</span>
            </div>
            <div class="breakdown">
                <span>S:${p.season_avg.toFixed(1)}</span>
                <span>F:${p.form_score.toFixed(1)}</span>
                <span>xG:${p.xg_score.toFixed(1)}</span>
                <span style="color:${fixtureColor(p.fixture_ease)}">FD:${fixtureLabel(p.fixture_ease)}</span>
            </div>
            ${p.ep_next > 0 ? `<div class="ep-next-label">FPL: ${p.ep_next.toFixed(1)} ep</div>` : ''}
        </div>`;
}

// ─── Players table ────────────────────────────────────────────────────────────

function renderTable() {
    const search = ($("#search").value || "").toLowerCase();
    let filtered = allPlayers.filter((p) => {
        if (posFilter !== "ALL" && p.position !== posFilter) return false;
        if (search && !p.name.toLowerCase().includes(search) && !p.team.toLowerCase().includes(search)) return false;
        return true;
    });

    filtered.sort((a, b) => {
        const va = a[currentSort.key];
        const vb = b[currentSort.key];
        if (typeof va === "string") return currentSort.asc ? va.localeCompare(vb) : vb.localeCompare(va);
        return currentSort.asc ? va - vb : vb - va;
    });

    $("#table-body").innerHTML = filtered
        .slice(0, 200)
        .map((p) => {
            const slPct = Math.round(p.start_likelihood * 100);
            const slColor = startColor(p.start_likelihood);
            const inSquad = isInSquad(p.id);
            const full = mySquad[p.position].length >= slotLimits[p.position];
            const addDisabled = inSquad || full;
            const addLabel = inSquad ? "Added" : full ? "Full" : "+ Add";
            return `
        <tr>
            <td>${p.name}${penBadge(p)}</td>
            <td>${p.team}</td>
            <td>${p.position}</td>
            <td>£${p.cost.toFixed(1)}m</td>
            <td style="color:#00ff87;font-weight:600">${p.predicted_points.toFixed(1)}</td>
            <td style="color:#58a6ff;font-weight:600">${p.predicted_points_4gw != null ? p.predicted_points_4gw.toFixed(1) : "-"}</td>
            <td style="color:${slColor}">${slPct}%</td>
            <td style="color:${startColor(p.exp_minutes)}">${Math.round(p.exp_minutes * 100)}%</td>
            <td style="color:${fixtureColor(p.fixture_ease)}">${fixtureLabel(p.fixture_ease)}</td>
            <td style="color:#a78bfa;font-weight:600">${p.ep_next != null ? p.ep_next.toFixed(1) : '-'}</td>
            <td>${p.season_avg.toFixed(1)}</td>
            <td>${p.total_points}</td>
            <td>${p.form_score.toFixed(1)}</td>
            <td>${p.xg_score.toFixed(1)}</td>
            <td>
                <button class="add-btn ${addDisabled ? "add-btn-disabled" : ""}"
                    ${addDisabled ? "disabled" : `onclick='addPlayerToSquad(${JSON.stringify(p)})'`}>
                    ${addLabel}
                </button>
            </td>
        </tr>`;
        })
        .join("");
}

function sortTable(key) {
    if (currentSort.key === key) {
        currentSort.asc = !currentSort.asc;
    } else {
        currentSort = { key, asc: false };
    }
    renderTable();
}

// ─── Prediction Accuracy: shared top/bottom-by-gameweek grouping ──────────────

// Splits rows into per-gameweek groups of the N biggest positive errors (over-predicted)
// and N biggest negative errors (under-predicted), skipping the middle. Caps each side
// at half the gameweek's rows so a sparse gameweek never shows the same row twice.
function groupTopBottomByGw(rows, gws, n = 5) {
    const gwsDesc = [...gws].sort((a, b) => b - a);
    return gwsDesc.map(gw => {
        const sorted = rows.filter(r => r.gw === gw).sort((a, b) => b.error - a.error);
        const total = sorted.length;
        const topCount = Math.min(n, Math.ceil(total / 2));
        const botCount = Math.min(n, total - topCount);
        return {
            gw,
            top: sorted.slice(0, topCount),
            bottom: sorted.slice(total - botCount).reverse(),
        };
    });
}

function groupHeaderRow(label, colspan) {
    return `<tr><td colspan="${colspan}" style="background:#1c2128;font-weight:700;color:#8b949e;padding:8px 12px">${label}</td></tr>`;
}

// Penalty-taker badge: P1 = FPL's first-choice taker. Dimmed when he's unlikely to take the next
// one (injured / rotation risk) — pen_share already folds in availability and the order chain.
function penBadge(p) {
    if (!p.pen_order) return "";
    const share = p.pen_share == null ? null : Math.round(p.pen_share * 100);
    const dim = share != null && share < 10;
    const tip = `Penalty taker #${p.pen_order} (FPL order)` +
        (share != null ? ` — about ${share}% chance he takes his team's next penalty given current availability` : "");
    return `<span class="pen-badge${dim ? " dim" : ""}" title="${tip}">P${p.pen_order}</span>`;
}

// ─── Prediction Accuracy: live accuracy (archived predictions) ───────────────

async function runLiveAccuracy() {
    const btn = $("#btn-live-accuracy");
    btn.disabled = true;
    btn.textContent = "Loading…";
    try {
        const resp = await fetch("/api/backtest/live-accuracy");
        if (!resp.ok) throw new Error(`Error ${resp.status}`);
        renderLiveAccuracy(await resp.json());
    } catch (e) {
        $("#live-accuracy-results").innerHTML = `<p style="color:#ff6b6b">Failed to load: ${e.message}</p>`;
    } finally {
        btn.disabled = false;
        btn.textContent = "Load Live Accuracy";
    }
}

function renderLiveAccuracy(data) {
    const el = $("#live-accuracy-results");
    if (!data.gameweeks.length) {
        el.innerHTML = `<p class="section-hint" style="margin-top:12px">No snapshots saved yet.</p>`;
        return;
    }
    const f = (v, d = 2) => (v == null ? "–" : v.toFixed(d));
    const sgn = v => (v > 0 ? "+" : "") + v.toFixed(2);
    const better = (a, b) => (a < b ? "#00ff87" : "#e0e0e0");
    const scored = data.gameweeks.filter(g => g.status === "scored");
    const pending = data.gameweeks.filter(g => g.status !== "scored");

    let html = "";
    if (pending.length) {
        html += `<p class="section-hint" style="margin-top:12px">` + pending.map(g =>
            `GW${g.gameweek}: snapshot saved` +
            (g.fixtures_total != null ? ` — ${g.fixtures_finished}/${g.fixtures_total} fixtures finished, scored once all are done` : ` — ${g.note || "waiting"}`) +
            (g.odds_present ? "" : " (saved without odds)")).join("<br>") + `</p>`;
    }
    if (scored.length) {
        const rows = scored.map(g => `
            <tr>
                <td>GW${g.gameweek}</td><td>${g.n}</td>
                <td style="color:${better(g.ours.mae, g.fpl.mae)};font-weight:600">${f(g.ours.mae)}</td><td>${f(g.fpl.mae)}</td>
                <td style="color:${better(g.ours.rmse, g.fpl.rmse)};font-weight:600">${f(g.ours.rmse)}</td><td>${f(g.fpl.rmse)}</td>
                <td>${sgn(g.ours.bias)}</td><td>${sgn(g.fpl.bias)}</td>
                <td style="color:${g.spearman.ours > g.spearman.fpl ? "#00ff87" : "#e0e0e0"};font-weight:600">${f(g.spearman.ours)}</td><td>${f(g.spearman.fpl)}</td>
                <td>${g.top_overlap["10"].ours} / ${g.top_overlap["10"].fpl}</td>
                <td>${g.top_overlap["20"].ours} / ${g.top_overlap["20"].fpl}</td>
                <td>${g.captain.ours.name} <b>${g.captain.ours.actual}</b></td>
                <td>${g.captain.fpl.name} <b>${g.captain.fpl.actual}</b></td>
                <td>${g.captain.best_actual}</td>
            </tr>`).join("");
        const pooled = data.pooled
            ? `<tr style="border-top:2px solid #30363d"><td>All</td><td>${data.pooled.n}</td>
                <td style="font-weight:600">${f(data.pooled.ours.mae)}</td><td>${f(data.pooled.fpl.mae)}</td>
                <td style="font-weight:600">${f(data.pooled.ours.rmse)}</td><td>${f(data.pooled.fpl.rmse)}</td>
                <td>${sgn(data.pooled.ours.bias)}</td><td>${sgn(data.pooled.fpl.bias)}</td><td colspan="7"></td></tr>` : "";
        html += `
            <div style="overflow-x:auto;margin-top:12px"><table class="players-table">
                <thead><tr><th>GW</th><th>Players</th><th>MAE ours</th><th>FPL</th><th>RMSE ours</th><th>FPL</th>
                    <th>Bias ours</th><th>FPL</th><th>Rank corr ours</th><th>FPL</th>
                    <th>Top-10 hit (ours / FPL)</th><th>Top-20 hit</th><th>Our captain (pts)</th><th>FPL's captain (pts)</th><th>Best</th></tr></thead>
                <tbody>${rows}${pooled}</tbody></table></div>
            <p class="section-hint">Bias = predicted − actual. Rank corr = Spearman correlation of predicted vs actual points across the players.
            Top-N hit = how many of the N highest-scoring players were in the N we (or FPL) ranked highest. Captain = the highest-predicted player.
            Green = the better of the two.</p>`;
    } else {
        html += `<p class="section-hint">No archived gameweek has finished yet — results appear here once the first one does.</p>`;
    }
    el.innerHTML = html;
}

// ─── Prediction Accuracy: tier-weight fit ─────────────────────────────────────

// Bar list for a deviance curve (lower deviance = longer bar), best row highlighted.
function devianceCurve(points, labelOf, devOf, isBest) {
    const devs = points.map(devOf);
    const lo = Math.min(...devs), hi = Math.max(...devs), range = hi - lo || 1;
    return points.map((pt, i) => {
        const d = devs[i];
        const w = Math.max(4, Math.round((1 - (d - lo) / range) * 100));
        const best = isBest(pt);
        return `<div class="sens-row${best ? " sens-best" : ""}">
            <span class="sens-val" style="width:54px">${labelOf(pt)}</span>
            <div class="sens-bar-wrap"><div class="sens-bar" style="width:${w}%;background:${best ? "#00ff87" : "#58a6ff"}"></div></div>
            <span class="sens-mae" style="width:56px">${d.toFixed(3)}</span>
        </div>`;
    }).join("");
}

function renderTierWeightFit(summary, fit) {
    const f3 = v => (v == null ? "–" : v.toFixed(3));
    const names = { tier1: "Tier 1 (odds)", tier2: "Tier 2 (rolling xG)", tier3: "Tier 3 (FDR, centred)",
                    model: "Model blend (T2+T3)", production: "Production (live blend)" };
    const sumRows = Object.entries(names).map(([k, label]) => {
        const s = summary[k];
        if (!s || !s.n) return "";
        const skill = s.skill == null ? "–" : `${s.skill > 0 ? "+" : ""}${(s.skill * 100).toFixed(1)}%`;
        const skillColor = s.skill > 0 ? "#00ff87" : "#ff6b6b";
        return `<tr><td>${label}</td><td>${s.n}</td><td>${f3(s.deviance)}</td><td>${f3(s.baseline_deviance)}</td>
            <td style="color:${skillColor};font-weight:600">${skill}</td><td>${f3(s.mae)}</td>
            <td>${s.bias > 0 ? "+" : ""}${f3(s.bias)}</td></tr>`;
    }).join("");

    const t23 = fit.tier2_vs_tier3, three = fit.three_way;
    const boot = b => b ? `bootstrap median ${b.median.toFixed(2)}, 80% range ${b.p10.toFixed(2)}–${b.p90.toFixed(2)}` : "";
    const sampleTag = (x) => x.reliable
        ? `<span style="color:#00ff87">${x.n_fixtures} fixtures</span>`
        : `<span style="color:#f5a623">only ${x.n_fixtures} fixtures — indicative, not conclusive</span>`;

    let html = `
        <h2 style="font-size:16px;margin-top:20px">Tier accuracy vs league-average baseline</h2>
        <p class="section-hint">Skill = how much lower the Poisson deviance is than just predicting the league-average goals for every team (over the same rows). Negative = worse than guessing the average.</p>
        <div style="overflow-x:auto"><table class="players-table">
            <thead><tr><th>Source</th><th>Team-matches</th><th>Deviance</th><th>Baseline dev.</th><th>Skill</th><th>MAE</th><th>Bias</th></tr></thead>
            <tbody>${sumRows}</tbody></table></div>`;

    if (t23.curve) {
        html += `
        <h2 style="font-size:16px;margin-top:20px">Best Tier 2 weight (rest Tier 3) — GW2+</h2>
        <p class="section-hint">Best static Tier 2 weight: <b>${t23.best_w2.toFixed(2)}</b> (deviance ${t23.best_deviance.toFixed(3)} vs league-average ${f3(t23.league_avg_deviance)}, live model blend ${f3(t23.production_model_deviance)}). ${boot(t23.bootstrap)}. Sample: ${sampleTag(t23)}.</p>
        <div class="sens-chart-card">${devianceCurve(t23.curve.filter((_, i) => i % 2 === 0), c => `T2 ${c.w2.toFixed(1)}`, c => c.deviance, c => c.w2 === t23.best_w2 || (t23.curve.filter((_, i) => i % 2 === 0).every(x => x.deviance >= c.deviance)))}</div>`;
    }

    if (three.top) {
        const t = three.top[0];
        html += `
        <h2 style="font-size:16px;margin-top:20px">Tier 1 / 2 / 3 mix — GW${three.gameweeks.join(", ")} (odds archived)</h2>
        <p class="section-hint">Best mix: <b>Tier 1 ${t.w1.toFixed(2)} / Tier 2 ${t.w2.toFixed(2)} / Tier 3 ${t.w3.toFixed(2)}</b> (deviance ${t.deviance.toFixed(3)}). Best <code>odds_weight</code> vs the live Tier 2/3 model: <b>${three.best_odds_weight.toFixed(2)}</b> — ${boot(three.bootstrap_odds_weight)}. Sample: ${sampleTag(three)}. The live default is 0.60.</p>
        <div class="sens-chart-card">${devianceCurve(three.odds_weight_curve.filter((_, i) => i % 2 === 0), c => `odds ${c.odds_weight.toFixed(1)}`, c => c.deviance, c => Math.abs(c.odds_weight - three.best_odds_weight) < 0.051 && three.odds_weight_curve.filter((_, i) => i % 2 === 0).every(x => x.deviance >= c.deviance))}</div>`;
    } else {
        html += `<p class="section-hint" style="margin-top:16px">No gameweek has both odds and rolling-xG data yet, so the three-way fit isn't available.</p>`;
    }
    return html;
}

// ─── Prediction Accuracy: Team xG backtest ────────────────────────────────────

async function runTeamXgBacktest() {
    const btn = $("#btn-team-xg-backtest");
    btn.disabled = true;
    btn.textContent = "Running…";
    try {
        const resp = await fetch("/api/backtest/team-xg");
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Server error");
        }
        const data = await resp.json();
        renderTeamXgBacktest(data);
    } catch (e) {
        alert(`Team xG backtest failed: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = "Run Team xG Backtest";
    }
}

function renderTeamXgBacktest(data) {
    const gws = data.gameweeks;
    const mae = data.mae_by_tier;

    $("#team-xg-legend").innerHTML = `
        <span class="legend-dot" style="background:#a78bfa"></span> Tier 1 (odds) &nbsp;
        <span class="legend-dot" style="background:#f5a623"></span> Tier 2 (rolling) &nbsp;
        <span class="legend-dot" style="background:#8b949e"></span> Tier 3 (FDR) &nbsp;
        <span class="legend-dot" style="background:#00ff87"></span> Production blend
    `;
    $("#team-xg-chart").innerHTML = lineChart(
        [
            { label: "Tier 1", color: "#a78bfa", data: mae.tier1 },
            { label: "Tier 2", color: "#f5a623", data: mae.tier2 },
            { label: "Tier 3", color: "#8b949e", data: mae.tier3 },
            { label: "Production", color: "#00ff87", data: mae.production },
        ],
        gws
    );

    const counts = data.sample_counts;
    const countsStr = gws.map(gw => {
        const t1 = counts.tier1[gw] || 0;
        const t2 = counts.tier2[gw] || 0;
        const t3 = counts.tier3[gw] || 0;
        return `GW${gw}: Tier1=${t1}, Tier2=${t2}, Tier3=${t3} fixtures`;
    }).join(" · ");
    $("#team-xg-samples").textContent = `Sample sizes — ${countsStr}`;
    $("#team-xg-weights").innerHTML = renderTierWeightFit(data.summary, data.weight_fit);

    const teamRow = r => `
        <tr>
            <td>GW${r.gw}</td>
            <td>${r.team}</td>
            <td>${r.opponent}</td>
            <td>${r.is_home ? "H" : "A"}</td>
            <td>${r.tier1 != null ? r.tier1.toFixed(2) : "–"}</td>
            <td>${r.tier2 != null ? r.tier2.toFixed(2) : "–"}</td>
            <td>${r.tier3.toFixed(2)}</td>
            <td style="font-weight:600">${r.production.toFixed(2)}</td>
            <td style="color:#00ff87;font-weight:700">${r.actual_goals}</td>
            <td style="color:${r.error < 0 ? '#ff6b6b' : '#f5a623'};font-weight:700">${r.error > 0 ? "+" : ""}${r.error.toFixed(2)}</td>
        </tr>`;

    $("#team-xg-body").innerHTML = groupTopBottomByGw(data.rows, gws, 5).map(g => `
        ${g.top.length ? groupHeaderRow(`GW${g.gw} — Top ${g.top.length} Over-predicted`, 10) : ""}
        ${g.top.map(teamRow).join("")}
        ${g.bottom.length ? groupHeaderRow(`GW${g.gw} — Top ${g.bottom.length} Under-predicted`, 10) : ""}
        ${g.bottom.map(teamRow).join("")}
    `).join("");

    $("#team-xg-results").classList.remove("hidden");
    $("#team-xg-results").scrollIntoView({ behavior: "smooth" });
}

// ─── Prediction Accuracy: Player points backtest ──────────────────────────────

async function runPlayerPointsBacktest() {
    const btn = $("#btn-player-points-backtest");
    const shareWindow = document.querySelector('input[name="share-window"]:checked').value;
    btn.disabled = true;
    btn.textContent = "Running…";
    try {
        const resp = await fetch(`/api/backtest/player-points?share_window=${shareWindow}`);
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Server error");
        }
        const data = await resp.json();
        renderPlayerPointsBacktest(data);
    } catch (e) {
        alert(`Player points backtest failed: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = "Run Player Points Backtest";
    }
}

// ─── Prediction Accuracy: per-component breakdown ───────────────────────────

const _ppComp = { data: null, view: "played", pos: "ALL", query: "", sort: "actual_total", dir: -1 };
const _COMP_KEYS = ["appearance", "goals", "assists", "clean_sheet", "goals_conceded", "saves", "bonus", "defcon", "cards", "other"];

function renderPlayerComponents(data) {
    _ppComp.data = data;
    drawPlayerComponents();
}

function drawPlayerComponents() {
    const { data, view, pos, query, sort, dir } = _ppComp;
    const summary = data.component_summary[view][pos];
    const f = (v, d = 2) => (v > 0 ? "+" : "") + v.toFixed(d);

    const btn = (group, val, label) =>
        `<button class="filter-btn${_ppComp[group] === val ? " active" : ""}" data-pp-${group}="${val}">${label}</button>`;
    const controls = `
        <div style="margin:6px 0 10px;display:flex;gap:6px;flex-wrap:wrap;align-items:center">
            ${btn("view", "played", "Players who played (mins > 0)")}${btn("view", "all", "All predictions (incl. 0 mins)")}
            <span style="width:12px"></span>
            ${["ALL", "GKP", "DEF", "MID", "FWD"].map(p => btn("pos", p, p)).join("")}
        </div>`;

    const rows = _COMP_KEYS.map(k => summary.components[k]).filter(c => c && (c.mean_predicted || c.mean_actual))
        .map(c => {
            const beats = c.mae < c.baseline_mae;
            return `<tr><td>${c.label}</td><td>${c.mean_predicted.toFixed(3)}</td><td>${c.mean_actual.toFixed(3)}</td>
                <td style="color:${Math.abs(c.bias) < 0.05 ? "#8b949e" : c.bias < 0 ? "#ff6b6b" : "#f5a623"};font-weight:600">${f(c.bias, 3)}</td>
                <td style="font-weight:700">${c.mae.toFixed(3)}</td><td style="color:#8b949e">${c.baseline_mae.toFixed(3)}</td>
                <td style="color:${beats ? "#00ff87" : "#ff6b6b"}">${beats ? "better" : "no better"}</td></tr>`;
        }).join("");

    const caveat = view === "played"
        ? `<b>Heads-up:</b> filtering to players who played selects on an outcome the model is predicting — its predictions include the chance a player doesn't play, so appearance, clean-sheet and every minutes-scaled component look under-predicted here by construction. Use "All predictions" to judge calibration.`
        : `Every player-gameweek the model made a prediction for, including those who didn't play — the fair view for calibration.`;

    // Per-player table
    let players = data.player_components.filter(p =>
        (pos === "ALL" || p.position === pos) && (!query || p.name.toLowerCase().includes(query.toLowerCase())));
    const val = p => sort === "name" ? p.name : sort === "error" ? p.predicted_total - p.actual_total
        : sort === "predicted_total" ? p.predicted_total : sort === "games" ? p.games : p.actual_total;
    players = players.sort((a, b) => (val(a) > val(b) ? 1 : val(a) < val(b) ? -1 : 0) * dir);
    const shown = players.slice(0, 100);
    const cell = (p, k) => {
        const pr = p.predicted[k], ac = p.actual[k];
        if (!pr && !ac) return `<td style="color:#484f58">–</td>`;
        const d = pr - ac;
        return `<td title="predicted ${pr.toFixed(2)} vs actual ${ac.toFixed(2)}">${pr.toFixed(1)} / <b>${ac.toFixed(1)}</b></td>`;
    };
    const th = (key, label) => `<th data-pp-sort="${key}" style="cursor:pointer">${label}${sort === key ? (dir < 0 ? " ▼" : " ▲") : ""}</th>`;
    const playerTable = `
        <h2 style="font-size:16px;margin-top:24px">Per-player: predicted vs actual by component
            <small style="font-weight:400;color:#8b949e">(season total over the GWs each played, predicted / <b>actual</b>)</small></h2>
        <input id="pp-player-search" type="search" placeholder="Search player…" value="${query.replace(/"/g, "&quot;")}"
               style="margin:4px 0 8px;padding:6px 10px;background:#161b22;border:1px solid #30363d;border-radius:6px;color:#e6edf3;width:220px">
        <span class="section-hint" style="margin-left:8px">${players.length} players${players.length > shown.length ? ` — showing top ${shown.length}` : ""}; click a header to sort</span>
        <div style="overflow-x:auto"><table class="players-table">
            <thead><tr>${th("name", "Player")}<th>Team</th><th>Pos</th>${th("games", "GP")}<th>Mins</th>
                ${th("predicted_total", "Pred")}${th("actual_total", "Actual")}${th("error", "Err")}
                ${_COMP_KEYS.map(k => `<th>${({appearance:"App",goals:"Goals",assists:"Ast",clean_sheet:"CS",goals_conceded:"GC",saves:"Saves",bonus:"Bonus",defcon:"DefCon",cards:"Cards",other:"Other"})[k]}</th>`).join("")}</tr></thead>
            <tbody>${shown.map(p => {
                const err = p.predicted_total - p.actual_total;
                return `<tr><td>${p.name}</td><td>${p.team}</td><td>${p.position}</td><td>${p.games}</td><td>${p.minutes}</td>
                    <td>${p.predicted_total.toFixed(1)}</td><td style="color:#00ff87;font-weight:700">${p.actual_total.toFixed(1)}</td>
                    <td style="color:${err < 0 ? "#ff6b6b" : "#f5a623"};font-weight:600">${f(err, 1)}</td>
                    ${_COMP_KEYS.map(k => cell(p, k)).join("")}</tr>`;
            }).join("")}</tbody></table></div>`;

    $("#pp-components").innerHTML = `
        <h2 style="font-size:16px;margin-top:20px">Points by component — predicted vs actual
            <small style="font-weight:400;color:#8b949e">(per player-gameweek, n=${summary.n}; bias = predicted − actual)</small></h2>
        ${controls}
        <p class="section-hint">${caveat} "Baseline" is the MAE of guessing each component's own average for every row (hindsight, so a stiff test) — "better" means the model beats it. For rare components (DefCon, cards, goals) MAE can rise even when the prediction gets more accurate on average, so read it together with Bias. Form adjustment is zero for GW1–5 by construction (needs more than 4 prior games).</p>
        <div style="overflow-x:auto"><table class="players-table">
            <thead><tr><th>Component</th><th>Mean predicted</th><th>Mean actual</th><th>Bias</th><th>MAE</th><th>Baseline MAE</th><th>vs baseline</th></tr></thead>
            <tbody>${rows}</tbody></table></div>
        ${playerTable}`;

    const root = $("#pp-components");
    root.querySelectorAll("[data-pp-view]").forEach(b => b.onclick = () => { _ppComp.view = b.dataset.ppView; drawPlayerComponents(); });
    root.querySelectorAll("[data-pp-pos]").forEach(b => b.onclick = () => { _ppComp.pos = b.dataset.ppPos; drawPlayerComponents(); });
    root.querySelectorAll("[data-pp-sort]").forEach(h => h.onclick = () => {
        const k = h.dataset.ppSort;
        _ppComp.dir = _ppComp.sort === k ? -_ppComp.dir : (k === "name" ? 1 : -1);
        _ppComp.sort = k; drawPlayerComponents();
    });
    const search = $("#pp-player-search");
    search.oninput = () => {
        _ppComp.query = search.value; const pos = search.selectionStart; drawPlayerComponents();
        const again = $("#pp-player-search"); again.focus(); again.setSelectionRange(pos, pos);
    };
}

function renderPlayerPointsBacktest(data) {
    $("#pp-overall-mae").textContent = data.overall_mae.toFixed(3);
    $("#pp-starters-mae").textContent = data.starters_only_mae.toFixed(3);
    $("#pp-overall-rmse").textContent = data.overall_rmse.toFixed(3);
    $("#pp-overall-bias").textContent = (data.overall_bias > 0 ? "+" : "") + data.overall_bias.toFixed(3);
    $("#pp-predictions").textContent = data.total_predictions.toLocaleString();
    $("#pp-skipped").textContent = data.skipped_no_prior_data.toLocaleString();

    const posColors = { GKP: "#a78bfa", DEF: "#f5a623", MID: "#00ff87", FWD: "#ff6b6b" };
    $("#pp-legend").innerHTML = Object.entries(posColors)
        .map(([pos, color]) => `<span class="legend-dot" style="background:${color}"></span> ${pos} &nbsp;`)
        .join("");

    $("#pp-chart").innerHTML = lineChart(
        Object.entries(posColors).map(([pos, color]) => ({
            label: pos, color, data: data.mae_by_position_per_gw[pos],
        })),
        data.gameweeks
    );

    renderPlayerComponents(data);

    const playerRow = r => `
        <tr>
            <td>GW${r.gw}</td>
            <td>${r.name}</td>
            <td>${r.team}</td>
            <td>${r.position}</td>
            <td>${r.predicted.toFixed(2)}</td>
            <td style="color:#00ff87;font-weight:700">${r.actual}</td>
            <td style="color:${r.error < 0 ? '#ff6b6b' : '#f5a623'};font-weight:600">${r.error > 0 ? "+" : ""}${r.error.toFixed(2)}</td>
            <td>${r.started ? "Yes" : "No"}</td>
        </tr>`;

    $("#pp-misses-body").innerHTML = groupTopBottomByGw(data.rows, data.gameweeks, 5).map(g => `
        ${g.top.length ? groupHeaderRow(`GW${g.gw} — Top ${g.top.length} Over-predicted`, 8) : ""}
        ${g.top.map(playerRow).join("")}
        ${g.bottom.length ? groupHeaderRow(`GW${g.gw} — Top ${g.bottom.length} Under-predicted`, 8) : ""}
        ${g.bottom.map(playerRow).join("")}
    `).join("");

    $("#player-points-results").classList.remove("hidden");
    $("#player-points-results").scrollIntoView({ behavior: "smooth" });
}

// ─── Team Overview & Fixture Tracker ─────────────────────────────────────────

let _teamsData = null;

async function loadTeams(forceReload = false) {
    if (_teamsData && !forceReload) {
        renderTeamOverview(_teamsData);
        renderFixtureTracker(_teamsData);
        return;
    }
    const loadingRow = `<tr><td colspan="16" class="loading"><span class="spinner"></span>Loading teams…</td></tr>`;
    $("#teams-table-body").innerHTML = loadingRow;
    $("#fixture-tracker-grid").innerHTML = '<p class="section-hint" style="color:#8b949e">Loading fixtures…</p>';
    try {
        const resp = await fetch("/api/teams");
        if (!resp.ok) throw new Error(`Server error ${resp.status}`);
        _teamsData = await resp.json();
        renderTeamOverview(_teamsData);
        renderFixtureTracker(_teamsData);
    } catch (e) {
        $("#teams-table-body").innerHTML = `<tr><td colspan="16" class="loading" style="color:#ff6b6b">Failed to load: ${e.message}</td></tr>`;
        $("#fixture-tracker-grid").innerHTML = `<p style="color:#ff6b6b">Failed to load: ${e.message}</p>`;
    }
}

async function refreshOdds() {
    const btn = document.getElementById("refresh-odds-btn");
    if (!btn) return;
    btn.disabled = true;
    btn.textContent = "Refreshing…";
    try {
        const resp = await fetch("/api/refresh-odds", { method: "POST" });
        const body = await resp.json();
        if (!resp.ok) throw new Error(body.detail || `Error ${resp.status}`);
        _teamsData = null;
        await loadTeams(true);
        loadArchiveStatus();
        if (body.status === "stale") {
            alert("Live odds refresh failed (quota limit or API error) — still showing the last saved odds, no data was lost. Try again later.");
        }
    } catch (e) {
        alert("Odds refresh failed: " + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Refresh Odds";
    }
}

function renderOddsStatus(oddsStatus) {
    const el = document.getElementById("odds-status");
    if (!el) return;
    if (!oddsStatus || !oddsStatus.has_key) {
        el.innerHTML = '<span style="color:#8b949e">Odds API: no key configured</span>';
        return;
    }
    if (oddsStatus.fetched_at && oddsStatus.gameweek) {
        const dt = new Date(oddsStatus.fetched_at + "Z");
        const fmt = dt.toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
        el.innerHTML = `<span style="color:#00ff87">Odds API: GW${oddsStatus.gameweek} · ${oddsStatus.fixture_count} fixtures · updated ${fmt} UTC</span>`;
    } else {
        const gw = oddsStatus.next_gw ? `GW${oddsStatus.next_gw}` : "this gameweek";
        const last = oddsStatus.last_cached_gameweek ? ` (last pull was GW${oddsStatus.last_cached_gameweek}, not used)` : "";
        el.innerHTML = `<span style="color:#f5a623">Odds API: no odds for ${gw} yet${last} — predictions use the model only. Click Refresh Odds to pull (uses API quota).</span>`;
    }
}

function renderTeamOverview(data) {
    const { teams, odds_status } = data;
    renderOddsStatus(odds_status);
    $("#teams-table-body").innerHTML = teams.map(t => {
        const gd = t.goal_diff >= 0 ? `+${t.goal_diff}` : `${t.goal_diff}`;
        const gdColor = t.goal_diff > 0 ? "#00ff87" : t.goal_diff < 0 ? "#ff6b6b" : "#e0e0e0";
        const posLabel = t.position || "-";
        const attXg = t.attack_xg6 != null ? t.attack_xg6.toFixed(2) : "-";
        const defXg = t.defence_xg6 != null ? t.defence_xg6.toFixed(2) : "-";
        const attColor = t.attack_xg6 >= 2.0 ? "#00ff87" : t.attack_xg6 >= 1.2 ? "#f5a623" : "#ff6b6b";
        const defColor = t.defence_xg6 <= 1.0 ? "#00ff87" : t.defence_xg6 <= 1.8 ? "#f5a623" : "#ff6b6b";
        // Odds API columns (null = no data)
        const oddsAtt = t.odds_team_xg != null ? t.odds_team_xg.toFixed(2) : "–";
        const oddsDef = t.odds_opp_xg != null ? t.odds_opp_xg.toFixed(2) : "–";
        const oddsAttColor = t.odds_team_xg == null ? "#8b949e" : t.odds_team_xg >= 2.0 ? "#00ff87" : t.odds_team_xg >= 1.2 ? "#f5a623" : "#ff6b6b";
        const oddsDefColor = t.odds_opp_xg == null ? "#8b949e" : t.odds_opp_xg <= 1.0 ? "#00ff87" : t.odds_opp_xg <= 1.8 ? "#f5a623" : "#ff6b6b";
        return `<tr>
            <td style="color:#8b949e;text-align:center">${posLabel}</td>
            <td style="font-weight:600">${t.name}</td>
            <td style="text-align:center">${t.played}</td>
            <td style="text-align:center;color:#00ff87">${t.won}</td>
            <td style="text-align:center;color:#8b949e">${t.drawn}</td>
            <td style="text-align:center;color:#ff6b6b">${t.lost}</td>
            <td style="text-align:center">${t.goals_for}</td>
            <td style="text-align:center">${t.goals_against}</td>
            <td style="text-align:center;color:${gdColor};font-weight:600">${gd}</td>
            <td style="text-align:center;font-weight:700;color:#00ff87">${t.points}</td>
            <td style="text-align:center;color:#a78bfa">${t.strength_home}</td>
            <td style="text-align:center;color:#60a5fa">${t.strength_away}</td>
            <td style="text-align:center;font-weight:600;color:${attColor}">${attXg}</td>
            <td style="text-align:center;font-weight:600;color:${defColor}">${defXg}</td>
            <td style="text-align:center;font-weight:600;color:${oddsAttColor}">${oddsAtt}</td>
            <td style="text-align:center;font-weight:600;color:${oddsDefColor}">${oddsDef}</td>
        </tr>`;
    }).join("");
}

function renderFixtureTracker(data) {
    const { teams, gws } = data;
    if (!gws || gws.length === 0) {
        $("#fixture-tracker-grid").innerHTML = "<p style='color:#8b949e'>No upcoming fixtures available.</p>";
        return;
    }

    let html = '<div class="fixture-grid-wrap"><table class="fixture-grid">';

    // Header row
    html += "<thead><tr>";
    html += `<th class="team-col">Team</th>`;
    gws.forEach(gw => { html += `<th>GW${gw}</th>`; });
    html += "</tr></thead>";

    // Body rows
    html += "<tbody>";
    teams.forEach(t => {
        html += "<tr>";
        html += `<td class="team-name-cell">${t.short_name}</td>`;
        t.upcoming.forEach(gwData => {
            if (gwData.matches.length === 0) {
                html += `<td style="color:#30363d">—</td>`;
            } else {
                html += `<td><div class="fixture-cell">`;
                gwData.matches.forEach(m => {
                    const haClass = m.is_home ? "home" : "";
                    const haLabel = m.is_home ? "H" : "A";
                    html += `<div class="fixture-match">
                        <span class="fdr fdr-${m.fdr}">${m.opp_short}</span>
                        <span class="ha ${haClass}">${haLabel}</span>
                    </div>`;
                });
                html += `</div></td>`;
            }
        });
        html += "</tr>";
    });
    html += "</tbody></table></div>";

    $("#fixture-tracker-grid").innerHTML = html;
}

// ─── SVG line chart ───────────────────────────────────────────────────────────

function lineChart(seriesList, gws) {
    const W = 700, H = 200;
    const pad = { top: 15, right: 15, bottom: 30, left: 42 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const allVals = seriesList.flatMap(s => Object.values(s.data).filter(v => v > 0));
    if (!allVals.length) return "<p>No data</p>";

    const minVal = Math.min(...allVals) * 0.92;
    const maxVal = Math.max(...allVals) * 1.08;
    const valRange = maxVal - minVal || 1;

    const xScale = i => pad.left + (i / Math.max(gws.length - 1, 1)) * plotW;
    const yScale = v => pad.top + plotH * (1 - (v - minVal) / valRange);

    let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:200px;overflow:visible">`;

    // Y grid + labels
    for (let i = 0; i <= 4; i++) {
        const v = minVal + valRange * i / 4;
        const y = yScale(v);
        svg += `<line x1="${pad.left}" y1="${y}" x2="${pad.left + plotW}" y2="${y}" stroke="#21262d" stroke-width="1"/>`;
        svg += `<text x="${pad.left - 5}" y="${y + 4}" font-size="9" fill="#8b949e" text-anchor="end">${v.toFixed(2)}</text>`;
    }

    // X labels every 5 GWs
    gws.forEach((gw, i) => {
        if (i % 5 === 0 || i === gws.length - 1) {
            svg += `<text x="${xScale(i)}" y="${pad.top + plotH + 18}" font-size="9" fill="#8b949e" text-anchor="middle">GW${gw}</text>`;
        }
    });

    // Axes
    svg += `<line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + plotH}" stroke="#30363d" stroke-width="1"/>`;
    svg += `<line x1="${pad.left}" y1="${pad.top + plotH}" x2="${pad.left + plotW}" y2="${pad.top + plotH}" stroke="#30363d" stroke-width="1"/>`;

    // Series
    for (const s of seriesList) {
        const pts = gws
            .map((gw, i) => s.data[gw] !== undefined ? `${xScale(i)},${yScale(s.data[gw])}` : null)
            .filter(Boolean);
        if (pts.length > 1) {
            svg += `<polyline points="${pts.join(" ")}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" opacity="0.9"/>`;
        }
        gws.forEach((gw, i) => {
            if (s.data[gw] !== undefined) {
                svg += `<circle cx="${xScale(i)}" cy="${yScale(s.data[gw])}" r="3" fill="${s.color}"/>`;
            }
        });
    }

    svg += "</svg>";
    return svg;
}

