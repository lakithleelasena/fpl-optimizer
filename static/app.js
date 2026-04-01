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
    ["home-away", "season", "xgi", "fixture", "form", "threat", "xgc"].forEach((w) => {
        const slider = $(`#w-${w}`);
        const display = $(`#w-${w}-val`);
        slider.addEventListener("input", () => { display.textContent = slider.value; });
    });

    $("#btn-optimize").addEventListener("click", runOptimize);
    $("#search").addEventListener("input", renderTable);
    $("#btn-transfer-advice").addEventListener("click", runTransferAdvice);
    $("#btn-backtest").addEventListener("click", runBacktest);
    $("#btn-apply-weights").addEventListener("click", applyBestWeights);

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

// ─── Players ─────────────────────────────────────────────────────────────────

async function loadPlayers() {
    $("#table-body").innerHTML = '<tr><td colspan="11" class="loading"><span class="spinner"></span>Loading players...</td></tr>';
    try {
        const resp = await fetch("/api/players");
        allPlayers = await resp.json();
        renderTable();
        renderSquadBuilder();
    } catch (e) {
        $("#table-body").innerHTML = `<tr><td colspan="11" class="loading">Failed to load: ${e.message}</td></tr>`;
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
        w_home_away: parseFloat($("#w-home-away").value),
        w_season: parseFloat($("#w-season").value),
        w_xgi: parseFloat($("#w-xgi").value),
        w_fixture: parseFloat($("#w-fixture").value),
        w_form: parseFloat($("#w-form").value),
        w_threat: parseFloat($("#w-threat").value),
        w_xgc: parseFloat($("#w-xgc").value),
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
        renderSquad(data);

        // Refresh player table with the same weights so scores match pitch cards
        try {
            const playersResp = await fetch(
                `/api/players?w_home_away=${body.w_home_away}&w_season=${body.w_season}&w_xgi=${body.w_xgi}&w_fixture=${body.w_fixture}&w_form=${body.w_form}&w_threat=${body.w_threat}&w_xgc=${body.w_xgc}`
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
    $("#summary-cost").textContent = `£${data.total_cost.toFixed(1)}m`;
    $("#summary-points").textContent = data.total_predicted_points.toFixed(1);
    $("#summary-count").textContent = `${data.starters.length + data.bench.length}`;

    const groups = { GKP: [], DEF: [], MID: [], FWD: [] };
    data.starters.forEach((p) => groups[p.position].push(p));

    let html = "";
    for (const [pos, players] of Object.entries(groups)) {
        if (players.length === 0) continue;
        html += `<div class="position-row"><div class="position-row-label">${pos}</div>`;
        html += players.map((p) => cardHTML(p, p.id === captainId, p.id === viceCaptainId)).join("");
        html += `</div>`;
    }
    $("#pitch-starters").innerHTML = html;
    $("#pitch-bench").innerHTML = data.bench.map((p) => cardHTML(p, false, false)).join("");
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

function renderSquadBuilder() {
    const total = squadCount();
    $("#squad-total-count").textContent = total;

    for (const [pos, limit] of Object.entries(slotLimits)) {
        const players = mySquad[pos];
        $(`#count-${pos}`).textContent = `${players.length}/${limit}`;

        const slotsEl = $(`#slots-${pos}`);
        let html = players.map((p) => `
            <div class="squad-slot filled">
                <span class="slot-name">${p.name}</span>
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
                <span class="sr-name">${p.name}${note}</span>
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
    // Summary
    $("#ts-transfers").textContent = data.transfers.length;
    const hitsEl = $("#ts-hits");
    hitsEl.textContent = data.hits_required;
    hitsEl.style.color = data.hits_required > 0 ? "#ff6b6b" : "#00ff87";

    const gainEl = $("#ts-net-gain");
    gainEl.textContent = `+${data.net_points_gain.toFixed(1)}`;
    gainEl.style.color = data.net_points_gain >= 0 ? "#00ff87" : "#ff6b6b";

    $("#ts-total-3gw").textContent = data.total_predicted_3gw.toFixed(1);

    // Chip recommendation
    renderChipRec(data.chip_recommendation);

    // Transfer suggestions
    renderTransfers(data.transfers, data.free_transfers);

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

function renderTransfers(transfers, freeTransfers) {
    if (transfers.length === 0) {
        $("#transfers-content").innerHTML = `<p class="no-transfers">No beneficial transfers found this gameweek. Hold your transfers.</p>`;
        return;
    }

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
                        <div class="t-meta">${t.transfer_out.team} · £${t.transfer_out.cost.toFixed(1)}m · ${t.transfer_out.predicted_points.toFixed(1)} 3GW pts</div>
                    </div>
                    <div class="transfer-arrow">→</div>
                    <div class="transfer-in">
                        <div class="t-label">IN</div>
                        <div class="t-name">${t.transfer_in.name}</div>
                        <div class="t-meta">${t.transfer_in.team} · £${t.transfer_in.cost.toFixed(1)}m · ${t.transfer_in.predicted_points.toFixed(1)} 3GW pts</div>
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
    const groups = { GKP: [], DEF: [], MID: [], FWD: [] };
    data.starters.forEach((p) => groups[p.position].push(p));

    let html = "";
    for (const [pos, players] of Object.entries(groups)) {
        if (players.length === 0) continue;
        html += `<div class="position-row"><div class="position-row-label">${pos}</div>`;
        html += players.map((p) => cardHTML(p, p.id === data.captain_id, p.id === data.vice_captain_id, false)).join("");
        html += `</div>`;
    }
    $("#advice-pitch-starters").innerHTML = html;
    $("#advice-pitch-bench").innerHTML = data.bench.map((p) => cardHTML(p, false, false, false)).join("");
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

function cardHTML(p, isCaptain = false, isViceCaptain = false, show3gw = false) {
    const slPct = Math.round(p.start_likelihood * 100);
    const slColor = startColor(p.start_likelihood);
    const badge = isCaptain
        ? `<div class="captain-badge">C</div>`
        : isViceCaptain
        ? `<div class="captain-badge vc-badge">V</div>`
        : "";
    const displayPts = (!show3gw && p.gw_pts && p.gw_pts.length > 0)
        ? p.gw_pts[0]
        : p.predicted_points;
    const ptsLabel = show3gw ? "3GW" : "GW";
    const gwBreakdown = show3gw && p.gw_pts && p.gw_pts.length === 3
        ? `<div class="gw-breakdown">GW1:${p.gw_pts[0].toFixed(1)} GW2:${p.gw_pts[1].toFixed(1)} GW3:${p.gw_pts[2].toFixed(1)}</div>`
        : "";
    return `
        <div class="player-card pos-${p.position}" style="position:relative">
            ${badge}
            <div class="player-name">${p.name}</div>
            <div class="player-team">${p.team} · ${p.position}</div>
            <div class="player-pts">${displayPts.toFixed(1)}<span class="pts-label">${ptsLabel}</span></div>
            ${gwBreakdown}
            <div class="player-cost">£${p.cost.toFixed(1)}m</div>
            <div class="start-likelihood" style="color:${slColor}">${slPct}% start</div>
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
            <td>${p.name}</td>
            <td>${p.team}</td>
            <td>${p.position}</td>
            <td>£${p.cost.toFixed(1)}m</td>
            <td style="color:#00ff87;font-weight:600">${p.predicted_points.toFixed(1)}</td>
            <td style="color:${slColor}">${slPct}%</td>
            <td style="color:${fixtureColor(p.fixture_ease)}">${fixtureLabel(p.fixture_ease)}</td>
            <td style="color:#a78bfa;font-weight:600">${p.ep_next != null ? p.ep_next.toFixed(1) : '-'}</td>
            <td>${p.season_avg.toFixed(1)}</td>
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

// ─── Backtest ─────────────────────────────────────────────────────────────────

let bestWeightsFound = null;

async function runBacktest() {
    const btn = $("#btn-backtest");
    btn.disabled = true;
    btn.textContent = "Running… (may take 5–15s)";
    $("#backtest-results").classList.add("hidden");

    try {
        const resp = await fetch("/api/backtest");
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Server error");
        }
        const data = await resp.json();
        renderBacktest(data);
    } catch (e) {
        alert(`Backtest failed: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = "Run Backtest";
    }
}

function renderBacktest(data) {
    bestWeightsFound = data.best;

    // Summary cards
    $("#bt-best-mae").textContent = data.best_mae.toFixed(4);
    $("#bt-default-mae").textContent = data.default_mae.toFixed(4);
    const impEl = $("#bt-improvement");
    const imp = data.improvement_pct;
    impEl.textContent = `${imp > 0 ? "+" : ""}${imp.toFixed(1)}%`;
    impEl.style.color = imp > 0 ? "#00ff87" : imp < 0 ? "#ff6b6b" : "#e0e0e0";
    $("#bt-gws").textContent = data.gameweeks.length;
    $("#bt-datapoints").textContent = data.total_data_points.toLocaleString();
    $("#bt-combos").textContent = data.total_combinations_tested;

    // Best weights banner
    const b = data.best;
    $("#bt-best-weights").innerHTML = `
        <div class="best-weights-row">
            <div class="bw-chip"><span class="bw-label">H/A</span><span class="bw-val">${b.w_home_away.toFixed(2)}</span></div>
            <div class="bw-chip"><span class="bw-label">Season Avg</span><span class="bw-val">${b.w_season.toFixed(2)}</span></div>
            <div class="bw-chip"><span class="bw-label">xGI</span><span class="bw-val">${b.w_xgi.toFixed(2)}</span></div>
            <div class="bw-chip"><span class="bw-label">Fixture</span><span class="bw-val">${b.w_fixture.toFixed(2)}</span></div>
            <div class="bw-chip"><span class="bw-label">Form</span><span class="bw-val">${b.w_form.toFixed(2)}</span></div>
            <div class="bw-chip"><span class="bw-label">Threat</span><span class="bw-val">${b.w_threat.toFixed(2)}</span></div>
            <div class="bw-chip"><span class="bw-label">xGC</span><span class="bw-val">${b.w_xgc.toFixed(2)}</span></div>
            <div class="bw-chip bw-mae"><span class="bw-label">MAE</span><span class="bw-val">${b.mae.toFixed(4)}</span></div>
        </div>`;

    // Per-GW chart
    $("#bt-gw-chart").innerHTML = lineChart(
        [
            { label: "Best weights", color: "#00ff87", data: data.per_gw_best },
            { label: "Default weights", color: "#f5a623", data: data.per_gw_default },
        ],
        data.gameweeks
    );

    // Sensitivity
    const sens = data.sensitivity;
    $("#bt-sensitivity").innerHTML = [
        sensitivityChart(sens.w_home_away, "Home/Away Weight"),
        sensitivityChart(sens.w_season,    "Season Avg Weight"),
        sensitivityChart(sens.w_xgi,       "xG Involvement Weight"),
        sensitivityChart(sens.w_fixture,   "Fixture Difficulty Weight"),
        sensitivityChart(sens.w_form,      "Form Weight"),
        sensitivityChart(sens.w_threat,    "ICT Threat Weight"),
        sensitivityChart(sens.w_xgc,       "xGC (Clean Sheet) Weight"),
    ].join("");

    // Top combos table
    const defaultW = [0.05, 0.20, 0.10, 0.35, 0.10, 0.10, 0.20];
    $("#bt-combos-body").innerHTML = data.top_combinations.map((c, i) => {
        const isDefault = Math.abs(c.w_home_away - defaultW[0]) < 0.01 &&
                          Math.abs(c.w_season     - defaultW[1]) < 0.01 &&
                          Math.abs(c.w_xgi        - defaultW[2]) < 0.01 &&
                          Math.abs(c.w_fixture    - defaultW[3]) < 0.01 &&
                          Math.abs(c.w_form       - defaultW[4]) < 0.01 &&
                          Math.abs(c.w_threat     - defaultW[5]) < 0.01 &&
                          Math.abs(c.w_xgc        - defaultW[6]) < 0.01;
        const isBest = i === 0;
        const cls = isBest ? "row-best" : isDefault ? "row-default" : "";
        return `<tr class="${cls}">
            <td>${i + 1}${isBest ? " 🏆" : isDefault ? " (default)" : ""}</td>
            <td>${c.w_home_away.toFixed(2)}</td>
            <td>${c.w_season.toFixed(2)}</td>
            <td>${c.w_xgi.toFixed(2)}</td>
            <td>${c.w_fixture.toFixed(2)}</td>
            <td>${c.w_form.toFixed(2)}</td>
            <td>${c.w_threat.toFixed(2)}</td>
            <td>${c.w_xgc.toFixed(2)}</td>
            <td style="color:#00ff87;font-weight:700">${c.mae.toFixed(4)}</td>
        </tr>`;
    }).join("");

    $("#backtest-results").classList.remove("hidden");
    $("#backtest-results").scrollIntoView({ behavior: "smooth" });
}

function applyBestWeights() {
    if (!bestWeightsFound) return;
    const b = bestWeightsFound;
    const setSlider = (id, val) => {
        const el = $(`#${id}`);
        if (el) { el.value = val; el.dispatchEvent(new Event("input")); }
    };
    setSlider("w-home-away", b.w_home_away);
    setSlider("w-season",    b.w_season);
    setSlider("w-xgi",       b.w_xgi);
    setSlider("w-fixture",   b.w_fixture);
    setSlider("w-form",      b.w_form);
    setSlider("w-threat",    b.w_threat);
    setSlider("w-xgc",       b.w_xgc);

    // Switch to optimizer tab
    $$(".tab-btn").forEach(btn => btn.classList.remove("active"));
    $$(".tab-content").forEach(c => c.classList.add("hidden"));
    $(".tab-btn[data-tab='optimizer']").classList.add("active");
    $("#tab-optimizer").classList.remove("hidden");
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

// ─── Sensitivity bar chart ────────────────────────────────────────────────────

function sensitivityChart(data, title) {
    const maes = data.map(d => d.mae);
    const minMae = Math.min(...maes);
    const maxMae = Math.max(...maes);
    const range = maxMae - minMae || 1;

    let html = `<div class="sens-chart-card"><div class="sens-chart-title">${title}</div>`;
    html += data.map(d => {
        const normalized = (d.mae - minMae) / range; // 0=best, 1=worst
        const barW = Math.max(4, Math.round((1 - normalized) * 100));
        const color = normalized < 0.33 ? "#00ff87" : normalized < 0.66 ? "#f5a623" : "#ff6b6b";
        const isBest = d.mae === minMae;
        return `
        <div class="sens-row${isBest ? " sens-best" : ""}">
            <span class="sens-val">${d.value.toFixed(1)}</span>
            <div class="sens-bar-wrap"><div class="sens-bar" style="width:${barW}%;background:${color}"></div></div>
            <span class="sens-mae">${d.mae.toFixed(4)}</span>
        </div>`;
    }).join("");
    html += "</div>";
    return html;
}
