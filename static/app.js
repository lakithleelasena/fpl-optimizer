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
        n_gw: parseInt($("#n-gw").value) || 1,
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
        n_gw: parseInt($("#n-gw-transfer").value) || 3,
        form_factor: parseFloat($("#form-factor").value),
        cs_factor: parseFloat($("#cs-factor").value),
        atk_factor: parseFloat($("#atk-factor").value),
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
            <div class="player-name">${p.name}</div>
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
            <td>${p.name}</td>
            <td>${p.team}</td>
            <td>${p.position}</td>
            <td>£${p.cost.toFixed(1)}m</td>
            <td style="color:#00ff87;font-weight:600">${p.predicted_points.toFixed(1)}</td>
            <td style="color:${slColor}">${slPct}%</td>
            <td style="color:${startColor(p.exp_minutes)}">${Math.round(p.exp_minutes * 100)}%</td>
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

function getSelectedSignals() {
    return Array.from($$(".signal-check input:checked")).map(el => el.value);
}

async function runBacktest() {
    const btn = $("#btn-backtest");
    btn.disabled = true;
    btn.textContent = "Running… (may take 5–15s)";
    $("#backtest-results").classList.add("hidden");

    const signals = getSelectedSignals();
    if (signals.length === 0) {
        alert("Please select at least one signal.");
        btn.disabled = false;
        btn.textContent = "Run Backtest";
        return;
    }

    try {
        const resp = await fetch(`/api/backtest?signals=${signals.join(",")}`);
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
    const selectedSignals = getSelectedSignals();

    const setSlider = (id, val) => {
        const el = $(`#${id}`);
        if (el) { el.value = val; el.dispatchEvent(new Event("input")); }
    };

    // Signal → slider id mapping
    const signalToSlider = {
        home_away: "w-home-away",
        season:    "w-season",
        xgi:       "w-xgi",
        fixture:   "w-fixture",
        form:      "w-form",
        threat:    "w-threat",
        xgc:       "w-xgc",
    };

    // Apply best weight if signal was selected, otherwise zero it out
    setSlider("w-home-away", selectedSignals.includes("home_away") ? b.w_home_away : 0);
    setSlider("w-season",    selectedSignals.includes("season")    ? b.w_season    : 0);
    setSlider("w-xgi",       selectedSignals.includes("xgi")       ? b.w_xgi       : 0);
    setSlider("w-fixture",   selectedSignals.includes("fixture")   ? b.w_fixture   : 0);
    setSlider("w-form",      selectedSignals.includes("form")      ? b.w_form      : 0);
    setSlider("w-threat",    selectedSignals.includes("threat")    ? b.w_threat    : 0);
    setSlider("w-xgc",       selectedSignals.includes("xgc")       ? b.w_xgc       : 0);

    // Switch to optimizer tab
    $$(".tab-btn").forEach(btn => btn.classList.remove("active"));
    $$(".tab-content").forEach(c => c.classList.add("hidden"));
    $(".tab-btn[data-tab='optimizer']").classList.add("active");
    $("#tab-optimizer").classList.remove("hidden");
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
        el.innerHTML = `<span style="color:#f5a623">Odds API: key configured — no cache yet, click Refresh</span>`;
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
