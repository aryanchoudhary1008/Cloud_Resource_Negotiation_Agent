/**
 * script.js — Cloud Resource Negotiation Agent Frontend
 *
 * Handles all user interactions and communicates with the Flask backend
 * via the Fetch API. All API calls are async/await for clean error handling.
 *
 * Functions:
 *   showSection(name)      — Switch visible section
 *   updatePricePreview()   — Live base price estimate in the form
 *   handleNegotiate(event) — POST /api/negotiate
 *   renderResult(data)     — Display negotiation result
 *   renderLog(log)         — Parse and colorize log entries
 *   handleTrain()          — POST /api/train
 *   loadHistory()          — GET  /api/history
 *   loadStats()            — GET  /api/stats
 *   loadGraph()            — GET  /api/graph
 */

"use strict";

// ── Section Navigation ─────────────────────────────────────────────

/**
 * Show one section and hide all others.
 * Called from navbar links.
 *
 * @param {string} name - "negotiate" | "history" | "stats" | "train"
 */
function showSection(name) {
  const sections = {
    negotiate: "negotiate-section",
    history:   "history-section",
    stats:     "stats-section",
    train:     "train-section",
  };

  Object.entries(sections).forEach(([key, id]) => {
    const el = document.getElementById(id);
    if (el) {
      el.classList.toggle("active",  key === name);
      el.classList.toggle("hidden",  key !== name);
    }
  });

  // Auto-load data for sections that need it
  if (name === "history") loadHistory();
  if (name === "stats")   loadStats();
}

// ── Live Price Preview ─────────────────────────────────────────────

/**
 * Compute and display an estimated base price as the user changes
 * the form inputs. Uses the same formula as the backend:
 *   base = cpu*2.5 + ram*1.5 + storage*0.1
 */
function updatePricePreview() {
  const cpu     = parseFloat(document.getElementById("cpu").value)     || 0;
  const ram     = parseFloat(document.getElementById("ram").value)     || 0;
  const storage = parseFloat(document.getElementById("storage").value) || 0;

  const estimate = (cpu * 2.5) + (ram * 1.5) + (storage * 0.1);

  document.getElementById("base-price-display").textContent =
    `$${estimate.toFixed(2)}`;
}

// Wire up live preview to form inputs
document.addEventListener("DOMContentLoaded", () => {
  ["cpu", "ram", "storage", "budget"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", updatePricePreview);
  });
  updatePricePreview();  // initial render
});

// ── Negotiation ────────────────────────────────────────────────────

/**
 * Handle form submission: POST to /api/negotiate and render result.
 *
 * @param {Event} event - Form submit event (prevented from reloading)
 */
async function handleNegotiate(event) {
  event.preventDefault();

  const btn       = document.getElementById("negotiate-btn");
  const btnText   = document.getElementById("btn-text");
  const spinner   = document.getElementById("btn-spinner");

  // Build request body from form values
  const payload = {
    cpu:           parseInt(document.getElementById("cpu").value),
    ram:           parseInt(document.getElementById("ram").value),
    storage:       parseInt(document.getElementById("storage").value),
    budget:        parseFloat(document.getElementById("budget").value),
    training_mode: document.getElementById("training-toggle").checked,
  };

  // Show loading state
  btn.disabled    = true;
  btnText.textContent = "Negotiating...";
  spinner.classList.remove("hidden");

  try {
    const response = await fetch("/api/negotiate", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || `Server error ${response.status}`);
    }

    renderResult(data);

  } catch (err) {
    alert(`Negotiation failed: ${err.message}`);
    console.error("[Negotiate] Error:", err);
  } finally {
    // Restore button state
    btn.disabled          = false;
    btnText.textContent   = "▶ Start Negotiation";
    spinner.classList.add("hidden");
  }
}

/**
 * Render the negotiation result into the result panel.
 *
 * @param {Object} data - JSON response from /api/negotiate
 */
function renderResult(data) {
  // Show result panel, hide empty state
  document.getElementById("result-empty").classList.add("hidden");
  document.getElementById("result-content").classList.remove("hidden");

  // ── Outcome badge ───────────────────────────────────────────────
  const badge = document.getElementById("outcome-badge");
  if (data.success) {
    badge.className = "outcome-badge success";
    badge.textContent = `✓ Deal Reached in ${data.rounds} Round${data.rounds !== 1 ? "s" : ""}`;
  } else {
    badge.className = "outcome-badge failure";
    badge.textContent = `✗ Negotiation Failed after ${data.rounds} Round${data.rounds !== 1 ? "s" : ""}`;
  }

  // ── Metrics ─────────────────────────────────────────────────────
  document.getElementById("r-price").textContent  =
    data.success ? `$${data.final_price.toFixed(2)}` : "—";
  document.getElementById("r-budget").textContent  = `$${data.budget.toFixed(2)}`;
  document.getElementById("r-rounds").textContent  = `${data.rounds} / 5`;
  document.getElementById("r-reward").textContent  = data.total_reward.toFixed(1);
  document.getElementById("r-cpu").textContent     = `${data.final_cpu} cores`;
  document.getElementById("r-ram").textContent     = `${data.final_ram} GB`;

  // ── Savings bar (only for successful deals) ──────────────────────
  const savingsBar = document.getElementById("savings-bar");
  if (data.success && data.final_price > 0) {
    const saved   = data.budget - data.final_price;
    const pct     = Math.max(0, Math.min((saved / data.budget) * 100, 100));

    document.getElementById("savings-fill").style.width    = `${pct}%`;
    document.getElementById("savings-amount").textContent  =
      saved >= 0
        ? `Saved $${saved.toFixed(2)} (${pct.toFixed(0)}% under budget)`
        : `Over budget by $${Math.abs(saved).toFixed(2)}`;

    savingsBar.classList.remove("hidden");
  } else {
    savingsBar.classList.add("hidden");
  }

  // ── Log ─────────────────────────────────────────────────────────
  renderLog(data.log || []);
}

/**
 * Parse the log array and render styled log entries.
 * Applies different CSS classes based on entry content.
 *
 * @param {string[]} log - Array of log strings from the backend
 */
function renderLog(log) {
  const container = document.getElementById("negotiation-log");
  container.innerHTML = "";

  log.forEach(line => {
    const div = document.createElement("div");
    div.className = "log-entry";

    // Classify the line for styling
    if (/^Round \d+: Provider offers/i.test(line)) {
      div.classList.add("round-header");
    } else if (/^Round \d+: Agent action/i.test(line) || /User Agent action/i.test(line)) {
      div.classList.add("action");
    } else if (/Deal reached|accepted/i.test(line)) {
      div.classList.add("success");
    } else if (/failed|rejected|maximum rounds/i.test(line)) {
      div.classList.add("failure");
    } else {
      div.classList.add("detail");
    }

    div.textContent = line;
    container.appendChild(div);
  });

  // Auto-scroll to bottom
  container.scrollTop = container.scrollHeight;
}

// ── Training ───────────────────────────────────────────────────────

/**
 * POST to /api/train with the selected episode count.
 * Displays training results when complete.
 */
async function handleTrain() {
  const btn      = document.getElementById("train-btn");
  const btnText  = document.getElementById("train-btn-text");
  const spinner  = document.getElementById("train-spinner");
  const episodes = parseInt(document.getElementById("train-episodes").value);

  // Show loading
  btn.disabled          = true;
  btnText.textContent   = `Training ${episodes} episodes...`;
  spinner.classList.remove("hidden");

  try {
    const response = await fetch("/api/train", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ episodes }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || `Server error ${response.status}`);
    }

    // Show results
    document.getElementById("train-empty").classList.add("hidden");
    document.getElementById("train-results").classList.remove("hidden");

    document.getElementById("t-episodes").textContent = data.episodes_trained.toLocaleString();
    document.getElementById("t-success").textContent  = `${data.success_rate}%`;
    document.getElementById("t-reward").textContent   = data.avg_reward.toFixed(3);
    document.getElementById("t-states").textContent   = data.total_states_learned.toLocaleString();
    document.getElementById("t-epsilon").textContent  = data.current_epsilon.toFixed(4);

  } catch (err) {
    alert(`Training failed: ${err.message}`);
    console.error("[Train] Error:", err);
  } finally {
    btn.disabled          = false;
    btnText.textContent   = "▶ Start Training";
    spinner.classList.add("hidden");
  }
}

// ── History ────────────────────────────────────────────────────────

/**
 * GET /api/history and render results into the data table.
 */
async function loadHistory() {
  const tbody   = document.getElementById("history-tbody");
  const table   = document.getElementById("history-table");
  const empty   = document.getElementById("history-empty");

  tbody.innerHTML = '<tr><td colspan="10" style="text-align:center; color:var(--text-muted); padding:2rem;">Loading…</td></tr>';
  table.classList.remove("hidden");
  empty.classList.add("hidden");

  try {
    const response = await fetch("/api/history?limit=50");
    const sessions = await response.json();

    if (!response.ok) throw new Error(sessions.error || "Failed to load history");

    if (!sessions.length) {
      table.classList.add("hidden");
      empty.classList.remove("hidden");
      return;
    }

    // Build table rows
    tbody.innerHTML = sessions.map(s => {
      const ts     = new Date(s.timestamp + "Z").toLocaleString();
      const status = s.success
        ? `<span class="status-pill success">✓ Success</span>`
        : `<span class="status-pill failure">✗ Failed</span>`;
      const price  = s.success ? `$${s.final_price.toFixed(2)}` : "—";

      return `
        <tr>
          <td>${s.id}</td>
          <td>${status}</td>
          <td>${s.final_cpu} cores</td>
          <td>${s.final_ram} GB</td>
          <td>${s.storage} GB</td>
          <td>$${s.budget.toFixed(2)}</td>
          <td>${price}</td>
          <td>${s.rounds}</td>
          <td>${s.total_reward.toFixed(1)}</td>
          <td>${ts}</td>
        </tr>
      `;
    }).join("");

  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; color:var(--accent-red); padding:1rem;">${err.message}</td></tr>`;
    console.error("[History] Error:", err);
  }
}

// ── Stats ──────────────────────────────────────────────────────────

/**
 * GET /api/stats and populate the statistics cards.
 */
async function loadStats() {
  const empty   = document.getElementById("stats-empty");
  const content = document.getElementById("stats-content");

  try {
    const response = await fetch("/api/stats");
    const data     = await response.json();

    if (!response.ok) throw new Error(data.error || "Failed to load stats");

    if (data.total_sessions === 0) {
      empty.classList.remove("hidden");
      content.classList.add("hidden");
      return;
    }

    empty.classList.add("hidden");
    content.classList.remove("hidden");

    document.getElementById("s-success-rate").textContent = `${data.success_rate}%`;
    document.getElementById("s-total").textContent        = data.total_sessions;
    document.getElementById("s-avg-rounds").textContent   = data.avg_rounds.toFixed(2);
    document.getElementById("s-cost-eff").textContent     = `$${data.avg_cost_efficiency.toFixed(2)}`;
    document.getElementById("s-res-sat").textContent      = `${data.resource_satisfaction}%`;
    document.getElementById("s-failed").textContent       = data.failed;

  } catch (err) {
    console.error("[Stats] Error:", err);
  }
}

// ── Graph ──────────────────────────────────────────────────────────

/**
 * GET /api/graph — fetch a base64 PNG and display it in the graph container.
 * Requires matplotlib installed on the server.
 */
async function loadGraph() {
  const container = document.getElementById("graph-container");
  container.innerHTML = '<div class="empty-state"><div class="empty-icon">⏳</div><p>Generating graph…</p></div>';

  try {
    const response = await fetch("/api/graph");
    const data     = await response.json();

    if (!response.ok || data.error) {
      throw new Error(data.error || "Failed to generate graph");
    }

    // Display the base64 PNG image
    const img   = document.createElement("img");
    img.src     = `data:image/png;base64,${data.image}`;
    img.alt     = "Training and session outcome charts";
    img.style.maxWidth = "100%";

    container.innerHTML = "";
    container.appendChild(img);

  } catch (err) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">⚠</div>
        <p style="color:var(--accent-red)">${err.message}</p>
        <p style="font-size:0.82rem; margin-top:0.5rem">Install matplotlib: <code>pip install matplotlib</code></p>
      </div>
    `;
    console.error("[Graph] Error:", err);
  }
}
