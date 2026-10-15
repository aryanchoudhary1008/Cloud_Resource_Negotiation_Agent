
// Theme Toggle Logic
function toggleTheme() {
    document.body.classList.toggle('light-mode');
    const isLight = document.body.classList.contains('light-mode');
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
}

// Load Theme on Startup
document.addEventListener('DOMContentLoaded', () => {
    if (localStorage.getItem('theme') === 'light') {
        document.body.classList.add('light-mode');
    }
});

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
    test:      "test-section",
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
  if (name === "test")    loadDatasetSummary();
}


// ── Live Price Preview ─────────────────────────────────────────────

let CONFIG = null;   // filled from GET /api/providers

/**
 * Load providers / tiers / algorithm availability from the backend.
 */
async function loadConfig() {
  try {
    const res = await fetch("/api/providers");
    CONFIG = await res.json();
  } catch (err) {
    console.error("[Config] Error:", err);
  }
  updatePricePreview();
  updateAlgoWarning();
}

/**
 * Show the list (on-demand) monthly price for the selected provider,
 * tier and quantity — same formula as the backend (hourly x 730 x qty).
 */
function updatePricePreview() {
  const out = document.getElementById("base-price-display");
  if (!CONFIG) { out.textContent = "$--"; return; }

  const provider = document.getElementById("provider").value;
  const tier     = document.getElementById("tier").value;
  const qty      = parseInt(document.getElementById("quantity").value) || 0;
  const fmt = v => `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  const listFor = name => CONFIG.providers[name].pricing[tier].on_demand * CONFIG.hours_per_month * qty;

  if (provider === "auto") {                       // show the range across providers
    const prices = Object.keys(CONFIG.providers).map(listFor);
    out.textContent = `${fmt(Math.min(...prices))} – ${fmt(Math.max(...prices))}`;
  } else {
    out.textContent = fmt(listFor(provider));
  }
}

/** Warn when the chosen algorithm is unavailable or untrained. */
function updateAlgoWarning() {
  const box  = document.getElementById("algo-warning");
  const algo = document.getElementById("algorithm").value;
  if (!CONFIG) return;
  const info = CONFIG.algorithms[algo];
  let msg = "";
  if (!info.available)            msg = algo === "dqn" ? "DQN needs PyTorch: pip install torch" : "Agent unavailable.";
  else if (info.trained_episodes === 0) msg = "Not trained yet — run python train.py or use the Training tab.";
  box.textContent = msg;
  box.classList.toggle("hidden", !msg);
}

document.addEventListener("DOMContentLoaded", () => {
  ["provider", "tier", "quantity"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", updatePricePreview);
  });
  document.getElementById("algorithm").addEventListener("change", updateAlgoWarning);
  updateAlgoInfo();
  loadConfig();
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
    provider:      document.getElementById("provider").value,
    tier:          document.getElementById("tier").value,
    quantity:      parseInt(document.getElementById("quantity").value),
    budget:        parseFloat(document.getElementById("budget").value),
    algorithm:     document.getElementById("algorithm").value,
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
    loadConfig();   // refresh trained-episode info

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
    badge.textContent = data.auto
      ? `★ Agent selected ${data.provider} — deal in ${data.rounds} Round${data.rounds !== 1 ? "s" : ""}`
      : `✓ Deal Reached in ${data.rounds} Round${data.rounds !== 1 ? "s" : ""}`;
  } else {
    badge.className = "outcome-badge failure";
    badge.textContent = data.auto
      ? "✗ No provider could close a deal within your budget"
      : `✗ Negotiation Failed after ${data.rounds} Round${data.rounds !== 1 ? "s" : ""}`;
  }

  renderQuotes(data);

  // ── Metrics ─────────────────────────────────────────────────────
  const money = v => `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  const algoName = data.algorithm === "dqn" ? "DQN" : "Q-Learning";

  document.getElementById("r-price").textContent  = data.success ? money(data.final_price) : "—";
  document.getElementById("r-budget").textContent = money(data.budget);
  document.getElementById("r-rounds").textContent = `${data.rounds} / ${data.max_rounds}`;
  document.getElementById("r-reward").textContent = data.total_reward.toFixed(1);
  document.getElementById("r-cpu").textContent    =
    `${data.quantity} × ${data.tier} (${data.final_cpu} vCPU / ${data.final_ram} GB)`;
  document.getElementById("r-ram").textContent    = `${data.provider} · ${algoName}`;

  // ── Savings bar (vs list price) ──────────────────────────────────
  const savingsBar = document.getElementById("savings-bar");
  if (data.success && data.list_price > 0) {
    const saved = data.list_price - data.final_price;
    const pct   = Math.max(0, Math.min((saved / data.list_price) * 100, 100));
    document.getElementById("savings-fill").style.width   = `${pct}%`;
    document.getElementById("savings-amount").textContent =
      `Saved ${money(saved)} (${pct.toFixed(0)}% below list price ${money(data.list_price)})`;
    savingsBar.classList.remove("hidden");
  } else {
    savingsBar.classList.add("hidden");
  }

  // ── Log ─────────────────────────────────────────────────────────
  renderLog(data.log || []);
  if (data.warning) alert(data.warning);
}

let LAST_QUOTES = [];

/**
 * Auto mode: table with every provider's outcome; winner is starred.
 */
function renderQuotes(data) {
  const section = document.getElementById("quotes-section");
  if (!data.auto || !data.quotes) { section.classList.add("hidden"); LAST_QUOTES = []; return; }

  LAST_QUOTES = data.quotes;
  const money = v => `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  document.getElementById("quotes-table").innerHTML =
    `<thead><tr><th></th><th>Provider</th><th>Result</th><th>Tier</th><th>Final price</th><th>vs list</th><th>Rounds</th></tr></thead><tbody>` +
    data.quotes.map((q, i) => `
      <tr style="cursor:pointer" onclick="showQuoteLog(${i})">
        <td>${q.selected ? "★" : ""}</td>
        <td><strong>${q.provider}</strong></td>
        <td>${q.success ? "✓ Deal" : "✗ Failed"}</td>
        <td>${q.quantity} × ${q.tier}${q.downgrades ? ` (from ${q.requested_tier})` : ""}</td>
        <td>${q.success ? money(q.final_price) : "—"}</td>
        <td>${q.success && q.savings_pct !== null ? q.savings_pct + "%" : "—"}</td>
        <td>${q.rounds} / ${q.max_rounds}</td>
      </tr>`).join("") + "</tbody>";
  section.classList.remove("hidden");
}

function showQuoteLog(i) {
  if (LAST_QUOTES[i]) renderLog(LAST_QUOTES[i].log || []);
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
    } else if (/Deal closed|accepted/i.test(line)) {
      div.classList.add("success");
    } else if (/failed|rejected|patience|No deal|exceeds/i.test(line)) {
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
  const episodes  = parseInt(document.getElementById("train-episodes").value);
  const algorithm = document.getElementById("train-algorithm").value;

  // Show loading
  btn.disabled          = true;
  btnText.textContent   = `Training ${episodes} episodes...`;
  spinner.classList.remove("hidden");

  try {
    const response = await fetch("/api/train", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ episodes, algorithm }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || `Server error ${response.status}`);
    }

    // Show results
    document.getElementById("train-empty").classList.add("hidden");
    document.getElementById("train-results").classList.remove("hidden");

    document.getElementById("t-episodes").textContent = data.episodes_trained.toLocaleString();
    if (document.getElementById("t-total-episodes")) {
      document.getElementById("t-total-episodes").textContent = data.total_episodes.toLocaleString();
    }
    if (document.getElementById("t-train-success")) {
      document.getElementById("t-train-success").textContent = `${data.training_success_rate ?? data.success_rate}%`;
    }
    document.getElementById("t-success").textContent  = `${data.eval_success_rate ?? data.success_rate}%`;
    if (document.getElementById("t-feas-success")) {
      document.getElementById("t-feas-success").textContent = `${data.eval_feasible_success_rate ?? "—"}%`;
    }
    document.getElementById("t-reward").textContent   = (data.eval_avg_reward ?? data.avg_reward).toFixed(3);
    document.getElementById("t-states").textContent   = data.knowledge_size.toLocaleString();
    document.getElementById("t-states-label").textContent = data.knowledge_label;
    document.getElementById("t-saved-msg").textContent =
      `${algorithm === "dqn" ? "DQN model saved to dqn_model.pt" : "Q-table saved to q_table.pkl"} ` +
      `(${data.total_episodes.toLocaleString()} cumulative episodes, ${data.seconds}s).`;
    loadConfig();
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

const ALGO_INFO = {
  qlearning: ["How Q-Learning Works", [
    "Observe a discretised state: (provider, tier, price, budget, gap, patience)",
    "Pick an action via ε-greedy: explore or exploit",
    "Receive reward: savings vs list price + bonus, −5 on failure, −0.3 per round",
    "Update the table: Q(s,a) ← Q(s,a) + α[r + γ·maxQ(s') − Q(s,a)]",
    "Epsilon decays over time → more exploitation"]],
  dqn: ["How the DQN Works", [
    "Observe a continuous state vector (provider one-hot, tier, price ratios, patience…)",
    "A neural network estimates Q-values for all 4 actions",
    "Store each transition in the experience replay buffer",
    "Train on random mini-batches against a slowly-updated target network",
    "Epsilon decays over time → more exploitation"]],
};

function updateAlgoInfo() {
  const algo = document.getElementById("train-algorithm").value;
  const [title, steps] = ALGO_INFO[algo];
  document.getElementById("algo-title").textContent = title;
  document.getElementById("algo-steps").innerHTML = steps.map((t, i) =>
    `<div class="algo-step"><span class="algo-num">${i + 1}</span><span>${t}</span></div>`).join("");
}

// ── History ────────────────────────────────────────────────────────

/**
 * GET /api/history and render results into the data table.
 */
async function loadHistory() {
  const tbody   = document.getElementById("history-tbody");
  const table   = document.getElementById("history-table");
  const empty   = document.getElementById("history-empty");

  tbody.innerHTML = '<tr><td colspan="11" style="text-align:center; color:var(--text-muted); padding:2rem;">Loading…</td></tr>';
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
          <td>${s.provider}${s.auto_mode && s.selected ? " ★" : ""}</td>
          <td>${s.provider === "Legacy" ? "—" : s.auto_mode ? (s.selected ? "Auto (chosen)" : "Auto (quote)") : "Manual"}</td>
          <td>${s.algorithm === "dqn" ? "DQN" : "Q-Learning"}</td>
          <td>${s.provider === "Legacy" ? `${s.final_cpu} cores / ${s.final_ram} GB` : `${s.quantity} × ${s.tier}`}</td>
          <td>$${s.budget.toFixed(2)}</td>
          <td>${price}</td>
          <td>${s.rounds}</td>
          <td>${s.total_reward.toFixed(1)}</td>
          <td>${ts}</td>
        </tr>
      `;
    }).join("");

  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="11" style="text-align:center; color:var(--accent-red); padding:1rem;">${err.message}</td></tr>`;
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

    const table = (id, rows, label) => {
      document.getElementById(id).innerHTML =
        `<thead><tr><th>${label}</th><th>Sessions</th><th>Success rate</th><th>Avg rounds</th><th>Avg savings vs list</th><th>Chosen by agent</th></tr></thead><tbody>` +
        rows.map(r => `<tr><td>${r.name === "dqn" ? "DQN" : r.name === "qlearning" ? "Q-Learning" : r.name}</td><td>${r.total}</td>` +
          `<td>${r.success_rate}%</td><td>${r.avg_rounds}</td><td>${r.avg_savings_pct}%</td><td>${r.auto_wins}</td></tr>`).join("") + "</tbody>";
    };
    table("by-provider-table", data.by_provider, "Provider");
    table("by-algo-table", data.by_algorithm, "Algorithm");

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

// ── Testing & Benchmark ────────────────────────────────────────────

let DATASET_SUMMARY = null;

async function loadDatasetSummary() {
  const container = document.getElementById("dataset-summary-text");
  if (!container) return;
  try {
    const res = await fetch("/api/dataset/summary");
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to load dataset summary");
    DATASET_SUMMARY = data;

    const split = document.getElementById("test-split")?.value || "test";
    const current = data[split] || {};
    const total = current.total || 0;
    const ceiling = current.feasible_rate || 0;
    const diffs = current.difficulties || {};

    container.innerHTML = `
      <div><strong>${split.toUpperCase()} Split:</strong> ${total.toLocaleString()} scenarios (${ceiling}% feasible ceiling)</div>
      <div style="margin-top:0.3rem;"><strong>Difficulty:</strong> Easy: ${diffs.easy || 0} | Med: ${diffs.medium || 0} | Hard: ${diffs.hard || 0} | Exp: ${diffs.expert || 0}</div>
      <div style="margin-top:0.3rem;"><strong>Total Dataset:</strong> 6,000 scenarios stratified 70% Train / 15% Val / 15% Test</div>
    `;
  } catch (err) {
    container.innerHTML = `<span style="color:var(--accent-red)">${err.message}</span>`;
  }
}

async function handleEvaluate() {
  const btn = document.getElementById("test-btn");
  const btnText = document.getElementById("test-btn-text");
  const spinner = document.getElementById("test-spinner");
  const algorithm = document.getElementById("test-algorithm").value;
  const split = document.getElementById("test-split").value;
  const limit = document.getElementById("test-limit").value;

  btn.disabled = true;
  btnText.textContent = "Evaluating...";
  spinner.classList.remove("hidden");

  try {
    const payload = { algorithm, split };
    if (limit) payload.limit = parseInt(limit);

    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);

    const empty = document.getElementById("test-empty");
    const results = document.getElementById("test-results");
    if (empty) empty.classList.add("hidden");
    if (results) results.classList.remove("hidden");

    const ov = data.overall || {};
    const algoName = algorithm === "dqn" ? "DQN" : "Q-Learning";
    const splitName = split === "test" ? "Test (Unseen)" : "Validation";

    document.getElementById("test-results-subtitle").textContent =
      `${algoName} evaluated on ${ov.episodes} scenarios from ${splitName} split`;

    document.getElementById("test-success-rate").textContent = `${ov.success_rate}%`;
    document.getElementById("test-ceiling").textContent = `${ov.feasible_ceiling}%`;
    document.getElementById("test-feasible-rate").textContent = `${ov.feasible_success_rate}%`;
    document.getElementById("test-avg-reward").textContent = (ov.avg_reward >= 0 ? "+" : "") + ov.avg_reward.toFixed(2);
    document.getElementById("test-savings").textContent = `${ov.avg_savings_pct}%`;
    document.getElementById("test-tier-kept").textContent = `${ov.tier_kept_pct}%`;

    document.getElementById("test-headline-text").textContent =
      `${algoName} achieved ${ov.success_rate}% success (${ov.feasible_success_rate}% of the ${ov.feasible_ceiling}% achievable ceiling).`;

    // Provider table
    const provTable = document.getElementById("test-provider-table");
    if (provTable && data.by_provider) {
      provTable.innerHTML = `
        <thead>
          <tr>
            <th>Provider</th>
            <th>Scenarios</th>
            <th>Success %</th>
            <th>Achievable Ceiling %</th>
            <th>Avg Reward</th>
            <th>Avg Savings %</th>
            <th>Tier Kept %</th>
          </tr>
        </thead>
        <tbody>
          ${Object.entries(data.by_provider).map(([p, s]) => `
            <tr>
              <td><strong>${p}</strong></td>
              <td>${s.episodes}</td>
              <td>${s.success_rate}%</td>
              <td>${s.feasible_ceiling}%</td>
              <td>${(s.avg_reward >= 0 ? "+" : "") + s.avg_reward.toFixed(2)}</td>
              <td>${s.avg_savings_pct}%</td>
              <td>${s.tier_kept_pct}%</td>
            </tr>
          `).join("")}
        </tbody>
      `;
    }

    // Auto summary
    const autoSummary = document.getElementById("test-auto-summary");
    if (autoSummary && data.auto) {
      const a = data.auto;
      const winsStr = Object.entries(a.wins || {}).map(([p, count]) => `${p}: ${count}`).join(" · ");
      autoSummary.innerHTML = `
        <div><strong>Multi-Provider Success:</strong> ${a.success_rate}% (vs ${a.own_provider_success_rate}% if locked to a single provider)</div>
        <div style="margin-top:0.3rem;"><strong>Provider Wins:</strong> ${winsStr}</div>
        <div style="margin-top:0.3rem;"><strong>Average Savings:</strong> ${a.avg_savings_pct}% &nbsp;|&nbsp; <strong>Tier Maintained:</strong> ${a.tier_kept_pct}%</div>
      `;
    }

    // Failure reasons table
    const failuresTable = document.getElementById("test-failures-table");
    if (failuresTable) {
      const fails = data.failures || {};
      const failEntries = Object.entries(fails);
      if (failEntries.length === 0) {
        failuresTable.innerHTML = `<tbody><tr><td colspan="2" style="color:var(--text-muted); text-align:center;">No failures recorded in this run!</td></tr></tbody>`;
      } else {
        failuresTable.innerHTML = `
          <thead>
            <tr>
              <th>Termination Reason</th>
              <th>Occurrences</th>
            </tr>
          </thead>
          <tbody>
            ${failEntries.map(([reason, count]) => `
              <tr>
                <td>${reason}</td>
                <td>${count}</td>
              </tr>
            `).join("")}
          </tbody>
        `;
      }
    }

  } catch (err) {
    alert(`Evaluation failed: ${err.message}`);
    console.error("[Evaluate] Error:", err);
  } finally {
    btn.disabled = false;
    btnText.textContent = "▶ Run Evaluation";
    spinner.classList.add("hidden");
  }
}

