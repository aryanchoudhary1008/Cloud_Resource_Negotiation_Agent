"""
app.py
------
Flask REST API for the Cloud Resource Negotiation Agent.

Run this file to start the web server:
    python app.py

Then open http://127.0.0.1:5000 in your browser.

Routes:
    GET  /              → Serve single-page UI (index.html)
    POST /api/negotiate → Run a negotiation, return JSON result
    GET  /api/history   → Return past sessions (newest first)
    GET  /api/stats     → Return aggregate metrics
    POST /api/train     → Run N training episodes, update Q-table
    GET  /api/graph     → Return base64 PNG of training graphs (bonus)
"""

import random
from flask import Flask, request, jsonify, render_template

from agents.user_agent       import UserAgent
from agents.provider_agent   import ProviderAgent
from models.negotiation      import NegotiationSession
from database.db             import init_db, save_session, get_all_sessions, get_stats
from rl.state_space          import CPU_BINS, RAM_BINS

# ── App Setup ────────────────────────────────────────────────────────

app = Flask(__name__)

# Initialize SQLite database (creates file + table if they don't exist)
init_db()

# Global user agent instance — loads Q-table from disk at startup
# training_mode=False → Q-table is NOT updated during regular negotiations
user_agent = UserAgent(training_mode=False)


# ── HTML Route ───────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main single-page application UI."""
    return render_template("index.html")


# ── API: Run Negotiation ─────────────────────────────────────────────

@app.route("/api/negotiate", methods=["POST"])
def negotiate():
    """
    Run a complete cloud resource negotiation session.

    Request body (JSON):
        cpu           (int):            CPU cores [2, 4, 8, 16]
        ram           (int):            RAM in GB [4, 8, 16, 32]
        storage       (int, optional):  Storage in GB (default: 100)
        budget        (float):          Maximum budget in USD
        training_mode (bool, optional): If true, updates Q-table live

    Response (JSON):
        {
            "success":      bool,
            "rounds":       int,
            "final_price":  float,
            "final_cpu":    int,
            "final_ram":    int,
            "storage":      int,
            "budget":       float,
            "total_reward": float,
            "log":          [str, ...]
        }

    HTTP 400 if required fields are missing or invalid.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    # Validate and extract inputs
    try:
        cpu           = int(data["cpu"])
        ram           = int(data["ram"])
        storage       = int(data.get("storage", 100))
        budget        = float(data["budget"])
        training_mode = bool(data.get("training_mode", False))
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    # Basic range checks
    if cpu not in CPU_BINS:
        return jsonify({"error": f"cpu must be one of {CPU_BINS}"}), 400
    if ram not in RAM_BINS:
        return jsonify({"error": f"ram must be one of {RAM_BINS}"}), 400
    if budget <= 0:
        return jsonify({"error": "budget must be positive"}), 400
    if storage <= 0:
        return jsonify({"error": "storage must be positive"}), 400

    # Set training mode on the global agent
    user_agent.set_training_mode(training_mode)

    # Run the negotiation
    result = user_agent.negotiate(cpu, ram, storage, budget)

    # Persist session to SQLite
    session = NegotiationSession(
        cpu          = cpu,
        ram          = ram,
        storage      = storage,
        budget       = budget,
        final_price  = result.final_price,
        final_cpu    = result.final_cpu,
        final_ram    = result.final_ram,
        rounds       = result.rounds,
        success      = result.success,
        total_reward = result.total_reward,
    )
    session.id = save_session(session)

    # If live training was on, persist updated Q-table
    if training_mode:
        user_agent.save_q_table()

    return jsonify(result.to_dict())


# ── API: History ─────────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def history():
    """
    Return past negotiation sessions, newest first.

    Query parameters:
        limit (int, optional): Max results to return (default: 50).

    Response: JSON array of session objects.
    """
    limit    = request.args.get("limit", 50, type=int)
    sessions = get_all_sessions()[:limit]
    return jsonify([s.to_dict() for s in sessions])


# ── API: Stats ───────────────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def stats():
    """
    Return aggregate statistics across all recorded sessions.

    Response (JSON):
        {
            "total_sessions":        int,
            "successful":            int,
            "failed":                int,
            "success_rate":          float  (percent),
            "avg_rounds":            float,
            "avg_cost_efficiency":   float  (USD under budget on average),
            "resource_satisfaction": float  (percent with no downgrade)
        }
    """
    return jsonify(get_stats())


# ── API: Train ───────────────────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def train():
    """
    Run N training episodes and save the updated Q-table.

    Each episode uses a randomly generated resource/budget scenario
    so the agent learns to handle a wide variety of inputs.

    Request body (JSON, all optional):
        episodes (int): Number of training episodes (default: 500).

    Response (JSON):
        {
            "episodes_trained":     int,
            "success_rate":         float (%),
            "avg_reward":           float,
            "total_states_learned": int,
            "current_epsilon":      float
        }
    """
    data     = request.get_json(silent=True) or {}
    episodes = int(data.get("episodes", 500))

    # Clamp to a sensible range so the API doesn't hang
    episodes = max(1, min(episodes, 5000))

    # Create a fresh training agent (loads existing Q-table to continue learning)
    trainer = UserAgent(training_mode=True, q_table_path="q_table.pkl")

    successes    = 0
    total_reward = 0.0

    for ep in range(episodes):
        # Randomize scenario to maximize coverage of the state space
        cpu     = random.choice(CPU_BINS)
        ram     = random.choice(RAM_BINS)
        storage = random.randint(50, 500)
        budget  = random.uniform(10, 150)

        result = trainer.negotiate(cpu, ram, storage, budget)

        if result.success:
            successes += 1
        total_reward += result.total_reward

        # Decay exploration rate gradually
        trainer.ql_agent.decay_epsilon(decay_rate=0.995, min_epsilon=0.05)

    # Append a batch-average reward for the graph endpoint
    if episodes > 0:
        trainer.ql_agent.training_rewards.append(
            round(total_reward / episodes, 3)
        )
    trainer.ql_agent.episode_count += episodes

    # Persist Q-table
    trainer.save_q_table()

    # Reload updated Q-table into the global inference agent
    user_agent.ql_agent.load("q_table.pkl")

    return jsonify({
        "episodes_trained":     episodes,
        "success_rate":         round(successes / episodes * 100, 1),
        "avg_reward":           round(total_reward / episodes, 3),
        "total_states_learned": len(trainer.ql_agent.q_table),
        "current_epsilon":      round(trainer.ql_agent.epsilon, 4),
    })


# ── API: Graph (Bonus) ───────────────────────────────────────────────

@app.route("/api/graph", methods=["GET"])
def graph():
    """
    Generate and return training/session graphs as a base64 PNG.

    Chart 1: Line chart of average reward per training batch.
    Chart 2: Bar chart of success vs failure for last 20 sessions.

    Response (JSON):
        { "image": "<base64-encoded PNG string>" }

    Returns HTTP 500 if matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — no GUI window
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
    except ImportError:
        return jsonify({
            "error": "matplotlib not installed. Run: pip install matplotlib"
        }), 500

    # ── Set up figure with dark theme ────────────────────────────────
    BG_DARK   = "#0f172a"
    BG_CARD   = "#1e293b"
    BORDER    = "#334155"
    TEXT      = "#e2e8f0"
    BLUE      = "#38bdf8"
    GREEN     = "#22c55e"
    RED       = "#ef4444"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(BG_DARK)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG_CARD)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    # ── Chart 1: Training reward history ─────────────────────────────
    rewards = user_agent.ql_agent.training_rewards
    if rewards:
        ax1.plot(rewards, color=BLUE, linewidth=2, marker="o",
                 markersize=4, markerfacecolor=BLUE)
        ax1.fill_between(range(len(rewards)), rewards,
                         alpha=0.15, color=BLUE)
        ax1.set_title("Avg Reward per Training Batch", fontsize=11, pad=10)
        ax1.set_xlabel("Training Batch", fontsize=9)
        ax1.set_ylabel("Avg Reward", fontsize=9)
        ax1.grid(True, color=BORDER, alpha=0.5, linestyle="--")
    else:
        ax1.text(0.5, 0.5, "No training data yet.\nRun training first.",
                 ha="center", va="center", color=TEXT, fontsize=11,
                 transform=ax1.transAxes)
        ax1.set_title("Training Reward History", fontsize=11, pad=10)

    # ── Chart 2: Recent session outcomes ─────────────────────────────
    recent   = get_all_sessions()[:20]
    if recent:
        n_success = sum(1 for s in recent if s.success)
        n_fail    = len(recent) - n_success
        bars = ax2.bar(
            ["Success", "Failure"],
            [n_success, n_fail],
            color=[GREEN, RED],
            width=0.45,
            edgecolor=BORDER,
        )
        ax2.set_title("Last 20 Sessions: Outcomes", fontsize=11, pad=10)
        ax2.set_ylabel("Count", fontsize=9)
        # Value labels on top of bars
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2, h + 0.15,
                str(int(h)),
                ha="center", va="bottom", color=TEXT,
                fontsize=10, fontweight="bold",
            )
        ax2.grid(True, axis="y", color=BORDER, alpha=0.5, linestyle="--")
    else:
        ax2.text(0.5, 0.5, "No sessions yet.\nRun a negotiation first.",
                 ha="center", va="center", color=TEXT, fontsize=11,
                 transform=ax2.transAxes)
        ax2.set_title("Session Outcomes (Last 20)", fontsize=11, pad=10)

    plt.tight_layout(pad=2.0)

    # ── Encode as base64 PNG ──────────────────────────────────────────
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=120)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return jsonify({"image": img_b64})


# ── Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Cloud Resource Negotiation Agent")
    print("  http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host="127.0.0.1", port=5000)
