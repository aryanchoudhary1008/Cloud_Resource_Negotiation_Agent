"""
app.py
------
Flask REST API for the Cloud Resource Negotiation Agent.

    python app.py     →  http://127.0.0.1:5000

Routes
    GET  /               single-page UI
    GET  /api/providers  providers, tiers, prices, algorithm availability
    POST /api/negotiate  run a negotiation (provider + algorithm selectable)
    GET  /api/history    past sessions, newest first
    GET  /api/stats      aggregate + per-provider + per-algorithm statistics
    POST /api/train      train Q-learning or DQN for N episodes
    GET  /api/benchmark  latest results written by train.py
    GET  /api/graph      learning-curve + provider-comparison charts (PNG)
"""

import importlib.util
import json
import os
import time
import uuid

from flask import Flask, request, jsonify, render_template

from config import BENCHMARK_JSON, ALGORITHMS
from agents.user_agent     import UserAgent
from agents.provider_agent import (load_config, provider_names, tier_names,
                                   tier_specs, hours_per_month, ProviderAgent)
from models.negotiation    import NegotiationSession
from database.db           import init_db, save_session, get_all_sessions, get_stats
from rl.trainer            import train_agent, evaluate_agent, evaluate_auto
from rl.dataset            import load_split, dataset_summary, feasible_rate

app = Flask(__name__)
init_db()

# One inference agent per algorithm, created lazily
_agents = {}


def torch_available():
    return importlib.util.find_spec("torch") is not None


def get_agent(algorithm):
    """Return the shared UserAgent for `algorithm` (raises ImportError if DQN lacks torch)."""
    if algorithm not in _agents:
        _agents[algorithm] = UserAgent(algorithm=algorithm, training_mode=False)
    return _agents[algorithm]


def algorithm_info():
    info = {}
    for algo in ALGORITHMS:
        available = algo == "qlearning" or torch_available()
        trained = 0
        if available:
            try:
                trained = get_agent(algo).agent.episode_count
            except Exception:
                available = False
        info[algo] = {"available": available, "trained_episodes": trained}
    return info


@app.route("/")
def index():
    return render_template("index.html")


# ── API: providers / config ──────────────────────────────────────────

@app.route("/api/providers", methods=["GET"])
def providers():
    cfg = load_config()
    return jsonify({
        "hours_per_month": hours_per_month(),
        "tiers":           tier_specs(),
        "providers": {
            name: {
                "full_name":       p["full_name"],
                "region":          p["region"],
                "style":           p["style"],
                "pricing":         p["pricing"],
                "max_rounds_range": p["max_rounds_range"],
                "volume_discount_tiers": p["volume_discount_tiers"],
            }
            for name, p in cfg["providers"].items()
        },
        "algorithms": algorithm_info(),
    })


# ── API: negotiate ───────────────────────────────────────────────────

@app.route("/api/negotiate", methods=["POST"])
def negotiate():
    """
    Body (JSON):
        provider      'auto' (default — the agent negotiates with every provider
                      and picks the best deal) | 'AWS' | 'GCP' | 'Azure'
        tier          'Small' | 'Medium' | 'Large'
        quantity      int >= 1
        budget        float, max total USD / month
        algorithm     'qlearning' (default) | 'dqn'
        training_mode bool (optional) — learn live during this session
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        provider      = str(data.get("provider", "auto"))
        tier          = str(data["tier"])
        quantity      = int(data["quantity"])
        budget        = float(data["budget"])
        algorithm     = str(data.get("algorithm", "qlearning"))
        training_mode = bool(data.get("training_mode", False))
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    auto = provider.lower() == "auto"
    if not auto and provider not in provider_names():
        return jsonify({"error": f"provider must be 'auto' or one of {provider_names()}"}), 400
    if tier not in tier_names():
        return jsonify({"error": f"tier must be one of {tier_names()}"}), 400
    if algorithm not in ALGORITHMS:
        return jsonify({"error": f"algorithm must be one of {list(ALGORITHMS)}"}), 400
    if not 1 <= quantity <= 10000:
        return jsonify({"error": "quantity must be between 1 and 10000"}), 400
    if budget <= 0:
        return jsonify({"error": "budget must be positive"}), 400

    try:
        agent = get_agent(algorithm)
    except ImportError:
        return jsonify({"error": "DQN needs PyTorch. Run: pip install torch"}), 503

    agent.set_training_mode(training_mode)
    request_id = uuid.uuid4().hex[:12] if auto else ""

    if auto:
        winner, results = agent.negotiate_best(tier, quantity, budget)
    else:
        winner = agent.negotiate(provider, tier, quantity, budget)
        results = [winner]
    agent.set_training_mode(False)

    # Save one history row per negotiation; the winner is flagged `selected`
    session_ids = {}
    for r in results:
        session = NegotiationSession(
            cpu=r.requested_cpu, ram=r.requested_ram, storage=0,
            budget=budget, final_price=r.final_price,
            final_cpu=r.final_cpu, final_ram=r.final_ram,
            rounds=r.rounds, success=r.success, total_reward=r.total_reward,
            provider=r.provider, algorithm=algorithm, tier=tier,
            quantity=quantity, list_price=r.list_price,
            request_id=request_id, auto_mode=auto, selected=(r is winner),
        )
        session_ids[r.provider] = save_session(session)

    if training_mode:
        agent.save()

    payload = winner.to_dict()
    payload["session_id"]       = session_ids[winner.provider]
    payload["trained_episodes"] = agent.agent.episode_count
    payload["auto"]             = auto
    if auto:
        payload["request_id"]        = request_id
        payload["selected_provider"] = winner.provider if winner.success else None
        payload["quotes"] = [
            {**r.to_dict(), "selected": r is winner and r.success}
            for r in results
        ]
    if agent.agent.episode_count == 0:
        payload["warning"] = (f"The {algorithm} agent has not been trained yet — "
                              "run `python train.py` or use the Training tab.")
    return jsonify(payload)


# ── API: history / stats ─────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def history():
    limit = request.args.get("limit", 50, type=int)
    return jsonify([s.to_dict() for s in get_all_sessions()[:limit]])


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(get_stats())


# ── API: train ───────────────────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def train():
    """
    Body (JSON, all optional):
        algorithm  'qlearning' (default) | 'dqn'
        episodes   int, 1..5000 (default 500)
    """
    data      = request.get_json(silent=True) or {}
    algorithm = str(data.get("algorithm", "qlearning"))
    episodes  = max(1, min(int(data.get("episodes", 500)), 5000))

    if algorithm not in ALGORITHMS:
        return jsonify({"error": f"algorithm must be one of {list(ALGORITHMS)}"}), 400

    try:
        trainer = UserAgent(algorithm=algorithm, training_mode=True)
    except ImportError:
        return jsonify({"error": "DQN needs PyTorch. Run: pip install torch"}), 503

    start   = time.time()
    history = train_agent(trainer, episodes, batch=max(1, episodes // 10), verbose=False)
    trainer.save()

    # refresh the shared inference agent
    if algorithm in _agents:
        _agents[algorithm].reload()

    # Measure performance from the converged exploitation phase (final half of training)
    # so the dashboard reflects the trained policy rather than initial exploratory random actions
    converged = history[len(history) // 2:] if len(history) >= 2 else history
    success_rate = round(sum(h["success_rate"] for h in converged) / len(converged), 1)
    avg_reward   = round(sum(h["avg_reward"] for h in converged) / len(converged), 3)

    return jsonify({
        "algorithm":       algorithm,
        "episodes_trained": episodes,
        "success_rate":    success_rate,
        "avg_reward":      avg_reward,
        "knowledge_size":  trainer.agent.knowledge_size,
        "knowledge_label": trainer.agent.knowledge_label,
        "current_epsilon": round(trainer.agent.epsilon, 4),
        "total_episodes":  trainer.agent.episode_count,
        "seconds":         round(time.time() - start, 1),
    })



# ── API: benchmark ───────────────────────────────────────────────────

@app.route("/api/benchmark", methods=["GET"])
def benchmark():
    """Latest Q-learning vs DQN results written by `python train.py`."""
    if not os.path.exists(BENCHMARK_JSON):
        return jsonify({"error": "No benchmark yet. Run: python train.py"}), 404
    with open(BENCHMARK_JSON) as f:
        return jsonify(json.load(f))


# ── API: dataset & evaluation ────────────────────────────────────────

@app.route("/api/dataset/summary", methods=["GET"])
def get_dataset_summary():
    """Row counts, feasibility ceiling, difficulty and provider breakdowns."""
    try:
        return jsonify(dataset_summary())
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """
    Greedy evaluation on a dataset split.
    Body (JSON, optional):
        algorithm  'dqn' | 'qlearning' (default 'dqn')
        split      'test' | 'val' (default 'test')
        limit      int (optional subset limit)
    """
    d = request.get_json(silent=True) or {}
    algo  = str(d.get("algorithm", "dqn")).lower()
    split = str(d.get("split", "test")).lower()

    if algo not in ALGORITHMS:
        return jsonify({"error": f"algorithm must be one of {list(ALGORITHMS)}"}), 400
    if split not in ("test", "val", "train"):
        return jsonify({"error": "split must be 'test' or 'val'"}), 400

    try:
        agent = get_agent(algo)
    except ImportError:
        return jsonify({"error": "DQN needs PyTorch. Run: pip install torch"}), 503
    except Exception as ex:
        return jsonify({"error": f"Failed to initialize agent: {ex}"}), 500

    try:
        scenarios = load_split(split)
        limit = d.get("limit")
        if limit:
            scenarios = scenarios[:max(1, int(limit))]
        ev   = evaluate_agent(agent, scenarios)
        auto = evaluate_auto(agent, scenarios)
        ev["auto"] = auto
        ev["algorithm"] = algo
        ev["split"] = split
        return jsonify(ev)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500


@app.route("/api/validate", methods=["POST"])
def validate():
    """
    Run validation on saved models for Q-Learning and/or DQN against data/val.csv.
    """
    d = request.get_json(silent=True) or {}
    requested_algo = d.get("algorithm")
    algos = [requested_algo] if (requested_algo and requested_algo in ALGORITHMS) else list(ALGORITHMS)

    try:
        val_scenarios = load_split("val")
        limit = d.get("limit")
        if limit:
            val_scenarios = val_scenarios[:max(1, int(limit))]

        results = {}
        for a in algos:
            try:
                ag = get_agent(a)
                results[a] = evaluate_agent(ag, val_scenarios)
            except Exception as ex:
                results[a] = {"error": str(ex)}

        return jsonify({
            "split": "val",
            "ceiling": feasible_rate(val_scenarios),
            "episodes": len(val_scenarios),
            "results": results,
        })
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500



# ── API: graph ───────────────────────────────────────────────────────

@app.route("/api/graph", methods=["GET"])
def graph():
    """
    Chart 1: learning curves (avg reward per batch) for every trained algorithm.
    Chart 2: negotiation success rate per provider (from the history DB).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
    except ImportError:
        return jsonify({"error": "matplotlib not installed. Run: pip install matplotlib"}), 500

    BG_DARK, BG_CARD, BORDER, TEXT = "#0f172a", "#1e293b", "#334155", "#e2e8f0"
    COLORS = {"qlearning": "#38bdf8", "dqn": "#f59e0b"}
    NAMES  = {"qlearning": "Q-Learning", "dqn": "DQN"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(BG_DARK)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG_CARD)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

    # Chart 1 — learning curves
    plotted = False
    for algo in ALGORITHMS:
        if algo == "dqn" and not torch_available():
            continue
        try:
            hist = get_agent(algo).agent.training_history
        except Exception:
            continue
        if hist:
            ax1.plot([h["episode"] for h in hist], [h["avg_reward"] for h in hist],
                     color=COLORS[algo], linewidth=2, marker="o", markersize=4,
                     label=NAMES[algo])
            plotted = True
    if plotted:
        ax1.set_title("Learning curves — avg reward per batch", fontsize=11, pad=10)
        ax1.set_xlabel("Training episode"); ax1.set_ylabel("Avg reward")
        ax1.grid(True, color=BORDER, alpha=0.5, linestyle="--")
        ax1.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT)
    else:
        ax1.text(0.5, 0.5, "No training data yet.\nRun: python train.py", ha="center",
                 va="center", color=TEXT, transform=ax1.transAxes)
        ax1.set_title("Learning curves", fontsize=11, pad=10)

    # Chart 2 — success per provider
    by_provider = [p for p in get_stats()["by_provider"] if p["name"] != "Legacy"]
    if by_provider:
        names = [p["name"] for p in by_provider]
        rates = [p["success_rate"] for p in by_provider]
        bars = ax2.bar(names, rates, color=["#38bdf8", "#22c55e", "#a78bfa"][:len(names)],
                       width=0.5, edgecolor=BORDER)
        ax2.set_ylim(0, 105)
        ax2.set_title("Negotiation success rate by provider", fontsize=11, pad=10)
        ax2.set_ylabel("Success %")
        for bar, p in zip(bars, by_provider):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f"{p['success_rate']}%  (n={p['total']})", ha="center",
                     color=TEXT, fontsize=9)
        ax2.grid(True, axis="y", color=BORDER, alpha=0.5, linestyle="--")
    else:
        ax2.text(0.5, 0.5, "No sessions yet.\nRun a negotiation first.", ha="center",
                 va="center", color=TEXT, transform=ax2.transAxes)
        ax2.set_title("Success rate by provider", fontsize=11, pad=10)

    plt.tight_layout(pad=2.0)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=120)
    plt.close(fig)
    return jsonify({"image": base64.b64encode(buf.getvalue()).decode("utf-8")})


if __name__ == "__main__":
    print("=" * 55)
    print("  Cloud Resource Negotiation Agent")
    print("  http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host="127.0.0.1", port=5000)
