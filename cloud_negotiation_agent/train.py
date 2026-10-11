"""
train.py
--------
Standalone training + benchmarking script.

Usage
    python train.py                         # train BOTH algorithms, compare
    python train.py --algo qlearning        # Q-learning only
    python train.py --algo dqn              # DQN only
    python train.py --episodes 5000 --seed 7
    python train.py --resume                # continue from saved model

What it does
    1. Trains each selected algorithm on the SAME random scenario stream.
    2. Evaluates each one greedily (ε = 0) on the SAME fixed scenario set,
       with a per-provider breakdown.
    3. Saves q_table.pkl / dqn_model.pt.
    4. Writes results/benchmark.json, results/benchmark.md and (when both
       algorithms ran) results/learning_curves.png for the report.
"""

import argparse
import json
import os
import random
import time

import numpy as np

from config import RESULTS_DIR, BENCHMARK_JSON, MODEL_PATHS
from agents.user_agent import UserAgent
from rl.trainer import train_agent, evaluate_agent, evaluate_auto, make_eval_set

LABELS = {"qlearning": "Q-Learning", "dqn": "DQN"}


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def run_one(algo, episodes, eval_set, seed, resume):
    print("\n" + "=" * 62)
    print(f"  Training {LABELS[algo]}  ({episodes} episodes, seed {seed})")
    print("=" * 62)

    set_seeds(seed)
    agent = UserAgent(algorithm=algo, training_mode=True, load=resume)
    if not resume:
        agent.agent.epsilon = 0.5                      # same start for both algorithms

    start = time.time()
    history = train_agent(agent, episodes, seed=seed)
    elapsed = time.time() - start
    agent.save()

    evaluation = evaluate_agent(agent, eval_set)
    ov = evaluation["overall"]
    print(f"\n  Evaluation on {ov['episodes']} unseen scenarios (greedy policy):")
    print(f"    success {ov['success_rate']}%  |  reward {ov['avg_reward']:+.2f}  |  "
          f"rounds {ov['avg_rounds']}  |  savings {ov['avg_savings_pct']}%  |  "
          f"tier kept {ov['tier_kept_pct']}%")
    for p, s in evaluation["by_provider"].items():
        print(f"      {p:<6} success {s['success_rate']:>5}%   reward {s['avg_reward']:>+6.2f}   "
              f"savings {s['avg_savings_pct']:>5}%")
    auto = evaluate_auto(agent, eval_set)
    print(f"  Auto provider selection: success {auto['success_rate']}% "
          f"(vs {auto['own_provider_success_rate']}% locked to one provider)  |  "
          f"tier kept {auto['tier_kept_pct']}%  |  wins {auto['wins']}")
    print(f"  Training time: {elapsed:.1f}s   {agent.agent.knowledge_label}: "
          f"{agent.agent.knowledge_size:,}")

    return {
        "algorithm":      algo,
        "episodes":       episodes,
        "seed":           seed,
        "train_seconds":  round(elapsed, 1),
        "knowledge_label": agent.agent.knowledge_label,
        "knowledge_size": agent.agent.knowledge_size,
        "history":        history,
        "evaluation":     evaluation,
        "auto_selection": auto,
    }


def write_outputs(results, eval_n):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(BENCHMARK_JSON, "w") as f:
        json.dump({"eval_episodes": eval_n, "results": results}, f, indent=2)

    # Markdown table for the report
    lines = ["| Algorithm | Success % | Avg reward | Avg rounds | Avg savings % | Tier kept % | Train time (s) |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        o = r["evaluation"]["overall"]
        lines.append(f"| {LABELS[r['algorithm']]} | {o['success_rate']} | {o['avg_reward']} | "
                     f"{o['avg_rounds']} | {o['avg_savings_pct']} | {o['tier_kept_pct']} | {r['train_seconds']} |")
    lines += ["", "Success rate by provider:", "",
              "| Algorithm | AWS | GCP | Azure |", "|---|---|---|---|"]
    for r in results:
        bp = r["evaluation"]["by_provider"]
        lines.append(f"| {LABELS[r['algorithm']]} | " +
                     " | ".join(f"{bp[p]['success_rate']}" for p in ("AWS", "GCP", "Azure")) + " |")
    lines += ["", "Auto provider selection (best of AWS / GCP / Azure):", "",
              "| Algorithm | Success % (auto) | Success % (single provider) | Tier kept % | AWS wins | GCP wins | Azure wins |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        a = r["auto_selection"]
        lines.append(f"| {LABELS[r['algorithm']]} | {a['success_rate']} | {a['own_provider_success_rate']} | "
                     f"{a['tier_kept_pct']} | {a['wins']['AWS']} | {a['wins']['GCP']} | {a['wins']['Azure']} |")
    with open(os.path.join(RESULTS_DIR, "benchmark.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Learning-curve plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib missing — skipped learning_curves.png)")
        return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        eps = [h["episode"] for h in r["history"]]
        a1.plot(eps, [h["avg_reward"]   for h in r["history"]], marker="o", label=LABELS[r["algorithm"]])
        a2.plot(eps, [h["success_rate"] for h in r["history"]], marker="o", label=LABELS[r["algorithm"]])
    a1.set(title="Learning curve — average reward", xlabel="Training episode", ylabel="Avg reward / batch")
    a2.set(title="Learning curve — success rate",  xlabel="Training episode", ylabel="Success % / batch")
    for ax in (a1, a2):
        ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "learning_curves.png"), dpi=130)
    plt.close(fig)
    print(f"\n  Saved: {BENCHMARK_JSON}, benchmark.md, learning_curves.png")


def main():
    ap = argparse.ArgumentParser(description="Train / benchmark negotiation agents")
    ap.add_argument("--algo", choices=["qlearning", "dqn", "both"], default="both")
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--eval-episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="continue from the saved model")
    args = ap.parse_args()

    algos = ["qlearning", "dqn"] if args.algo == "both" else [args.algo]
    eval_set = make_eval_set(args.eval_episodes)

    results = [run_one(a, args.episodes, eval_set, args.seed, args.resume) for a in algos]
    write_outputs(results, args.eval_episodes)

    print("\n  Done. Next: python app.py  →  http://127.0.0.1:5000\n")


if __name__ == "__main__":
    main()
