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
import importlib.util
import json
import math
import os
import random
import time

import numpy as np

from config import (
    RESULTS_DIR,
    BENCHMARK_JSON,
    MULTISEED_BENCHMARK_JSON,
    MULTISEED_BENCHMARK_MD,
    MODEL_PATHS,
)
from agents.user_agent import UserAgent
from rl.dataset import load_split, feasible_rate
from rl.trainer import train_agent, evaluate_agent, evaluate_auto, make_eval_set

LABELS = {"qlearning": "Q-Learning", "dqn": "DQN"}
DEFAULT_SEEDS = [42, 0, 1, 7, 99]


def torch_available():
    return importlib.util.find_spec("torch") is not None


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def compute_mean_std(values):
    """Compute (mean, std) for a list of numbers; handles None or empty lists."""
    valid = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not valid:
        return None, None
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    return round(mean, 2), round(std, 2)


def format_mean_std(mean, std, pct=False, signed=False):
    if mean is None:
        return "N/A"
    sign_str = "+" if signed and mean > 0 else ""
    pct_str = "%" if pct else ""
    return f"{sign_str}{mean:.2f} ± {std:.2f}{pct_str}"


def run_one(algo, episodes, train_set, val_set, eval_set, seed, resume, curriculum,
            save_model=True, model_path=None):
    print("\n" + "=" * 62)
    print(f"  Training {LABELS[algo]}  ({episodes} episodes, seed {seed}, curriculum={curriculum})")
    print("=" * 62)

    set_seeds(seed)
    agent = UserAgent(algorithm=algo, training_mode=True, model_path=model_path, load=resume)
    if not resume:
        agent.agent.epsilon = 0.5                      # same start for both algorithms

    start = time.time()
    history = train_agent(agent, episodes=episodes, scenarios=train_set,
                          val_scenarios=val_set, seed=seed, curriculum=curriculum,
                          save_best=save_model)
    elapsed = time.time() - start
    if save_model:
        agent.save()

    evaluation = evaluate_agent(agent, eval_set)
    ov = evaluation["overall"]
    print(f"\n  Evaluation on {ov['episodes']} unseen scenarios (greedy policy):")
    print(f"    success {ov['success_rate']}% (achievable ceiling {ov['feasible_ceiling']}%, "
          f"feasible success {ov['feasible_success_rate']}%)")
    print(f"    reward {ov['avg_reward']:+.2f}  |  rounds {ov['avg_rounds']}  |  "
          f"savings {ov['avg_savings_pct']}%  |  tier kept {ov['tier_kept_pct']}%")
    for p, s in evaluation["by_provider"].items():
        print(f"      {p:<6} success {s['success_rate']:>5}% (ceiling {s['feasible_ceiling']:>5}%)  "
              f"reward {s['avg_reward']:>+6.2f}   savings {s['avg_savings_pct']:>5}%")
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


def aggregate_multiseed_results(seed_results):
    """Aggregate per-seed results into mean and std for key metrics."""
    metrics_to_collect = {
        "success_rate": [r["evaluation"]["overall"]["success_rate"] for r in seed_results],
        "feasible_success_rate": [r["evaluation"]["overall"]["feasible_success_rate"] for r in seed_results],
        "avg_reward": [r["evaluation"]["overall"]["avg_reward"] for r in seed_results],
        "avg_rounds": [r["evaluation"]["overall"]["avg_rounds"] for r in seed_results],
        "avg_savings_pct": [r["evaluation"]["overall"]["avg_savings_pct"] for r in seed_results],
        "tier_kept_pct": [r["evaluation"]["overall"]["tier_kept_pct"] for r in seed_results],
        "auto_success_rate": [r["auto_selection"]["success_rate"] for r in seed_results],
        "train_seconds": [r["train_seconds"] for r in seed_results],
    }

    # Walk-away accuracy metrics
    wa_precisions = [r["evaluation"].get("walk_away_accuracy", {}).get("precision_pct") for r in seed_results]
    wa_recalls = [r["evaluation"].get("walk_away_accuracy", {}).get("recall_pct") for r in seed_results]
    metrics_to_collect["walkaway_precision"] = wa_precisions
    metrics_to_collect["walkaway_recall"] = wa_recalls

    # By provider
    for p in ("AWS", "GCP", "Azure"):
        metrics_to_collect[f"provider_{p}_success"] = [
            r["evaluation"]["by_provider"][p]["success_rate"] for r in seed_results
        ]
        metrics_to_collect[f"provider_{p}_savings"] = [
            r["evaluation"]["by_provider"][p]["avg_savings_pct"] for r in seed_results
        ]

    # By difficulty
    for d in ("easy", "medium", "hard", "expert"):
        metrics_to_collect[f"diff_{d}_success"] = [
            r["evaluation"].get("by_difficulty", {}).get(d, {}).get("success_rate", 0.0)
            for r in seed_results
        ]

    aggregated = {}
    for k, v in metrics_to_collect.items():
        m, s = compute_mean_std(v)
        aggregated[k] = {"mean": m, "std": s, "values": v}

    return aggregated


def write_multiseed_outputs(multiseed_data, eval_n):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(MULTISEED_BENCHMARK_JSON, "w", encoding="utf-8") as f:
        json.dump({"eval_episodes": eval_n, "multiseed": multiseed_data}, f, indent=2)

    lines = [
        "# Multi-Seed Benchmarking Results",
        "",
        f"Evaluated on held-out test split ({eval_n} scenarios across seeds: {multiseed_data.get('seeds', [])}).",
        "",
        "## Overall Aggregate Metrics (Mean ± Std Dev)",
        "",
        "| Algorithm | Success % | Feasible Success % | Avg Reward | Avg Rounds | Avg Savings % | Tier Kept % | Auto Success % | WA Precision % | WA Recall % |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for algo, data in multiseed_data.get("algorithms", {}).items():
        agg = data["aggregated"]
        s_rate = format_mean_std(agg["success_rate"]["mean"], agg["success_rate"]["std"], pct=True)
        fs_rate = format_mean_std(agg["feasible_success_rate"]["mean"], agg["feasible_success_rate"]["std"], pct=True)
        reward = format_mean_std(agg["avg_reward"]["mean"], agg["avg_reward"]["std"], signed=True)
        rounds = format_mean_std(agg["avg_rounds"]["mean"], agg["avg_rounds"]["std"])
        savings = format_mean_std(agg["avg_savings_pct"]["mean"], agg["avg_savings_pct"]["std"], pct=True)
        tier_k = format_mean_std(agg["tier_kept_pct"]["mean"], agg["tier_kept_pct"]["std"], pct=True)
        auto_s = format_mean_std(agg["auto_success_rate"]["mean"], agg["auto_success_rate"]["std"], pct=True)
        wa_p = format_mean_std(agg["walkaway_precision"]["mean"], agg["walkaway_precision"]["std"], pct=True)
        wa_r = format_mean_std(agg["walkaway_recall"]["mean"], agg["walkaway_recall"]["std"], pct=True)

        lines.append(f"| {LABELS.get(algo, algo)} | {s_rate} | {fs_rate} | {reward} | {rounds} | {savings} | {tier_k} | {auto_s} | {wa_p} | {wa_r} |")

    lines += [
        "",
        "## Provider Success Rate (Mean ± Std Dev)",
        "",
        "| Algorithm | AWS | GCP | Azure |",
        "|---|---|---|---|",
    ]
    for algo, data in multiseed_data.get("algorithms", {}).items():
        agg = data["aggregated"]
        aws_s = format_mean_std(agg["provider_AWS_success"]["mean"], agg["provider_AWS_success"]["std"], pct=True)
        gcp_s = format_mean_std(agg["provider_GCP_success"]["mean"], agg["provider_GCP_success"]["std"], pct=True)
        az_s = format_mean_std(agg["provider_Azure_success"]["mean"], agg["provider_Azure_success"]["std"], pct=True)
        lines.append(f"| {LABELS.get(algo, algo)} | {aws_s} | {gcp_s} | {az_s} |")

    lines += [
        "",
        "## Difficulty Success Rate (Mean ± Std Dev)",
        "",
        "| Algorithm | Easy | Medium | Hard | Expert |",
        "|---|---|---|---|---|",
    ]
    for algo, data in multiseed_data.get("algorithms", {}).items():
        agg = data["aggregated"]
        easy_s = format_mean_std(agg["diff_easy_success"]["mean"], agg["diff_easy_success"]["std"], pct=True)
        med_s = format_mean_std(agg["diff_medium_success"]["mean"], agg["diff_medium_success"]["std"], pct=True)
        hard_s = format_mean_std(agg["diff_hard_success"]["mean"], agg["diff_hard_success"]["std"], pct=True)
        exp_s = format_mean_std(agg["diff_expert_success"]["mean"], agg["diff_expert_success"]["std"], pct=True)
        lines.append(f"| {LABELS.get(algo, algo)} | {easy_s} | {med_s} | {hard_s} | {exp_s} |")

    lines += [
        "",
        "## Per-Seed Breakdown",
        "",
        "| Algorithm | Seed | Success % | Feasible Success % | Avg Reward | Avg Savings % | WA Prec % | WA Rec % | Train Time |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for algo, data in multiseed_data.get("algorithms", {}).items():
        for r in data["seed_results"]:
            seed = r["seed"]
            ov = r["evaluation"]["overall"]
            wa = r["evaluation"].get("walk_away_accuracy", {})
            wa_p_str = f"{wa.get('precision_pct'):.1f}%" if wa.get('precision_pct') is not None else "N/A"
            wa_r_str = f"{wa.get('recall_pct'):.1f}%" if wa.get('recall_pct') is not None else "N/A"
            lines.append(
                f"| {LABELS.get(algo, algo)} | {seed} | {ov['success_rate']:.1f}% | "
                f"{ov['feasible_success_rate']:.1f}% | {ov['avg_reward']:+.2f} | "
                f"{ov['avg_savings_pct']:.1f}% | {wa_p_str} | {wa_r_str} | {r['train_seconds']}s |"
            )

    with open(MULTISEED_BENCHMARK_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Saved multi-seed results: {MULTISEED_BENCHMARK_JSON}, {MULTISEED_BENCHMARK_MD}")


def run_multiseed_benchmark(algos, seeds, episodes, eval_episodes, eval_split, curriculum, resume):
    train_set = load_split("train")
    val_set = load_split("val")
    eval_set = load_split(eval_split)
    if eval_episodes and eval_episodes < len(eval_set):
        eval_set = eval_set[:eval_episodes]

    multiseed_data = {
        "seeds": seeds,
        "episodes": episodes,
        "eval_split": eval_split,
        "eval_episodes": len(eval_set),
        "algorithms": {},
    }

    # Isolated directory for intermediate per-seed checkpoints
    checkpoints_dir = os.path.join(RESULTS_DIR, "multiseed_checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    for algo in algos:
        seed_results = []
        for s in seeds:
            ext = "pkl" if algo == "qlearning" else "pt"
            seed_model_path = os.path.join(checkpoints_dir, f"{algo}_seed_{s}.{ext}")
            res = run_one(
                algo=algo,
                episodes=episodes,
                train_set=train_set,
                val_set=val_set,
                eval_set=eval_set,
                seed=s,
                resume=resume,
                curriculum=curriculum,
                save_model=True,
                model_path=seed_model_path,
            )
            seed_results.append(res)

        aggregated = aggregate_multiseed_results(seed_results)
        multiseed_data["algorithms"][algo] = {
            "seed_results": seed_results,
            "aggregated": aggregated,
        }

    write_multiseed_outputs(multiseed_data, len(eval_set))
    return multiseed_data


def write_outputs(results, eval_n):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(BENCHMARK_JSON, "w") as f:
        json.dump({"eval_episodes": eval_n, "results": results}, f, indent=2)

    # Markdown table for the report
    lines = ["| Algorithm | Success % | Feasible Ceiling % | Feasible Success % | Avg reward | Avg rounds | Avg savings % | Tier kept % | Train time (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        o = r["evaluation"]["overall"]
        lines.append(f"| {LABELS[r['algorithm']]} | {o['success_rate']}% | {o['feasible_ceiling']}% | "
                     f"{o['feasible_success_rate']}% | {o['avg_reward']:+.2f} | {o['avg_rounds']} | "
                     f"{o['avg_savings_pct']}% | {o['tier_kept_pct']}% | {r['train_seconds']}s |")
    lines += ["", "Success rate by provider:", "",
              "| Algorithm | AWS | GCP | Azure |", "|---|---|---|---|"]
    for r in results:
        bp = r["evaluation"]["by_provider"]
        lines.append(f"| {LABELS[r['algorithm']]} | " +
                     " | ".join(f"{bp[p]['success_rate']}%" for p in ("AWS", "GCP", "Azure")) + " |")
    lines += ["", "Auto provider selection (best of AWS / GCP / Azure):", "",
              "| Algorithm | Success % (auto) | Success % (single provider) | Tier kept % | AWS wins | GCP wins | Azure wins |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        a = r["auto_selection"]
        lines.append(f"| {LABELS[r['algorithm']]} | {a['success_rate']}% | {a['own_provider_success_rate']}% | "
                     f"{a['tier_kept_pct']}% | {a['wins']['AWS']} | {a['wins']['GCP']} | {a['wins']['Azure']} |")
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
    ap.add_argument("--eval-episodes", type=int, default=None, help="max eval scenarios to test")
    ap.add_argument("--eval-split", choices=["test", "val"], default="test", help="split to evaluate on (test | val)")
    ap.add_argument("--curriculum", action="store_true", default=True, help="use curriculum learning")
    ap.add_argument("--no-curriculum", dest="curriculum", action="store_false", help="disable curriculum learning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="continue from the saved model")
    ap.add_argument("--multi-seed", action="store_true", help="run multi-seed benchmark across [42, 0, 1, 7, 99]")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="seeds list for multi-seed benchmark")
    args = ap.parse_args()

    requested_algos = ["qlearning", "dqn"] if args.algo == "both" else [args.algo]
    algos = []
    for a in requested_algos:
        if a == "dqn" and not torch_available():
            print("[Warning] PyTorch is not installed. Skipping DQN algorithm.")
            continue
        algos.append(a)

    if not algos:
        print("[Error] No available algorithm to run.")
        return

    if args.multi_seed:
        run_multiseed_benchmark(
            algos=algos,
            seeds=args.seeds,
            episodes=args.episodes,
            eval_episodes=args.eval_episodes,
            eval_split=args.eval_split,
            curriculum=args.curriculum,
            resume=args.resume,
        )
        print("\n  Multi-seed benchmark completed.")
        return

    train_set = load_split("train")
    val_set = load_split("val")
    eval_set = load_split(args.eval_split)
    if args.eval_episodes and args.eval_episodes < len(eval_set):
        eval_set = eval_set[:args.eval_episodes]

    results = [run_one(a, args.episodes, train_set, val_set, eval_set, args.seed, args.resume, args.curriculum) for a in algos]
    write_outputs(results, len(eval_set))

    print("\n  Done. Next: python app.py  ->  http://127.0.0.1:5000\n")


if __name__ == "__main__":
    main()

