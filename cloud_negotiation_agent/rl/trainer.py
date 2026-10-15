"""
rl/trainer.py
-------------
Training + evaluation helpers shared by train.py and the /api/train & /api/evaluate routes.

    sample_scenario(rng)            random provider / tier / quantity / budget (legacy)
    train_agent(user_agent, ...)    dataset-driven curriculum training with validation checkpointing
    evaluate_agent(user_agent, ...) greedy evaluation on a fixed scenario set with feasibility ceiling
    evaluate_auto(user_agent, ...)  greedy evaluation for multi-provider "Auto" mode
"""

import math
import random

from agents.provider_agent import ProviderAgent, provider_names, tier_names
from rl.dataset import load_split, feasible_rate


def sample_scenario(rng, providers=None):
    """
    Random negotiation scenario (legacy fallback).
    """
    provider = rng.choice(providers or provider_names())
    tier     = rng.choice(tier_names())
    quantity = max(1, int(round(10 ** rng.uniform(0, 2.7))))
    list_p   = ProviderAgent(provider).list_price(tier, quantity)
    budget   = round(list_p * rng.uniform(0.55, 1.05), 2)
    return {"provider": provider, "tier": tier, "quantity": quantity, "budget": budget}


def make_eval_set(n, seed=123):
    """A fixed, reproducible list of scenarios (same set for every algorithm)."""
    rng = random.Random(seed)
    return [sample_scenario(rng) for _ in range(n)]


def train_agent(user_agent, episodes=500, scenarios=None, val_scenarios=None,
                seed=None, batch=100, decay_rate=0.995, min_epsilon=0.05,
                verbose=True, curriculum=True, val_every=None, save_best=True):
    """
    Train `user_agent` using dataset scenarios, curriculum difficulty progression,
    and best-model validation checkpointing.

    Returns:
        list[dict]: one entry per `batch` episodes —
                    {"episode", "avg_reward", "success_rate", "epsilon", "val_success_rate"}
    """
    rng = random.Random(seed)
    was_training = user_agent.training_mode
    user_agent.set_training_mode(True)
    learner = user_agent.agent

    # Load dataset if not passed explicitly
    if scenarios is None:
        try:
            scenarios = load_split("train")
        except Exception:
            scenarios = None

    if val_scenarios is None and save_best:
        try:
            val_scenarios = load_split("val")[:150]
        except Exception:
            val_scenarios = None

    if episodes is None:
        episodes = len(scenarios) if scenarios else 500

    batch = max(1, min(batch, episodes))
    val_interval = val_every or max(batch, episodes // 5)

    history = []
    b_reward, b_success = 0.0, 0
    start = learner.episode_count
    best_val_success = -1.0

    # Build per-difficulty dual sub-pools (feasible + infeasible) for curriculum training.
    # infeasible_ratio controls what fraction of sampled scenarios are infeasible cases;
    # exposing the agent to unsolvable negotiations teaches it to walk away correctly.
    infeasible_ratio = 0.15
    diff_pools = {d: {"feasible": [], "infeasible": []} for d in ["easy", "medium", "hard", "expert"]}
    has_curriculum_pools = False
    if scenarios and curriculum and isinstance(scenarios[0], dict) and "difficulty" in scenarios[0]:
        for sc in scenarios:
            d = sc.get("difficulty", "medium")
            if d in diff_pools:
                feas_key = "feasible" if sc.get("feasible", True) else "infeasible"
                diff_pools[d][feas_key].append(sc)
        has_curriculum_pools = any(
            len(diff_pools[d]["feasible"]) > 0 or len(diff_pools[d]["infeasible"]) > 0
            for d in diff_pools
        )

    def _sample_from_pool(difficulty_keys):
        """Sample one scenario from the union of requested difficulty pools,
        respecting infeasible_ratio.  Falls back gracefully for empty sub-pools."""
        feas_pool = []
        infeas_pool = []
        for d in difficulty_keys:
            feas_pool.extend(diff_pools[d]["feasible"])
            infeas_pool.extend(diff_pools[d]["infeasible"])
        if not feas_pool and not infeas_pool:
            return None
        if not infeas_pool:
            return rng.choice(feas_pool)
        if not feas_pool:
            return rng.choice(infeas_pool)
        # Weighted draw: infeasible_ratio of draws come from infeasible pool
        if rng.random() < infeasible_ratio:
            return rng.choice(infeas_pool)
        return rng.choice(feas_pool)

    # Adaptive per-episode epsilon decay:
    # Decays smoothly down to min_epsilon by ~60% of total episodes,
    # ensuring the final 40% is dedicated to pure exploitation & convergence.
    decay_target_ep = max(30, int(0.60 * episodes))
    step_decay = (min_epsilon / max(learner.epsilon, min_epsilon + 0.01)) ** (1.0 / decay_target_ep)
    step_decay = max(0.990, min(0.9995, step_decay))

    if verbose:
        print(f"  {'Episode':>8}  {'Success%':>9}  {'Avg Reward':>11}  {'Epsilon':>8}  {'Val Success':>11}")
        print("  " + "-" * 58)

    for ep in range(1, episodes + 1):
        progress = ep / max(episodes, 1)

        # Pick scenario based on curriculum stage or shuffle
        if has_curriculum_pools:
            if progress <= 0.30:
                active_keys = ["easy", "medium"]
            elif progress <= 0.65:
                active_keys = ["easy", "medium", "hard"]
            else:
                active_keys = ["easy", "medium", "hard", "expert"]
            scenario = _sample_from_pool(active_keys)
            if scenario is None:
                scenario = scenarios[(ep - 1) % len(scenarios)]
        elif scenarios:
            scenario = scenarios[(ep - 1) % len(scenarios)]
        else:
            scenario = sample_scenario(rng)

        result = user_agent.negotiate(**scenario)

        b_reward  += result.total_reward
        b_success += int(result.success)

        # Decay epsilon per episode so exploration smoothly reduces
        learner.decay_epsilon(step_decay, min_epsilon)
        learner.episode_count = start + ep

        # Validation Checkpointing (evaluate on greedy policy)
        cur_val_success = None
        if val_scenarios and (ep % val_interval == 0 or ep == episodes):
            val_res = evaluate_agent(user_agent, val_scenarios)
            cur_val_success = val_res["overall"]["success_rate"]
            # Save checkpoint once agent has made progress and beats best score
            if save_best and progress >= 0.30 and cur_val_success >= best_val_success:
                best_val_success = cur_val_success
                user_agent.save()
            user_agent.set_training_mode(True)

        if ep % batch == 0 or ep == episodes:
            n = batch if ep % batch == 0 else ep % batch
            entry = {
                "episode":          start + ep,
                "avg_reward":       round(b_reward / n, 3),
                "success_rate":     round(b_success / n * 100, 1),
                "epsilon":          round(learner.epsilon, 4),
                "val_success_rate": cur_val_success,
            }
            history.append(entry)
            learner.training_rewards.append(entry["avg_reward"])
            learner.training_history.append(entry)

            if verbose:
                val_str = f"{cur_val_success:>9.1f}%" if cur_val_success is not None else "        -"
                print(f"  {entry['episode']:>8}  {entry['success_rate']:>8.1f}%  "
                      f"{entry['avg_reward']:>+11.3f}  {entry['epsilon']:>8.4f}  {val_str}")
            b_reward, b_success = 0.0, 0

    learner.episode_count = start + episodes
    user_agent.set_training_mode(was_training)

    # Reload best model if checkpointing was active, preserving total episodes
    if save_best and best_val_success >= 0.0:
        try:
            user_agent.reload()
            learner.episode_count = start + episodes
        except Exception:
            pass

    return history


def evaluate_agent(user_agent, scenarios):
    """
    Greedy evaluation (ε = 0, no learning) on a fixed list of scenarios.

    Returns:
        {"overall": {...}, "by_provider": {name: {...}}, "failures": {...}}
        each block includes: episodes, success_rate (%), feasible_ceiling (%),
        feasible_success_rate (%), avg_reward, avg_rounds, avg_savings_pct, tier_kept_pct
    """
    was_training, old_eps = user_agent.training_mode, user_agent.agent.epsilon
    user_agent.training_mode = False
    user_agent.agent.epsilon = 0.0

    rows = []
    failure_counts = {}
    for sc in scenarios:
        r = user_agent.negotiate(**sc)
        saving = ((r.list_price - r.final_price) / r.list_price * 100) if r.success else None
        feasible = sc.get("feasible", True)

        if not r.success:
            reason = "patience exhausted"
            if r.log:
                last_line = r.log[-1].lower()
                if "patience" in last_line:
                    reason = "patience exhausted"
                elif "budget" in last_line:
                    reason = "budget exceeded"
                elif "rejected" in last_line:
                    reason = "offer rejected / floor not met"
                elif "walk away" in last_line:
                    reason = "walked away"
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

        rows.append((r.provider, r.success, r.total_reward, r.rounds, saving, r.downgrades, feasible))

    user_agent.training_mode = was_training
    user_agent.agent.epsilon = old_eps

    def summarise(subset):
        n = len(subset)
        if n == 0:
            return {
                "episodes": 0, "success_rate": 0.0, "feasible_ceiling": 0.0,
                "feasible_success_rate": 0.0, "avg_reward": 0.0, "avg_rounds": 0.0,
                "avg_savings_pct": 0.0, "tier_kept_pct": 0.0
            }
        savings = [row[4] for row in subset if row[4] is not None]
        ok_rows = [row for row in subset if row[1]]

        feasible_subset = [row for row in subset if row[6]]
        feasible_n = len(feasible_subset)
        feasible_ok = sum(1 for row in feasible_subset if row[1])

        ceiling = round(feasible_n / n * 100, 1)
        feasible_sr = round(feasible_ok / feasible_n * 100, 1) if feasible_n > 0 else 0.0

        return {
            "episodes":              n,
            "success_rate":          round(len(ok_rows) / n * 100, 1),
            "feasible_ceiling":      ceiling,
            "feasible_success_rate": feasible_sr,
            "avg_reward":            round(sum(row[2] for row in subset) / n, 3),
            "avg_rounds":            round(sum(row[3] for row in subset) / n, 2),
            "avg_savings_pct":       round(sum(savings) / len(savings), 1) if savings else 0.0,
            "tier_kept_pct":         round(sum(1 for row in ok_rows if row[5] == 0) / len(ok_rows) * 100, 1) if ok_rows else 0.0,
        }

    return {
        "overall":     summarise(rows),
        "by_provider": {p: summarise([r for r in rows if r[0] == p]) for p in provider_names()},
        "failures":    failure_counts,
        "ceiling":     feasible_rate(scenarios),
    }


def evaluate_auto(user_agent, scenarios):
    """
    Evaluate the "Auto" provider-selection mode.

    Every scenario's (tier, quantity, budget) is offered to ALL providers and
    the agent keeps the best deal. Compared with `own_provider_success_rate`.
    """
    was_training, old_eps = user_agent.training_mode, user_agent.agent.epsilon
    user_agent.training_mode = False
    user_agent.agent.epsilon = 0.0

    wins = {p: 0 for p in provider_names()}
    ok = own_ok = kept = 0
    savings = []
    for sc in scenarios:
        winner, results = user_agent.negotiate_best(sc["tier"], sc["quantity"], sc["budget"])
        own = next(r for r in results if r.provider == sc["provider"])
        own_ok += int(own.success)
        if winner.success:
            ok += 1
            wins[winner.provider] += 1
            kept += int(winner.downgrades == 0)
            savings.append((winner.list_price - winner.final_price) / winner.list_price * 100)

    user_agent.training_mode = was_training
    user_agent.agent.epsilon = old_eps
    n = max(len(scenarios), 1)
    return {
        "episodes":                  len(scenarios),
        "success_rate":              round(ok / n * 100, 1),
        "own_provider_success_rate": round(own_ok / n * 100, 1),
        "avg_savings_pct":           round(sum(savings) / len(savings), 1) if savings else 0.0,
        "tier_kept_pct":             round(kept / ok * 100, 1) if ok else 0.0,
        "wins":                      wins,
        "ceiling":                   feasible_rate(scenarios),
    }
