"""
rl/trainer.py
-------------
Training + evaluation helpers shared by train.py and the /api/train route.

    sample_scenario(rng)            random provider / tier / quantity / budget
    train_agent(user_agent, ...)    run N training episodes, return learning curve
    evaluate_agent(user_agent, ...) greedy evaluation on a FIXED scenario set,
                                    with a per-provider breakdown
"""

import math
import random

from agents.provider_agent import ProviderAgent, provider_names, tier_names


def sample_scenario(rng, providers=None):
    """
    Random negotiation scenario.

    quantity : log-uniform 1 … ~500 instances (so volume tiers are exercised)
    budget   : 55 % – 105 % of the requested configuration's list price, so
               some scenarios are easy, some force a downgrade, and a few
               are impossible within the provider's patience.
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


def train_agent(user_agent, episodes, seed=None, batch=100,
                decay_rate=0.995, min_epsilon=0.05, verbose=True):
    """
    Train `user_agent` on random scenarios.

    Returns:
        list[dict]: one entry per `batch` episodes —
                    {"episode", "avg_reward", "success_rate", "epsilon"}
    """
    rng = random.Random(seed)
    was_training = user_agent.training_mode
    user_agent.set_training_mode(True)
    learner = user_agent.agent

    history, b_reward, b_success = [], 0.0, 0
    start = learner.episode_count

    if verbose:
        print(f"  {'Episode':>8}  {'Success%':>9}  {'Avg Reward':>11}  {'Epsilon':>8}")
        print("  " + "-" * 44)

    for ep in range(1, episodes + 1):
        scenario = sample_scenario(rng)
        result   = user_agent.negotiate(**scenario)

        b_reward  += result.total_reward
        b_success += int(result.success)
        learner.decay_epsilon(decay_rate, min_epsilon)

        if ep % batch == 0 or ep == episodes:
            n = batch if ep % batch == 0 else ep % batch
            entry = {
                "episode":      start + ep,
                "avg_reward":   round(b_reward / n, 3),
                "success_rate": round(b_success / n * 100, 1),
                "epsilon":      round(learner.epsilon, 4),
            }
            history.append(entry)
            learner.training_rewards.append(entry["avg_reward"])
            learner.training_history.append(entry)
            if verbose:
                print(f"  {entry['episode']:>8}  {entry['success_rate']:>8.1f}%  "
                      f"{entry['avg_reward']:>+11.3f}  {entry['epsilon']:>8.4f}")
            b_reward, b_success = 0.0, 0

    learner.episode_count = start + episodes
    user_agent.set_training_mode(was_training)
    return history


def evaluate_agent(user_agent, scenarios):
    """
    Greedy evaluation (ε = 0, no learning) on a fixed list of scenarios.

    Returns:
        {"overall": {...}, "by_provider": {name: {...}}}
        each block: episodes, success_rate (%), avg_reward, avg_rounds,
                    avg_savings_pct (vs list price, successful deals only)
    """
    was_training, old_eps = user_agent.training_mode, user_agent.agent.epsilon
    user_agent.training_mode = False
    user_agent.agent.epsilon = 0.0

    rows = []
    for sc in scenarios:
        r = user_agent.negotiate(**sc)
        saving = ((r.list_price - r.final_price) / r.list_price * 100) if r.success else None
        rows.append((r.provider, r.success, r.total_reward, r.rounds, saving, r.downgrades))

    user_agent.training_mode = was_training
    user_agent.agent.epsilon = old_eps

    def summarise(subset):
        n = len(subset)
        if n == 0:
            return {"episodes": 0, "success_rate": 0.0, "avg_reward": 0.0,
                    "avg_rounds": 0.0, "avg_savings_pct": 0.0, "tier_kept_pct": 0.0}
        savings = [row[4] for row in subset if row[4] is not None]
        ok_rows = [row for row in subset if row[1]]
        return {
            "episodes":        n,
            "success_rate":    round(len(ok_rows) / n * 100, 1),
            "avg_reward":      round(sum(row[2] for row in subset) / n, 3),
            "avg_rounds":      round(sum(row[3] for row in subset) / n, 2),
            "avg_savings_pct": round(sum(savings) / len(savings), 1) if savings else 0.0,
            # % of successful deals that kept the originally requested tier
            "tier_kept_pct":   round(sum(1 for row in ok_rows if row[5] == 0) / len(ok_rows) * 100, 1) if ok_rows else 0.0,
        }

    return {
        "overall": summarise(rows),
        "by_provider": {p: summarise([r for r in rows if r[0] == p]) for p in provider_names()},
    }


def evaluate_auto(user_agent, scenarios):
    """
    Evaluate the "Auto" provider-selection mode.

    Every scenario's (tier, quantity, budget) is offered to ALL providers and
    the agent keeps the best deal. Compared with `own_provider_success_rate`
    (what a user locked to the scenario's original provider would get).

    Returns: {"episodes", "success_rate", "own_provider_success_rate",
              "avg_savings_pct", "tier_kept_pct", "wins": {provider: count}}
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
    }
