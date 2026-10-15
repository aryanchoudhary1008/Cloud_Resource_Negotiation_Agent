"""
rl/dataset.py
-------------
Controlled scenario generation and stratified dataset splits (train / val / test).

Eliminates the arbitrary random scenario ceiling by:
  1. Drawing realistic enterprise cluster sizes.
  2. Generating balanced difficulty bands (easy, medium, hard, expert).
  3. Pre-labeling each scenario with feasibility (is there ANY deal achievable
     within provider patience and budget, including downgrades).
  4. Producing fixed, stratified 70 / 15 / 15 splits for honest, reproducible benchmarking.
"""

import csv
import os
import random
from collections import defaultdict

from config import DATA_DIR
from agents.provider_agent import ProviderAgent, provider_names, tier_names
from rl.environment import NegotiationEnv

COLUMNS = ["id", "provider", "tier", "quantity", "budget", "difficulty", "feasible", "split"]
REALISTIC_QUANTITIES = [1, 2, 5, 10, 20, 50, 100, 200, 500]


def is_scenario_feasible(provider, tier, quantity, budget):
    """
    Check if a scenario is solvable under the provider's concession and patience rules,
    allowing for downgrades down to Small if needed.
    """
    env = NegotiationEnv()
    env.reset(provider, tier, quantity, budget)
    tiers = tier_names()
    start_idx = tiers.index(tier)
    for t in tiers[:start_idx + 1]:
        if env._feasible(t):
            return True
    return False


def build_dataset(n=6000, seed=7, out_dir=DATA_DIR):
    """
    Generate `n` scenarios, label feasibility, stratify 70/15/15 into train/val/test,
    and save them to CSV files.
    """
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)

    providers = provider_names()
    tiers = tier_names()
    difficulties = ["easy", "medium", "hard", "expert"]
    diff_weights = [0.30, 0.30, 0.25, 0.15]

    # Step 1: Generate raw scenarios
    raw_rows = []
    for i in range(n):
        provider = rng.choice(providers)
        tier = rng.choice(tiers)
        quantity = rng.choice(REALISTIC_QUANTITIES)
        p = ProviderAgent(provider)
        list_p = p.list_price(tier, quantity)
        floor_r = p.floor_ratio

        level = rng.choices(difficulties, diff_weights)[0]
        lo, hi = {
            "easy":   (1.00, 1.05),
            "medium": (0.85, 1.00),
            "hard":   (floor_r + 0.03, 0.85),
            "expert": (floor_r - 0.02, floor_r + 0.03),
        }[level]

        budget = round(list_p * rng.uniform(lo, max(hi, lo + 0.01)), 2)
        feasible = is_scenario_feasible(provider, tier, quantity, budget)

        raw_rows.append({
            "id": i,
            "provider": provider,
            "tier": tier,
            "quantity": quantity,
            "budget": budget,
            "difficulty": level,
            "feasible": feasible,
        })

    # Step 2: Stratify by (provider, difficulty, feasible) — three-dimensional stratification
    # ensures each split has proportional representation of provider, difficulty, AND feasibility.
    # Groups smaller than MIN_GROUP_SIZE are assigned entirely to train to avoid empty eval splits.
    MIN_GROUP_SIZE = 3
    groups = defaultdict(list)
    for row in raw_rows:
        groups[(row["provider"], row["difficulty"], row["feasible"])].append(row)

    train_rows, val_rows, test_rows = [], [], []

    for group_key in sorted(groups.keys()):
        group = groups[group_key]
        rng.shuffle(group)
        g_len = len(group)

        if g_len < MIN_GROUP_SIZE:
            # Tiny groups go entirely to train to avoid empty splits
            for row in group:
                row["split"] = "train"
                train_rows.append(row)
            continue

        n_train = int(round(g_len * 0.70))
        n_val = int(round(g_len * 0.15))
        # Ensure at least 1 row in val and test when group is large enough
        n_train = min(n_train, g_len - 2)
        n_val   = max(1, min(n_val, g_len - n_train - 1))

        for row in group[:n_train]:
            row["split"] = "train"
            train_rows.append(row)

        for row in group[n_train:n_train + n_val]:
            row["split"] = "val"
            val_rows.append(row)

        for row in group[n_train + n_val:]:
            row["split"] = "test"
            test_rows.append(row)

    # Sort each split deterministically by id
    train_rows.sort(key=lambda r: r["id"])
    val_rows.sort(key=lambda r: r["id"])
    test_rows.sort(key=lambda r: r["id"])

    # Step 3: Write out CSV files
    for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        file_path = os.path.join(out_dir, f"{split_name}.csv")
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    return {
        "train": len(train_rows),
        "val":   len(val_rows),
        "test":  len(test_rows),
    }


def parse_row(row):
    """Cast CSV string values to their native Python types."""
    return {
        "id":         int(row["id"]),
        "provider":   str(row["provider"]),
        "tier":       str(row["tier"]),
        "quantity":   int(row["quantity"]),
        "budget":     float(row["budget"]),
        "difficulty": str(row["difficulty"]),
        "feasible":   str(row["feasible"]).strip().lower() in ("true", "1", "t"),
        "split":      str(row["split"]),
    }


def load_split(split="test", data_dir=DATA_DIR):
    """
    Read `data/<split>.csv`. Automatically builds dataset if files are missing.
    Returns list of parsed dicts.
    """
    file_path = os.path.join(data_dir, f"{split}.csv")
    if not os.path.exists(file_path):
        build_dataset(out_dir=data_dir)

    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(parse_row(row))
    return rows


def feasible_rate(scenarios):
    """Percentage of scenarios in `scenarios` where feasible is True (0.0 to 100.0)."""
    if not scenarios:
        return 0.0
    feasible_count = sum(1 for s in scenarios if s.get("feasible", True))
    return round(feasible_count / len(scenarios) * 100, 1)


def dataset_summary(data_dir=DATA_DIR):
    """
    Summary statistics across train, val, and test splits.

    Returns per-split totals, feasible/infeasible counts and rates,
    difficulty distribution, and provider distribution.
    """
    summary = {}
    for split in ["train", "val", "test"]:
        file_path = os.path.join(data_dir, f"{split}.csv")
        if not os.path.exists(file_path):
            build_dataset(out_dir=data_dir)
        rows = load_split(split, data_dir=data_dir)
        n = len(rows)
        diff_counts = defaultdict(int)
        prov_counts = defaultdict(int)
        feasible_count = 0
        for r in rows:
            diff_counts[r["difficulty"]] += 1
            prov_counts[r["provider"]] += 1
            if r["feasible"]:
                feasible_count += 1

        infeasible_count = n - feasible_count
        summary[split] = {
            "total":           n,
            "feasible_count":  feasible_count,
            "infeasible_count": infeasible_count,
            "feasible_rate":   round(feasible_count / n * 100, 1) if n > 0 else 0.0,
            "infeasible_rate": round(infeasible_count / n * 100, 1) if n > 0 else 0.0,
            "difficulties":    dict(diff_counts),
            "providers":       dict(prov_counts),
        }
    return summary
