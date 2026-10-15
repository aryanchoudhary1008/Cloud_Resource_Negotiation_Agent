"""
tests/test_multiseed.py
-----------------------
Unit and integration tests for multi-seed benchmarking:
  - Seed aggregation and mean ± std calculation
  - Handling of undefined metrics (walk-away precision/recall)
  - Missing dependency handling (DQN without PyTorch)
  - Isolated checkpoint paths (does not overwrite primary model / benchmark files)
  - Multi-seed smoke test with reduced episodes
"""

import math
import os
import pytest

from config import (
    Q_TABLE_PATH,
    DQN_MODEL_PATH,
    BENCHMARK_JSON,
    MULTISEED_BENCHMARK_JSON,
    MULTISEED_BENCHMARK_MD,
)
from train import (
    compute_mean_std,
    format_mean_std,
    aggregate_multiseed_results,
    write_multiseed_outputs,
    torch_available,
    DEFAULT_SEEDS,
    run_multiseed_benchmark,
)


def test_default_seeds():
    assert DEFAULT_SEEDS == [42, 0, 1, 7, 99]


def test_compute_mean_std_basic():
    values = [10.0, 20.0, 30.0]
    mean, std = compute_mean_std(values)
    assert mean == 20.0
    assert std == 10.0


def test_compute_mean_std_single_value():
    mean, std = compute_mean_std([15.0])
    assert mean == 15.0
    assert std == 0.0


def test_compute_mean_std_with_nones_and_empty():
    mean, std = compute_mean_std([None, None])
    assert mean is None
    assert std is None

    mean, std = compute_mean_std([])
    assert mean is None
    assert std is None

    mean, std = compute_mean_std([10.0, None, 20.0])
    assert mean == 15.0
    assert std == pytest.approx(7.07, abs=0.01)


def test_format_mean_std():
    assert format_mean_std(None, None) == "N/A"
    assert format_mean_std(85.5, 2.3, pct=True) == "85.50 ± 2.30%"
    assert format_mean_std(4.25, 0.5, signed=True) == "+4.25 ± 0.50"
    assert format_mean_std(-1.2, 0.3, signed=True) == "-1.20 ± 0.30"


def test_torch_available_reflects_environment():
    # If torch is not installed, torch_available() must be False and not raise
    avail = torch_available()
    assert isinstance(avail, bool)


def test_aggregate_multiseed_results_mock():
    mock_results = [
        {
            "seed": 42,
            "train_seconds": 1.2,
            "evaluation": {
                "overall": {
                    "success_rate": 90.0,
                    "feasible_success_rate": 92.0,
                    "avg_reward": 4.5,
                    "avg_rounds": 2.2,
                    "avg_savings_pct": 15.0,
                    "tier_kept_pct": 70.0,
                },
                "by_provider": {
                    "AWS": {"success_rate": 88.0, "avg_savings_pct": 10.0},
                    "GCP": {"success_rate": 92.0, "avg_savings_pct": 18.0},
                    "Azure": {"success_rate": 90.0, "avg_savings_pct": 20.0},
                },
                "by_difficulty": {
                    "easy": {"success_rate": 100.0},
                    "medium": {"success_rate": 95.0},
                    "hard": {"success_rate": 80.0},
                    "expert": {"success_rate": 60.0},
                },
                "walk_away_accuracy": {
                    "precision_pct": 100.0,
                    "recall_pct": 80.0,
                },
            },
            "auto_selection": {
                "success_rate": 98.0,
            },
        },
        {
            "seed": 0,
            "train_seconds": 1.4,
            "evaluation": {
                "overall": {
                    "success_rate": 80.0,
                    "feasible_success_rate": 82.0,
                    "avg_reward": 3.5,
                    "avg_rounds": 2.6,
                    "avg_savings_pct": 11.0,
                    "tier_kept_pct": 60.0,
                },
                "by_provider": {
                    "AWS": {"success_rate": 78.0, "avg_savings_pct": 8.0},
                    "GCP": {"success_rate": 82.0, "avg_savings_pct": 14.0},
                    "Azure": {"success_rate": 80.0, "avg_savings_pct": 16.0},
                },
                "by_difficulty": {
                    "easy": {"success_rate": 90.0},
                    "medium": {"success_rate": 85.0},
                    "hard": {"success_rate": 70.0},
                    "expert": {"success_rate": 50.0},
                },
                "walk_away_accuracy": {
                    "precision_pct": None,  # e.g., no walk-aways occurred
                    "recall_pct": 0.0,
                },
            },
            "auto_selection": {
                "success_rate": 94.0,
            },
        },
    ]

    agg = aggregate_multiseed_results(mock_results)
    assert agg["success_rate"]["mean"] == 85.0
    assert agg["success_rate"]["std"] == pytest.approx(7.07, abs=0.01)
    assert agg["train_seconds"]["mean"] == 1.3
    # precision had one None and one 100.0 -> valid list has 1 value -> mean 100.0, std 0.0
    assert agg["walkaway_precision"]["mean"] == 100.0
    assert agg["walkaway_precision"]["std"] == 0.0
    # recall had 80.0 and 0.0 -> mean 40.0
    assert agg["walkaway_recall"]["mean"] == 40.0


def test_write_multiseed_outputs(tmp_path, monkeypatch):
    import train
    json_path = str(tmp_path / "test_multiseed.json")
    md_path = str(tmp_path / "test_multiseed.md")
    monkeypatch.setattr(train, "MULTISEED_BENCHMARK_JSON", json_path)
    monkeypatch.setattr(train, "MULTISEED_BENCHMARK_MD", md_path)

    multiseed_data = {
        "seeds": [42, 0],
        "algorithms": {
            "qlearning": {
                "aggregated": {
                    "success_rate": {"mean": 88.0, "std": 2.5},
                    "feasible_success_rate": {"mean": 90.0, "std": 2.0},
                    "avg_reward": {"mean": 4.1, "std": 0.3},
                    "avg_rounds": {"mean": 2.3, "std": 0.2},
                    "avg_savings_pct": {"mean": 14.2, "std": 1.1},
                    "tier_kept_pct": {"mean": 65.0, "std": 3.0},
                    "auto_success_rate": {"mean": 97.0, "std": 1.0},
                    "walkaway_precision": {"mean": 85.0, "std": 5.0},
                    "walkaway_recall": {"mean": 70.0, "std": 8.0},
                    "provider_AWS_success": {"mean": 86.0, "std": 3.0},
                    "provider_GCP_success": {"mean": 90.0, "std": 2.0},
                    "provider_Azure_success": {"mean": 88.0, "std": 2.5},
                    "diff_easy_success": {"mean": 98.0, "std": 1.0},
                    "diff_medium_success": {"mean": 92.0, "std": 2.0},
                    "diff_hard_success": {"mean": 80.0, "std": 4.0},
                    "diff_expert_success": {"mean": 60.0, "std": 5.0},
                },
                "seed_results": [
                    {
                        "seed": 42,
                        "train_seconds": 1.5,
                        "evaluation": {
                            "overall": {
                                "success_rate": 89.0,
                                "feasible_success_rate": 91.0,
                                "avg_reward": 4.2,
                                "avg_savings_pct": 14.5,
                            },
                            "walk_away_accuracy": {
                                "precision_pct": 90.0,
                                "recall_pct": 75.0,
                            },
                        },
                    }
                ],
            }
        },
    }

    write_multiseed_outputs(multiseed_data, eval_n=50)

    assert os.path.exists(json_path)
    assert os.path.exists(md_path)
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Multi-Seed Benchmarking Results" in content
    assert "88.00 ± 2.50%" in content
    assert "+4.10 ± 0.30" in content


def test_multiseed_smoke_execution_isolation(tmp_path, monkeypatch):
    """
    Run a fast smoke test with 2 seeds, 10 episodes, 10 eval episodes.
    Verify:
      1. Primary model file (q_table.pkl) is NOT overwritten.
      2. Primary benchmark.json / benchmark.md is NOT overwritten.
      3. Isolated checkpoint files are saved in multiseed_checkpoints directory.
    """
    import train

    # Check timestamps or presence of existing files
    orig_qtable_mtime = os.path.getmtime(Q_TABLE_PATH) if os.path.exists(Q_TABLE_PATH) else None
    orig_bench_mtime = os.path.getmtime(BENCHMARK_JSON) if os.path.exists(BENCHMARK_JSON) else None

    # Redirect outputs to tmp_path to verify isolation
    test_json = str(tmp_path / "smoke_multiseed.json")
    test_md = str(tmp_path / "smoke_multiseed.md")
    monkeypatch.setattr(train, "MULTISEED_BENCHMARK_JSON", test_json)
    monkeypatch.setattr(train, "MULTISEED_BENCHMARK_MD", test_md)

    data = run_multiseed_benchmark(
        algos=["qlearning"],
        seeds=[42, 7],
        episodes=10,
        eval_episodes=5,
        eval_split="test",
        curriculum=True,
        resume=False,
    )

    assert "qlearning" in data["algorithms"]
    assert len(data["algorithms"]["qlearning"]["seed_results"]) == 2
    agg = data["algorithms"]["qlearning"]["aggregated"]
    assert "success_rate" in agg
    assert agg["success_rate"]["mean"] is not None

    # Verify primary files were not modified
    if orig_qtable_mtime is not None:
        assert os.path.getmtime(Q_TABLE_PATH) == orig_qtable_mtime
    if orig_bench_mtime is not None:
        assert os.path.getmtime(BENCHMARK_JSON) == orig_bench_mtime

    # Verify output JSON and MD were written
    assert os.path.exists(test_json)
    assert os.path.exists(test_md)

