"""
config.py
---------
Central place for every file path used by the project.

WHY THIS FILE EXISTS
    Before, paths such as "negotiation_history.db" and "q_table.pkl" were
    relative to the *current working directory*. Running the app from the
    project root and from inside cloud_negotiation_agent/ therefore created
    two different databases / Q-tables. All paths are now absolute and
    anchored to this file's folder, so there is exactly one copy of each.
"""

import os

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))

PROVIDERS_JSON = os.path.join(BASE_DIR, "providers.json")
DB_PATH        = os.path.join(BASE_DIR, "negotiation_history.db")

# Saved agents
Q_TABLE_PATH   = os.path.join(BASE_DIR, "q_table.pkl")
DQN_MODEL_PATH = os.path.join(BASE_DIR, "dqn_model.pt")

# Benchmark output (train.py writes here, /api/benchmark reads from here)
RESULTS_DIR            = os.path.join(BASE_DIR, "results")
BENCHMARK_JSON         = os.path.join(RESULTS_DIR, "benchmark.json")
MULTISEED_BENCHMARK_JSON = os.path.join(RESULTS_DIR, "multiseed_benchmark.json")
MULTISEED_BENCHMARK_MD   = os.path.join(RESULTS_DIR, "multiseed_benchmark.md")

# Dataset splits directory
DATA_DIR       = os.path.join(BASE_DIR, "data")

MODEL_PATHS = {
    "qlearning": Q_TABLE_PATH,
    "dqn":       DQN_MODEL_PATH,
}
ALGORITHMS = ("qlearning", "dqn")
