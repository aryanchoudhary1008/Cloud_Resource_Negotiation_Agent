"""
rl/q_learning.py
----------------
Tabular Q-Learning agent (the "classical" baseline for the DQN comparison).

    Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]

State key  : tuple from state_space.encode_tabular()
Actions    : 0 raise bid · 1 downgrade tier · 2 accept offer · 3 walk away

Changes in this version
    * Saved files carry STATE_VERSION. A q_table.pkl produced by an older
      state encoding is detected and ignored instead of silently
      mis-reading its keys.
    * training_history stores per-batch reward AND success rate, so the
      Q-learning vs DQN learning curves can be compared.
"""

import os
import pickle
import numpy as np
from collections import defaultdict

from rl.state_space import STATE_VERSION


class QLearningAgent:
    NUM_ACTIONS = 4

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.5):
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon

        self.q_table = defaultdict(lambda: np.zeros(self.NUM_ACTIONS))

        self.training_rewards = []   # avg reward per batch  (used by /api/graph)
        self.training_history = []   # [{"episode", "avg_reward", "success_rate", "epsilon"}]
        self.episode_count    = 0

    # ── Action selection ─────────────────────────────────────────────

    def choose_action(self, state):
        """ε-greedy: random action with probability ε, otherwise argmax Q(state, ·)."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.NUM_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def q_values(self, state):
        return self.q_table[state].copy()

    # ── Learning ─────────────────────────────────────────────────────

    def update(self, state, action, reward, next_state, done=False):
        """One Bellman update of Q(state, action)."""
        current_q = self.q_table[state][action]
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (td_target - current_q)

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.05):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    # ── Size / info ──────────────────────────────────────────────────

    @property
    def knowledge_size(self):
        """Number of distinct states stored in the Q-table."""
        return len(self.q_table)

    knowledge_label = "States in Q-table"

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path):
        data = {
            "state_version":    STATE_VERSION,
            "q_table":          dict(self.q_table),
            "training_rewards": self.training_rewards,
            "training_history": self.training_history,
            "episode_count":    self.episode_count,
            "epsilon":          self.epsilon,
            "alpha":            self.alpha,
            "gamma":            self.gamma,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[Q-Agent] Saved {len(self.q_table)} states to '{path}'")

    def load(self, path):
        """
        Load a saved Q-table. Returns False (and keeps the empty table) if the
        file is missing or was created by an incompatible state encoding.
        """
        if not os.path.exists(path):
            print(f"[Q-Agent] No Q-table found at '{path}'. Starting fresh.")
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        if data.get("state_version") != STATE_VERSION:
            print(f"[Q-Agent] '{path}' uses an old state encoding "
                  f"(v{data.get('state_version')} ≠ v{STATE_VERSION}) — ignoring it. "
                  "Run `python train.py` to build a new Q-table.")
            return False

        self.q_table = defaultdict(lambda: np.zeros(self.NUM_ACTIONS))
        self.q_table.update(data.get("q_table", {}))

        self.training_rewards = data.get("training_rewards", [])
        self.training_history = data.get("training_history", [])
        self.episode_count    = data.get("episode_count", 0)
        self.epsilon          = data.get("epsilon", self.epsilon)
        self.alpha            = data.get("alpha", self.alpha)
        self.gamma            = data.get("gamma", self.gamma)

        print(f"[Q-Agent] Loaded {len(self.q_table)} states from '{path}' "
              f"(trained for {self.episode_count} episodes, eps={self.epsilon:.4f})")
        return True

