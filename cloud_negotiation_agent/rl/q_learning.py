"""
rl/q_learning.py
----------------
Q-Learning agent implementation.

Q-learning is a MODEL-FREE, OFF-POLICY reinforcement learning algorithm.
The agent does NOT need a model of the environment — it learns purely
by trial and error, storing the expected future reward for every
(state, action) pair it encounters.

Core update rule (Bellman Equation):
    Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

Where:
    Q(s, a)  = current estimated value of taking action a in state s
    α (alpha) = learning rate   → how fast to overwrite old estimates
    r         = reward received after taking action a
    γ (gamma) = discount factor → how much future rewards are worth now
    max Q(s') = best action's Q-value from the next state

Action space (4 actions):
    0 → Increase user's offered price by 10%
    1 → Decrease resource request (downgrade CPU/RAM tier)
    2 → Accept the provider's current offer
    3 → Reject and end negotiation
"""

import os
import pickle
import numpy as np
from collections import defaultdict


class QLearningAgent:
    """
    Tabular Q-Learning agent with epsilon-greedy exploration.

    The Q-table is a defaultdict where:
        key   = state tuple  e.g. (2, 1, 2, 3, 0)
        value = np.array([q0, q1, q2, q3])  one float per action

    New (unseen) states automatically start with Q-values of 0,
    meaning the agent has no preference until it has experienced them.
    """

    NUM_ACTIONS = 4  # Increase price / Decrease resources / Accept / Reject

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.3):
        """
        Initialize the Q-learning agent.

        Args:
            alpha   (float): Learning rate in [0, 1].
                             High → learns fast but may oscillate.
                             Low  → learns slowly but more stable.
            gamma   (float): Discount factor in [0, 1].
                             High (≈1) → agent values long-term rewards.
                             Low  (≈0) → agent is short-sighted (greedy now).
            epsilon (float): Exploration probability in [0, 1].
                             High → random actions (explore unknown states).
                             Low  → pick the best known action (exploit).
        """
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon

        # Q-table: state_tuple → array of shape (NUM_ACTIONS,)
        # defaultdict auto-creates zero arrays for new states
        self.q_table = defaultdict(lambda: np.zeros(self.NUM_ACTIONS))

        # Metadata for tracking training progress
        self.training_rewards = []   # average reward per training batch (for graphs)
        self.episode_count    = 0    # total episodes trained so far

    # ── Action Selection ─────────────────────────────────────────────

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.

        Strategy:
          - With probability ε   → EXPLORE: pick a uniformly random action.
                                   Helps discover better strategies.
          - With probability 1-ε → EXPLOIT: pick argmax Q(state, *).
                                   Uses accumulated knowledge.

        Args:
            state (tuple): Encoded state from state_space.encode_state().

        Returns:
            int: Action index in {0, 1, 2, 3}.
        """
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.NUM_ACTIONS)
        else:
            # Exploitation: greedy action with highest Q-value
            q_values = self.q_table[state]
            return int(np.argmax(q_values))

    # ── Q-Table Update ───────────────────────────────────────────────

    def update(self, state, action, reward, next_state, done=False):
        """
        Apply the Bellman equation to update Q(state, action).

        Bellman update:
            target   = r + γ * max_a' Q(s', a')    [if not done]
            target   = r                            [if done/terminal]
            Q(s, a) += α * (target - Q(s, a))

        The term (target - Q(s,a)) is the "TD error" — the surprise
        between what we expected and what we actually got.

        Args:
            state      (tuple): State before action was taken.
            action     (int):   Action that was taken (0–3).
            reward     (float): Reward received immediately after action.
            next_state (tuple): State after action was applied.
            done       (bool):  True if the episode ended (no future rewards).
        """
        current_q = self.q_table[state][action]

        if done:
            # Terminal state: no future Q-value to bootstrap from
            td_target = reward
        else:
            # Non-terminal: reward + discounted best future Q-value
            max_future_q = np.max(self.q_table[next_state])
            td_target    = reward + self.gamma * max_future_q

        # TD error = how wrong our current estimate was
        td_error = td_target - current_q

        # Gradient step: move Q(s,a) toward the target
        self.q_table[state][action] += self.alpha * td_error

    # ── Epsilon Decay ────────────────────────────────────────────────

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.05):
        """
        Gradually reduce the exploration rate over training.

        As the agent gains experience, it should exploit more and
        explore less. Epsilon is multiplied by decay_rate each call,
        but never drops below min_epsilon (to retain a tiny amount
        of exploration even in deployment).

        Args:
            decay_rate  (float): Multiplicative decay per episode.
            min_epsilon (float): Floor for epsilon.
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path="q_table.pkl"):
        """
        Serialize the Q-table and metadata to a pickle file.

        Saves:
          - q_table          (as a plain dict for portability)
          - training_rewards (list of batch averages for graphing)
          - episode_count    (total episodes seen so far)
          - epsilon          (current exploration rate)
          - alpha, gamma     (hyperparameters, for reference)

        Args:
            path (str): File path to write the pickle to.
        """
        data = {
            "q_table":          dict(self.q_table),
            "training_rewards": self.training_rewards,
            "episode_count":    self.episode_count,
            "epsilon":          self.epsilon,
            "alpha":            self.alpha,
            "gamma":            self.gamma,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[Q-Agent] Saved {len(self.q_table)} states to '{path}'")

    def load(self, path="q_table.pkl"):
        """
        Deserialize a previously saved Q-table from a pickle file.

        Safe to call even when the file doesn't exist — returns False
        and keeps the current (empty) Q-table.

        Args:
            path (str): File path to load from.

        Returns:
            bool: True if loaded successfully, False if file not found.
        """
        if not os.path.exists(path):
            print(f"[Q-Agent] No Q-table found at '{path}'. Starting fresh.")
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        # Restore Q-table as a defaultdict (plain dict from pickle → upgrade)
        loaded = data.get("q_table", {})
        self.q_table = defaultdict(lambda: np.zeros(self.NUM_ACTIONS))
        self.q_table.update(loaded)

        self.training_rewards = data.get("training_rewards", [])
        self.episode_count    = data.get("episode_count",    0)
        self.epsilon          = data.get("epsilon",          self.epsilon)
        self.alpha            = data.get("alpha",            self.alpha)
        self.gamma            = data.get("gamma",            self.gamma)

        print(f"[Q-Agent] Loaded {len(self.q_table)} states from '{path}' "
              f"(trained for {self.episode_count} episodes, ε={self.epsilon:.4f})")
        return True
