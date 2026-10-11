"""
rl/dqn_agent.py
---------------
Deep Q-Network (DQN) agent — PyTorch.

Why DQN?
    Tabular Q-learning stores one number per (state, action) and cannot say
    anything about a state it has never seen. A DQN replaces the table with
    a neural network Q(s, a; θ) that GENERALISES: similar states (a price
    ratio of 0.81 vs 0.83, an unseen quantity) get similar action values.

Ingredients (all required by the roadmap)
    * Neural network      MLP  STATE_DIM → 64 → 64 → 4 actions
    * Experience replay   random mini-batches break the correlation
                          between consecutive negotiation steps
    * Target network      a slowly-updated copy of the network produces the
                          TD target, which stabilises training
    * ε-greedy            same exploration scheme as Q-learning

    loss   = Huber( Q(s,a;θ) , r + γ · max_a' Q_target(s',a') )   (r only if done)

The public interface mirrors QLearningAgent (choose_action / update /
decay_epsilon / save / load) so UserAgent can use either one.
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.state_space import STATE_DIM, STATE_VERSION


class QNetwork(nn.Module):
    """Fully-connected network: state vector → one Q-value per action."""

    def __init__(self, state_dim=STATE_DIM, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size FIFO store of (s, a, r, s', done) transitions."""

    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.as_tensor(np.array(s),  dtype=torch.float32),
            torch.as_tensor(a,            dtype=torch.int64).unsqueeze(1),
            torch.as_tensor(r,            dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(np.array(s2), dtype=torch.float32),
            torch.as_tensor(d,            dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    NUM_ACTIONS = 4

    def __init__(self, state_dim=STATE_DIM, lr=1e-3, gamma=0.95, epsilon=0.5,
                 batch_size=64, buffer_size=20000, target_sync=200, hidden=64):
        self.state_dim   = state_dim
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.batch_size  = batch_size
        self.target_sync = target_sync
        self.hidden      = hidden
        self.lr          = lr

        self.policy_net = QNetwork(state_dim, self.NUM_ACTIONS, hidden)
        self.target_net = QNetwork(state_dim, self.NUM_ACTIONS, hidden)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

        self.learn_steps      = 0
        self.last_loss        = None
        self.training_rewards = []
        self.training_history = []
        self.episode_count    = 0

    # ── Action selection ─────────────────────────────────────────────

    def choose_action(self, state):
        """ε-greedy over the network's Q-values. `state` is a float vector."""
        if random.random() < self.epsilon:
            return random.randrange(self.NUM_ACTIONS)
        return int(np.argmax(self.q_values(state)))

    def q_values(self, state):
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.policy_net(t).squeeze(0).numpy()

    # ── Learning ─────────────────────────────────────────────────────

    def update(self, state, action, reward, next_state, done=False):
        """Store the transition, then do one gradient step once the buffer is warm."""
        self.buffer.push(state, action, reward, next_state, done)
        if len(self.buffer) < self.batch_size:
            return None
        return self._learn()

    def _learn(self):
        s, a, r, s2, d = self.buffer.sample(self.batch_size)

        q_sa = self.policy_net(s).gather(1, a)                      # Q(s, a; θ)
        with torch.no_grad():
            q_next = self.target_net(s2).max(1, keepdim=True)[0]    # max_a' Q_target
            target = r + self.gamma * q_next * (1.0 - d)

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_sync == 0:                # hard target update
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.last_loss = float(loss.item())
        return self.last_loss

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.05):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    # ── Size / info ──────────────────────────────────────────────────

    @property
    def knowledge_size(self):
        """Trainable parameters in the policy network."""
        return sum(p.numel() for p in self.policy_net.parameters())

    knowledge_label = "Network parameters"

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path):
        torch.save({
            "state_version":    STATE_VERSION,
            "state_dim":        self.state_dim,
            "hidden":           self.hidden,
            "policy":           self.policy_net.state_dict(),
            "target":           self.target_net.state_dict(),
            "optimizer":        self.optimizer.state_dict(),
            "epsilon":          self.epsilon,
            "gamma":            self.gamma,
            "learn_steps":      self.learn_steps,
            "episode_count":    self.episode_count,
            "training_rewards": self.training_rewards,
            "training_history": self.training_history,
        }, path)
        print(f"[DQN-Agent] Saved model ({self.knowledge_size} parameters) to '{path}'")

    def load(self, path):
        """Load a saved model. Returns False if missing or incompatible."""
        if not os.path.exists(path):
            print(f"[DQN-Agent] No model found at '{path}'. Starting fresh.")
            return False

        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("state_version") != STATE_VERSION or data.get("state_dim") != self.state_dim:
            print(f"[DQN-Agent] '{path}' was trained with a different state encoding — ignoring it.")
            return False

        self.policy_net.load_state_dict(data["policy"])
        self.target_net.load_state_dict(data["target"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.epsilon          = data.get("epsilon", self.epsilon)
        self.gamma            = data.get("gamma", self.gamma)
        self.learn_steps      = data.get("learn_steps", 0)
        self.episode_count    = data.get("episode_count", 0)
        self.training_rewards = data.get("training_rewards", [])
        self.training_history = data.get("training_history", [])

        print(f"[DQN-Agent] Loaded model from '{path}' "
              f"(trained for {self.episode_count} episodes, ε={self.epsilon:.4f})")
        return True
