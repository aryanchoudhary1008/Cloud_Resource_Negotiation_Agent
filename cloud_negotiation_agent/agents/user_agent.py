"""
agents/user_agent.py
--------------------
User Agent — the RL customer that negotiates with a cloud provider.

It can be driven by either learning algorithm:
    algorithm="qlearning"  tabular Q-learning   (rl/q_learning.py)
    algorithm="dqn"        Deep Q-Network       (rl/dqn_agent.py, needs PyTorch)

The negotiation itself (rules, rewards, logging) lives in
rl/environment.py, so both algorithms face exactly the same problem.

Training mode : the agent learns after every step.
Inference mode: weights/table are frozen, ε = 0.05.
"""

from config import MODEL_PATHS, ALGORITHMS
from rl.environment import NegotiationEnv
from rl.state_space  import encode_tabular, encode_vector
from agents.provider_agent import tier_specs, provider_names


class NegotiationResult:
    """Outcome of one negotiation session."""

    def __init__(self, env, algorithm):
        specs = tier_specs()
        self.algorithm      = algorithm
        self.provider       = env.provider_name
        self.success        = env.success
        self.rounds         = env.round_num
        self.max_rounds     = env.max_rounds
        self.final_price    = env.final_price
        # list_price = on-demand price of the config actually delivered, so
        # savings measure negotiation skill, not just downgrading
        self.list_price     = env._list_price()
        self.requested_list_price = env.requested_list_price
        self.requested_tier = env.requested_tier
        self.tier           = env.tier
        self.quantity       = env.quantity
        self.downgrades     = env.downgrades
        self.final_cpu      = specs[env.tier]["vcpu"]
        self.final_ram      = specs[env.tier]["ram_gb"]
        self.requested_cpu  = specs[env.requested_tier]["vcpu"]
        self.requested_ram  = specs[env.requested_tier]["ram_gb"]
        self.budget         = env.budget
        self.total_reward   = env.total_reward
        self.log            = env.log

    def to_dict(self):
        saving = None
        if self.success and self.list_price > 0:
            saving = round((self.list_price - self.final_price) / self.list_price * 100, 1)
        return {
            "success":        self.success,
            "provider":       self.provider,
            "algorithm":      self.algorithm,
            "rounds":         self.rounds,
            "max_rounds":     self.max_rounds,
            "final_price":    round(self.final_price, 2),
            "list_price":     round(self.list_price, 2),
            "requested_list_price": round(self.requested_list_price, 2),
            "savings_pct":    saving,
            "requested_tier": self.requested_tier,
            "tier":           self.tier,
            "quantity":       self.quantity,
            "downgrades":     self.downgrades,
            "final_cpu":      self.final_cpu,
            "final_ram":      self.final_ram,
            "budget":         self.budget,
            "total_reward":   round(self.total_reward, 2),
            "log":            self.log,
        }


def choose_best(results):
    """
    Pick the winning negotiation among several providers.

    Ranking (first rule that differs wins):
      1. a closed deal beats a failed one
      2. fewer tier downgrades  (the user's requirement is kept)
      3. lower final price
      4. fewer rounds
    If every provider failed, return the attempt with the best reward.
    """
    closed = [r for r in results if r.success]
    if closed:
        return min(closed, key=lambda r: (r.downgrades, r.final_price, r.rounds))
    return max(results, key=lambda r: r.total_reward)


def make_learner(algorithm):
    """Create the underlying learning agent (imports PyTorch only for DQN)."""
    if algorithm == "qlearning":
        from rl.q_learning import QLearningAgent
        return QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.5)
    if algorithm == "dqn":
        from rl.dqn_agent import DQNAgent          # raises ImportError without torch
        return DQNAgent()
    raise ValueError(f"algorithm must be one of {ALGORITHMS}")


class UserAgent:
    """RL negotiator wrapping a Q-learning or DQN learner and a NegotiationEnv."""

    def __init__(self, algorithm="qlearning", training_mode=False,
                 model_path=None, load=True):
        self.algorithm     = algorithm
        self.training_mode = training_mode
        self.model_path    = model_path or MODEL_PATHS[algorithm]

        self.agent  = make_learner(algorithm)
        self.encode = encode_tabular if algorithm == "qlearning" else encode_vector
        self.env    = NegotiationEnv()

        if load:
            self.agent.load(self.model_path)
        if not training_mode:
            self.agent.epsilon = 0.05

    # ── Public API ───────────────────────────────────────────────────

    def negotiate(self, provider, tier, quantity, budget, max_rounds=None):
        """
        Run one full negotiation and return a NegotiationResult.

        Each round: observe → encode state → choose action → env.step() →
        (training only) learn from the transition.
        """
        obs = self.env.reset(provider, tier, quantity, budget, max_rounds=max_rounds)
        done = False

        while not done:
            state  = self.encode(obs)
            action = self.agent.choose_action(state)
            next_obs, reward, done, _ = self.env.step(action)

            if self.training_mode:
                self.agent.update(state, action, reward, self.encode(next_obs), done)
            obs = next_obs

        return NegotiationResult(self.env, self.algorithm)

    def negotiate_best(self, tier, quantity, budget, providers=None):
        """
        "Auto" mode: negotiate with every provider and keep the best deal.

        Returns:
            (winner, results) — the winning NegotiationResult and the list of
            all results (one per provider, in provider order).
        """
        results = [self.negotiate(p, tier, quantity, budget)
                   for p in (providers or provider_names())]
        return choose_best(results), results

    def save(self):
        self.agent.save(self.model_path)

    def reload(self):
        self.agent.load(self.model_path)

    def set_training_mode(self, mode: bool):
        self.training_mode = mode
        if mode:
            self.agent.epsilon = max(self.agent.epsilon, 0.1)
        else:
            self.agent.epsilon = 0.05
