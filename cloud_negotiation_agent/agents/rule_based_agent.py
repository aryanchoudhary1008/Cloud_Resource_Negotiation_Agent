"""
agents/rule_based_agent.py
--------------------------
Deterministic rule-based USER-side negotiation baseline.

This is NOT the ProviderAgent (provider-side, rule-based by design).
This is a CUSTOMER-side heuristic that negotiates against the same
ProviderAgent as Q-learning and DQN, giving a no-learning comparison point.

Strategy (in priority order each round):
  1. If provider's current offer is within budget  → ACCEPT immediately.
  2. If current tier is provably infeasible (cannot close in remaining
     rounds even with maximum raising)             → try DOWNGRADE.
       a. If already at Small and still infeasible → WALK_AWAY.
  3. Otherwise                                     → RAISE_BID.

Design rationale
  * The rule is deterministic (no randomness), so it is perfectly
    reproducible without seeding.
  * It is the *strongest reasonable heuristic*: it never misses an
    in-budget offer, always tries downgrading before giving up, and
    only walks away when no deal is mathematically possible. This
    makes it a meaningful baseline—if RL doesn't clearly beat it, the
    training pipeline needs work.
  * It uses the same NegotiationEnv so the environment dynamics,
    rewards, and termination conditions are identical to RL agents.

Interface
  negotiate(provider, tier, quantity, budget, **kwargs)
      → NegotiationResult  (same object as UserAgent returns)
  The **kwargs absorb unused fields (e.g. 'id', 'difficulty', 'feasible',
  'split') so the agent can be called directly with a dataset row dict.

Edge cases
  * Budget exactly equals offer: accepted (≤ is used, matching env logic).
  * Already at Small, still infeasible: walks away rather than looping.
  * Downgrade while at Medium/Large where a deal is still feasible for
    the current tier: the env applies UNNECESSARY_DOWNGRADE_PENALTY, but
    this baseline never does that—it only downgrades when _feasible()
    returns False for the current tier.
  * Patience exhausted mid-strategy: the env ends the episode regardless;
    RAISE_BID in the final round still counts as the agent's last action.
"""

from rl.environment import NegotiationEnv, RAISE_BID, DOWNGRADE, ACCEPT, WALK_AWAY
from agents.provider_agent import tier_names


class _DummyAgent:
    """Minimal stub so evaluate_agent() can read epsilon/episode_count."""
    epsilon = 0.0
    episode_count = 0
    training_history = []


class RuleBasedAgent:
    """
    Deterministic rule-based negotiation baseline.

    Exposes the same negotiate() / negotiate_best() surface as UserAgent
    so it can be passed directly to evaluate_agent() and evaluate_auto().
    """

    # Used by evaluate_agent() when it reads user_agent.training_mode
    training_mode = False
    # Stub so evaluate_agent() epsilon-save/restore works without crashing
    agent = _DummyAgent()

    def __init__(self):
        self.env = NegotiationEnv()
        self.algorithm = "rule_based"

    # ── Compatibility stubs (called by evaluate_agent internals) ──────

    def set_training_mode(self, mode: bool):
        self.training_mode = mode   # no-op for a rule-based agent

    def save(self):
        pass                        # nothing to persist

    def reload(self):
        pass

    # ── Core negotiation ──────────────────────────────────────────────

    def negotiate(self, provider, tier, quantity, budget, max_rounds=None, **_kwargs):
        """
        Run one full negotiation deterministically.

        Returns a NegotiationResult-compatible object whose attributes
        match exactly what evaluate_agent() reads:
            .provider, .success, .total_reward, .rounds,
            .list_price, .final_price, .downgrades, .log
        """
        obs = self.env.reset(provider, tier, quantity, budget,
                             max_rounds=max_rounds)
        done = False

        while not done:
            action = self._choose_action(obs)
            obs, _reward, done, _info = self.env.step(action)

        from agents.user_agent import NegotiationResult
        return NegotiationResult(self.env, self.algorithm)

    def negotiate_best(self, tier, quantity, budget, providers=None):
        """Auto mode: negotiate with every provider, pick the best deal."""
        from agents.provider_agent import provider_names
        from agents.user_agent import choose_best
        results = [self.negotiate(p, tier, quantity, budget)
                   for p in (providers or provider_names())]
        return choose_best(results), results

    # ── Decision logic ────────────────────────────────────────────────

    def _choose_action(self, obs):
        """
        Deterministic action selection based on the current observation.

        Priority:
          1. Offer <= budget  -> ACCEPT
          2. Current tier infeasible  -> DOWNGRADE or WALK_AWAY
          3. Otherwise  -> RAISE_BID
        """
        # 1. Accept if the provider's current offer is within budget
        if obs["offer"] <= obs["budget"]:
            return ACCEPT

        # 2. Check feasibility of remaining rounds for current tier
        if not self.env._feasible(obs["tier"]):
            tiers = tier_names()
            current_idx = tiers.index(obs["tier"])
            if current_idx > 0:
                return DOWNGRADE
            # Already at the smallest tier and still infeasible
            return WALK_AWAY

        # 3. Raise bid — keep negotiating
        return RAISE_BID

