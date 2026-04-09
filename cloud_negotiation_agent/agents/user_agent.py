"""
agents/user_agent.py
--------------------
User Agent — RL-based negotiation using Q-learning.

The User Agent represents a cloud customer who negotiates resource
pricing. It uses a Q-learning policy to decide what to do each round:
    - Pay more (increase offered price)
    - Ask for less (downgrade CPU/RAM tier)
    - Accept the current offer
    - Reject and walk away

Over many training episodes, the agent learns which actions lead to
the best outcomes (good price, within budget, fewer rounds wasted).

Training mode:  Q-table is updated after every step.
Inference mode: Uses the pre-trained Q-table with minimal exploration.
"""

from rl.q_learning   import QLearningAgent
from rl.state_space  import encode_state, CPU_BINS, RAM_BINS
from agents.provider_agent import ProviderAgent

# Default file path for the saved Q-table
Q_TABLE_PATH = "q_table.pkl"


# ── Result Container ─────────────────────────────────────────────────

class NegotiationResult:
    """
    Holds the complete outcome of a single negotiation session.

    Attributes:
        success      (bool):  Whether a deal was reached.
        rounds       (int):   How many rounds the negotiation took.
        final_price  (float): Agreed price (0.0 if no deal).
        final_cpu    (int):   CPU cores in final deal (may be downgraded).
        final_ram    (int):   RAM in final deal (may be downgraded).
        storage      (int):   Storage requested (never negotiated down).
        budget       (float): User's original budget.
        total_reward (float): Cumulative RL reward earned this episode.
        log          (list):  Human-readable round-by-round description.
    """

    def __init__(self, success, rounds, final_price, final_cpu, final_ram,
                 storage, budget, total_reward, log):
        self.success      = success
        self.rounds       = rounds
        self.final_price  = final_price
        self.final_cpu    = final_cpu
        self.final_ram    = final_ram
        self.storage      = storage
        self.budget       = budget
        self.total_reward = total_reward
        self.log          = log

    def to_dict(self):
        """Serialize to a JSON-safe dict for API responses."""
        return {
            "success":      self.success,
            "rounds":       self.rounds,
            "final_price":  round(self.final_price, 2),
            "final_cpu":    self.final_cpu,
            "final_ram":    self.final_ram,
            "storage":      self.storage,
            "budget":       self.budget,
            "total_reward": round(self.total_reward, 2),
            "log":          self.log,
        }


# ── User Agent ───────────────────────────────────────────────────────

class UserAgent:
    """
    RL-based user agent that negotiates cloud resource prices.

    Internally wraps a QLearningAgent and a ProviderAgent.
    The negotiate() method runs the full multi-round loop.

    Training mode (training_mode=True):
        - Q-table is updated after every (state, action, reward, next_state).
        - Epsilon stays at 0.3 initially (lots of exploration).
        - Call save_q_table() after training to persist progress.

    Inference mode (training_mode=False):
        - Q-table is frozen (no updates).
        - Epsilon is 0.05 (mostly greedy, a little randomness).
    """

    MAX_ROUNDS = 5

    def __init__(self, training_mode=False, q_table_path=Q_TABLE_PATH):
        """
        Args:
            training_mode (bool): Enable Q-table updates during negotiation.
            q_table_path  (str):  Path to load/save the Q-table pickle.
        """
        self.training_mode = training_mode
        self.q_table_path  = q_table_path

        # Core RL agent
        self.ql_agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.3)

        # Rule-based opponent
        self.provider = ProviderAgent()

        # Attempt to load pre-trained Q-table
        self.ql_agent.load(q_table_path)

        # In inference mode, mostly exploit the learned policy
        if not training_mode:
            self.ql_agent.epsilon = 0.05

    # ── Public API ───────────────────────────────────────────────────

    def negotiate(self, cpu, ram, storage, budget):
        """
        Run the full negotiation loop for up to MAX_ROUNDS rounds.

        Each round:
          1. Provider makes an offer (price decreases each round).
          2. Encode the current negotiation state → Q-table key.
          3. Agent picks an action (epsilon-greedy from Q-table).
          4. Apply action: adjust price / downgrade resources / accept / reject.
          5. Compute reward.
          6. Update Q-table (only in training mode).
          7. If deal accepted or rejected → return result immediately.
          8. Otherwise → continue to next round.

        If MAX_ROUNDS pass with no decision → negotiation fails.

        Args:
            cpu     (int):   CPU cores originally requested.
            ram     (int):   RAM in GB originally requested.
            storage (int):   Storage in GB (not negotiated down).
            budget  (float): User's maximum acceptable price.

        Returns:
            NegotiationResult: Full outcome with log and metrics.
        """
        log          = []       # round-by-round narrative
        total_reward = 0.0

        # Mutable resources — may be downgraded during negotiation
        current_cpu = cpu
        current_ram = ram

        # The user's initial "willingness to pay" equals their full budget
        # (they'll try to get a lower price, but can pay up to budget)
        user_offered_price = budget

        # Get the undiscounted base price for this configuration
        base_price = self.provider.calculate_base_price(cpu, ram, storage)

        # ── Negotiation loop ─────────────────────────────────────────
        for round_num in range(1, self.MAX_ROUNDS + 1):

            # Step 1: Provider makes its offer for this round
            offer          = self.provider.make_offer(current_cpu, current_ram, storage, round_num)
            provider_price = offer["price"]
            final_flag     = "  [FINAL OFFER]" if offer["is_final"] else ""

            log.append(
                f"Round {round_num}: Provider offers ${provider_price:.2f} "
                f"for {current_cpu} CPU, {current_ram}GB RAM, "
                f"{storage}GB storage{final_flag}"
            )

            # Step 2: Encode current state → hashable tuple for Q-table lookup
            state = encode_state(
                cpu=current_cpu,
                ram=current_ram,
                budget=budget,
                current_price=provider_price,
                round_num=round_num,
                max_price=base_price,
            )

            # Step 3: Agent selects an action
            action = self.ql_agent.choose_action(state)
            action_labels = [
                "Increase offered price by 10%",
                "Decrease resource request",
                "Accept provider's offer",
                "Reject — end negotiation",
            ]
            log.append(f"Round {round_num}: Agent action → {action_labels[action]}")

            # Step 4 & 5: Apply action and compute immediate reward
            # Base reward: -1 per round (penalizes slow deals)
            reward = -1.0
            done   = False

            # ── ACTION 0: Raise user's offered price ─────────────────
            if action == 0:
                # User signals willingness to pay more (up to their budget cap)
                user_offered_price = min(user_offered_price * 1.10, budget)

                log.append(f"  → User now offering ${user_offered_price:.2f}")

                # Check if provider accepts this counter-offer
                if self.provider.evaluate_user_offer(user_offered_price, base_price):
                    # Deal struck — use the lower of the two prices
                    final_price = min(user_offered_price, provider_price)
                    done = True

                    if final_price <= budget:
                        reward += 10.0   # good deal — within budget
                        log.append(f"  Provider accepted! Deal at ${final_price:.2f}")
                    else:
                        reward += -5.0   # deal but over budget
                        log.append(f"  Provider accepted, but over budget: ${final_price:.2f}")

                    self._update_q(state, action, reward, state, done=True)
                    total_reward += reward

                    return NegotiationResult(
                        success=True,  rounds=round_num,
                        final_price=final_price,
                        final_cpu=current_cpu,   final_ram=current_ram,
                        storage=storage,          budget=budget,
                        total_reward=total_reward, log=log,
                    )
                else:
                    log.append(
                        f"  → Provider rejected ${user_offered_price:.2f} "
                        f"(minimum: ${base_price * self.provider.MIN_PROFIT_RATIO:.2f})"
                    )

            # ── ACTION 1: Downgrade resources ─────────────────────────
            elif action == 1:
                # Reduce CPU and/or RAM by one tier to lower the price
                cpu_idx = CPU_BINS.index(current_cpu) if current_cpu in CPU_BINS else 1
                ram_idx = RAM_BINS.index(current_ram) if current_ram in RAM_BINS else 1

                downgraded = False
                if cpu_idx > 0:
                    current_cpu = CPU_BINS[cpu_idx - 1]
                    downgraded  = True
                if ram_idx > 0:
                    current_ram = RAM_BINS[ram_idx - 1]
                    downgraded  = True

                if downgraded:
                    log.append(
                        f"  → Resources reduced to {current_cpu} CPU, {current_ram}GB RAM"
                    )
                    # Recalculate base price with the new lower config
                    base_price = self.provider.calculate_base_price(current_cpu, current_ram, storage)
                else:
                    log.append("  → Already at minimum resources, cannot downgrade further")

            # ── ACTION 2: Accept provider's current offer ─────────────
            elif action == 2:
                final_price = provider_price
                done = True

                if final_price <= budget:
                    reward += 10.0
                    log.append(f"  Deal accepted at ${final_price:.2f}!")
                else:
                    reward += -5.0
                    log.append(f"  Accepted but over budget: ${final_price:.2f}")

                self._update_q(state, action, reward, state, done=True)
                total_reward += reward

                return NegotiationResult(
                    success=True,  rounds=round_num,
                    final_price=final_price,
                    final_cpu=current_cpu,   final_ram=current_ram,
                    storage=storage,          budget=budget,
                    total_reward=total_reward, log=log,
                )

            # ── ACTION 3: Reject — terminate negotiation ──────────────
            elif action == 3:
                reward += -10.0   # heavy penalty for walking away
                done    = True
                log.append("  Negotiation rejected by agent. No deal.")

                self._update_q(state, action, reward, state, done=True)
                total_reward += reward

                return NegotiationResult(
                    success=False, rounds=round_num,
                    final_price=0.0,
                    final_cpu=current_cpu,   final_ram=current_ram,
                    storage=storage,          budget=budget,
                    total_reward=total_reward, log=log,
                )

            # ── Q-table update for non-terminal steps ─────────────────
            # Compute what the state WILL look like next round
            if not done:
                next_offer = self.provider.make_offer(
                    current_cpu, current_ram, storage, round_num + 1
                )
                next_state = encode_state(
                    cpu=current_cpu,
                    ram=current_ram,
                    budget=budget,
                    current_price=next_offer["price"],
                    round_num=round_num + 1,
                    max_price=base_price,
                )
                self._update_q(state, action, reward, next_state, done=False)
                total_reward += reward

        # ── Max rounds exhausted without a deal ──────────────────────
        reward = -10.0
        total_reward += reward
        log.append(f"Maximum {self.MAX_ROUNDS} rounds reached. Negotiation failed.")

        return NegotiationResult(
            success=False, rounds=self.MAX_ROUNDS,
            final_price=0.0,
            final_cpu=current_cpu,    final_ram=current_ram,
            storage=storage,           budget=budget,
            total_reward=total_reward, log=log,
        )

    # ── Internal Helpers ─────────────────────────────────────────────

    def _update_q(self, state, action, reward, next_state, done):
        """Update Q-table only when in training mode."""
        if self.training_mode:
            self.ql_agent.update(state, action, reward, next_state, done)

    def save_q_table(self):
        """Persist the Q-table to disk."""
        self.ql_agent.save(self.q_table_path)

    def set_training_mode(self, mode: bool):
        """
        Toggle training mode at runtime.

        Args:
            mode (bool): True to enable Q-table updates; False to freeze.
        """
        self.training_mode = mode
        # Adjust exploration rate based on mode
        if mode:
            self.ql_agent.epsilon = max(self.ql_agent.epsilon, 0.1)
        else:
            self.ql_agent.epsilon = 0.05
