"""
rl/environment.py
-----------------
NegotiationEnv — the single negotiation environment shared by Q-learning,
DQN, the Flask API and the tests.

One episode = one negotiation between the user agent and ONE provider.

Actions (same 4 as before)
    0  RAISE_BID   raise the user's bid by BID_STEP x list price
    1  DOWNGRADE   move one tier down (Large → Medium → Small)
    2  ACCEPT      accept the provider's current offer
    3  WALK_AWAY   end the negotiation with no deal

Reward (Multi_Provider_Cloud_Data_Report.md §6.2)
    every step                 -0.3                       (encourages speed)
    deal closed                +5 success bonus
                               +10 x (list - price) / list     ← primary reward
                               +2  x remaining_rounds / max_rounds   (efficiency)
                               -1.5 x tiers_downgraded
    failure (walk away / patience exhausted / accepted over budget)  -5
    bid below the provider's floor                        -1 (extra)
    downgrading when the CURRENT tier is still feasible   -3 (extra)  ← new
                                                          (stops the agent
                                                          shrinking a request
                                                          it could have afforded)
    price used in the reward is clipped at the provider's floor, so the
    reward can never reflect a price the provider would not really give.
"""

import random

from agents.provider_agent import ProviderAgent, tier_names, tier_specs

ACTION_LABELS = [
    "Raise bid",
    "Downgrade resources (one tier down)",
    "Accept provider's offer",
    "Walk away — end negotiation",
]
NUM_ACTIONS = len(ACTION_LABELS)

RAISE_BID, DOWNGRADE, ACCEPT, WALK_AWAY = range(NUM_ACTIONS)

# Reward constants
STEP_COST       = -0.3
SUCCESS_BONUS   = 5.0
SAVINGS_SCALE   = 10.0
EFFICIENCY_SCALE = 2.0
DOWNGRADE_COST  = 1.5
FAILURE_PENALTY = -5.0
LOWBALL_PENALTY = -1.0
UNNECESSARY_DOWNGRADE_PENALTY = -3.0
JUSTIFIED_WALKAWAY_REWARD = 1.0     # reward bonus for correctly refusing an infeasible deal

# Bidding constants
INITIAL_BID_RATIO = 0.50     # opening bid = 50 % of list price (capped at budget)
BID_STEP          = 0.06     # each "raise" adds 6 % of list price


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


class NegotiationEnv:
    """Gym-style environment: reset() → obs, step(action) → (obs, reward, done, info)."""

    def __init__(self):
        self.tiers = tier_names()
        self.done  = False

    # ── Episode setup ────────────────────────────────────────────────

    def reset(self, provider, tier, quantity, budget, max_rounds=None):
        """
        Start a new negotiation.

        Args:
            provider   (str):   'AWS' | 'GCP' | 'Azure'
            tier       (str):   'Small' | 'Medium' | 'Large'
            quantity   (int):   number of instances requested
            budget     (float): max total monthly USD the user can pay
            max_rounds (int):   override provider patience (curriculum hook)
        """
        if tier not in self.tiers:
            raise ValueError(f"tier must be one of {self.tiers}")
        if quantity < 1:
            raise ValueError("quantity must be >= 1")
        if budget <= 0:
            raise ValueError("budget must be positive")

        self.provider       = ProviderAgent(provider, max_rounds=max_rounds)
        self.provider_name  = provider
        self.requested_tier = tier
        self.tier           = tier
        self.quantity       = int(quantity)
        self.budget         = float(budget)
        self.max_rounds     = self.provider.max_rounds
        self.round_num      = 1
        self.downgrades     = 0
        self.done           = False
        self.success        = False
        self.final_price    = 0.0
        self.total_reward   = 0.0

        # List price of the ORIGINAL request (used to report savings)
        self.requested_list_price = self.provider.list_price(tier, self.quantity)

        self.bid   = min(INITIAL_BID_RATIO * self.provider.list_price(tier, self.quantity),
                         self.budget)
        self.offer = self.provider.make_offer(self.tier, self.quantity, self.round_num)

        self.log = []
        self._log_offer()
        return self._obs()

    # ── Observation ──────────────────────────────────────────────────

    def _list_price(self):
        return self.provider.list_price(self.tier, self.quantity)

    def _obs(self):
        list_p = self._list_price()
        gap = _clip((self.offer - self.bid) / self.offer, 0.0, 1.0) if self.offer > 0 else 0.0
        return {
            "provider":     self.provider_name,
            "provider_idx": list(_provider_order()).index(self.provider_name),
            "tier":         self.tier,
            "tier_idx":     self.tiers.index(self.tier),
            "quantity":     self.quantity,
            "budget":       self.budget,
            "round":        self.round_num,
            "max_rounds":   self.max_rounds,
            "remaining":    self.max_rounds - self.round_num,
            "offer":        self.offer,
            "bid":          self.bid,
            "list_price":   list_p,
            "floor_price":  self.provider.floor_price(self.tier, self.quantity),
            "offer_ratio":  self.offer  / list_p,
            "bid_ratio":    self.bid    / list_p,
            "budget_ratio": self.budget / list_p,
            "gap":          gap,
        }

    # ── Feasibility ──────────────────────────────────────────────────

    def _feasible(self, tier):
        """
        Can a deal for `tier` still close within budget and patience?

        Looks ahead from the current round assuming the agent raises its bid
        every round (or simply accepts the offer): a deal is feasible if, in
        some remaining round, the provider's offer is within budget, or the
        bid has reached max(provider floor, offer x (1 - accept_margin)).
        """
        p      = self.provider
        list_p = p.list_price(tier, self.quantity)
        floor  = p.floor_price(tier, self.quantity)
        bid    = self.bid * list_p / self._list_price()   # bid expressed for `tier`
        for k, r in enumerate(range(self.round_num, self.max_rounds + 1), start=1):
            offer = p.make_offer(tier, self.quantity, r)
            if offer <= self.budget:
                return True
            bid_r = min(self.budget, bid + k * BID_STEP * list_p)
            if bid_r >= max(floor, offer * (1.0 - p.accept_margin)):
                return True
        return False

    # ── Logging helpers ──────────────────────────────────────────────

    def _config_text(self):
        spec = tier_specs()[self.tier]
        return (f"{self.quantity} x {self.tier} "
                f"({spec['vcpu']} vCPU / {spec['ram_gb']} GB)")

    def _log_offer(self):
        final = "  [FINAL ROUND]" if self.provider.is_final_round(self.round_num) else ""
        self.log.append(
            f"Round {self.round_num}: Provider offers ${self.offer:,.2f}/month "
            f"for {self._config_text()}{final}"
        )

    # ── Step ─────────────────────────────────────────────────────────

    def step(self, action):
        """
        Apply `action`. Returns (obs, reward, done, info).

        info (always) : {"event": str}
        info (on done): also success, final_price, rounds, reason
        """
        if self.done:
            raise RuntimeError("Episode finished — call reset() first.")
        if action not in range(NUM_ACTIONS):
            raise ValueError(f"action must be 0..{NUM_ACTIONS - 1}")

        self.log.append(f"Round {self.round_num}: Agent action → {ACTION_LABELS[action]}")
        reward = STEP_COST
        info   = {"event": ACTION_LABELS[action]}

        # ── 0: raise bid ─────────────────────────────────────────────
        if action == RAISE_BID:
            list_p   = self._list_price()
            self.bid = min(self.bid + BID_STEP * list_p, self.budget)
            self.log.append(f"  → Agent bids ${self.bid:,.2f}")

            if self.bid < self.provider.floor_price(self.tier, self.quantity):
                reward += LOWBALL_PENALTY          # bidding below absolute limit

            accepted, why = self.provider.evaluate_bid(
                self.bid, self.offer, self.tier, self.quantity)
            if accepted:
                price = min(self.bid, self.offer)  # never pay above the offer
                return self._close(price, reward, info)
            self.log.append(f"  → Provider rejected: {why}")

        # ── 1: downgrade one tier ────────────────────────────────────
        elif action == DOWNGRADE:
            idx = self.tiers.index(self.tier)
            if idx > 0:
                if self._feasible(self.tier):
                    reward += UNNECESSARY_DOWNGRADE_PENALTY
                    self.log.append("  (note: current tier was still affordable — penalised)")
                old_list = self._list_price()
                self.tier = self.tiers[idx - 1]
                self.downgrades += 1
                # keep the bid at the same fraction of list price
                self.bid *= self._list_price() / old_list
                self.bid  = min(self.bid, self.budget)
                self.offer = self.provider.make_offer(self.tier, self.quantity, self.round_num)
                self.log.append(f"  → Resources reduced to {self._config_text()}")
            else:
                self.log.append("  → Already at the smallest tier, cannot downgrade")

        # ── 2: accept provider's offer ───────────────────────────────
        elif action == ACCEPT:
            if self.offer <= self.budget:
                return self._close(self.offer, reward, info)
            self.log.append(
                f"  Accepted ${self.offer:,.2f} but it exceeds budget ${self.budget:,.2f}")
            return self._fail(reward, info, "accepted an over-budget offer")

# ── 3: walk away ─────────────────────────────────────────────
        else:
            self.log.append("  Negotiation ended by agent. No deal.")
            tiers = tier_names()
            curr_idx = tiers.index(self.tier)
            can_deal = any(self._feasible(t) for t in tiers[:curr_idx + 1])
            if can_deal:
                # Abandoned a deal that was still mathematically achievable
                return self._fail(reward, info, "agent walked away (deal was feasible)")
            else:
                # Justified refusal of an impossible negotiation
                justified_reward = reward + JUSTIFIED_WALKAWAY_REWARD
                self.done, self.success, self.final_price = True, False, 0.0
                self.total_reward += justified_reward
                info.update(success=False, final_price=0.0, rounds=self.round_num, reason="agent walked away (justified)")
                return self._obs(), justified_reward, True, info

        # ── Round finished without a deal ────────────────────────────
        if self.provider.is_final_round(self.round_num):
            self.log.append(
                f"Provider patience ({self.max_rounds} rounds) exhausted. Negotiation failed.")
            return self._fail(reward, info, "provider patience exhausted")

        self.round_num += 1
        self.offer = self.provider.make_offer(self.tier, self.quantity, self.round_num)
        self._log_offer()
        self.total_reward += reward
        return self._obs(), reward, False, info

    # ── Terminal helpers ─────────────────────────────────────────────

    def _close(self, price, reward, info):
        """Deal closed at `price`. Applies the success reward."""
        list_p = self._list_price()
        floor  = self.provider.floor_price(self.tier, self.quantity)

        effective = max(price, floor)                 # per-provider floor clipping
        savings   = (list_p - effective) / list_p

        reward += SUCCESS_BONUS
        reward += SAVINGS_SCALE * savings
        reward += EFFICIENCY_SCALE * (self.max_rounds - self.round_num) / self.max_rounds
        reward -= DOWNGRADE_COST * self.downgrades

        self.done, self.success, self.final_price = True, True, price
        self.total_reward += reward
        self.log.append(f"  Deal closed at ${price:,.2f}/month "
                        f"({savings * 100:.1f}% below list price)")
        info.update(success=True, final_price=price, rounds=self.round_num,
                    reason="deal closed")
        return self._obs(), reward, True, info

    def _fail(self, reward, info, reason):
        reward += FAILURE_PENALTY
        self.done, self.success, self.final_price = True, False, 0.0
        self.total_reward += reward
        info.update(success=False, final_price=0.0, rounds=self.round_num, reason=reason)
        return self._obs(), reward, True, info


def _provider_order():
    from rl.state_space import PROVIDERS
    return PROVIDERS
