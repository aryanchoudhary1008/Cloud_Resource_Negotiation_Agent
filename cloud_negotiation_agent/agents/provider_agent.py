"""
agents/provider_agent.py
------------------------
Cloud Provider Agent — rule-based, configuration-driven (AWS / GCP / Azure).

The provider is intentionally RULE-BASED (not RL):
  1. Real providers have fixed, transparent pricing models.
  2. The RL user agent needs a stable adversary to learn against.
  3. Each provider has a distinct "personality", so the RL agent must learn
     provider-specific policies instead of one generic policy.

All numbers come from providers.json — nothing is hard-coded here.

Pricing model
    list_price = hourly_on_demand(tier) x 730 hours x quantity     (USD / month)

Negotiation behaviour (per provider, from providers.json)
    concession_rate    how much of the list price it drops each round
    price_floor_ratio  it never sells below  floor_ratio x list_price
    max_rounds         its patience; after that it walks away
    accept_margin      it accepts a bid within this fraction of its own offer
    volume discount    quantity-based discount applied to its offers
"""

import json
from functools import lru_cache

from config import PROVIDERS_JSON


# ── Config loading ───────────────────────────────────────────────────

@lru_cache(maxsize=4)
def load_config(path=PROVIDERS_JSON):
    """Load and cache providers.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def provider_names():
    """Provider names in file order, e.g. ['AWS', 'GCP', 'Azure']."""
    return list(load_config()["providers"].keys())


def tier_names():
    """Tier names ordered cheapest → most expensive: ['Small', 'Medium', 'Large']."""
    return list(load_config()["tiers"].keys())


def tier_specs():
    """Mapping tier → {'vcpu', 'ram_gb', 'examples'}."""
    return load_config()["tiers"]


def hours_per_month():
    return load_config()["_meta"]["hours_per_month"]


# ── Provider agent ───────────────────────────────────────────────────

class ProviderAgent:
    """
    Rule-based provider for ONE negotiation episode.

    Args:
        name       (str): 'AWS' | 'GCP' | 'Azure' (any key in providers.json).
        max_rounds (int): Patience for this episode. Defaults to the
                          provider's configured max_rounds. The curriculum
                          pipeline (Phase 2) overrides this per difficulty.
    """

    def __init__(self, name, max_rounds=None):
        providers = load_config()["providers"]
        if name not in providers:
            raise ValueError(f"Unknown provider '{name}'. Choose from {list(providers)}")

        self.name   = name
        self.cfg    = providers[name]
        self.floor_ratio     = self.cfg["price_floor_ratio"]
        self.concession_rate = self.cfg["concession_rate"]
        self.accept_margin   = self.cfg["accept_margin"]
        self.max_rounds      = int(max_rounds if max_rounds is not None
                                   else self.cfg["max_rounds"])

    # ── Prices ───────────────────────────────────────────────────────

    def hourly_price(self, tier, kind="on_demand"):
        """Hourly price for one instance. kind: on_demand|reserved_1yr|reserved_3yr|spot."""
        return self.cfg["pricing"][tier][kind]

    def list_price(self, tier, quantity):
        """Undiscounted monthly list (on-demand) price for `quantity` instances."""
        return self.hourly_price(tier) * hours_per_month() * quantity

    def floor_price(self, tier, quantity):
        """Absolute minimum monthly price this provider will ever accept."""
        return self.list_price(tier, quantity) * self.floor_ratio

    def volume_discount(self, quantity):
        """Largest discount whose quantity threshold has been reached (0.0 if none)."""
        discount = 0.0
        for threshold, d in self.cfg.get("volume_discount_tiers", []):
            if quantity >= threshold:
                discount = d
        return discount

    # ── Negotiation behaviour ────────────────────────────────────────

    def make_offer(self, tier, quantity, round_num):
        """
        Provider's asking price for the given round (monthly USD).

            factor = max(floor_ratio, 1 - concession_rate x (round - 1))
            offer  = list_price x (1 - volume_discount) x factor
            offer  = max(offer, floor_price)          # never below the floor

        Round 1 → full (volume-discounted) list price; each later round
        concedes `concession_rate` more, until the floor is reached.
        """
        list_p = self.list_price(tier, quantity)
        factor = max(self.floor_ratio, 1.0 - self.concession_rate * (round_num - 1))
        offer  = list_p * (1.0 - self.volume_discount(quantity)) * factor
        return round(max(offer, self.floor_price(tier, quantity)), 2)

    def evaluate_bid(self, bid, offer, tier, quantity):
        """
        Decide whether to accept the user's counter-bid.

        Accepts only if the bid is
            (a) at or above the provider's absolute floor, AND
            (b) within `accept_margin` of the provider's current offer.

        Returns:
            tuple[bool, str]: (accepted, reason)
        """
        floor = self.floor_price(tier, quantity)
        if bid < floor:
            return False, f"bid ${bid:,.2f} is below provider floor ${floor:,.2f}"
        if bid < offer * (1.0 - self.accept_margin):
            return False, (f"bid ${bid:,.2f} is too far below current offer "
                           f"${offer:,.2f}")
        return True, "bid accepted"

    def is_final_round(self, round_num):
        """True when the provider's patience is exhausted after this round."""
        return round_num >= self.max_rounds
