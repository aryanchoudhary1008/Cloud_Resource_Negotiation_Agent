"""
agents/provider_agent.py
------------------------
Cloud Provider Agent — Rule-based pricing and negotiation logic.

This agent simulates a cloud service provider (like AWS, GCP, Azure).
It is intentionally RULE-BASED (not RL) because:
  1. Real providers have fixed, transparent pricing models.
  2. The RL user agent needs a stable adversary to learn against.
  3. It makes the system easier to understand and explain.

Pricing model:
    base_price = (cpu * 2.5) + (ram * 1.5) + (storage * 0.1)

Negotiation strategy:
    - Round 1: offer at full base price
    - Rounds 2–4: reduce price by 5% per round (to attract the customer)
    - Round 4+: final offer, no further reductions
    - Reject any user counter-offer below 60% of base price (unprofitable)
"""


class ProviderAgent:
    """
    Rule-based cloud provider agent.

    Pricing coefficients (USD per unit per hour, simplified):
        CPU:     $2.50 per core
        RAM:     $1.50 per GB
        Storage: $0.10 per GB
    """

    # ── Pricing Constants ────────────────────────────────────────────
    CPU_RATE     = 2.5   # USD per core
    RAM_RATE     = 1.5   # USD per GB
    STORAGE_RATE = 0.1   # USD per GB

    # ── Negotiation Constants ────────────────────────────────────────
    DISCOUNT_PER_ROUND   = 0.05   # 5% price reduction per round
    MIN_PROFIT_RATIO     = 0.60   # reject if user offers < 60% of base
    FINAL_OFFER_ROUND    = 4      # no further discounts at/after this round

    # ─────────────────────────────────────────────────────────────────

    def calculate_base_price(self, cpu, ram, storage):
        """
        Compute the undiscounted base price for a resource configuration.

        Formula:
            base = (cpu × 2.5) + (ram × 1.5) + (storage × 0.1)

        Args:
            cpu     (int): Number of CPU cores.
            ram     (int): RAM in GB.
            storage (int): Storage in GB.

        Returns:
            float: Base price in USD (2 decimal places).

        Examples:
            4 CPU, 8GB RAM, 100GB → (10 + 12 + 10) = $32.00
            8 CPU, 16GB RAM, 200GB → (20 + 24 + 20) = $64.00
        """
        price = (cpu * self.CPU_RATE) + (ram * self.RAM_RATE) + (storage * self.STORAGE_RATE)
        return round(price, 2)

    def make_offer(self, cpu, ram, storage, round_num):
        """
        Generate the provider's price offer for the current round.

        Discount schedule:
            Round 1 → 0% off   (full base price)
            Round 2 → 5% off
            Round 3 → 10% off
            Round 4 → 15% off  (final — no further reduction)
            Round 5 → 15% off  (same as round 4, already final)

        Args:
            cpu      (int): CPU cores.
            ram      (int): RAM in GB.
            storage  (int): Storage in GB.
            round_num(int): Current negotiation round (1–5).

        Returns:
            dict: {
                "price":      float  — offered price this round,
                "cpu":        int,
                "ram":        int,
                "storage":    int,
                "round":      int,
                "base_price": float  — undiscounted base price,
                "is_final":   bool   — True if no further reductions coming
            }
        """
        base_price = self.calculate_base_price(cpu, ram, storage)

        # Cap effective round for discount calculation
        # (rounds beyond FINAL_OFFER_ROUND get the same max discount)
        effective_round = min(round_num, self.FINAL_OFFER_ROUND)

        # Each round (starting at round 2) shaves off 5%
        # Round 1: factor = 1.00 (0 discounts applied)
        # Round 2: factor = 0.95 (1 discount)
        # Round 3: factor = 0.90 (2 discounts)
        # Round 4: factor = 0.85 (3 discounts, final)
        num_discounts    = effective_round - 1
        discount_factor  = 1.0 - (num_discounts * self.DISCOUNT_PER_ROUND)
        offered_price    = round(base_price * discount_factor, 2)

        return {
            "price":      offered_price,
            "cpu":        cpu,
            "ram":        ram,
            "storage":    storage,
            "round":      round_num,
            "base_price": base_price,
            "is_final":   round_num >= self.FINAL_OFFER_ROUND,
        }

    def evaluate_user_offer(self, user_price, base_price):
        """
        Decide whether the provider accepts the user's counter-offer.

        The provider has a minimum margin requirement: it will only accept
        if the user pays at least 60% of the base price. Below this
        threshold the deal is unprofitable.

        Args:
            user_price (float): The price the user is currently willing to pay.
            base_price (float): The original undiscounted base price.

        Returns:
            bool: True if the provider accepts; False if rejected.
        """
        min_acceptable = base_price * self.MIN_PROFIT_RATIO
        return user_price >= min_acceptable

    def is_final_offer(self, round_num):
        """
        Returns True if the provider has reached its last possible discount.

        Args:
            round_num (int): Current round number.

        Returns:
            bool
        """
        return round_num >= self.FINAL_OFFER_ROUND
