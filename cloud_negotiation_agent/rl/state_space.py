"""
rl/state_space.py
-----------------
State encoding for BOTH learning agents.

The environment (rl/environment.py) produces an *observation dict*.
Two encoders turn it into what each algorithm needs:

    encode_tabular(obs) → small hashable tuple   (Q-learning dictionary key)
    encode_vector(obs)  → float32 numpy vector   (DQN neural-network input)

Why two encoders?
    A Q-table needs a coarse, discrete key; adding a provider one-hot there
    would just blow up the table. A neural network instead takes a
    continuous normalised vector and generalises across nearby states —
    that generalisation is exactly what the DQN-vs-Q-learning comparison
    is meant to show.

Observation fields used (all produced by NegotiationEnv):
    provider_idx, tier_idx, quantity, round, max_rounds, remaining,
    offer_ratio  = provider's current offer / list price
    bid_ratio    = user's current bid       / list price
    budget_ratio = user's budget            / list price
    gap          = (offer - bid) / offer, clipped to [0, 1]
"""

import math
import numpy as np

from agents.provider_agent import provider_names, tier_names

PROVIDERS = provider_names()
TIERS     = tier_names()

# Bump this whenever the encoding OR the reward function changes (v4: price_gap_to_floor feature).
# Saved models carry the version and are ignored if it does not match (old q_table.pkl/dqn_model.pt are stale).
STATE_VERSION = 4

# Dimension of the DQN input vector:
#   provider one-hot + tier one-hot + 8 continuous features
STATE_DIM = len(PROVIDERS) + len(TIERS) + 8

# Tabular bin edges
BUDGET_EDGES = [0.55, 0.70, 0.85, 1.00]      # → 5 budget bins
GAP_EDGES    = [0.03, 0.10, 0.20, 0.30]      # → 5 gap bins
PRICE_BINS   = 6                              # offer_ratio 0.5..1.0+ in steps of 0.1


def _bin(value, edges):
    """Index of the first edge that `value` does not exceed (len(edges) if above all)."""
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def _patience_bin(remaining):
    if remaining <= 0:
        return 0
    if remaining <= 2:
        return 1
    if remaining <= 4:
        return 2
    return 3


def encode_tabular(obs):
    """
    Encode an observation as a hashable tuple for the Q-table.

    Returns:
        (provider, tier, price_bin, budget_bin, gap_bin, patience_bin)
        3 x 3 x 6 x 5 x 5 x 4 = 5,400 possible states.
    """
    price_bin = int(np.clip((obs["offer_ratio"] - 0.5) / 0.1, 0, PRICE_BINS - 1))
    return (
        obs["provider_idx"],
        obs["tier_idx"],
        price_bin,
        _bin(obs["budget_ratio"], BUDGET_EDGES),
        _bin(obs["gap"], GAP_EDGES),
        _patience_bin(obs["remaining"]),
    )


def encode_vector(obs):
    """
    Encode an observation as a normalised float32 vector for the DQN.

    Layout (STATE_DIM entries):
        [0 : P]        provider one-hot
        [P : P+T]      tier one-hot
        qty            log10(quantity) / 3            (1 → 0.0, 1000 → 1.0)
        round_frac     round / max_rounds
        remaining      remaining patience / 10
        offer_ratio    provider offer / list price
        bid_ratio      user bid / list price
        budget_ratio   budget / list price, clipped to 1.5 then / 1.5
        gap            (offer - bid) / offer
        floor_gap      (bid - floor) / list price, clipped to [-1.0, 1.0]
    """
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    p, t = len(PROVIDERS), len(TIERS)

    vec[obs["provider_idx"]]     = 1.0
    vec[p + obs["tier_idx"]]     = 1.0

    base = p + t
    vec[base + 0] = min(math.log10(max(obs["quantity"], 1)) / 3.0, 1.0)
    vec[base + 1] = obs["round"] / max(obs["max_rounds"], 1)
    vec[base + 2] = min(obs["remaining"] / 10.0, 1.0)
    vec[base + 3] = obs["offer_ratio"]
    vec[base + 4] = obs["bid_ratio"]
    vec[base + 5] = min(obs["budget_ratio"], 1.5) / 1.5
    vec[base + 6] = obs["gap"]

    list_p = obs.get("list_price", 1.0)
    floor_p = obs.get("floor_price", 0.0)
    bid_p = obs.get("bid", 0.0)
    gap_floor = (bid_p - floor_p) / list_p if list_p > 0 else 0.0
    vec[base + 7] = float(np.clip(gap_floor, -1.0, 1.0))
    return vec

