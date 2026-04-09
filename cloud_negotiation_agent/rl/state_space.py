"""
rl/state_space.py
-----------------
State space encoding for the Q-learning agent.

In Q-learning, the agent needs a DISCRETE, HASHABLE representation
of the world to use as a dictionary key in the Q-table.

Problem: cpu, ram, budget, price are continuous/large integers.
Solution: Discretize them into small integer "bins".

Example:
    cpu=7 cores → closest to bin [2,4,8,16] → bin index 2 (value 8)
    budget=37   → closest to bin [10,20,50,100] → bin index 2 (value 50)

Final state = (cpu_bin, ram_bin, budget_bin, price_bin, round_bin)
            = e.g. (2, 1, 2, 3, 0)  → hashable tuple, usable as dict key
"""

# ── Discrete bins for each state dimension ─────────────────────────

# CPU tier options (cores)
CPU_BINS = [2, 4, 8, 16]

# RAM tier options (GB)
RAM_BINS = [4, 8, 16, 32]

# Budget tiers (USD) — rough breakpoints for low/medium/high/premium budgets
BUDGET_BINS = [10, 20, 50, 100]

# Number of price bins (price is normalized to [0,1] then mapped to 0..4)
PRICE_BINS = 5


def discretize(value, bins):
    """
    Map a continuous value to the index of the nearest bin.

    Uses minimum absolute distance — no rounding needed.

    Args:
        value (float/int): The value to discretize.
        bins  (list):      Sorted list of representative values.

    Returns:
        int: Index (0-based) of the nearest bin.

    Examples:
        discretize(3,  [2, 4, 8, 16]) → 1   (4 is closest to 3)
        discretize(15, [2, 4, 8, 16]) → 3   (16 is closest to 15)
        discretize(9,  [2, 4, 8, 16]) → 2   (8 is closest to 9)
    """
    return min(range(len(bins)), key=lambda i: abs(bins[i] - value))


def discretize_price(current_price, max_price):
    """
    Normalize current_price relative to max_price, then map to 5 bins.

    Rationale: Absolute price values vary wildly between sessions
    (e.g. $5 for 2 CPU vs $80 for 16 CPU). Normalizing keeps the
    state space compact and generalizable.

    Args:
        current_price (float): The price being offered right now.
        max_price     (float): Upper bound reference (usually base_price).

    Returns:
        int: Bin index 0..4 where 0 = very cheap, 4 = very expensive.
    """
    if max_price <= 0:
        return 0

    # Clamp ratio to [0, 1] to handle prices that exceed max_price
    ratio = min(current_price / max_price, 1.0)

    # Map [0.0, 1.0] → [0, 4]
    bin_idx = min(int(ratio * PRICE_BINS), PRICE_BINS - 1)
    return bin_idx


def encode_state(cpu, ram, budget, current_price, round_num, max_price=None):
    """
    Encode the full negotiation state into a compact, hashable tuple.

    This tuple serves as the KEY in the Q-table dictionary:
        q_table[(2, 1, 2, 3, 0)] → np.array([q0, q1, q2, q3])

    Args:
        cpu           (int):   CPU cores currently being negotiated.
        ram           (int):   RAM in GB currently being negotiated.
        budget        (float): User's stated maximum budget.
        current_price (float): Provider's current offered price.
        round_num     (int):   Negotiation round number (1–5).
        max_price     (float): Reference max price for normalization.
                               Defaults to 2× budget if not provided.

    Returns:
        tuple: (cpu_bin, ram_bin, budget_bin, price_bin, round_bin)
               All elements are small integers → fully hashable.
    """
    # Discretize each dimension
    cpu_bin    = discretize(cpu,    CPU_BINS)
    ram_bin    = discretize(ram,    RAM_BINS)
    budget_bin = discretize(budget, BUDGET_BINS)

    # Use budget × 2 as a rough max_price if not explicitly provided
    if max_price is None or max_price <= 0:
        max_price = budget * 2.0

    price_bin = discretize_price(current_price, max_price)

    # Clamp round to valid range [0, 4]
    round_bin = max(0, min(round_num - 1, 4))

    return (cpu_bin, ram_bin, budget_bin, price_bin, round_bin)
