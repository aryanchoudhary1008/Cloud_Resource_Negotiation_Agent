"""
models/negotiation.py
---------------------
NegotiationSession data model.

A dataclass that represents one complete negotiation session.
Used for:
  - Storing results in SQLite (via database/db.py)
  - Returning structured data from the Flask API
  - Computing aggregate statistics

Python dataclasses (introduced in Python 3.7) are like lightweight
data containers — they auto-generate __init__, __repr__, etc.
"""

from dataclasses import dataclass, field
from datetime    import datetime
from typing      import Optional


@dataclass
class NegotiationSession:
    """
    Complete record of a single cloud resource negotiation.venv\Scripts\activatevenv\Scripts\activate

    Fields marked 'Optional' or with defaults are not required
    at construction time — they get filled in after the fact
    (e.g. `id` is assigned by SQLite on INSERT).

    Attributes:
        cpu          (int):   CPU cores originally requested by the user.
        ram          (int):   RAM in GB originally requested.
        storage      (int):   Storage in GB requested.
        budget       (float): User's stated maximum budget (USD).
        final_price  (float): Agreed price; 0.0 if negotiation failed.
        final_cpu    (int):   CPU cores in the final deal (may be < cpu
                              if the agent downgraded resources).
        final_ram    (int):   RAM in final deal (may be < ram).
        rounds       (int):   Number of negotiation rounds used.
        success      (bool):  True if a deal was reached.
        total_reward (float): Cumulative RL reward for this episode.
        id           (int):   SQLite primary key (None until persisted).
        timestamp    (str):   ISO-8601 UTC time of session completion.
    """

    # Required fields (must be provided at construction)
    cpu:         int
    ram:         int
    storage:     int
    budget:      float
    final_price: float
    final_cpu:   int
    final_ram:   int
    rounds:      int
    success:     bool

    # Optional fields with defaults
    total_reward: float         = 0.0
    id:           Optional[int] = None
    timestamp:    str           = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def to_dict(self):
        """
        Convert the session to a JSON-serializable dictionary.

        Used by Flask routes to return session data as JSON.
        Rounds float values to 2 decimal places for cleanliness.

        Returns:
            dict: All fields with JSON-safe types.
        """
        return {
            "id":           self.id,
            "cpu":          self.cpu,
            "ram":          self.ram,
            "storage":      self.storage,
            "budget":       round(self.budget, 2),
            "final_price":  round(self.final_price, 2),
            "final_cpu":    self.final_cpu,
            "final_ram":    self.final_ram,
            "rounds":       self.rounds,
            "success":      self.success,
            "total_reward": round(self.total_reward, 2),
            "timestamp":    self.timestamp,
        }
