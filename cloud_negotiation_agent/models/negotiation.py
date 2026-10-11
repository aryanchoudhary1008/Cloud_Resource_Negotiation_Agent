"""
models/negotiation.py
---------------------
NegotiationSession data model — one row of negotiation history.

New in this version: provider, algorithm, tier, quantity and list_price,
so history and statistics can be compared across AWS / GCP / Azure and
across Q-learning / DQN.

Backward compatibility: `storage` is kept (always 0 now — storage is no
longer negotiated) so rows from the old schema still load.
"""

from dataclasses import dataclass, field
from datetime    import datetime, timezone
from typing      import Optional


def _utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


@dataclass
class NegotiationSession:
    """
    Attributes:
        cpu, ram     vCPU / GB of the originally requested tier
        storage      legacy column, always 0
        budget       user's max monthly budget (USD)
        final_price  agreed monthly price; 0.0 if no deal
        final_cpu    vCPU in the final deal (lower if the agent downgraded)
        final_ram    GB RAM in the final deal
        rounds       negotiation rounds used
        success      True if a deal was reached
        total_reward cumulative RL reward
        provider     'AWS' | 'GCP' | 'Azure'  ('Legacy' for pre-upgrade rows)
        algorithm    'qlearning' | 'dqn'
        tier         originally requested tier ('Small' | 'Medium' | 'Large')
        quantity     number of instances requested
        list_price   monthly list price of the configuration actually delivered
        request_id   groups the 3 negotiations of one Auto request ('' for manual)
        auto_mode    True if the provider was chosen by the agent (Auto mode)
        selected     True for the provider that was chosen / the only provider used
    """

    cpu:         int
    ram:         int
    storage:     int
    budget:      float
    final_price: float
    final_cpu:   int
    final_ram:   int
    rounds:      int
    success:     bool

    total_reward: float         = 0.0
    provider:     str           = "Legacy"
    algorithm:    str           = "qlearning"
    tier:         str           = "Custom"
    quantity:     int           = 1
    list_price:   float         = 0.0
    request_id:   str           = ""
    auto_mode:    bool          = False
    selected:     bool          = True
    id:           Optional[int] = None
    timestamp:    str           = field(default_factory=_utc_now)

    def to_dict(self):
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
            "provider":     self.provider,
            "algorithm":    self.algorithm,
            "tier":         self.tier,
            "quantity":     self.quantity,
            "list_price":   round(self.list_price, 2),
            "request_id":   self.request_id,
            "auto_mode":    self.auto_mode,
            "selected":     self.selected,
            "timestamp":    self.timestamp,
        }
