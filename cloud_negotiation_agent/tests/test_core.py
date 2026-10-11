"""Core unit tests: provider logic, environment rewards, state encoding, Q/DQN updates, DB migration."""
import sqlite3
import numpy as np
import pytest

from agents.provider_agent import ProviderAgent, provider_names
from rl.environment import (NegotiationEnv, RAISE_BID, DOWNGRADE, ACCEPT, WALK_AWAY,
                            FAILURE_PENALTY)
from rl.state_space import encode_tabular, encode_vector, STATE_DIM
from rl.q_learning import QLearningAgent


# ── Provider ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["AWS", "GCP", "Azure"])
def test_offers_never_increase_and_respect_floor(name):
    p = ProviderAgent(name)
    offers = [p.make_offer("Medium", 10, r) for r in range(1, 12)]
    assert all(a >= b for a, b in zip(offers, offers[1:]))
    assert min(offers) >= round(p.floor_price("Medium", 10), 2)

def test_provider_personalities_differ():
    rates = {n: ProviderAgent(n).concession_rate for n in provider_names()}
    assert rates["AWS"] < rates["GCP"] < rates["Azure"]
    pat = {n: ProviderAgent(n).max_rounds for n in provider_names()}
    assert pat["AWS"] > pat["GCP"] > pat["Azure"]

def test_volume_discount_tiers():
    aws = ProviderAgent("AWS")
    assert aws.volume_discount(99) == 0 and aws.volume_discount(100) == 0.05
    assert aws.volume_discount(500) == 0.10

def test_bid_below_floor_rejected():
    p = ProviderAgent("AWS")
    ok, _ = p.evaluate_bid(1.0, 100.0, "Small", 1)
    assert not ok

def test_unknown_provider():
    with pytest.raises(ValueError):
        ProviderAgent("Oracle")


# ── Environment ──────────────────────────────────────────────────────
def env(provider="AWS", tier="Medium", qty=10, budget=1000.0, rounds=None):
    e = NegotiationEnv(); e.reset(provider, tier, qty, budget, max_rounds=rounds); return e

def test_accept_within_budget_gives_positive_reward():
    e = env(); _, r, done, info = e.step(ACCEPT)
    assert done and info["success"] and r > 0

def test_accept_over_budget_fails():
    e = env(budget=10.0); _, r, done, info = e.step(ACCEPT)
    assert done and not info["success"] and r < FAILURE_PENALTY + 0.01

def test_walk_away_penalty():
    e = env(); _, r, done, info = e.step(WALK_AWAY)
    assert done and not info["success"] and r < 0

def test_patience_exhausted():
    e = env(rounds=2)
    e.step(RAISE_BID); _, _, done, info = e.step(RAISE_BID)
    assert done and not info["success"] and "patience" in info["reason"]

def test_downgrade_lowers_price_and_costs_reward():
    e = env(tier="Large"); before = e.offer
    e.step(DOWNGRADE)
    assert e.tier == "Medium" and e.offer < before and e.downgrades == 1

def test_cannot_downgrade_below_small():
    e = env(tier="Small"); e.step(DOWNGRADE)
    assert e.tier == "Small" and e.downgrades == 0

def test_reward_price_clipped_at_floor():
    e = env("AWS")
    res = e._close(0.01, 0.0, {})            # absurdly low price → clipped to the floor
    max_reward = 5 + 10 * (1 - e.provider.floor_ratio) + 2
    assert res[1] <= max_reward + 1e-9

def test_step_after_done_raises():
    e = env(); e.step(WALK_AWAY)
    with pytest.raises(RuntimeError):
        e.step(RAISE_BID)

def test_bad_inputs():
    with pytest.raises(ValueError): NegotiationEnv().reset("AWS", "Huge", 1, 100)
    with pytest.raises(ValueError): NegotiationEnv().reset("AWS", "Small", 0, 100)


# ── State encoding ───────────────────────────────────────────────────
def test_tabular_encoding_hashable_and_bounded():
    obs = env("GCP").reset("GCP", "Large", 50, 500)
    key = encode_tabular(obs); hash(key)
    assert len(key) == 6 and key[0] == 1 and key[1] == 2

def test_vector_encoding_shape_and_range():
    v = encode_vector(env("Azure").reset("Azure", "Small", 1000, 50))
    assert v.shape == (STATE_DIM,) and v.dtype == np.float32
    assert 0.0 <= v.min() and v.max() <= 1.5
    assert v[2] == 1.0     # Azure one-hot


# ── Q-learning ───────────────────────────────────────────────────────
def test_bellman_update():
    a = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0)
    a.q_table["s2"][1] = 10.0
    a.update("s1", 0, 1.0, "s2", done=False)
    assert a.q_table["s1"][0] == pytest.approx(0.5 * (1 + 0.9 * 10))
    a.update("s1", 0, 2.0, "s2", done=True)
    assert a.q_table["s1"][0] == pytest.approx(5.0 + 0.5 * (2.0 - 5.0))

def test_q_save_load_and_stale_version(tmp_path):
    import pickle
    a = QLearningAgent(); a.update((0,)*6, 2, 5.0, (0,)*6, True)
    f = str(tmp_path / "q.pkl"); a.save(f)
    b = QLearningAgent(); assert b.load(f) and len(b.q_table) == 1
    pickle.dump({"q_table": {(1, 2, 3, 4, 0): np.zeros(4)}}, open(f, "wb"))   # old format
    assert not QLearningAgent().load(f)


# ── DQN (skipped without PyTorch) ────────────────────────────────────
def test_dqn_learns_and_roundtrips(tmp_path):
    pytest.importorskip("torch")
    from rl.dqn_agent import DQNAgent
    a = DQNAgent(batch_size=8, target_sync=5)
    s = np.zeros(STATE_DIM, dtype=np.float32)
    losses = [a.update(s, i % 4, 1.0, s, i % 3 == 0) for i in range(40)]
    assert any(l is not None for l in losses) and a.learn_steps > 0
    f = str(tmp_path / "m.pt"); a.save(f)
    b = DQNAgent(); assert b.load(f)
    assert np.allclose(a.q_values(s), b.q_values(s))


# ── Database migration ───────────────────────────────────────────────
def test_db_migrates_legacy_schema(tmp_path, monkeypatch):
    from database import db
    path = str(tmp_path / "old.db"); monkeypatch.setattr(db, "DB_PATH", path)
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE negotiations (id INTEGER PRIMARY KEY AUTOINCREMENT, cpu INTEGER, ram INTEGER,
        storage INTEGER, budget REAL, final_price REAL, final_cpu INTEGER, final_ram INTEGER,
        rounds INTEGER, success INTEGER, total_reward REAL, timestamp TEXT)""")
    conn.execute("INSERT INTO negotiations VALUES (NULL,4,8,100,40,30,4,8,2,1,8,'2025-01-01T00:00:00')")
    conn.commit(); conn.close()

    db.init_db()
    old = db.get_all_sessions()[0]
    assert old.provider == "Legacy" and old.algorithm == "qlearning"

    from models.negotiation import NegotiationSession
    db.save_session(NegotiationSession(cpu=2, ram=8, storage=0, budget=500, final_price=300,
        final_cpu=2, final_ram=8, rounds=3, success=True, provider="GCP", algorithm="dqn",
        tier="Medium", quantity=10, list_price=489.1))
    stats = db.get_stats()
    names = {p["name"] for p in stats["by_provider"]}
    assert {"Legacy", "GCP"} <= names and {a["name"] for a in stats["by_algorithm"]} == {"qlearning", "dqn"}


# ── Auto provider selection ──────────────────────────────────────────
class _R:
    def __init__(self, provider, success, downgrades, price, rounds=3, reward=0.0):
        self.provider, self.success, self.downgrades = provider, success, downgrades
        self.final_price, self.rounds, self.total_reward = price, rounds, reward

def test_choose_best_prefers_kept_tier_over_cheaper_downgrade():
    from agents.user_agent import choose_best
    cheap_downgraded = _R("GCP", True, 1, 700.0)
    pricier_full     = _R("Azure", True, 0, 1700.0)
    assert choose_best([cheap_downgraded, pricier_full]).provider == "Azure"

def test_choose_best_lowest_price_then_rounds():
    from agents.user_agent import choose_best
    a, b, c = _R("AWS", True, 0, 900.0), _R("GCP", True, 0, 800.0, rounds=5), _R("Azure", True, 0, 800.0, rounds=2)
    assert choose_best([a, b, c]).provider == "Azure"

def test_choose_best_ignores_failures_unless_all_fail():
    from agents.user_agent import choose_best
    assert choose_best([_R("AWS", False, 0, 0, reward=-5), _R("GCP", True, 0, 500.0)]).provider == "GCP"
    assert choose_best([_R("AWS", False, 0, 0, reward=-6), _R("GCP", False, 0, 0, reward=-5)]).provider == "GCP"

def test_unnecessary_downgrade_is_penalised_but_needed_one_is_not():
    from rl.environment import STEP_COST, UNNECESSARY_DOWNGRADE_PENALTY
    easy = env("GCP", "Large", 18, 2000.0)          # Large is affordable
    _, r_easy, _, _ = easy.step(DOWNGRADE)
    assert r_easy == pytest.approx(STEP_COST + UNNECESSARY_DOWNGRADE_PENALTY)
    hard = env("GCP", "Large", 18, 500.0)           # Large is impossible → downgrade is justified
    _, r_hard, _, _ = hard.step(DOWNGRADE)
    assert r_hard == pytest.approx(STEP_COST)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from database import db
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    db.init_db()                       # app.py only initialises the DB once, at import
    import app as app_module
    return app_module.app.test_client()

def test_api_auto_mode_returns_quotes_and_saves_all_rows(client):
    r = client.post("/api/negotiate", json={"provider": "auto", "tier": "Large", "quantity": 18,
                                            "budget": 2000, "algorithm": "qlearning"})
    d = r.get_json()
    assert r.status_code == 200 and d["auto"] and len(d["quotes"]) == 3
    assert sum(q["selected"] for q in d["quotes"]) == (1 if d["success"] else 0)
    hist = client.get("/api/history").get_json()
    assert len(hist) == 3 and len({h["request_id"] for h in hist}) == 1
    stats = client.get("/api/stats").get_json()
    assert sum(p["auto_wins"] for p in stats["by_provider"]) == (1 if d["success"] else 0)

def test_api_manual_provider_and_validation(client):
    ok = client.post("/api/negotiate", json={"provider": "AWS", "tier": "Small", "quantity": 2,
                                             "budget": 100, "algorithm": "qlearning"})
    assert ok.status_code == 200 and not ok.get_json()["auto"]
    bad = client.post("/api/negotiate", json={"provider": "Oracle", "tier": "Small", "quantity": 2, "budget": 100})
    assert bad.status_code == 400
