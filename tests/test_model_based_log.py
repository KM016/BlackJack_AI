from copy import deepcopy
from decimal import Decimal
import math

import pytest

from blackjack.engine import BlackjackEngine
from blackjack.player import Player
from research.calibration_model import CalibrationModel
from strategies.basic import BasicStrategy
from strategies.model_based_log import ModelBasedLogBetting, expected_log_gain
from strategies.runner import run_round
from strategies.wager_grid import DiscreteBetting
from test_wager_grid import make_view


# create a fitted model with a matching two-outcome return distribution
def make_model(losses=1, wins=3, lower_ci="0.01", supported=True):
    count = losses + wins
    values_mean = Decimal(wins - losses) / count
    variance = (Decimal(count) - Decimal(wins - losses) ** 2 / count) / (count - 1)
    return CalibrationModel({
        "version": "calibration_v1", "role": "calibration",
        "population": "two_seats_fixed_one_unit_s17_das",
        "settings": {"bins": ["0"], "min_variance": "0.000000000001",
                     "min_samples": 2, "min_sessions": 2, "max_ci_width": "20",
                     "bootstrap_replicates": 100},
        "metadata": {"rules": {"decks": 6}},
        "summary": {"bins": [
            {"label": "[-inf, 0)", "supported": False, "count": 0,
             "distribution": [], "mean": None, "variance": None},
            {"label": "[0, inf)", "supported": supported,
             "mean": str(values_mean), "variance": str(variance),
             "mean_ci": [lower_ci, "0.8"], "count": count,
             "sessions": 2, "bootstrap_coverage": 100,
             "variance_bootstrap_coverage": 100,
             "distribution": [{"return": "-1", "count": losses},
                              {"return": "1", "count": wins}]}
        ]}
    })


# TEST 1: Check expected log gain matches a manually calculated distribution
def test_expected_log_gain():
    distribution = ((Decimal(-8), 2), (Decimal(-2), 3),
                    (Decimal(0), 1), (Decimal("1.5"), 4))
    actual = expected_log_gain(distribution, Decimal(1), Decimal(100))
    expected = (2 * math.log(0.92) + 3 * math.log(0.98)
                + 4 * math.log(1.015)) / 10
    assert actual == pytest.approx(expected, abs=1e-15)


# TEST 2: Check an unaffordable wager would make log wealth undefined
@pytest.mark.parametrize("wager,bankroll", [(1, 8), (8, 8), (9, 8)])
def test_invalid_log_wager(wager, bankroll):
    with pytest.raises(ValueError):
        expected_log_gain(((Decimal(-8), 1), (Decimal(1), 1)), wager, bankroll)


# TEST 3: Check an empty or invalid return distribution cannot be used
def test_invalid_log_distribution():
    with pytest.raises(ValueError):
        expected_log_gain([], 1, 1000)
    with pytest.raises(ValueError):
        expected_log_gain(((Decimal(1), 0),), 1, 1000)
    with pytest.raises(ValueError):
        expected_log_gain(((Decimal(1), 1),), 1, 0)


# TEST 4: Check the strategy chooses the best positive log-growth wager
def test_best_wager():
    policy = ModelBasedLogBetting(make_model())
    assert policy.choose_wager(make_view()) == 8
    assert policy.last_decision["action_index"] == 5
    assert policy.last_decision["estimated_log_gain"] > 0


# TEST 5: Check that only wagers meeting bankroll and table limits are considered
def test_betting_limits():
    policy = ModelBasedLogBetting(make_model())
    assert policy.choose_wager(make_view(bankroll=Decimal(100))) == 2
    assert policy.choose_wager(make_view(max_bet=Decimal(5))) == 4
    assert policy.choose_wager(make_view(min_bet=Decimal(11))) is None


# TEST 6: Check a weak confidence interval makes the strategy sit out
def test_uncertain_edge():
    policy = ModelBasedLogBetting(make_model(lower_ci="-0.1"))
    assert policy.choose_wager(make_view()) is None
    assert policy.last_decision["sit_out_reason"] == "no_supported_positive_edge"


# TEST 7: Check an unsupported count range cannot produce a wager
def test_unsupported_bin():
    policy = ModelBasedLogBetting(make_model())
    events = [{"type": "round_start", "round_number": 1, "num_decks": 6,
               "reshuffled": False},
              {"type": "card", "card": {"value": "Ace"}},
              {"type": "round_end", "state": "complete"}]
    assert policy.choose_wager(make_view(round_number=1, state="complete", events=events)) is None


# TEST 8: Check losing distributions produce negative expected log growth
def test_negative_log_gain():
    assert expected_log_gain(((Decimal(-1), 3), (Decimal(1), 1)), 8, 1000) < 0


# TEST 9: Check ties select the smaller wager without relying on float rounding
def test_tie_prefers_smallest(monkeypatch):
    policy = ModelBasedLogBetting(make_model())
    monkeypatch.setattr("strategies.model_based_log.expected_log_gain", lambda *args: 0.001)
    assert policy.choose_wager(make_view()) == 1
    monkeypatch.setattr("strategies.model_based_log.expected_log_gain", lambda *args: 0.0)
    assert policy.choose_wager(make_view()) is None


# TEST 10: Check invalid or mismatched distribution frequencies are rejected
def test_distribution_mismatch():
    document = make_model().get_document()
    document["summary"]["bins"][1]["distribution"][0]["count"] += 1
    with pytest.raises(ValueError, match="count"):
        ModelBasedLogBetting(CalibrationModel(document))


# TEST 11: Check a changed return distribution cannot silently change moments
def test_moment_mismatch():
    document = make_model().get_document()
    document["summary"]["bins"][1]["distribution"] = [
        {"return": "-2", "count": 1}, {"return": "1", "count": 3}]
    with pytest.raises(ValueError, match="moments"):
        ModelBasedLogBetting(CalibrationModel(document))


# TEST 12: Check model population and deck count must match the simulated game
def test_population_and_decks():
    document = make_model().get_document()
    document["population"] = "other_population"
    with pytest.raises(ValueError, match="population"):
        ModelBasedLogBetting(CalibrationModel(document))
    with pytest.raises(ValueError, match="Deck count"):
        ModelBasedLogBetting(make_model()).choose_wager(make_view(num_decks=2))


# TEST 13: Check the model is loaded once and never read again during decisions
def test_load_once(tmp_path):
    path = tmp_path / "model.json"
    make_model().save(path)
    policy = ModelBasedLogBetting.from_file(path)
    path.unlink()
    assert policy.choose_wager(make_view()) == 8


# TEST 14: Check betting decisions never change a player's observation
def test_public_observation_unchanged():
    view = make_view()
    original = deepcopy(view)
    policy = ModelBasedLogBetting(make_model())
    assert policy.choose_wager(view) == 8
    assert view == original


# TEST 15: Check model-based betting actually runs through the blackjack engine
def test_real_engine():
    engine = BlackjackEngine({"p": Player(1000)}, seed=42)
    policy = ModelBasedLogBetting(make_model())
    result = run_round(engine, {"p": BasicStrategy()}, {"p": policy})
    assert result["status"] == "complete"
    assert result["wagers"]["p"] == 8
    assert engine.players["p"].bets == {}


# TEST 16: Check the reference still deals cards when the research agent sits out
def test_reference_continues_when_agent_sits_out():
    from strategies.flat import FlatBetting

    engine = BlackjackEngine({"reference": Player(1000000), "agent": Player(1000)}, seed=42)
    policy = ModelBasedLogBetting(make_model(lower_ci="-0.1"))
    result = run_round(engine,
                       {"reference": BasicStrategy(), "agent": BasicStrategy()},
                       {"reference": FlatBetting(1), "agent": policy})
    assert result["status"] == "complete"
    assert engine.round_number == 1
    assert list(result["wagers"]) == ["reference"]
    assert engine.players["agent"].bankroll == 1000


# TEST 17: Check tiny decimal-rounding differences do not reject valid fitted moments
def test_decimal_rounding():
    document = make_model().get_document()
    document["summary"]["bins"][1]["variance"] = "1.00000000000000000000001"
    assert ModelBasedLogBetting(CalibrationModel(document)).choose_wager(make_view()) == 8
