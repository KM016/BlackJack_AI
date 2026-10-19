from copy import deepcopy
from decimal import Decimal

import pytest

from blackjack.engine import BlackjackEngine
from blackjack.player import Player
from research.calibration_model import CalibrationModel
from strategies.basic import BasicStrategy
from strategies.calibrated_kelly import CalibratedKellyBetting
from strategies.runner import run_round


# make a small fitted model without using the real research datasets
def make_model(mean="0.02", variance="1", mean_ci=("0.01", "0.03"), supported=True):
    return CalibrationModel({
        "version": "calibration_v1", "role": "calibration",
        "settings": {"bins": ["0"], "min_variance": "0.000000000001",
                     "min_samples": 2, "min_sessions": 2, "max_ci_width": "20",
                     "bootstrap_replicates": 100},
        "metadata": {"rules": {"decks": 6}},
        "summary": {"bins": [
            {"label": "[-inf, 0)", "supported": False},
            {"label": "[0, inf)", "supported": supported, "mean": mean,
             "variance": variance, "mean_ci": list(mean_ci),
             "count": 4, "sessions": 2, "bootstrap_coverage": 100,
             "variance_bootstrap_coverage": 100,
             "distribution": [{"return": "-1", "count": 10},
                              {"return": "1.5", "count": 10}]}
        ]}
    })


# make a public pre-round observation like the engine supplies
def make_view(**changes):
    observation = {"state": "betting", "round_number": 0, "num_decks": 6,
                   "bankroll": Decimal("1000"), "min_bet": Decimal(1),
                   "max_bet": Decimal(1000), "can_bet": True,
                   "reshuffle_pending": False, "events": []}
    observation.update(changes)
    return observation


# TEST 1: Check half-Kelly uses the empirical mean and variance
def test_half_kelly():
    policy = CalibratedKellyBetting(make_model())
    assert policy.choose_wager(make_view()) == Decimal("10.000")
    assert policy.last_decision["ideal_fraction"] == Decimal("0.010")


# TEST 2: Check the original two-percent initial stake cap
def test_bankroll_cap():
    policy = CalibratedKellyBetting(make_model("0.5", "1", ("0.1", "0.7")))
    assert policy.choose_wager(make_view()) == Decimal("20.00")
    assert policy.last_decision["capped"]


# TEST 3: Check that uncertain positive means do not justify a wager
def test_uncertain_edge():
    policy = CalibratedKellyBetting(make_model("0.02", "1", ("-0.01", "0.03")))
    assert policy.choose_wager(make_view()) is None
    assert policy.last_decision["sit_out_reason"] == "uncertain_positive_edge"


# TEST 4: Check that nonpositive returns lead to sitting out
def test_negative_edge():
    policy = CalibratedKellyBetting(make_model("-0.02", "1", ("-0.03", "-0.01")))
    assert policy.choose_wager(make_view()) is None
    assert policy.last_decision["sit_out_reason"] == "no_estimated_edge"


# TEST 5: Check that unsupported count bins cannot produce a wager
def test_unsupported_bin():
    policy = CalibratedKellyBetting(make_model())
    assert policy.choose_wager(make_view(round_number=0, events=[], reshuffle_pending=False,
                                         bankroll=Decimal(1000), num_decks=6)) == 10
    # a sufficiently negative running count changes the selected bin
    events = [{"type": "round_start", "round_number": 1, "num_decks": 6,
               "reshuffled": False},
              {"type": "card", "card": {"value": "King"}},
              {"type": "round_end", "state": "complete"}]
    assert policy.choose_wager(make_view(round_number=1, state="complete", events=events)) is None
    assert policy.last_decision["sit_out_reason"] == "unsupported_bin"


# TEST 6: Check the table maximum and minimum are respected
def test_betting_limits():
    policy = CalibratedKellyBetting(make_model())
    assert policy.choose_wager(make_view(max_bet=Decimal(7))) == 7
    assert policy.choose_wager(make_view(min_bet=Decimal(11))) is None
    assert policy.choose_wager(make_view(bankroll=Decimal("0.5"), can_bet=False)) is None


# TEST 7: Check the split-and-double exposure safety bound
def test_exposure_safety():
    policy = CalibratedKellyBetting(
        make_model("0.5", "1", ("0.1", "0.7")), max_bankroll_fraction="1"
    )
    assert policy.choose_wager(make_view()) == 120
    with pytest.raises(ValueError, match="1/8"):
        CalibratedKellyBetting(make_model(), max_exposure_safe_fraction="0.125")


# TEST 8: Check invalid Kelly settings are rejected
@pytest.mark.parametrize("name,value", [
    ("kelly_multiplier", "0"), ("kelly_multiplier", "1.01"),
    ("max_bankroll_fraction", "0"), ("max_bankroll_fraction", "1.01"),
    ("max_exposure_safe_fraction", "0"), ("max_exposure_safe_fraction", "0.125"),
    ("kelly_multiplier", True), ("kelly_multiplier", 0.5)
])
def test_invalid_parameters(name, value):
    with pytest.raises(ValueError):
        CalibratedKellyBetting(make_model(), **{name: value})


# TEST 9: Check the game and calibration deck counts must agree
def test_wrong_deck_count():
    policy = CalibratedKellyBetting(make_model())
    with pytest.raises(ValueError, match="Deck count"):
        policy.choose_wager(make_view(num_decks=2))


# TEST 10: Check that the runner can actually play using the Kelly policy
def test_real_round():
    engine = BlackjackEngine({"player_1": Player(1000)}, seed=42)
    policy = CalibratedKellyBetting(make_model())
    result = run_round(engine, {"player_1": BasicStrategy()}, {"player_1": policy})
    assert result["status"] == "complete"
    assert result["wagers"]["player_1"] == Decimal("10.000")
    assert engine.players["player_1"].bets == {}
    assert engine.players["player_1"].bankroll >= 0


# TEST 11: Check sitting out does not consume cards or advance the RNG
def test_no_wager_no_cards():
    engine = BlackjackEngine({"player_1": Player(1000)}, seed=42)
    policy = CalibratedKellyBetting(make_model("-0.02", "1", ("-0.03", "-0.01")))
    rng_before = engine.shoe.rng.getstate()
    cards_before = list(engine.shoe.shoe.cards)
    result = run_round(engine, {"player_1": BasicStrategy()}, {"player_1": policy})
    assert result["status"] == "no_active_seats"
    assert list(engine.shoe.shoe.cards) == cards_before
    assert engine.shoe.rng.getstate() == rng_before


# TEST 12: Check the strategy does not edit its input observations
def test_observation_unchanged():
    policy = CalibratedKellyBetting(make_model())
    observation = make_view()
    original = deepcopy(observation)
    policy.choose_wager(observation)
    assert observation == original


# TEST 13: Check a saved model is loaded once and can be reused
def test_loaded_model(tmp_path):
    path = tmp_path / "calibration_model.json"
    make_model().save(path)
    policy = CalibratedKellyBetting.from_file(path)
    assert policy.choose_wager(make_view()) == 10
    path.unlink()
    assert policy.choose_wager(make_view()) == 10


# TEST 14: Check a loaded model cannot falsely mark weak bins as supported
def test_reject_false_supported_bin():
    document = make_model().get_document()
    document["summary"]["bins"][1]["count"] = 1
    with pytest.raises(ValueError, match="evidence thresholds"):
        CalibrationModel(document)
