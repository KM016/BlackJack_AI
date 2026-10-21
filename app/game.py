"""Play one human seat and the frozen DQN against a shared blackjack dealer."""

from decimal import Decimal
from pathlib import Path

from blackjack.engine import BlackjackEngine, ShoeExhausted
from blackjack.player import Player
from blackjack.rules import Rules
from strategies.basic import BasicStrategy
from strategies.hi_lo import HiLoCounter
from strategies.wager_grid import WAGERS, legal_wager_mask
from rl.policy import load_policy


START_BANKROLL = Decimal("1000")
REFERENCE_BANKROLL = Decimal("1000000000")
MAX_ROUNDS = 500
MODEL_PATHS = {
    "Selected DQN": "models/dqn_primary",
    "Exploratory DQN": "models/dqn_exploratory",
}


class BlackjackMatch:
    """Manage a live game without changing the original simulator or models.

    The reference seat automatically plays a one unit wager each round. This
    mirrors the DQN training environment and keeps cards circulating even if
    both visible players choose to sit out. Its bankroll is not included in
    the displayed results.
    """

    def __init__(self, seed=2026, model_name="Selected DQN", root=None):
        if model_name not in MODEL_PATHS:
            raise ValueError("Unknown saved model")
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")

        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
        self.model_name = model_name
        self.model = load_policy(self.root / MODEL_PATHS[model_name])
        self.engine = BlackjackEngine(
            players={
                "reference": Player(REFERENCE_BANKROLL),
                "you": Player(START_BANKROLL),
                "model": Player(START_BANKROLL),
            },
            rules=Rules(),
            seed=seed,
            retain_history=False,
        )
        self.playing = BasicStrategy()
        self.card_counter = HiLoCounter(num_decks=self.engine.rules.num_decks)
        self.seed = seed
        self.rounds = [0]
        self.bankroll = {"you": [float(START_BANKROLL)], "model": [float(START_BANKROLL)]}
        self.drawdown = {"you": [0.0], "model": [0.0]}
        self.peak = {"you": float(START_BANKROLL), "model": float(START_BANKROLL)}
        self.wagers = {"you": [0.0], "model": [0.0]}
        self.round_log = []
        self.action_log = []
        self.current_wagers = {"you": 0, "model": 0}
        self.last_logged_round = 0

    @property
    def round_number(self):
        return self.engine.round_number

    @property
    def active(self):
        return self.engine.state == "player_turn"

    @property
    def finished(self):
        return self.engine.state in ("complete", "void")

    @property
    def can_start(self):
        return self.engine.state in ("betting", "complete", "void") and self.round_number < MAX_ROUNDS

    def legal_human_wagers(self):
        if not self.can_start:
            return []
        observation = self.engine.betting_observation("you")
        mask = legal_wager_mask(observation)
        return [int(WAGERS[i]) if WAGERS[i] is not None else 0
                for i, permitted in enumerate(mask) if permitted]

    def start_round(self, human_wager):
        if not self.can_start:
            raise ValueError("Complete the current round before dealing again")
        if type(human_wager) is not int or human_wager not in self.legal_human_wagers():
            raise ValueError("Your wager is not allowed by the betting limits")

        # The trained network chooses its own wager before seeing any new cards.
        model_bet = self.model.choose_wager(self.engine.betting_observation("model"))
        model_amount = Decimal(0) if model_bet is None else model_bet
        self.current_wagers = {"you": human_wager, "model": int(model_amount)}

        # The reference seat ensures a round is dealt even if both sit out.
        wagers = {"reference": Decimal(1)}
        if human_wager:
            wagers["you"] = Decimal(human_wager)
        if model_bet is not None:
            wagers["model"] = model_bet

        prior_round = self.round_number
        try:
            self.engine.start_round(wagers)
            self.play_automatic_turns()
        except ShoeExhausted:
            if self.engine.state != "void" or self.round_number == prior_round:
                raise
            self._record_if_finished()

    def play_automatic_turns(self):
        """Auto play the model and reference, pausing for the human's decision."""
        while self.engine.state == "player_turn":
            seat = self.engine.current_player
            if seat == "you":
                break
            decision = self.playing.choose_action(self.engine.observation(seat))
            self.action_log.append({"round": self.round_number, "seat": seat, "action": decision})
            prior_round = self.round_number
            try:
                self.engine.act(seat, decision)
            except ShoeExhausted:
                if self.engine.state != "void" or self.round_number == prior_round:
                    raise
                break

        self._record_if_finished()

    def human_action(self, action):
        if self.engine.state != "player_turn" or self.engine.current_player != "you":
            raise ValueError("It is not your turn")
        if action not in self.engine.legal_actions("you"):
            raise ValueError("That action is not legal for this hand")

        self.action_log.append({"round": self.round_number, "seat": "you", "action": action})
        prior_round = self.round_number
        try:
            self.engine.act("you", action)
        except ShoeExhausted:
            if self.engine.state != "void" or self.round_number == prior_round:
                raise
        self.play_automatic_turns()

    def observation(self):
        # The public observation hides the dealer's unrevealed card.
        return self.engine.observation("you")

    def visible_count(self):
        # Card counting consumes public events, not the private shoe.
        self.card_counter.observe(self.observation())
        return self.card_counter.running_count, float(self.card_counter.get_true_count())

    def basic_strategy_hint(self):
        if self.engine.current_player != "you":
            return None
        return self.playing.choose_action(self.observation())

    def _record_if_finished(self):
        if self.engine.state not in ("complete", "void"):
            return
        if self.round_number <= self.last_logged_round:
            return

        # Advance the public count even when no one opens the live view.
        self.card_counter.observe(self.observation())
        result = self.engine.history.last_round
        row = {"round": self.round_number, "state": self.engine.state}
        for seat in ("you", "model"):
            info = result["players"][seat]
            end_balance = float(info["bankroll_end"])
            self.peak[seat] = max(self.peak[seat], end_balance)
            drawdown = (self.peak[seat] - end_balance) / self.peak[seat]

            self.bankroll[seat].append(end_balance)
            self.drawdown[seat].append(drawdown * 100)
            self.wagers[seat].append(float(info["initial_wager"]))

            row[f"{seat}_bet"] = float(info["initial_wager"])
            row[f"{seat}_profit"] = float(info["net_profit"])
            row[f"{seat}_bankroll"] = end_balance
            outcomes = [hand["outcome"] for hand in info["hands"]]
            row[f"{seat}_outcome"] = ", ".join(outcomes) if outcomes else "Sat out"

        self.rounds.append(self.round_number)
        self.round_log.append(row)
        self.last_logged_round = self.round_number

    def stats(self, seat):
        ending = self.bankroll[seat][-1]
        played = [r for r in self.round_log if r[f"{seat}_bet"] > 0]
        wins = sum(r[f"{seat}_profit"] > 0 for r in played)
        return {
            "bankroll": ending,
            "profit": ending - float(START_BANKROLL),
            "peak_drawdown": max(self.drawdown[seat]),
            "bets": len(played),
            "win_rate": (100 * wins / len(played)) if played else 0.0,
        }
