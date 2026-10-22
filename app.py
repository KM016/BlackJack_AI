"""Play blackjack against the saved DQNs and traditional betting strategies.

Run from the BlackJackAI project root with: streamlit run app.py
"""

from decimal import Decimal
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from blackjack.engine import BlackjackEngine, ShoeExhausted
from blackjack.player import Player
from blackjack.rules import Rules
from rl.policy import load_policy
from strategies.calibrated_kelly import CalibratedKellyBetting
from strategies.flat import FlatBetting
from strategies.hi_lo import HiLoBetting
from strategies.model_based_log import ModelBasedLogBetting
from strategies.wager_grid import DiscreteBetting
from strategies.basic import BasicStrategy
from strategies.wager_grid import WAGERS, legal_wager_mask


ROOT = Path(__file__).resolve().parent
START_BALANCE = 1000
MODEL_PATHS = {
    "Extended DQN 3 (selected)": "models/dqn_primary",
    "Original DQN 1 (exploratory)": "models/dqn_exploratory",
}
OPPONENTS = {
    "Extended DQN 3 (selected)": (
        "The formally selected reinforcement learning model. Trained for 500 episodes "
        "(250,000 betting opportunities) and chosen from six candidates using 30 "
        "selection sessions. It uses six public features to estimate which wager "
        "offers the highest learned value. Final experiment mean session log growth: -0.01971."
    ),
    "Original DQN 1 (exploratory)": (
        "An earlier version trained for 100 episodes (50,000 betting opportunities). "
        "It uses the same six inputs and 6-64-64-6 network architecture as Extended DQN 3, "
        "but has different trained weights and received less training. "
        "Included as an exploratory comparison, not the selected research model. "
        "Final experiment mean session log growth: -0.01419."
    ),
    "Flat betting": (
        "Always bets 1 unit when permitted, without considering card counts, "
        "bankroll growth or previous results. A simple baseline."
    ),
    "Hi Lo card counting": (
        "Uses the public cards already revealed to estimate the true count. "
        "It bets 1, 2, 4, 6 or 8 units as the count increases, subject to the same limits."
    ),
    "Calibrated half Kelly": (
        "Uses returns estimated from separate blackjack simulations to size bets "
        "according to estimated advantage and variance. Bets only when the "
        "calibration supports a positive edge."
    ),
    "Model based log betting": (
        "Uses the saved calibrated return distribution to choose the legal wager "
        "with the highest estimated log bankroll growth. "
        "This was the main benchmark for the selected DQN."
    ),
    "Never bet (control)": (
        "Always sits out, keeping its starting bankroll. "
        "A zero risk control rather than a strategy intended to win."
    ),
}
SUITS = {"Spades": "♠", "Hearts": "♥", "Diamonds": "♦", "Clubs": "♣"}
RANKS = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}

plt.rcParams.update({
    "figure.facecolor": "lightsteelblue",
    "axes.facecolor": "lemonchiffon",
})

st.set_page_config(page_title="BlackJackAI", page_icon="♠️", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
.blackjack-table {
    background: radial-gradient(ellipse at center, #236651, #103b33);
    border: 8px solid #493626;
    border-radius: 40px;
    padding: 24px;
    margin: 15px 0;
    color: white;
}
.seat {color: #f1e3b7; font-weight: bold; margin: 12px 0 6px;}
.playing-card {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 54px;
    height: 78px;
    margin: 4px;
    background: white;
    color: #20252c;
    border-radius: 7px;
    font-size: 23px;
    font-weight: bold;
    box-shadow: 0 3px 7px #0005;
}
.playing-card.red {color: #c22836;}
.playing-card.hidden {background: #594396; color: white;}
</style>
""", unsafe_allow_html=True)


def create_opponent(name):
    """Use the same saved agents and betting policies as the experiment."""
    if name in MODEL_PATHS:
        return load_policy(ROOT / MODEL_PATHS[name])
    if name == "Flat betting":
        return DiscreteBetting(FlatBetting(stake=1))
    if name == "Hi Lo card counting":
        return DiscreteBetting(HiLoBetting(base_unit=1))
    if name == "Calibrated half Kelly":
        return DiscreteBetting(CalibratedKellyBetting.from_file(ROOT / "models/calibration_model.json"))
    if name == "Model based log betting":
        return DiscreteBetting(ModelBasedLogBetting.from_file(ROOT / "models/calibration_model.json"))
    if name == "Never bet (control)":
        return DiscreteBetting()
    raise ValueError("Unknown betting strategy")


def new_session(model_name, seed):
    """Use the original simulator, with one hidden reference player."""
    engine = BlackjackEngine(
        players={
            "reference": Player(Decimal("1000000000")),
            "you": Player(Decimal(START_BALANCE)),
            "dqn": Player(Decimal(START_BALANCE)),
        },
        rules=Rules(),
        seed=seed,
        retain_history=False,
    )
    return {
        "engine": engine,
        "model": create_opponent(model_name),
        "basic": BasicStrategy(),
        "results": [],
        "wagers": {"you": 0, "dqn": 0},
        "model_name": model_name,
        "seed": seed,
        "phase": "betting",
    }


def legal_bets(game):
    if game["engine"].state not in ("betting", "complete", "void"):
        return []
    mask = legal_wager_mask(game["engine"].betting_observation("you"))
    return [0 if WAGERS[i] is None else int(WAGERS[i])
            for i, allowed in enumerate(mask) if allowed]


def save_result(game):
    """Save each completed round once so the graph updates correctly."""
    engine = game["engine"]
    if engine.state not in ("complete", "void") or len(game["results"]) >= engine.round_number:
        return

    players = engine.history.last_round["players"]
    result = {"Round": engine.round_number}
    for seat, label in (("you", "You"), ("dqn", "Opponent")):
        row = players[seat]
        result[f"{label} bankroll"] = float(row["bankroll_end"])
        result[f"{label} profit"] = float(row["net_profit"])
        result[f"{label} bet"] = float(row["initial_wager"])
    game["results"].append(result)
    game["phase"] = "result"


def advance_ai(game):
    """Play the automated seats until it is your turn or the round ends."""
    engine = game["engine"]
    while engine.state == "player_turn" and engine.current_player != "you":
        action = game["basic"].choose_action(engine.observation(engine.current_player))
        try:
            engine.act(engine.current_player, action)
        except ShoeExhausted:
            if engine.state != "void":
                raise
            break
    save_result(game)


def deal_round(game, wager):
    engine = game["engine"]
    if wager not in legal_bets(game):
        raise ValueError("Invalid wager")

    model_bet = game["model"].choose_wager(engine.betting_observation("dqn"))
    wagers = {"reference": Decimal(1)}
    if wager:
        wagers["you"] = Decimal(wager)
    if model_bet is not None:
        wagers["dqn"] = model_bet

    game["wagers"] = {"you": wager, "dqn": int(model_bet or 0)}
    game["phase"] = "playing"
    try:
        engine.start_round(wagers)
    except ShoeExhausted:
        if engine.state != "void":
            raise
    advance_ai(game)


def take_action(game, action):
    engine = game["engine"]
    if engine.current_player != "you" or action not in engine.legal_actions("you"):
        raise ValueError("Not a legal action")
    try:
        engine.act("you", action)
    except ShoeExhausted:
        if engine.state != "void":
            raise
    advance_ai(game)


def card_html(cards, hidden=False):
    output = ""
    for card in cards:
        suit = SUITS[card["suit"]]
        rank = RANKS.get(card["value"], str(card["value"]))
        color = " red" if suit in ("♥", "♦") else ""
        output += f'<span class="playing-card{color}">{escape(rank)}{suit}</span>'
    if hidden:
        output += '<span class="playing-card hidden">♠</span>'
    return output or "No cards"


def hand_html(hands):
    if not hands:
        return "Sitting out"
    output = ""
    for i, hand in enumerate(hands, 1):
        label = f"Hand {i}" if len(hands) > 1 else "Hand"
        status = hand["outcome"]
        if status is not None:
            amount = float(hand["net_profit"])
            description = f" · {status.capitalize()} ({amount:+g} units)"
        else:
            description = ""
        output += f'<div>{label}, {hand["total"]} points{description}</div>'
        output += card_html(hand["cards"])
    return output


def render_table(game):
    engine = game["engine"]
    waiting_for_bet = game["phase"] == "betting"

    if waiting_for_bet:
        # do not show the completed hand as if it belongs to the next round
        dealer_cards = '<span style="color:#d5e6df">Cards appear after the deal</span>'
        you_cards = "No cards yet"
        dqn_cards = "No cards yet"
        dealer_points = ""
        you_bet = 0
        dqn_bet = 0
    else:
        observation = engine.observation("you")
        dealer = observation["dealer"]
        people = observation["players"]
        dealer_cards = card_html(dealer["cards"], dealer["hole_hidden"])
        you_cards = hand_html(people["you"]["hands"])
        dqn_cards = hand_html(people["dqn"]["hands"])
        dealer_points = "" if dealer["total"] is None else f' ({dealer["total"]})'
        you_bet = game["wagers"]["you"]
        dqn_bet = game["wagers"]["dqn"]

    st.markdown(f"""
    <div class="blackjack-table">
        <div class="seat">Dealer{dealer_points}</div>
        <div>{dealer_cards}</div>
        <hr style="border-color:#ffffff50">
        <div style="display:flex; gap:5%">
            <div style="flex:1">
                <div class="seat">You, bet {you_bet}</div>
                {you_cards}
            </div>
            <div style="flex:1">
                <div class="seat">Opponent, bet {dqn_bet}</div>
                {dqn_cards}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def plot_results(results, choice):
    rounds = [0] + [row["Round"] for row in results]
    fig, ax = plt.subplots(figsize=(10, 4))

    for label, color in (("You", "#425e87"), ("Opponent", "#a34c6c")):
        bankroll = [START_BALANCE] + [row[f"{label} bankroll"] for row in results]
        if choice == "Bankroll":
            values = bankroll
        elif choice == "Profit and loss":
            values = [balance - START_BALANCE for balance in bankroll]
        else:
            peak = START_BALANCE
            values = []
            for balance in bankroll:
                peak = max(peak, balance)
                values.append(100 * (peak - balance) / peak)

        ax.plot(rounds, values, label=label, color=color, linewidth=2)

    ax.set_xlabel("Completed rounds")
    ax.set_ylabel("Drawdown (%)" if choice == "Drawdown" else "Units")
    ax.set_title(choice)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def play_page():
    st.title("♠ Blackjack Strategy Arena")
    st.caption(
        "You and the selected opponent face the same dealer. "
        "The opponent decides how much to bet, while both players use the same "
        "blackjack rules. The opponent's hands always follow BasicStrategy."
    )

    with st.sidebar:
        st.header("Choose your opponent")
        model_name = st.selectbox("Betting strategy", list(OPPONENTS))
        st.write(OPPONENTS[model_name])
        with st.expander("How do the DQNs differ?"):
            st.markdown(
                "**Extended DQN 3:** 500 training episodes, 250,000 opportunities. "
                "Selected from six trained candidates using a separate 30 session evaluation. "
                "\n\n**Original DQN 1:** 100 episodes, 50,000 opportunities. "
                "An earlier model included for exploration. "
                "\n\nBoth have six input features, two hidden layers of 64 neurons "
                "and six possible wager actions. Neither retrains in the app. "
                "In the final experiment, the selected DQN underperformed "
                "the model based log betting strategy."
            )
        seed = int(st.number_input("Shuffle seed", min_value=0, value=2026, step=1))
        restart = st.button("New session", use_container_width=True)
        st.caption("Six decks, dealer stands on soft 17, blackjack pays 3:2.")
        st.caption("Both players start with 1,000 units. Wagers are 0, 1, 2, 4, 6 or 8 units, subject to the original betting limits.")
        st.caption("Only the two DQNs are neural networks. The other options are existing benchmark policies.")

    if restart or "blackjack_session" not in st.session_state:
        st.session_state.blackjack_session = new_session(model_name, seed)

    game = st.session_state.blackjack_session
    if model_name != game["model_name"]:
        st.sidebar.warning("Press New session to apply the newly selected opponent.")
    engine = game["engine"]
    results = game["results"]

    first, second, third = st.columns(3)
    for column, label in ((first, "You"), (second, "Opponent")):
        balance = START_BALANCE if not results else results[-1][f"{label} bankroll"]
        column.metric(f"{label} bankroll", f"{balance:,.2f}", f"{balance - START_BALANCE:+.2f}")
    third.metric("Rounds", len(results))
    st.caption(f'Current opponent: **{game["model_name"]}**')

    render_table(game)

    if engine.state == "player_turn" and engine.current_player == "you":
        st.subheader("Your turn")
        legal = set(engine.legal_actions("you"))
        actions = st.columns(4)
        for column, action in zip(actions, ("hit", "stand", "double", "split")):
            if column.button(action.capitalize(), disabled=action not in legal,
                             use_container_width=True, key=f"{engine.round_number}_{action}_{engine.current_hand.hand_id}"):
                take_action(game, action)
                st.rerun()
    elif game["phase"] == "result":
        last = results[-1]
        st.subheader(f"Round {engine.round_number} complete")
        if engine.state == "void":
            st.warning("Round voided. Your wagers have been refunded.")
        else:
            st.success(
                f'You: {last["You profit"]:+g} units  |  '
                f'Opponent: {last["Opponent profit"]:+g} units'
            )
        st.caption("Review the final hands above before starting the next round.")
        if engine.round_number < 500:
            if st.button("New game", type="primary", use_container_width=True):
                game["phase"] = "betting"
                game["wagers"] = {"you": 0, "dqn": 0}
                st.rerun()
        else:
            st.info("Session complete. Start a new session from the sidebar.")
    elif engine.round_number < 500:
        st.subheader("Next round")
        bets = legal_bets(game)
        wager = st.select_slider("Your wager", options=bets, value=bets[0],
                                 format_func=lambda bet: "Sit out" if bet == 0 else f"{bet} units")
        if st.button("Deal cards", type="primary"):
            deal_round(game, wager)
            st.rerun()
    else:
        st.success("Session complete. Start a new session from the sidebar.")

    if results:
        last = results[-1]
        st.caption(f'Last round: you {last["You profit"]:+g} units, opponent {last["Opponent profit"]:+g} units.')

    st.divider()
    st.subheader("Session statistics")
    chart = st.selectbox("Show graph", ("Bankroll", "Profit and loss", "Drawdown"))
    plot_results(results, chart)

    for column, label in ((first, "You"), (second, "Opponent")):
        bets = [r for r in results if r[f"{label} bet"] > 0]
        wins = sum(r[f"{label} profit"] > 0 for r in bets)
        column.caption(f"{label}: {len(bets)} rounds played, {wins} profitable rounds")

    if results:
        with st.expander("Round history"):
            st.dataframe(list(reversed(results[-20:])), hide_index=True, use_container_width=True)

    st.caption("Simulation only. Saved DQNs and calibration models are not retrained during play. A hidden one unit reference seat keeps the shoe moving, as in the original training setup.")


if __name__ == "__main__":
    play_page()
