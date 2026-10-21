"""Play blackjack alongside the saved DQN and watch the results live.

Run from the project root with: streamlit run app.py
"""

import matplotlib.pyplot as plt
import streamlit as st

from app.game import BlackjackMatch, MAX_ROUNDS, MODEL_PATHS


st.set_page_config(page_title="Blackjack | You vs DQN", page_icon="♠️", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(ellipse at 55% -10%, #204b45 0%, #0a2525 48%, #08171b 100%);
    color: #f3f3e9;
}
.block-container { max-width: 1450px; padding-top: 2rem; }
h1, h2, h3 { letter-spacing: -.025em; }
[data-testid="stMetric"] {
    background: #132d30;
    border: 1px solid #355352;
    border-radius: 14px;
    padding: .7rem 1rem;
}
.table-surface {
    background: linear-gradient(145deg, #1b554a, #114238);
    border: 3px solid #af8c52;
    box-shadow: 0 12px 30px #05181975;
    border-radius: 32px;
    padding: 1.4rem 1.5rem 1.5rem;
    margin: .55rem 0 1.4rem;
}
.seat-label {
    color: #e8d9b7;
    letter-spacing: .12em;
    font-size: .79rem;
    font-weight: 800;
    text-transform: uppercase;
    margin-bottom: .5rem;
}
.card {
    display: inline-flex;
    width: 48px;
    height: 72px;
    border-radius: 7px;
    margin: 0 4px 6px 0;
    background: #fffdf6;
    color: #182934;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    box-shadow: 2px 3px 2px #041b1b88;
    border: 1px solid #e7dfc7;
    font-size: 1.15rem;
    font-weight: 900;
    line-height: 1.2;
}
.card.red { color: #be2e4b; }
.card.hidden {
    background: repeating-linear-gradient(45deg, #4d58a4, #4d58a4 5px, #353d78 5px, #353d78 10px);
    color: #fff;
    border: 2px solid #e7dcab;
}
.hand-meta { margin: .2rem 0 .8rem; color: #f5eee0; font-size: .83rem; }
.table-note { color: #d5e9d8; font-size: .87rem; margin-top: .3rem; }
.hero-note { color: #aec5c2; margin-bottom: 1.2rem; }
.stButton > button[kind="primary"] { border: 1px solid #e0bc70; }
</style>
""", unsafe_allow_html=True)


SUITS = {
    "Spades": "♠", "Hearts": "♥", "Diamonds": "♦", "Clubs": "♣",
}
RANKS = {"Jack": "J", "Queen": "Q", "King": "K", "Ace": "A"}


def cards_html(cards, hidden=False):
    if not cards and not hidden:
        return '<span class="table-note">No cards dealt</span>'
    blocks = []
    for card in cards:
        suit = SUITS.get(card["suit"], "♠")
        rank = RANKS.get(card["value"], card["value"])
        color = " red" if suit in ("♥", "♦") else ""
        blocks.append(f'<span class="card{color}">{rank}<br>{suit}</span>')
    if hidden:
        blocks.append('<span class="card hidden">✦</span>')
    return "".join(blocks)


def hands_html(hands):
    if not hands:
        return '<div class="table-note">Sitting out this round</div>'
    result = []
    for number, hand in enumerate(hands, 1):
        title = f"Hand {number}" if len(hands) > 1 else "Hand"
        status = hand["outcome"] or ("Finished" if hand["finished"] else "Playing")
        if hand["awaiting_card"]:
            status = "Waiting for split card"
        result.append(
            f'<div class="hand-meta">{title} · {hand["total"]} points · '
            f'{status} · Wager {hand["wager"] or "pending"}</div>'
            f'<div>{cards_html(hand["cards"])}</div>'
        )
    return "".join(result)


def plot_series(match, kind):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor("lightsteelblue")
    ax.set_facecolor("lemonchiffon")

    for seat, label, color in (("you", "You", "#165e82"), ("model", "DQN", "#9d426c")):
        if kind == "bankroll":
            data = match.bankroll[seat]
        elif kind == "profit":
            data = [amount - 1000 for amount in match.bankroll[seat]]
        else:
            data = match.drawdown[seat]
        ax.plot(match.rounds, data, label=label, color=color, linewidth=2, marker="o", markersize=3)

    if kind == "bankroll":
        ax.set_title("Bankroll over time")
        ax.set_ylabel("Bankroll (units)")
        ax.axhline(1000, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    elif kind == "profit":
        ax.set_title("Cumulative profit and loss")
        ax.set_ylabel("Profit / loss (units)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    else:
        ax.set_title("Drawdown from each player's bankroll peak")
        ax.set_ylabel("Drawdown (%)")
        ax.set_ylim(bottom=0)

    ax.set_xlabel("Completed rounds")
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(alpha=0.23)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


with st.sidebar:
    st.header("Table settings")
    model_choice = st.selectbox("Trained model", list(MODEL_PATHS))
    seed = st.number_input("Shoe seed", min_value=0, max_value=999999999, value=2026, step=1)
    if st.button("Start a new session", use_container_width=True):
        st.session_state.match = BlackjackMatch(seed=int(seed), model_name=model_choice)
        st.rerun()

    st.divider()
    st.markdown("**Table rules**")
    st.caption("Six decks · dealer stands on soft 17 · blackjack pays 3:2 · doubling and splitting allowed")
    st.caption("Starting bankroll: 1,000 units each")
    st.caption("Wager choices: sit out, 1, 2, 4, 6 or 8 units. Positive wagers must respect the 2% bankroll limit.")
    st.caption("The DQN selects wagers. Its hands use the original BasicStrategy. You control your own hand.")
    with st.expander("Why is there a reference seat?"):
        st.write("The original DQN was trained with a background reference player who always plays one unit. This app keeps that hidden automated seat so the shoe continues moving when you or the DQN sit out. It never affects either displayed bankroll.")

if "match" not in st.session_state:
    st.session_state.match = BlackjackMatch()

match = st.session_state.match

st.title("♠️ Blackjack | You vs DQN")
st.markdown(
    '<div class="hero-note">Play alongside your trained blackjack agent against the same dealer. '
    'Make your own decisions and see how your results compare after every round.</div>',
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns([1, 1, 0.85])
for panel, seat, heading in ((col_a, "you", "Your bankroll"), (col_b, "model", "DQN bankroll")):
    data = match.stats(seat)
    with panel:
        st.metric(heading, f'{data["bankroll"]:,.2f}', delta=f'{data["profit"]:+,.2f} units')
with col_c:
    st.metric("Rounds completed", f"{len(match.round_log)} / {MAX_ROUNDS}")

observation = match.observation()
dealer = observation["dealer"]
players = observation["players"]
running_count, true_count = match.visible_count()

with st.container():
    dealer_markup = cards_html(dealer["cards"], hidden=dealer["hole_hidden"])
    dealer_note = f'{dealer["total"]} points' if dealer["total"] is not None else "Hidden card not revealed"
    you_markup = hands_html(players["you"]["hands"])
    model_markup = hands_html(players["model"]["hands"])
    st.markdown(
        f'<div class="table-surface">'
        f'<div class="seat-label">Dealer · {dealer_note}</div>'
        f'<div>{dealer_markup}</div>'
        f'<hr style="border-color:#ffffff30; margin:1.2rem 0">'
        f'<div style="display:flex; gap:4%;">'
        f'<div style="flex:1; min-width:0"><div class="seat-label">You · Wager {match.current_wagers["you"]}</div>{you_markup}</div>'
        f'<div style="flex:1; min-width:0"><div class="seat-label">DQN · Wager {match.current_wagers["model"]}</div>{model_markup}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

if match.active and match.engine.current_player == "you":
    st.subheader("Your turn")
    hint = match.basic_strategy_hint()
    st.caption(f"BasicStrategy suggests: **{hint.capitalize()}**. You can make your own choice.")
    available = set(match.engine.legal_actions("you"))
    options = ("hit", "stand", "double", "split")
    buttons = st.columns(4)
    for column, action in zip(buttons, options):
        with column:
            if st.button(action.capitalize(), disabled=action not in available,
                         use_container_width=True, key=f"move_{match.round_number}_{action}_{match.engine.current_hand.hand_id}"):
                match.human_action(action)
                st.rerun()

elif match.can_start:
    st.subheader(f"Place your bet for round {match.round_number + 1}")
    allowed = match.legal_human_wagers()
    next_bet = st.select_slider(
        "Your initial wager", options=allowed, value=allowed[0],
        format_func=lambda wager: "Sit out" if wager == 0 else f"{wager} units",
        key=f"stake_{match.round_number + 1}",
    )
    if st.button("Deal next round", type="primary", use_container_width=True):
        match.start_round(next_bet)
        st.rerun()

else:
    st.success("Session complete. Start a new session from the sidebar to play again.")

if match.round_log:
    latest = match.round_log[-1]
    st.caption(
        f'Last round: you {latest["you_profit"]:+g} units ({latest["you_outcome"]}), '
        f'DQN {latest["model_profit"]:+g} units ({latest["model_outcome"]}).'
    )

st.caption(f"Running count: **{running_count:+d}** | True count: **{true_count:+.2f}** | Model: **{match.model_name}**")

st.divider()
st.header("Live statistics")
row1a, row1b = st.columns(2)
with row1a:
    fig = plot_series(match, "bankroll")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with row1b:
    fig = plot_series(match, "profit")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

row2a, row2b = st.columns([1.2, 0.8])
with row2a:
    fig = plot_series(match, "drawdown")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with row2b:
    st.subheader("Session comparison")
    for seat, label in (("you", "You"), ("model", "DQN")):
        stats = match.stats(seat)
        st.markdown(f"**{label}**")
        s1, s2 = st.columns(2)
        s1.metric("Rounds bet", stats["bets"])
        s2.metric("Rounds won", f'{stats["win_rate"]:.1f}%')
        st.caption(f'Max drawdown: {stats["peak_drawdown"]:.2f}%')

with st.expander("Round history", expanded=False):
    if match.round_log:
        st.dataframe(list(reversed(match.round_log[-20:])), use_container_width=True, hide_index=True)
    else:
        st.write("Your completed rounds will appear here.")

with st.expander("Recent playing decisions", expanded=False):
    moves = [move for move in match.action_log if move["seat"] != "reference"][-20:]
    if moves:
        st.dataframe(list(reversed(moves)), use_container_width=True, hide_index=True)
    else:
        st.write("Hands have not been played yet.")

st.caption("Simulation only. Bankroll values are game units, not real money. The DQN is the original frozen model and does not learn from this session.")
