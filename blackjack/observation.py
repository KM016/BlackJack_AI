from copy import deepcopy
from blackjack.history import card_record

# controls the info players and betting strategies can see
def observe(engine, player_id):
    # check that the player exists
    if player_id not in engine.players:
        raise ValueError("Unknown player")

    # only show the dealer's hidden card after it is revealed
    dealer_cards = engine.dealer.hand.cards

    if engine.dealer.revealed:
        visible_cards = dealer_cards
        dealer_total = engine.dealer.hand.get_total()
    else:
        visible_cards = dealer_cards[:1]
        dealer_total = None

    # store the information the player can see about the dealer
    dealer = {
        "cards": [card_record(card) for card in visible_cards],
        "hole_hidden": len(dealer_cards) > 1 and not engine.dealer.revealed,
        "total": dealer_total
    }

    # collect the visible information about each player's hands
    players = {}

    for seat_id, player in engine.players.items():
        hands = []

        for player_hand in player.hands:
            hands.append({
                "hand_id": player_hand.hand_id,
                "cards": [card_record(card) for card in player_hand.hand.cards],
                "total": player_hand.hand.get_total(),
                "soft": player_hand.hand.is_soft(),
                "from_split": player_hand.hand.from_split,
                "finished": player_hand.finished,
                "awaiting_card": player_hand.awaiting_card,
                "wager": engine.wagers.get(player_hand.hand_id),
                "outcome": player_hand.outcome,
                "net_profit": player_hand.net_profit
            })

        players[seat_id] = {"hands": hands}

    # return only the information available to the player
    return deepcopy({
        "round_number": engine.round_number,
        "state": engine.state,
        "player_id": player_id,
        "bankroll": engine.players[player_id].bankroll,
        "dealer": dealer,
        "players": players,
        "current_player": engine.current_player,
        "current_hand": engine.current_hand.hand_id if engine.current_hand else None,
        "legal_actions": engine.legal_actions(player_id),
        "events": engine.public_events
    })


def observe_betting(engine, player_id):
    # check that the player exists
    if player_id not in engine.players:
        raise ValueError("Unknown player")

    # betting decisions can only be made between rounds
    if engine.state not in ("betting", "complete", "void"):
        raise ValueError("Cannot observe betting during an active round")

    # get the player's available bankroll
    bankroll = engine.players[player_id].bankroll

    # return only the information available before betting
    return deepcopy({
        "state": engine.state,
        "round_number": engine.round_number,
        "next_round_number": engine.round_number + 1,
        "player_id": player_id,
        "bankroll": bankroll,
        "min_bet": engine.rules.min_bet,
        "max_bet": engine.rules.max_bet,
        "max_affordable_bet": min(bankroll, engine.rules.max_bet),
        "can_bet": bankroll >= engine.rules.min_bet,
        "num_decks": engine.rules.num_decks,
        "reshuffle_pending": engine.shoe.needs_reshuffle(),
        "events": engine.public_events
    })
