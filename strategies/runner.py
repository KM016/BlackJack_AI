from blackjack.engine import ShoeExhausted

# auto pilot for the blackjack simulator

# run a complete blackjack round using betting and playing strategies
def run_round(engine, playing_strategies, betting_strategies):
    # require one playing strategy and one betting strategy for every seat
    seats = set(engine.players)
    if set(playing_strategies) != seats or set(betting_strategies) != seats:
        raise ValueError("Supply playing and betting strategies for every seat")

    # ask each players betting strategy for pre round wager
    wagers = {}
    for player_id in engine.players:
        observation = engine.betting_observation(player_id)
        wager = betting_strategies[player_id].choose_wager(observation)

        # None means the player sits out this round
        if wager is not None:
            wagers[player_id] = wager

    # do not deal any cards when everyone sits out
    if not wagers:
        return {"status": "no_active_seats", "round_number": engine.round_number, "wagers": {}}

    previous_round = engine.round_number

    try:
        # start the round with the chosen wagers
        engine.start_round(wagers)

        # let Basic Strategy play each active hand until the round is finished
        while engine.state == "player_turn":
            player_id = engine.current_player
            observation = engine.observation(player_id)
            action = playing_strategies[player_id].choose_action(observation)
            engine.act(player_id, action)

    # when the shoe is exhausted void the round
    except ShoeExhausted:
        # a compulsory missing card voids the round and refunds all wagers
        # other shortages are real errors and should not be hidden
        if engine.state != "void" or engine.round_number == previous_round:
            raise

    # return a short summary of the completed or voided round
    return {"status": engine.state, "round_number": engine.round_number, "wagers": wagers}
