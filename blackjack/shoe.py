import pydealer
import random

# a shoe is multiple (6) decks of cards shuffled together to make one deck 
class Shoe:
    def __init__(self, num_decks=6, seed=None, penetration_limit=0.75):
        # check that the number of decks is valid
        if type(num_decks) is not int or num_decks < 1:
            raise ValueError("Number of decks must be a positive integer")

        # check that penetration is between 0 and 1 (% of cards that have been already dealt from the shoe)
        if not 0 < penetration_limit <= 1:
            raise ValueError("Penetration must be between 0 and 1")

        # store the shoe settings
        self.num_decks = num_decks
        self.penetration_limit = penetration_limit

        # use a seed so we can reproduce the same shuffled cards
        self.rng = random.Random(seed)

        # track the number of times the shoe has been shuffled
        self.shuffle_count = 0

        # create and shuffle the initial shoe
        self.reset()

    def reset(self):
        # create an empty list to hold all the cards
        cards = []

        # add a standard 52 card deck for the number of decks in the shoe 
        for _ in range(self.num_decks):
            deck = pydealer.Deck()
            cards.extend(deck.cards)

        # shuffle all the cards together
        self.rng.shuffle(cards)

        # store the shuffled cards in a PyDealer stack
        self.shoe = pydealer.Stack(cards=cards)

        # record that the shoe has been shuffled
        self.shuffle_count += 1

    def deal(self, n=1):
        # check that we are dealing a valid number of cards
        if type(n) is not int or n < 1 or n > self.get_remaining():
            raise ValueError("Invalid number of cards requested")

        # deal the cards and remove them from the shoe
        return self.shoe.deal(n)

    def get_remaining(self):
        # get the number of cards left in the shoe
        return self.shoe.size

    def get_penetration(self):
        # calculate the number of cards in a full shoe
        total_cards = self.num_decks * 52

        # calculate the proportion of cards already dealt
        return 1 - self.get_remaining() / total_cards

    def needs_reshuffle(self):
        # check if the penetration limit has been reached
        return self.get_penetration() >= self.penetration_limit
