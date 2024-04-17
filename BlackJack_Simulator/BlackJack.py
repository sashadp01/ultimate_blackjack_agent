"""
    Modified BlackJack Simulator into a Gym environment
    Comments with # Addition: indicate modifications
"""

import sys
from random import shuffle

import numpy as np
from BlackJack_Simulator.importer.StrategyImporter import StrategyImporter

# Addition: new imports
import gymnasium as gym
from gymnasium import spaces

SHOE_SIZE = 1
SHOE_PENETRATION = 0.25
BET_SPREAD = 20.0

DECK_SIZE = 52.0
CARDS = {
    "Ace": 11,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
    "Six": 6,
    "Seven": 7,
    "Eight": 8,
    "Nine": 9,
    "Ten": 10,
    "Jack": 10,
    "Queen": 10,
    "King": 10,
}
BASIC_OMEGA_II = {
    "Ace": 0,
    "Two": 1,
    "Three": 1,
    "Four": 2,
    "Five": 2,
    "Six": 2,
    "Seven": 1,
    "Eight": 0,
    "Nine": -1,
    "Ten": -2,
    "Jack": -2,
    "Queen": -2,
    "King": -2,
}

BLACKJACK_RULES = {
    "triple7": False,  # Count 3x7 as a blackjack
}

ACTIONS = {"H": 0, "S": 1, "D": 2, "Sr": 3, "P": 4}

HARD_STRATEGY = {}
SOFT_STRATEGY = {}
PAIR_STRATEGY = {}


# Addition: new class
class UltimateBlackjackRoundEnv(gym.Env):
    """Custom Environment that follows gym interface. Episode is a single round of Blackjack."""

    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode="console", card_counting=True, n_decks=1):
        global SHOE_SIZE
        super(UltimateBlackjackRoundEnv, self).__init__()
        self.render_mode = render_mode
        self.card_counting = card_counting
        # define action and observation space
        self.action_space = spaces.Discrete(
            5
        )  # Actions are Hit, Stand, Double, Surrender, Split
        # observation space:
        # State space:
        # Player sum: 3-31,
        # Dealer's upcard: 1-10,
        # Usable ace: 0 or 1,
        # pair in hand: 0 or 1,
        # first action in hand: 0 or 1
        # aces, 2-9, 10 or face card: 0-4 except 10 or face card: 0-16 (assuming 1 deck)
        SHOE_SIZE = n_decks
        c = 4 * SHOE_SIZE
        if self.card_counting:
            self.observation_space = spaces.Box(
                low=np.array([3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                high=np.array([31, 10, 1, 1, 1, c, c, c, c, c, c, c, c, c, 4 * c]),
                dtype=int,
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array([3, 1, 0, 0, 0]),
                high=np.array([31, 10, 1, 1, 1]),
                dtype=int,
            )
        self.game = Game(randomize_shoe_state=True)
        self.actions_played = []
        self.illegal_moves = 0
        self.won = 0
        self.returns = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # create new game
        self.game = Game(randomize_shoe_state=True)
        self.game.start_gym_round()
        # return initial observation
        observation = self.get_observation()
        self.actions_played = []
        self.illegal_move_made = 0
        self.won = 0
        self.returns = []
        return (
            observation,
            self.game.get_all_cards(),
        )  # Addition: return observation and list of all cards

    # get observation in the format [Player sum: 3-31, Dealer's upcard: 1-10, Usable ace: 0 or 1]
    def get_observation(self):
        player_sum = self.game.player.hands[0].value
        dealer_sum = self.game.dealer.hand.value
        usable_ace = 1 if self.game.player.hands[0].aces_soft > 0 else 0
        pair_in_hand = 1 if self.game.player.hands[0].splitable() else 0
        first_action_in_hand = 1 if self.game.player.hands[0].length() == 2 else 0

        cards_played = self.game.shoe.cards_played
        return_array = np.array(
            [player_sum, dealer_sum, usable_ace, pair_in_hand, first_action_in_hand]
        )
        # concatenate the cards played
        if self.card_counting:
            return_array = np.concatenate([return_array, cards_played])
        return return_array.astype(int)

    # take action in the round
    def step(self, action):
        self.actions_played.append(action)
        # take action
        round_over, illegal = self.game.player.take_action(action, self.game.shoe)
        if illegal:
            reward = -1
            terminated = True
            status = None
            self.illegal_move_made = 1
        elif not round_over:
            reward = 0
            terminated = False
            status = None
        else:
            # let dealer play
            self.game.dealer.play(self.game.shoe)
            # get winnings
            win, bet, status = self.game.get_hand_winnings(self.game.player.hands[0])
            if status == "WON" or status == "WON 3:2":
                self.won = 1
            reward = win
            terminated = True

        self.returns.append(reward)

        # return observation, reward, terminated, truncated, info
        if terminated:
            info = {}
            info["illegal"] = self.illegal_move_made
            info["won"] = self.won
            info["hit"] = self.actions_played.count(0)
            info["stand"] = self.actions_played.count(1)
            info["double"] = self.actions_played.count(2)
            info["surrender"] = self.actions_played.count(3)
            info["split"] = self.actions_played.count(4)
            info["return"] = sum(self.returns)
            return self.get_observation(), reward, terminated, False, info
        else:
            return self.get_observation(), reward, terminated, False, {}

    # render the environment
    def render(self):
        if self.render_mode == "console":
            print("Player Hand: %s" % self.game.player.hands[0])
            print("Dealer Hand: %s" % self.game.dealer.hand)
            # print if player won or lost

    def close(self):
        pass


class Card(object):
    """
    Represents a playing card with name and value.
    """

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return "%s" % self.name


class Shoe(object):
    """
    Represents the shoe, which consists of a number of card decks.
    """

    reshuffle = False

    def __init__(self, decks, randomize_shoe_state=False):
        self.count = 0
        self.count_history = []
        self.ideal_count = {}
        self.decks = decks
        self.cards = self.init_cards()
        self.cards_played = np.zeros(
            10
        )  # Addition: keep track of number of cards played (ace,2-9,10 or face card)
        self.init_count()

        # Addition: randomize shoe state by removing a random number of cards
        if randomize_shoe_state:
            cards_to_remove = np.random.randint(
                0, len(self.cards) * (1 - SHOE_PENETRATION)
            )
            for i in range(cards_to_remove):
                self.do_count(self.cards.pop())

    def __str__(self):
        s = ""
        for c in self.cards:
            s += "%s\n" % c
        return s

    def init_cards(self, randomize_shoe_state=False):
        """
        Initialize the shoe with shuffled playing cards and set count to zero.
        """
        self.count = 0
        self.count_history.append(self.count)

        cards = []
        for d in range(self.decks):
            for c in CARDS:
                for i in range(0, 4):
                    cards.append(Card(c, CARDS[c]))
        shuffle(cards)

        return cards

    def init_count(self):
        """
        Keep track of the number of occurrences for each card in the shoe in the course over the game. ideal_count
        is a dictionary containing (card name - number of occurrences in shoe) pairs
        """
        for card in CARDS:
            self.ideal_count[card] = 4 * SHOE_SIZE

    def deal(self):
        """
        Returns:    The next card off the shoe. If the shoe penetration is reached,
                    the shoe gets reshuffled.
        """
        if self.shoe_penetration() < SHOE_PENETRATION:
            self.reshuffle = True
        if len(self.cards) == 0:  # prevent index out of range
            self.cards = self.init_cards()
            self.init_count()
            self.reshuffle = False
        card = self.cards.pop()

        assert self.ideal_count[card.name] > 0, "Either a cheater or a bug!"
        self.ideal_count[card.name] -= 1

        self.do_count(card)
        return card

    def do_count(self, card):
        """
        Add the dealt card to current count.
        """
        self.count += BASIC_OMEGA_II[card.name]
        self.count_history.append(self.truecount())
        # Addition: keep track of number of cards played
        card_index = card.value if not (card.value == 11) else 1
        self.cards_played[card_index - 1] += 1

    def truecount(self):
        """
        Returns: The current true count.
        """
        return self.count / (self.decks * self.shoe_penetration())

    def shoe_penetration(self):
        """
        Returns: Ratio of cards that are still in the shoe to all initial cards.
        """
        return len(self.cards) / (DECK_SIZE * self.decks)


class Hand(object):
    """
    Represents a hand, either from the dealer or from the player
    """

    _value = 0
    _aces = []
    _aces_soft = 0
    splithand = False
    surrender = False
    doubled = False
    first_action = True

    def __init__(self, cards):
        self.cards = cards

    def __str__(self):
        h = ""
        for c in self.cards:
            h += "%s " % c
        return h

    @property
    def value(self):
        """
        Returns: The current value of the hand (aces are either counted as 1 or 11).
        """
        self._value = 0
        for c in self.cards:
            self._value += c.value

        if self._value > 21 and self.aces_soft > 0:
            for ace in self.aces:
                if ace.value == 11:
                    self._value -= 10
                    ace.value = 1
                    if self._value <= 21:
                        break

        return self._value

    @property
    def aces(self):
        """
        Returns: The all aces in the current hand.
        """
        self._aces = []
        for c in self.cards:
            if c.name == "Ace":
                self._aces.append(c)
        return self._aces

    @property
    def aces_soft(self):
        """
        Returns: The number of aces valued as 11
        """
        self._aces_soft = 0
        for ace in self.aces:
            if ace.value == 11:
                self._aces_soft += 1
        return self._aces_soft

    def soft(self):
        """
        Determines whether the current hand is soft (soft means that it consists of aces valued at 11).
        """
        if self.aces_soft > 0:
            return True
        else:
            return False

    def splitable(self):
        """
        Determines if the current hand can be splitted.
        """
        if self.length() == 2 and self.cards[0].name == self.cards[1].name:
            return True
        else:
            return False

    def blackjack(self):
        """
        Check a hand for a blackjack, taking the defined BLACKJACK_RULES into account.
        """
        if not self.splithand and self.value == 21:
            if all(c.value == 7 for c in self.cards) and BLACKJACK_RULES["triple7"]:
                return True
            elif self.length() == 2:
                return True
            else:
                return False
        else:
            return False

    def busted(self):
        """
        Checks if the hand is busted.
        """
        if self.value > 21:
            return True
        else:
            return False

    def add_card(self, card):
        """
        Add a card to the current hand.
        """
        self.cards.append(card)

    def split(self):
        """
        Split the current hand.
        Returns: The new hand created from the split.
        """
        self.splithand = True
        c = self.cards.pop()
        new_hand = Hand([c])
        new_hand.splithand = True
        return new_hand

    def length(self):
        """
        Returns: The number of cards in the current hand.
        """
        return len(self.cards)


class Player(object):
    """
    Represent a player
    """

    def __init__(self, hand=None, dealer_hand=None):
        self.hands = [hand]
        self.dealer_hand = dealer_hand

    def set_hands(self, new_hand, new_dealer_hand):
        self.hands = [new_hand]
        self.dealer_hand = new_dealer_hand

    def play(self, shoe):
        for hand in self.hands:
            # print("Player Hand from play(): " + str(hand))
            self.play_hand(hand, shoe)

    def play_hand(self, hand, shoe):
        if hand.length() < 2:
            if hand.cards[0].name == "Ace":
                hand.cards[0].value = 11
            self.hit(hand, shoe)
        # print("Player Hand from play_hand(): " + str(hand))
        while not hand.busted() and not hand.blackjack():
            # print("Value: " + str(hand.value))
            # print("Hand: " + str(hand))
            # print("Soft: " + str(hand.soft()))
            # print("Splitable: " + str(hand.splitable()))
            if hand.soft():
                flag = SOFT_STRATEGY[hand.value][self.dealer_hand.cards[0].name]
            elif hand.splitable():
                flag = PAIR_STRATEGY[hand.value][self.dealer_hand.cards[0].name]
            else:
                flag = HARD_STRATEGY[hand.value][self.dealer_hand.cards[0].name]

            if flag == "D":
                if hand.length() == 2:
                    # print "Double Down"
                    hand.doubled = True
                    self.hit(hand, shoe)
                    break
                else:
                    flag = "H"

            if flag == "Sr":
                if hand.length() == 2:
                    # print "Surrender"
                    hand.surrender = True
                    break
                else:
                    flag = "H"

            if flag == "H":
                self.hit(hand, shoe)

            if flag == "P":
                self.split(hand, shoe)

            if flag == "S":
                break
            # print("Flag: " + flag)
        # print("Player Hand: " + str(hand))

    # Addition: take action in the round
    def take_action(self, action, shoe):
        hand = self.hands[0]
        round_over = False
        illegal = False
        if action == 0:  # Hit
            self.hit(hand, shoe)
        elif action == 1:  # Stand
            round_over = True
        elif hand.first_action:
            if action == 2:  # Double
                hand.doubled = True
                self.hit(hand, shoe)
                round_over = True
            elif action == 3:  # Surrender
                hand.surrender = True
                round_over = True
            elif action == 4:  # Split
                if hand.splitable():
                    self.split(hand, shoe)
                else:
                    # invalid action
                    illegal = True
        else:
            # invalid action
            illegal = True

        if not illegal and action != 4:
            hand.first_action = False

        if hand.busted() or hand.blackjack():
            round_over = True
        return round_over, illegal

    def hit(self, hand, shoe):
        c = shoe.deal()
        hand.add_card(c)
        # print "Hitted: %s" % c

    def split(self, hand, shoe):
        # Addition: override split function to new behavior:
        # - Remove one of the duplicate cards from the current hand
        # - Hit the current hand
        # - set splithand to True: will double the bet
        c = hand.cards.pop()
        if hand.cards[0].name == "Ace":
            hand.cards[0].value = 11
        self.hit(hand, shoe)
        hand.splithand = True

        # self.hands.append(hand.split())
        # # print "Splitted %s" % hand
        # self.play_hand(hand, shoe)


class Dealer(object):
    """
    Represent the dealer
    """

    def __init__(self, hand=None):
        self.hand = hand

    def set_hand(self, new_hand):
        self.hand = new_hand

    def play(self, shoe):
        while self.hand.value < 17:
            self.hit(shoe)
        # print("Dealer Hand: " + str(self.hand))

    def hit(self, shoe):
        c = shoe.deal()
        self.hand.add_card(c)
        # print "Dealer hitted: %s" %c

    # Returns an array of 6 numbers representing the probability that the final score of the dealer is
    # [17, 18, 19, 20, 21, Busted] '''
    # TODO Differentiate 21 and BJ
    # TODO make an actual tree, this is false AF
    def get_probabilities(self):
        start_value = self.hand.value
        # We'll draw 5 cards no matter what an count how often we got 17, 18, 19, 20, 21, Busted


class Tree(object):
    """
    A tree that opens with a statistical card and changes as a new
    statistical card is added. In this context, a statistical card is a list of possible values, each with a probability.
    e.g : [2 : 0.05, 3 : 0.1, ..., 22 : 0.1]
    Any value above 21 will be truncated to 22, which means 'Busted'.
    """

    # TODO to test
    def __init__(self, start=[]):
        self.tree = []
        self.tree.append(start)

    def add_a_statistical_card(self, stat_card):
        # New set of leaves in the tree
        leaves = []
        for p in self.tree[-1]:
            for v in stat_card:
                new_value = v + p
                proba = self.tree[-1][p] * stat_card[v]
                if new_value > 21:
                    # All busted values are 22
                    new_value = 22
                if new_value in leaves:
                    leaves[new_value] = leaves[new_value] + proba
                else:
                    leaves[new_value] = proba


class Game(object):
    """
    A sequence of Blackjack Rounds that keeps track of total money won or lost
    """

    def __init__(self, randomize_shoe_state=False):
        # print("SHOE_SIZE: ", SHOE_SIZE)
        self.shoe = Shoe(SHOE_SIZE, randomize_shoe_state)
        self.money = 0.0
        self.bet = 0.0
        self.stake = 1.0
        self.player = Player()
        self.dealer = Dealer()

    def get_hand_winnings(self, hand):
        win = 0.0
        bet = self.stake
        if not hand.surrender:
            if hand.busted():
                status = "LOST"
            else:
                if hand.blackjack():
                    if self.dealer.hand.blackjack():
                        status = "PUSH"
                    else:
                        status = "WON 3:2"
                elif self.dealer.hand.busted():
                    status = "WON"
                elif self.dealer.hand.value < hand.value:
                    status = "WON"
                elif self.dealer.hand.value > hand.value:
                    status = "LOST"
                elif self.dealer.hand.value == hand.value:
                    if self.dealer.hand.blackjack():
                        status = "LOST"  # player's 21 vs dealers blackjack
                    else:
                        status = "PUSH"
        else:
            status = "SURRENDER"

        if status == "LOST":
            win += -1
        elif status == "WON":
            win += 1
        elif status == "WON 3:2":
            win += 1.5
        elif status == "SURRENDER":
            win += -0.5
        if hand.doubled:
            win *= 2
            bet *= 2
        if hand.splithand:
            win *= 2
            bet *= 2
            # print(f"Splitted hand. Win: {win}, Bet: {bet}")

        win *= self.stake

        return win, bet, status

    def play_round(self):
        if self.shoe.truecount() > 6:
            # print("Bet Spread: %f" % BET_SPREAD)
            self.stake = BET_SPREAD
        else:
            self.stake = 1.0

        player_hand = Hand([self.shoe.deal(), self.shoe.deal()])
        dealer_hand = Hand([self.shoe.deal()])
        self.player.set_hands(player_hand, dealer_hand)
        self.dealer.set_hand(dealer_hand)
        # print "Dealer Hand: %s" % self.dealer.hand
        # print "Player Hand: %s\n" % self.player.hands[0]

        self.player.play(self.shoe)
        self.dealer.play(self.shoe)

        # print ""

        for hand in self.player.hands:
            win, bet, status = self.get_hand_winnings(hand)
            self.money += win
            self.bet += bet
            # print "Player Hand: %s %s (Value: %d, Busted: %r, BlackJack: %r, Splithand: %r, Soft: %r, Surrender: %r, Doubled: %r)" % (hand, status, hand.value, hand.busted(), hand.blackjack(), hand.splithand, hand.soft(), hand.surrender, hand.doubled)

        # print "Dealer Hand: %s (%d)" % (self.dealer.hand, self.dealer.hand.value)

    # Addition: deals the initial cards until player's turn
    def start_gym_round(self):
        self.stake = 1.0

        player_hand = Hand([self.shoe.deal(), self.shoe.deal()])
        dealer_hand = Hand([self.shoe.deal()])
        self.player.set_hands(player_hand, dealer_hand)
        self.dealer.set_hand(dealer_hand)
        return

    # Addition
    def get_all_cards(self):
        """
        Returns all cards in the game where each card is a tuple (card name, card value)
        """
        return {
            "cards": [
                card
                for hand in self.player.hands + [self.dealer.hand]
                for card in hand.cards
            ],
        }

    def get_money(self):
        return self.money

    def get_bet(self):
        return self.bet


# Use this function for baselines
# Problems with adding a new class:
# Name of dealer card in strategy dictionary
# Deterministic true in predict
# Classifier
# >= instead of > for threshold
def omega_II_baseline_eval(
    basic_strategy_filename, nbr_rounds=100000, bet_spread=20.0, n_decks=1
):
    global HARD_STRATEGY, SOFT_STRATEGY, PAIR_STRATEGY, BET_SPREAD, SHOE_SIZE
    BET_SPREAD = bet_spread
    SHOE_SIZE = n_decks
    importer = StrategyImporter(basic_strategy_filename)
    HARD_STRATEGY, SOFT_STRATEGY, PAIR_STRATEGY = importer.import_player_strategy()
    # print("Hard: ", HARD_STRATEGY)
    # print("Soft: ", SOFT_STRATEGY)
    # print("Pair: ", PAIR_STRATEGY)
    moneys = []
    bets = []
    # countings = []
    # nb_hands = 0
    for g in range(nbr_rounds):
        game = Game(randomize_shoe_state=True)
        game.play_round()
        # # print '%s GAME no. %d %s' % (20 * '#', i + 1, 20 * '#')
        # game.play_round()
        # nb_hands += 1
        moneys.append(game.get_money())
        bets.append(game.get_bet())
        # countings += game.shoe.count_history

        # print(
        #     "WIN for Game no. %d: %s (%s bet)"
        #     % (
        #         g + 1,
        #         "{0:.2f}".format(game.get_money()),
        #         "{0:.2f}".format(game.get_bet()),
        #     )
        # )
    average_return = np.mean(moneys)
    print(f"""Average returns: {average_return}""")
    sume = 0.0
    total_bet = 0.0
    for value in moneys:
        sume += value
    for value in bets:
        total_bet += value

    # print(
    #     "\n%d hands overall, %0.2f hands per game on average"
    #     % (nb_hands, float(nb_hands) / GAMES)
    # )
    # print("%0.2f total bet" % total_bet)
    print(
        "Overall winnings: {} (edge = {} %)".format(
            "{0:.2f}".format(sume), "{0:.3f}".format(100.0 * sume / total_bet)
        )
    )
    return average_return
    # moneys = sorted(moneys)
    # Addition: remove plotting
    # fit = stats.norm.pdf(moneys, np.mean(moneys), np.std(moneys))  # this is a fitting indeed
    # pl.plot(moneys, fit, '-o')
    # pl.hist(moneys, normed=True)
    # pl.show()

    # plt.ylabel('count')
    # plt.plot(countings, label='x')
    # plt.legend()
    # plt.show()
