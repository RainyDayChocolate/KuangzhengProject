#!/usr/bin/env python
"""Several tools for the Mini-Doudizhu
"""

import random

import numpy as np

from pyminiddz.poker import (CARD_NAME, CARD_TO_RANK, empty_hand, empty_vector,
                             legal_hands)

PLAYERS = ['C', 'D', 'E']

PASS = '-'

generate_deck = lambda: '999TTTJJJQQQKKKAAA222B'

rotate = lambda cards_list, shift: cards_list[-shift:] + cards_list[:-shift]

def encoding():
    helper = [2, 2, 2, 2, 2, 2, 2, 1]
    all_hands, all_vector = legal_hands(empty_hand(), helper)
    all_hands.append(empty_hand())
    all_vector.append(empty_vector())
    encoding = {hand: ind for ind, hand in enumerate(all_hands)}
    decoding = {ind: move_vec for ind, move_vec in enumerate(all_vector)}

    return encoding, decoding

MOVE_ENCODING, MOVE_DECODING = encoding()


def next_player(current_player):
    """The sequence of player
    C -> D -> E
    """

    if current_player == 'C':
        return 'D'
    elif current_player == 'D':
        return 'E'
    elif current_player == 'E':
        return 'C'


def lower_upper(player):
    """Get the next player and the previous one.
    """

    lower_player = (player + 1) % 3
    upper_player = (player + 2) % 3
    return lower_player, upper_player


def check_wins(winner, player_idx):
    """To find out if the player_idx has won the game.
    """

    if winner == 0:
        return player_idx == 0
    return player_idx != 0


def card_trans_np(cards):
    """Transfer the cards from string format to the vector format
    """
    init_v = np.zeros(8, dtype=np.int8)
    if cards == PASS:
        return init_v

    for card in cards:
        init_v[CARD_TO_RANK[card]] += 1
    return init_v


def card_trans_np_reverse(cards_np):
    """Transfer the cards from the vector format to str format
    """

    cards_str = "".join([x * int(y) for x, y in zip(CARD_NAME, cards_np)])
    if not cards_str:
        return PASS
    return cards_str


def random_sample(cards, num):
    """Randoms sample num cards from cards
    Divide cards to two piles. upper(num) lower(cards' num - num)

    Params:
    ------
    cards(15X1 np.ndarray):
        Contains all cards you decide to random distribute
    num(Int):
        The number of cards in upper player's hand

    Returns:
    ------
    upper(15X1 np.ndarray):
        Random Distributed cards, sum(upper) = num
    lower(15X1 np.ndarray)
        Random Distributed cards, sum(lower) = sum(cards) - num
    """

    cardlist = np.array(list(card_trans_np_reverse(cards)))
    random.shuffle(cardlist)
    upper = card_trans_np("".join(cardlist[np.random.permutation(num)]))
    lower = cards.copy()
    lower -= upper

    return upper, lower


def deal_cards(seed=None):
    """Deal Cards at the begining of each game

    Params:
    ------
    Seed:
        Just a random Seed

    Returns:
    ------
    cards(3X8 np.ndarray):
        A array contains C's init_cards, D's init_cards, E's init_cards
    """

    random.seed(seed)
    cards = list(generate_deck())
    random.shuffle(cards)
    cards_C, cards_D, cards_E = cards[:8], cards[8:15], cards[15:]

    cards = np.vstack([card_trans_np(cards_C),
                       card_trans_np(cards_D),
                       card_trans_np(cards_E)])
    return cards
