"""
This script shows basic settings about Mini-Doudizhu.

Mini-Doudizhu settings:

Total KIND: 8
Total NUM: 22 CARDS
Total Player: 3, One landlord and TWO peasants
A deck of Cards: B222AAAKKKQQQJJJTTT999
    RANKING: B > 2 > A > K > Q > J > T > 9
    where B can beat any hand and doubles the score.

Five kinds of plays:
    0: PASS
    1: SINGLE CARD: 2, A, K, Q, J, T, 9
    2: PAIR CARD: 22, AA, KK, QQ, JJ, TT, 99
    3: STRAIGHT: AKQ, KQJ, QJT, JT9, AKQJ, KQJT, QJT9, AKQJT, KQJT9, AKQJT9
    4: BOMB(Super-card): B

For each Player:
    Landlord: 8 Cards
    Peasant Down: 7 Cards
    Peasant Up: 7 Cards
"""


from random import randint

import numpy as np


CARD_NAME = '9TJQKA2B'
CARD_TO_RANK = dict(zip(CARD_NAME, range(len(CARD_NAME))))

INVALID = -1
PASS = 0
SINGLE = 1  # 单牌
PAIR = 2  # 对牌
STRAIGHT = 3 # 顺子
SUPER = 4 # 炸弹 B

CARD_PER_RANK = {PASS: 0,
                 SINGLE: 1,
                 PAIR: 2,
                 STRAIGHT: 1,
                 SUPER: 1}

MIN_RANK = CARD_NAME[0]
MIN_STRAIGHT_LEN = 3


empty_hand = lambda: (0, 0, 0)
empty_vector = lambda: np.zeros(len(CARD_NAME), int)
is_pass = lambda cards: (not cards.any())


def is_super(play):
    """Checks if play is a super play

    Params:
    ------
    play(np.ndarray(1D)):
        A choice played, dim=8,

    Return:
    ------
    A tuple: (SUPER, PRIMARY, 1) for super-card, () otherwise.
    """

    owned_ranks = np.where(play >= 1)[0]
    if len(owned_ranks) != 1:
        return ()

    _card = int(owned_ranks[0])
    if _card != CARD_TO_RANK['B'] or play[_card] != 1:
        return ()

    return (SUPER, _card, 1)


def is_single(play):
    """Checks if play is a single

    Params:
    ------
    play(np.ndarray(1D)):
        A choice played, dim=8,

    Return:
    ------
    A tuple: (SINGLE, PRIMARY, 1) for single, () otherwise.
    """

    owned_ranks = np.where(play >= 1)[0]
    if len(owned_ranks) != 1:
        return ()

    _card = int(owned_ranks[0])
    if _card > CARD_TO_RANK['2'] or play[_card] != 1:
        return ()

    return (SINGLE, _card, 1)


def is_pair(play):
    """Checks if play is a pair

    Params:
    ------
    play(np.ndarray(1D)):
        A choice played, dim=8,

    Return:
    ------
    A tuple: (PAIR, PRIMARY, 1) for pair, () otherwise.
    """

    owned_ranks = np.where(play >= 2)[0]
    if len(owned_ranks) != 1:
        return ()

    _card = int(owned_ranks[0])
    if _card > CARD_TO_RANK['2'] or play[_card] != 2:
        return ()

    return (PAIR, _card, 1)


def is_straight(play):
    """Checks if play is a straight

    Params:
    ------
    play(np.ndarray(1D)):
        A choice played, dim=8,

    Return:
    ------

    A tuple: (STRAIGHT, PRIMARY, STRAIGHT_LENGTH) for straight, () otherwise
    """

    if is_pass(play):
        return ()

    owned_ranks = np.where(play == 1)[0]
    max_card = int(owned_ranks[-1])
    if max_card >= CARD_TO_RANK['2']:
        return ()

    num_owned_ranks = len(owned_ranks)
    if num_owned_ranks < 3:
        return ()

    if owned_ranks[-1] - owned_ranks[0] + 1 != num_owned_ranks:
        return ()

    return (STRAIGHT, max_card, num_owned_ranks)


def recognize(play):
    """Pattern recognization.

    Returns the pattern if the play is valid, (INVALID(-1), 0, 0) otherwise.

    Params:
    -----
    play(np.ndarray(1D)):
        A choice played, dim=8, np.ndarray(1D)

    Return:
    ------
    pattern(tuple):
        The format is (Pattern, Primary, Length)
    """

    if is_pass(play):
        return (0, 0, 0)

    single_pattern = is_single(play)
    if single_pattern:
        return single_pattern

    pair_pattern = is_pair(play)
    if pair_pattern:
        return pair_pattern

    straight_pattern = is_straight(play)
    if straight_pattern:
        return straight_pattern

    super_pattern = is_super(play)
    if super_pattern:
        return super_pattern

    return (INVALID, 0, 0)


def get_longest_straights(unduplicate_ranks):
    """All longest straights of each primary, ranks are sorted
    """

    straight_len = len(unduplicate_ranks)
    if straight_len < 3:
        return []
    left_pointer, right_pointer = straight_len - 2, straight_len - 1

    interval, intervals = [unduplicate_ranks[right_pointer], unduplicate_ranks[right_pointer]], []
    while left_pointer >= 0:
        if unduplicate_ranks[left_pointer] == interval[0] - 1:
            interval[0] = unduplicate_ranks[left_pointer]
            left_pointer -= 1
        else:
            if interval[1] - interval[0] > 1: # straight diff between head and tail
                intervals.append(interval)
            right_pointer = left_pointer
            left_pointer = right_pointer - 1
            interval = [unduplicate_ranks[right_pointer], unduplicate_ranks[right_pointer]]
    if interval[1] - interval[0] > 1: # straight diff between head and tail
        intervals.append(interval)

    straights = [(STRAIGHT, interval[1], interval[1] - interval[0] + 1)
                 for interval in intervals]

    return straights


def analyze(cards):
    """Divide cards, finding out the longest straight of each primary, all singles, all pairs

    Params:
    ------
    cards(np.ndarray):
        Current cards

    Returns:
    ------
    in_hands(Dict):
        Return a dict contains different bucket hand
        if the cards is the vector of B2KQJT99
        {SINGLE(1): [2, K, Q, J, T, 9],
        PAIR(2):   [99],
        STRAIGHT(3):[KQJT9], # The longest uncrossed straights.
                    Cross means (KQJ cross JT9 = J),
                    we need all s, in which (s1 cross s2 = empty)
        SUPER(4):   [B]
        }
    """

    cards = np.array(cards)
    in_hands = {}
    singles = np.where(cards >= 1)[0]
    single_hands = [(SINGLE, card, 1) for card in singles if card < CARD_TO_RANK['B']]
    in_hands[SINGLE] = single_hands

    pairs = np.where(cards >= 2)[0]
    pair_hands = [(PAIR, card, 1) for card in pairs]
    in_hands[PAIR] = pair_hands

    straight_primaries = [card for card in singles if card < CARD_TO_RANK['2']]
    straights = get_longest_straights(straight_primaries)
    in_hands[STRAIGHT] = straights

    if cards[-1] == 1:
        in_hands[SUPER] = (SUPER, CARD_TO_RANK['B'], 1)
    return in_hands


def hand2vector(hand):
    """Hands such as (STRAIGHT, 4, 4) to vector format like[0, 1, 1, 1, 1, 0, 0, 0]

    Params:
    ------
    hand(Tuple):
        A 3 dim tuple

    Returns:
    -----
    vector(ndarray):
        An 8 dim np.ndarray
    """

    pattern, primary, length = hand[0], hand[1], hand[2]
    if pattern == INVALID:
        raise ValueError('Could not convert this hand to cards')

    vector = empty_vector()
    if not pattern:
        return vector

    for rank in range(primary - length + 1, primary + 1):
        vector[rank] = CARD_PER_RANK[pattern]

    return vector


def hands_can_beat(last_hand, in_hands):
    """Find out all hands stronger than last_hand from in_hands

    Params:
    ------
    last_hand(Tuple):
        Hand to be compared
    in_hands(List):
        Set of legal hands of this pattern

    Returns:
    ------
    hands(List):
        Avalible hands in this specific pattern
    """
    last_pattern = last_hand[0]
    if not last_pattern:
        return in_hands

    hands_primary = lambda hand: hand[1]
    is_stronger = lambda hand: hand and hands_primary(hand) > hands_primary(last_hand)

    return filter(is_stronger, in_hands)


straight_min = lambda _max, _l: _max - _l + 1


def available_straight_hand(max_card, hand_len, is_lead,
                            min_len=MIN_STRAIGHT_LEN,
                            min_max_rank=MIN_STRAIGHT_LEN-1):
    """Return available rank

    Params:
    ------
    max_card(Int):
        The primary of this hand
    hand_len(Int):
        The length of straight
    is_lead(Bool):
        Whether the player in his lead_round,
    min_len(Int):
        The smallest length of available straight
        default In Mini game, the length of straight should be at least 3
    min_max_rank(Int):
        The smallest Primary of available straight
        In Mini game, the primary of straights should be at least 2 (i.e. J)

    Return:
    ------
    available_hands(Set):
        contains available straight_hand each element dim=3
    """
    # h_min, h_max, h_len just describe those infos in THE HAND
    # if we want to apply to certain kind of straights, we have to limit the top
    hand_min = straight_min(max_card, hand_len)
    if is_lead:
        # Count from the smallest straight
        return [(STRAIGHT, top_card, l)
                for l in range(min_len, hand_len + 1)
                for top_card in range(hand_min + l - 1, max_card + 1)]

    if max_card >= min_max_rank and hand_len >= min_len:
        # Count from the constrained straight
        min_top = max(hand_min + min_len - 1, min_max_rank)
        return [(STRAIGHT, top_card, min_len)
                for top_card in range(min_top, max_card + 1)]

    return []


def available_straights(last_hand, in_hands):
    """Find out straights can beat current_cards

    Params:
    ------
    last_hand(Tuple):
        To be compared, than_hand = (0, 0, 0) means last play in Pass
        or we should choose hands that bigger than that hand
    in_hands(List):
        A list of players choice(tuples)
    cards_per_rank(Int):
        single straight or pair straight

    Returns:
    ------
    hands(List):
        Avaliable hands
    """

    if not in_hands:
        return []
#    if len(last_hand) == 4:
#        last_hand = [last_hand[0], last_hand[1], last_hand[-1]]
    pattern, primary, length = last_hand

    # Follow or not follow settings
    min_len = length if length else MIN_STRAIGHT_LEN
    min_max_rank = primary + 1 if pattern else MIN_STRAIGHT_LEN - 1

    # Give legal straights
    straight_hands = []
    is_lead = not pattern 
    for (_, hand_max, hand_len) in in_hands:
        legal_straights = available_straight_hand(hand_max, hand_len, is_lead,
                                                  min_len=min_len,
                                                  min_max_rank=min_max_rank)
        straight_hands.extend(legal_straights)

    return straight_hands


def follow_legal_hands(last_hand, in_cards, in_hands):
    """Give legal hands in follow round
    """
    play_hands = []

    play_hands.append(empty_hand())

    pattern = last_hand[0]
    if pattern == SUPER:
        return play_hands

    hands = []
    if pattern in set([SINGLE, PAIR]):
        hands = hands_can_beat(last_hand, in_hands[pattern])
    elif pattern == STRAIGHT:
        hands = available_straights(last_hand, in_hands[pattern])

    if hands:
        play_hands.extend(hands)

    # super card
    if in_cards[-1] == 1:
        super_card = np.array([0] * 7 + [1])
        helper = recognize(super_card)
        play_hands.append(helper)

    return play_hands


def lead_legal_hands(in_cards, in_hands=None):
    """Give legal hands in lead round
    """

    play_hands = []

    for pattern in set([SINGLE, PAIR]):
        hands = hands_can_beat(empty_hand(), in_hands[pattern])
        play_hands.extend(hands)

    pattern = STRAIGHT
    hands = available_straights(empty_hand(), in_hands[pattern])
    play_hands.extend(hands)
    if in_cards[-1] >= 1:
        super_card = np.array([0] * 7 + [1])
        helper = recognize(super_card)
        play_hands.append(helper)

    return play_hands


def legal_hands(last_hand, in_cards):
    """
    Last Play(Tuple):
        Last choice
    In cards(np.ndarray):
        Current cards in hand
    """
    if not last_hand:
        raise ValueError('last_hand should be a list, at least 3 dimension')

    in_hands = analyze(in_cards)
    is_lead = not last_hand[0]
    if is_lead:
        hands = lead_legal_hands(in_cards, in_hands)
    else:
        hands = follow_legal_hands(last_hand, in_cards, in_hands)

    plays = [hand2vector(hand) for hand in hands]

    return hands, plays
