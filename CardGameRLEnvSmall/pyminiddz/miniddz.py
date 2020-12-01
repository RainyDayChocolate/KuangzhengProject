#!/usr/bin/env python


import copy as cp
import random
from collections import OrderedDict

import numpy as np

import pyminiddz.poker as poker
from pyminiddz.utils import (MOVE_ENCODING, MOVE_DECODING, card_trans_np,
                             card_trans_np_reverse, deal_cards, lower_upper,
                             random_sample, rotate)


class Move:
    """Determined by its vec and chang's encoding(4 dim)
    """
    def __init__(self, code=None, vec=None):
        if not code:
            self._code, self._vec = self.recognize_move(vec)
        else:
            self._code = code
            self._vec = np.array(vec, int)

    @staticmethod
    def recognize_move(vec):
        if vec is None:
            return poker.empty_hand(), poker.empty_vector()

        if isinstance(vec, str):
            vec = card_trans_np(vec)
        elif isinstance(vec, (tuple, list)):
            vec = np.array(vec, int)
        elif isinstance(vec, Move):
            return vec.get_code(), vec.get_vec()
        if not any(vec):
            return poker.empty_hand(), poker.empty_vector()

        return poker.recognize(vec), vec

    @property
    def move_type(self):
        return self._code[0]

    def get_vec(self):
        return cp.copy(self._vec)

    def get_str(self):
        helper = card_trans_np_reverse(self._vec)
        if helper == 'P':
            return ''
        return helper

    def get_code(self):
        return self._code

    def __str__(self):
        return card_trans_np_reverse(self._vec)

    def __repr__(self):
        return card_trans_np_reverse(self._vec)

    def __hash__(self):

        return hash(self._code)

    def __eq__(self, other):
        if not isinstance(other, Move):
            other = Move(vec=other)
        other_code = other.get_code()
        return self._code == other_code

    def copy(self):

        cp_move = Move(vec=self.get_vec().copy())
        return cp_move

    def is_pass(self):
        return self._code[0] == 0

    def is_bomb(self):
        return self._code[0] == poker.SUPER

    def to_dict(self):

        return {"code": self._code,
                "cards": card_trans_np_reverse(self._vec)}

    def from_dict(self, d):

        move = Move(d['code'], card_trans_np(d['cards']))
        return move


def move_encoder(move):
    """Transform Move(object) to decoding(int)
    """
    if not isinstance(move, Move):
        move = Move(vec=move)
    move_code = move.get_code()
    return MOVE_ENCODING[move_code]


def move_decoder(move_ind):
    """From decoding(int) to move
    """
    move_vec = MOVE_DECODING[int(move_ind)]
    return Move(vec=move_vec)


PLAYERS = ['C', 'D', 'E']

PASS = Move()
LEGAL_MOVES = OrderedDict()


def legal_moves(last_move, hand_vec, cache_len=10000):
    """Get legal moves according to hands that in pyminiddz/hands.py
    Params:
    -----
    last_move(Move):
        Last choice
    hand_vec(np.ndarray):
        cards owned

    Returns:
    -----
    legal_moves(List):
        A list of legal moves(Move)
    """

    situ_str = str(last_move) + "|" + card_trans_np_reverse(hand_vec)
    if situ_str in LEGAL_MOVES:
        return LEGAL_MOVES[situ_str]
    last_hand = last_move.get_code()
    if np.all(hand_vec <= 0):
        return []

    _, vec_plays = poker.legal_hands(last_hand, hand_vec)
    moves = [Move(vec=play) for play in vec_plays]
    LEGAL_MOVES[situ_str] = moves
    if len(LEGAL_MOVES) > cache_len:
        LEGAL_MOVES.popitem()

    return moves


class BaseGame:
    """This is an object with virtual methods
    """

    def __init__(self):
        self.current_player = None
        self.history = None
        self.player_cards = None
        self.last_player = None
        self.last_move = None

    def get_winner(self):
        raise NotImplementedError('Object does not contain this method')

    def is_end_of_game(self):
        return self.get_winner() is not None

    def get_current_player(self):
        return self.current_player

    def get_last_player(self):
        return self.last_player

    def get_last_move(self):
        return cp.copy(self.last_move)

    def get_history(self):
        return cp.copy(self.history)

    def check_wins(self, player_idx):
        assert isinstance(player_idx, int)
        if self.get_winner() is None:
            return None

        if self.get_winner() == 0:
            return player_idx == 0
        return player_idx != 0

    def get_bomb_num(self):
        bomb_num = [move.is_bomb() for move in self.history]
        return sum(bomb_num)

    def get_score(self, player_idx):

        if self.check_wins(player_idx) is None:
            return None

        bomb_num = sum([h.is_bomb() for h in self.history])
        raw_score = 2 ** bomb_num

        if self.check_wins(player_idx):
            return raw_score
        return -raw_score

    def next_player(self):
        return (self.current_player + 1) % 3

    def upper_player(self):
        return (self.current_player + 2) % 3

    def get_current_player_card(self):
        raise NotImplementedError('Object does not contain this method')

    def get_legal_moves(self):
        cards = cp.copy(self.get_current_player_card())

        if sum(cards) == 0:
            return tuple()
        if self.last_player == self.current_player:
            return legal_moves(PASS, cards)
        return legal_moves(self.last_move, cards)

    def get_cards_played(self, pos):
        """Return certain player's total play
        Total play is a 8-dim vec
        """

        total_play = poker.empty_vector()
        have_played_history = self.history[pos::3]
        for play in have_played_history:
            total_play += play.get_vec()

        return total_play

    def cards_played(self, pos):
        cards = poker.empty_vector()
        for move in self.history[pos:][::3]:
           cards += move.get_vec()
        return cards


    def is_lead_round(self):
        return self.last_player == self.current_player

    def get_last_play_player(self, history, current_player):
        """History and Situation to Last round information"""

        if not history:
            return PASS, 0

        for back_index, last_play in enumerate(history[:-4:-1]):
            if not last_play.is_pass():
                break
        last_player = (current_player - back_index - 1) % 3
        if last_player == current_player:
            last_play = PASS
        return last_play, last_player


class GameState(BaseGame):
    """This Object is designed for describing the Unfaced GameState, MUST be this point
    3 players are ['C', 'D' ,'E']
    cards should be [C's cards, D's cards, E's cards]
    !!!! So guessing phase in GameStatePov should follow this rule strictly
    """

    def __init__(self, seed=None,
                player_cards=None, last_player=0,
                 last_move=PASS, history=None, current_player=0, init=True):
        if not init:
            return

        if player_cards is None:
            player_cards = deal_cards()
        self.player_cards = np.vstack(player_cards)

        if not isinstance(last_move, Move):
            last_move = Move(vec=last_move)
        self.last_move = last_move

        if  history is None:
            history = []
        helper = lambda h: Move(vec=h) if not isinstance(h, Move) else h
        self.history = [helper(h) for h in history]
        self.last_player = last_player
        self.current_player = current_player

    def get_player_cards(self, player_idx):
        """Get someone's cards or several plays' total cards
        """
        if isinstance(player_idx, (int, float)):
            return cp.copy(self.player_cards[int(player_idx)])
        else:
            return sum([self.player_cards[p] for p in player_idx])

    def get_winner(self):
        """Who is the winner

        Returns:
        ------
        0 means, C
        1 mean, D
        2 mean, E
        """
        card_sum = self.player_cards.sum(axis=1)
        for player_idx, card_num in enumerate(card_sum):
            if card_num == 0:
                return player_idx

        return None

    def get_current_player_card(self):
        return cp.copy(self.player_cards[self.current_player])

    def do_move(self, move):
        """Finish follow operations:
        1 Append history
        2 Minus current_position's cards
        3 Change current player
        4 Record last play(if current player chooses a valid move)

        """
        if move is None:
            return
        if not isinstance(move, Move):
            move = Move(vec=move)
        move_vec = move.get_vec()
        self.player_cards[self.current_player] -= move_vec
        if (self.player_cards[self.current_player] < 0).any():
            raise ValueError('Cards num wrong')

        self.history.append(move)

        if not move.is_pass():
            self.last_move = move
            self.last_player = self.current_player
        self.current_player = self.next_player()

    def copy(self):
        other = GameState()
        other.last_player = self.last_player
        other.last_move = self.last_move.copy()
        other.history = self.history.copy()
        other.player_cards = self.player_cards.copy()
        other.current_player = self.current_player
        return other

    def player_min_card_num(self):
        return self.player_cards.sum(axis=1).min()

    def recall(self):
        """Regenerate history scenarios from current situation

        Return:
        -----
        """

        def state_recall_onestep(game_state):
            history = game_state.get_history()
            history_length = len(history)
            if history_length == 0:
                return None, None
            else:
                tail_move = history[-1]
                recall_player = (game_state.get_current_player() - 1) % 3
                recall_history = history[:-1]

                game_state_back = game_state.copy()
                game_state_back.history = recall_history
                game_state_back.current_player = recall_player

                game_state_back.player_cards[recall_player] += tail_move.get_vec()
                last_move, last_player = self.get_last_play_player(recall_history, recall_player)
                game_state_back.last_move = last_move
                game_state_back.last_player = last_player

                return game_state_back, tail_move

        s = self.copy()
        while True:
            one_back_state, one_back_move = state_recall_onestep(s)
            if not one_back_state:
                break
            yield one_back_state, one_back_move
            s = one_back_state

    def to_dict(self):
        # Recording helper
        res = {
            "cards": [card_trans_np_reverse(rec) for rec in self.player_cards],
            "current_player": self.current_player,
            "history": [h.to_dict() for h in self.history],
            "last_player": self.last_player,
            "last_move": self.last_move.to_dict()
        }
        return res

    def from_dict(self, d):
        # Recording helper
        self.player_cards = np.array([card_trans_np(cardstr)
                                        for cardstr in d['cards']])
        self.current_player = d['current_player']
        self.history = [Move().from_dict(h) for h in d['history']]
        # load history
        self.last_player = d['last_player']
        self.last_move = Move().from_dict(d['last_move'])

    def __repr__(self):
        res = "######Explicit Game State \n" + \
            "######C cards are: {} \n".format(card_trans_np_reverse(self.player_cards[0])) + \
            "######D cards are : {} \n".format(card_trans_np_reverse(self.player_cards[1])) +\
            "######E cards are : {} \n".format(card_trans_np_reverse(self.player_cards[2])) +\
            "######history is: {}\n".format(self.history) + \
            "######current player: {}".format(self.current_player)
        return res

    def sample_upper_lower(self):
        game_state = self.copy()
        lower, upper = lower_upper(self.current_player)
        other_cards = self.get_player_cards([lower, upper])
        sampled_lower, sampled_upper = random_sample(other_cards,
                                                     self.player_cards[lower].sum())
        game_state.player_cards[lower] = sampled_lower
        game_state.player_cards[upper] = sampled_upper
        return game_state


class GameStatePov(BaseGame):
    """Partial situation for the certain view of someone

    Pov = Point of view
    Thus self._pov shouldn't be changed
    """
    def __init__(self, pov):
        self._mycards = None
        self.set_pov(pov)

    def get_pov(self):
        return self._pov

    def set_pov(self, pov):
        self._pov= pov
        lower, upper = lower_upper(self._pov)
        self._lower = lower
        self._upper = upper

    def get_player_cards(self, player_idx):
        """Get someone's cards or several plays' total cards
        """
        if player_idx != self._pov:
            raise ValueError('The player_idx in Hidden State Must be the viewer')
        return self._mycards

    def from_pure_state(self, game_state):
        """Get Basic information
        Convert current state to the current decision maker's own knowledge
        All attributes are decided by my point of view
        """
        self._mycards = game_state.get_player_cards(self._pov)

        self.current_player = game_state.get_current_player()
        self.last_player = game_state.get_last_player()

        self.history = game_state.get_history()
        self.last_move = game_state.get_last_move()

        # Sythesis others' information
        lower, upper = lower_upper(self._pov)
        self._lower = lower
        self._upper = upper
        self._other_cards = game_state.get_player_cards([lower, upper])

        self.lower_cards_num = game_state.get_player_cards(lower).sum()
        self.upper_cards_num = game_state.get_player_cards(upper).sum()

    def get_mycards(self):
        return self._mycards

    def get_other_cards(self):
        return self._other_cards

    def get_other_cards_num(self):
        return self.lower_cards_num, self.upper_cards_num

    def get_current_player_card(self):
        return self._mycards

    def do_move(self, move):
        if not isinstance(move, Move):
            move = Move(vec=move)
        move_vec = move.get_vec()
        if self.current_player == self._pov:
            self._mycards -= move_vec
        elif self.current_player == self._lower:
            self.lower_cards_num -= move_vec.sum()
            self._other_cards -= move_vec
        elif self.current_player == self._upper:
            self.upper_cards_num -= move_vec.sum()
            self._other_cards -= move_vec
        self.history.append(move)

        assert self.lower_cards_num >= 0
        assert self.upper_cards_num >= 0

        if not move.is_pass():
            self.last_move = move
            self.last_player = self.current_player
        self.current_player = self.next_player()

    def sample_upper_lower(self):
        """current player get no idea what the real state looks like but guess
        on position player

        Similar to the method that transfer faced situation to unfaced situation
        """
        #Init an unfacaed GameState
        #if self.current_player != self._pov:
        #    raise ValueError('Current player should be the same as the viewer')

        unfaced_state = GameState(init=False)
        unfaced_state.current_player = self.current_player
        unfaced_state.last_player = self.last_player
        unfaced_state.last_move = cp.copy(self.last_move)
        unfaced_state.history = cp.copy(self.history)

        lower_cards, upper_cards = random_sample(self._other_cards, self.lower_cards_num)

        assert lower_cards.sum() == self.lower_cards_num
        assert upper_cards.sum() == self.upper_cards_num

        #Assign sampled cards
        cards = [self._mycards, lower_cards, upper_cards]
        # unfaced's cards should be CDE, so we have to rotate cards
        cards = rotate(cards, self._pov)
        cards = np.array(cards)
        unfaced_state.player_cards = cards

        return unfaced_state


    def get_winner(self):
        card_sum = [self._mycards.sum(), self.lower_cards_num, self.upper_cards_num]
        for player_idx, card_num in enumerate(card_sum):
            if card_num == 0:
                winner = (self._pov + player_idx) % 3
                return winner
        return None

    def player_min_card_num(self):
        """card number of the player who got least cards
        """
        return min(self._mycards.sum(),
            min(self.lower_cards_num, self.upper_cards_num))

    def __repr__(self):
        res = "######Hidden Game State on : {} \n".format(PLAYERS[self._pov]) + \
            "######My cards are: {} \n".format(card_trans_np_reverse(self._mycards)) + \
            "######Other Left cards are : {} \n".format(card_trans_np_reverse(self._other_cards)) +\
            "######history is: {}".format(self.history)

        return res

    def to_dict(self):
        res = {
            "mycards" : card_trans_np_reverse(self._mycards),
            "other_cards" : card_trans_np_reverse(self._other_cards),
            "current_player": self.current_player,
            "history" : [play.to_dict() for play in self.history],
            "last_player": self.last_player,
            "last_move": self.last_move.to_dict()
        }
        return res

    def from_dict(self, d):
        self._mycards = card_trans_np(d['mycards'])
        self._other_cards = card_trans_np(d['other_cards'])
        self.current_player = d['current_player']
        self.history = [Move().from_dict(h) for h in d['history']]
        # load history
        self.last_player = d['last_player']
        self.last_move = Move().from_dict(d['last_move'])

        history_num = [h.get_vec().sum() for h in self.history]
        played_num = [sum(history_num[::3]), sum(history_num[1::3]), sum(history_num[2::3])]
        left_num = [8 - played_num[0], 7 - played_num[1], 7 - played_num[2]]
        pov = self._pov
        left_num = [left_num[pov], left_num[(pov + 1) % 3], left_num[(pov + 2) % 3]]
        assert(self._mycards.sum() == left_num[0])
        self.lower_cards_num = left_num[1]
        self.upper_cards_num = left_num[2]

    def copy(self):
        other = GameStatePov(self._pov)
        other.last_player = self.last_player
        other.last_move = self.last_move.copy()
        other.history = self.history.copy()
        other._mycards = self._mycards.copy()
        other._other_cards = self._other_cards.copy()
        other.current_player = self.current_player

        other.lower_cards_num = self.lower_cards_num
        other.upper_cards_num = self.upper_cards_num

        other._lower = self._lower
        other._upper = self._upper
        return other
