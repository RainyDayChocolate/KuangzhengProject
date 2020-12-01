

import random

from pyminiddz.miniddz import legal_moves, PASS


class RandomPlayer:

    def __init__(self, name):
        self.PLAYERS = {'C': 0, 'D': 1,'E': 2}
        self._name = name
        self.player_idx = self.PLAYERS[name]

    def get_move(self, state):
        """state if of type GameState or GameStatePov
        """
        assert state.get_current_player() == state.get_pov()
        moves = state.get_legal_moves()
        move = random.choice(moves)
        return move