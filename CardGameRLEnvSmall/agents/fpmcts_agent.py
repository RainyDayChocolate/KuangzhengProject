"""This script provides a MCTS player
"""

import random

from pymodels.search.fpmcts import MCTS
from pymodels.utils import dummy_policy, dummy_policy_value, dummy_value


class FPMCTSPlayer:

    def __init__(self, player_pos,
                 policy_value=dummy_policy_value,
                 lower_policy=dummy_policy,
                 upper_policy=dummy_policy,
                 **kwargs):
        """
        Params:
        ------
        player_pos(Int):
            player_position_idx
        """
        self._player = player_pos

        self._mcts = MCTS(player_pos,
                          policy_value,
                          lower_policy,
                          upper_policy,
                          **kwargs)

    def get_move(self, state):
        """Get move by implementing the MCTS
        """

        self._mcts.set_root()
        assert(self._player == state.get_current_player())
        move = self._mcts.get_move(state)

        return move

    def get_mcts(self):
        """Operation for the players mcts
        """

        return self._mcts
