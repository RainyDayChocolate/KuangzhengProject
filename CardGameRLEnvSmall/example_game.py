import time
from itertools import combinations, cycle

import numpy as np

from agents.policy_agent import PolicyPlayer
from agents.random_agent import RandomPlayer
from agents.fpmcts_agent import FPMCTSPlayer
from agents.pimc_agent import PIMCPlayer

from pyminiddz.miniddz import PLAYERS, GameState, GameStatePov, Move
from pyminiddz.utils import card_trans_np, card_trans_np_reverse
from pymodels.policy.residual_policy import ResidualPolicyValue
from pymodels.search.fpmcts import MCTS

from tools.plot_psmcts import plot
# Can still import Other player from other files

def run_a_game(player_c, player_d, player_e, game=None):
    """Runs a game with the given players.
    """

    if not game:
        game = GameState()
    else:
        game = game.copy()
    players = [player_c, player_d, player_e]
    state_povs = [GameStatePov(0), GameStatePov(1), GameStatePov(2)]
    player_seq = cycle(range(3))
    print('======= Game Start =======')
    for player_idx in player_seq:
        current_player = players[player_idx]
        state_pov = state_povs[player_idx]
        state_pov.from_pure_state(game)
        move = current_player.get_move(state_pov)
        helper = (PLAYERS[player_idx],
                  card_trans_np_reverse(game.get_current_player_card()),
                  ' ' * player_idx * 4,
                  move)
        print("""[{0}] {1:>8} | {2}{3}""".format(*helper))
        game.do_move(move)
        if game.is_end_of_game():
            break

    winner = PLAYERS[game.get_winner()]
    score = {PLAYERS[idx]: game.get_score(idx) for idx in range(3)}
    print('-' * 25)
    print('Winner: {0}'.format(winner))
    print('Scores: ' + ', '.join([PLAYERS[idx] + ':' + str(game.get_score(idx)) for idx in range(3)]))
    print('======= Game End ========\n')
    return winner

if __name__ == '__main__':

    from collections import defaultdict
    res1, res2 = defaultdict(int), defaultdict(int)

    pimc_player = PIMCPlayer()
    mcts_c = FPMCTSPlayer(0)
    mcts_d = FPMCTSPlayer(1)
    mcts_e = FPMCTSPlayer(2)
    for _ in range(100):

        game = GameState()
        winner = run_a_game(mcts_c, pimc_player, pimc_player, game.copy())
        res1[winner] += 1 / 100
        winner = run_a_game(pimc_player, mcts_d, mcts_e, game.copy())
        res2[winner] += 1 / 100
        print(res1)
        print(res2)
