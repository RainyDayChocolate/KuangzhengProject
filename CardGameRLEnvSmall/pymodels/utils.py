import random

from pyminiddz.miniddz import GameStatePov

def rollout(state):
    if isinstance(state, GameStatePov):
        state = state.sample_upper_lower()
    player = state.get_current_player()
    state_cp = state.copy()
    double = 2 ** (state_cp.get_bomb_num())
    while not state_cp.is_end_of_game():
        move = random.choice(state_cp.get_legal_moves())
        state_cp.do_move(move)

    score = state_cp.get_score(player)

    return  score / double


def dummy_value(state):
    if not isinstance(state, list):
        return rollout(state)
    return [dummy_value(s) for s in state]


def dummy_policy(state):
    if not isinstance(state, list):
        moves = state.get_legal_moves()
        _policy = {move: 1.0 / len(moves) for move in moves}
        return _policy
    return [dummy_policy(s) for s in state]


dummy_policy_value = lambda state: (dummy_policy(state), dummy_value(state))
