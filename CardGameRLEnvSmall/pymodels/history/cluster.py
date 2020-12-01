from pymodels.utils import rollout
import numpy as np

def sample_to_vector(sample, policy=None):

    legal_moves = sample.get_legal_moves()
    current_player = sample.get_current_player()
    future_states = []
    for move in legal_moves:
        sample_cp = sample.copy()
        sample_cp.do_move(move)
        transfer = {}
        transfer['action'] = move
        transfer['proba'] = 1
        transfer['states'] = sample_cp
        future_states.append(transfer)

    for transfer in future_states:
        1

