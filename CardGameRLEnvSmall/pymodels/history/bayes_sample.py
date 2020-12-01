"""This script provides a Bayesian Sampling Method for the Mini-game

The key formula for this is
if H = [h1, h2, h3, ..., hn],
for each State sj whose prior probability is P(sj) belong to S

The posterior probability is
                    P(sj|H) = P(sj) * P(H|sj) / P(H)
in which P(H) = sum {P(H|s) * P(S)} for all s belong to S
"""
#!/usr/bin/env python

import numpy as np

from pyminiddz.utils import card_trans_np_reverse

get_state_str = lambda s: "|".join([card_trans_np_reverse(rec)
                                    for rec in s.player_cards])

class Sampler:
    """Bayesian Sampler method"""

    def __init__(self, policies, history_temperature=2):
        self.policies = policies
        self.history_temperature = history_temperature
        self._state_cache = {}

    @staticmethod
    def softmax_normalized_prob(move_probs, temperature):
        """Softmax the policy with temperature,
        and then normalized by the maximum. i,e
        [p1, p2,..., pn] -> [p1/max, p2/max,..., pn/max]

        Params:
        ------
        move_probs(Dict):
            Representing the prior possibilty for each move
        temperature(Float):
            if temperature is 0, return the greedy policy
            if temperature is infinite, return the totally average policy

        Returns:
        ------
        softmax_move_probs(Dict):
            The move_prob after normalized
        """

        moves, probs = zip(*list(move_probs.items()))
        probs = np.array(probs)
        probs = np.clip(probs, 0.0001, 0.9999)
        probs = probs ** (1.0 / temperature)
        probs /= probs.sum()
        probs /= probs.max()
        return dict(zip(moves, probs))

    def get_sample_conditioal_prob(self, sample_state):
        """Compute p(H|s) for s, conditional prob
        H = [h1, h2, ..., hn], and then S could be recalled to
        S = [s1, s2, ..., sn] where hi in A(si)
        The conditional probability of this sample is
        prob = Multiply(P(ai|si)) where current player of si not equal
        to that of sample state

        Params:
        ------
        sample_state(GameState):
            A Gamestate sampled from a hidden state

        Returns:
        ------
        condition_prob(Float):
            Condition probability under the current history
        """

        state_str = get_state_str(sample_state)
        if state_str in self._state_cache:
            return self._state_cache[state_str]

        player_position = sample_state.get_current_player()
        condition_prob = 1.0
        for recall_state, recall_move in sample_state.recall():
            moves_length = recall_state.get_legal_moves()
            recall_player = recall_state.get_current_player()
            is_me = recall_player == player_position

            if moves_length == 1 or is_me:
                continue

            player_poli = self.policies[recall_player]
            move_probs = player_poli(recall_state)
            move_probs = self.softmax_normalized_prob(move_probs,
                                                      self.history_temperature)
            prob = move_probs.get(recall_move, 0.0001)
            condition_prob *= prob
        self._state_cache[state_str] = condition_prob

        return condition_prob

    def get_samples_conditional_probas(self, samples):
        """Bacth computation for samples postories probabilities
        The formula is same to that of get_sample_conditioal_prob

        Params:
        ------
        samples(List):
            A list of samples(GameState)

        Returns:
        ------
        samples_probas(List)
        """
        me = samples[0].get_current_player()
        to_predict = {}
        samples_frequency, samples_hashs = {}, {}
        for sample in samples:
            sample_hash = get_state_str(sample)
            samples_hashs[sample_hash] = sample
            freq = samples_frequency.get(sample_hash, 0)
            freq += 1
            samples_frequency[sample_hash] = freq
        for sample_hash, sample in samples_hashs.items():
            for recall_state, recall_move in sample.recall():
                current_player = recall_state.get_current_player()
                if current_player == me:
                    continue
                to_predict.setdefault(current_player, []).append({'origin': sample_hash,
                                                                  'state': recall_state,
                                                                  'query_move': recall_move})
        states_probas = {}
        for player, predict_nodes in to_predict.items():
            policy = self.policies[player]
            states = [node['state'] for node in predict_nodes]
            state_policies = policy(states)
            for state_obj, _policy in zip(predict_nodes, state_policies):
                softmax_probas = self.softmax_normalized_prob(_policy,
                                                              self.history_temperature)
                move_proba = softmax_probas.get(state_obj['query_move'], 0.00001)
                state_proba = states_probas.get(state_obj['origin'], 1)
                state_proba *= move_proba
                states_probas[state_obj['origin']] = state_proba
        states_probas = [(samples_hashs[_hash_key], proba * samples_frequency[_hash_key])
                         for _hash_key, proba in states_probas.items()]
        return states_probas

    def get_samples(self, state, num=20):
        """Get several samples for the current state

        Params:
        ------
        state(GameState, GameStatePov):
            State that will analysis its samples
        num(Int):
            Get such amount of samples.

        Returns:
        ------
        samples(List):
            A list contains several samples(GameState)
            consisit with the input state
        """
        pool_length = num * 10
        self._state_cache.clear()
        samples = [state.sample_upper_lower()
                   for _ in range(pool_length)]

        sample_probas = self.get_samples_conditional_probas(samples)
        if not sample_probas:
            return [np.random.choice(samples) for _ in range(num)]
        else:
            samples, probas = zip(*sample_probas)
            probas = np.array(probas) / sum(probas)
            samples = [np.random.choice(samples, p=probas) for _ in range(num)]
            return samples
