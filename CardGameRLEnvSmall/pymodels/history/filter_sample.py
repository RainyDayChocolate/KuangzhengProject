"""This script describes bayesian-like sample method with prior recognization about
current plays
"""
import numpy as np

from pymodels.history.bayes_sample import Sampler as BayesianSampler
from pymodels.history.bayes_sample import get_state_str


class Sampler(BayesianSampler):

    def __init__(self, sample_filters, **kwargs):
        super().__init__(**kwargs)
        self._sample_filters = sample_filters

    def get_samples(self, state, num=20, pool_length=3000):
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
        if not state.get_history():
            return [state.sample_upper_lower()
                    for _ in range(num)]

        current_player = state.get_current_player()
        to_predict, counter = [], {}
        for _ in range(pool_length):
            sample = state.sample_upper_lower()
            sample_str = get_state_str(sample)
            if sample_str in counter:
                counter[sample_str] += 1
                continue
            counter[sample_str] = 1
            to_predict.append(sample)

        _sample_filter = self._sample_filters[current_player]
        samples = _sample_filter.filter_out(to_predict, num)
        sample_probas = self.get_samples_conditional_probas(samples)

        post_probas = []
        for sample, proba in sample_probas:
            post_probas.append(proba * counter[get_state_str(sample)])
        post_probas = np.array(post_probas) / sum(post_probas)
        return [np.random.choice(samples, p=post_probas)
                for _ in range(num)]