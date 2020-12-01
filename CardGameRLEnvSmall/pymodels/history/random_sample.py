"""This script provide random sample for miniddz
"""

class Sampler:

    @staticmethod
    def get_samples(state, num=20):

        samples = [state.sample_upper_lower()
                   for _ in range(num)]

        return samples