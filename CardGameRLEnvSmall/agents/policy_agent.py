"""This script gives an policy agent
"""
import random

from pymodels.policy.residual_policy import ResidualPolicyValue
from pymodels.policy.sep_policy_value import PolicyNetwork
from pymodels.nn_utils import NeuralNetBase
from pymodels.utils import dummy_policy


class PolicyPlayer:

    def __init__(self, policy=dummy_policy,
                 policy_path='',
                 policy_name='Default'):

        self._neural_base = NeuralNetBase()
        if not policy_path:
            self._policy = policy
        else:
            self._policy = self.load_policy(policy_path)
        self._name = policy_name

    def load_policy(self, policy_path):
        """Load policy from path

        Params:
        ------
        policy_path(String):
            The model path of the Residual Policy

        Returns:
        ------
        policy(Func):
            A loaded policy
        """

        model_path = policy_path + '.model'
        weights = policy_path + '.weights'
        nn_policy = self._neural_base.load_model(model_path, weights)
        policy = nn_policy.get_policy
        return policy

    def get_name(self):
        """get the policy name(sometimes equal to the ID)"""

        return self._name

    def get_move(self, state):
        """Get the best move at the current state based on the policy

        Params:
        ------
        state(GameState, GameStatePov):
            The current state for the player
        Returns:
        -----
        best_move(Move):
            The best move determined the the policy function
        """

        move_probs = self._policy(state)
        best_move = max(move_probs, key=move_probs.get)
        return best_move
