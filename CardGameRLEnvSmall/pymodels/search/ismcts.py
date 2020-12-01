"""An implementation of ISMCTS(Information set MCTS)
mentioned in the Daniel Whitehouse's Ph.D Thesis
    Monte Carlo Tree Search for games with Hidden
    Information and Uncertainty
"""

import random

import numpy as np

from pyminiddz.utils import card_trans_np_reverse
from pymodels.history.random_sample import Sampler
from pymodels.utils import dummy_policy, dummy_policy_value, dummy_value

get_state_str = lambda s: "|".join([card_trans_np_reverse(rec)
                                    for rec in s.player_cards])



def is_same_team(player_A, player_B):

    if player_A != 0:
        return player_B != 0
    return player_B == 0


class NodeBase:

    def get_value(self, action, noise):
        """Get Q(s, a) + c * P * sqrt(N(s)) / (N(s, a) + 1)

        Params:
        ------
        action(Move):
            The move to be queried
        noise(float):
            dir(noise)
        """
        action_P = self.actions_probas[action]
        if not self._visits:
            return action_P
        noise_weight = 0.35 if noise else 0
        noise_part = noise_weight * noise

        action_visit = self.actions_visits[action]
        visit_exploit = np.sqrt(self._visits) / (1 + action_visit)
        exploit = self._c_puct * action_P * visit_exploit
        action_part = (1 - noise_weight) * (self.actions_values[action] \
                                            + exploit)
        return noise_part + action_part

    def update(self, action, value):
        self._visits += 1
        self.actions_visits[action] += 1
        delta = (value - self.actions_values[action])
        delta /= self.actions_visits[action]
        self.actions_values[action] += delta

    def expand(self):
        for action in self.actions_probas:
            self.actions_values[action] = 0
            self.actions_visits[action] = 0

    def select(self, is_selfplay=False):
        if len(self.actions_probas) == 1:
            return list(self.actions_probas.keys())[0]
        if is_selfplay and self.is_root():
            noise = np.random.dirichlet([0.3] * len(self.actions_probas))
        else:
            noise = np.zeros(len(self.actions_probas))

        action_with_noise_value = [(action, self.get_value(action, noise))
                                   for action, noise in
                                   zip(self.actions_probas, noise)]
        action = max(action_with_noise_value, key=lambda x: x[1])[0]
        return action

    def get_pos(self):
        return self._pos

class SubTreeNode(NodeBase):

    def __init__(self, player_pos,
                 actions_probas, c_puct=2.8):
        """The current player's nodes

        Params:
        ------
        parent(TreeNode):
            The player's decision node which represents the information set
        player_pos(Int):
            The player's position
        actions_probas(Dict):
            Policy for the current node like {action: proba}
        c_puct(Float):
            Constant that control the
        """
        self._pos = player_pos
        self._c_puct = c_puct

        self.actions_probas = actions_probas
        self.actions_values, self.actions_visits = {}, {}
        self._visits = 0

        self.expand()


class SubTree:
    """Could be viewed the information of Other players' information set
    """
    def __init__(self, player_pos, c_puct,
                 state, lower_policy, upper_policy):

        self._pos = player_pos
        self._c_puct = c_puct
        self.structure = {}
        self.expand(state, lower_policy, upper_policy)

    def select(self):
        """Continuous two selections.
        """
        action_lower = self.root.select()
        next_node = self.structure[action_lower]
        action_upper = next_node.select() if next_node else None
        return (action_lower, self.root), (action_upper, next_node)

    def expand(self, state, lower_policy, upper_policy):
        """To expand a sub-structure contains other players' choices
        for the ISMCTS
        Params:
        ------
        state(GameState):
            An instance of determinzation in ISMCTS
        lower_policy(PolicyFunc):
            Policy for the lower player
        upper_policy(PolicyFunc):
            Policy for the upper player
        """
        if state.is_end_of_game():
            raise ValueError('State should not be finished')
        lower_actions_probas = lower_policy(state)
        self.root = SubTreeNode(player_pos=self._pos,
                                actions_probas=lower_actions_probas,
                                c_puct=self._c_puct)

        to_predict_upper = []
        nodes_recorder = []
        for action in lower_actions_probas:
            state_cp = state.copy()
            state_cp.do_move(action)
            nodes_recorder.append([action, state_cp])
            if not state_cp.is_end_of_game():
                to_predict_upper.append(state_cp)

        upper_predict_results = upper_policy(to_predict_upper)
        for action, next_state in nodes_recorder:
            if next_state.is_end_of_game():
                self.structure[action] = None
            else:
                upper_action_probas = upper_predict_results.pop(0)
                self.structure[action] = SubTreeNode(player_pos=(self._pos + 1) % 3,
                                                     actions_probas=upper_action_probas,
                                                     c_puct=self._c_puct)


class TreeNode(NodeBase):

    def __init__(self, player_pos, actions_probas, c_puct, parent=None):
        """The current player's nodes

        Params:
        ------
        parent(TreeNode):
            The player's decision node which represents the information set
        player_pos(Int):
            The player's position
        actions_probas(Dict):
            Policy for the current node like {action: proba}
        c_puct(Float):
            Constant that control the
        """
        self._pos = player_pos
        self._c_puct = c_puct

        self._visits = 0
        self._parent = parent
        self.actions_probas = actions_probas
        self.actions_values, self.actions_visits = {}, {}
        # children should also be TreeNode
        self._children = {}
        self.subtrees = {}

    def is_root(self):
        return self._parent is None

    def is_expanded(self):
        return self.actions_values != {}

    def is_leaf(self):
        return self._children == {}

    def get_subtree(self):
        return self.subtrees

    def to_next_node(self, preivous_actions, state, policy):
        if preivous_actions not in self._children:
            actions_probas = policy(state)
            self._children[preivous_actions] = TreeNode(self._pos, actions_probas,
                                                        self._c_puct, parent=self)
        return self._children[preivous_actions]

    def add_subtree(self, action, state_repr, subtree):
        self.subtrees.setdefault(action, {}).update({state_repr: subtree})

class ISMCTS:

    def __init__(self, player_pos, policy_value=dummy_policy_value,
                 lower_policy=dummy_policy,
                 upper_policy=dummy_policy,
                 c_puct=2.8, playout_num=200,
                 temperature=0, samples_num=20, sampler=Sampler()):

        self._policy_value = policy_value
        self._policy = lambda s: policy_value(s)[0]
        self._c_puct = c_puct
        self._playout_num = playout_num
        self._main_player = player_pos
        self._lower_policy = lower_policy
        self._upper_policy = upper_policy
        # used in expand not in rollout
        self._temperature = temperature
        self._is_selfplay = (temperature != 0)
        self._sampler = sampler
        self._samples_num = samples_num
        self._mcts_policy = {}

    def playout(self, sample):

        tree_node = self._root
        sample_str = get_state_str(sample)
        playout_path = []
        while tree_node.is_expanded():
            action = tree_node.select()
            previous_key = [action]
            sample.do_move(action)
            playout_path.append([tree_node, action])
            if sample.is_end_of_game():
                break
            if action in tree_node.subtrees:
                subtree_helper = tree_node.subtrees[action]
                subtree = subtree_helper.setdefault(
                        sample_str,
                        SubTree(player_pos=(self._main_player + 1) % 3,
                                c_puct=self._c_puct, state=sample,
                                lower_policy=self._lower_policy,
                                upper_policy=self._upper_policy))
            else:
                subtree = SubTree(player_pos=(self._main_player + 1) % 3,
                                  c_puct=self._c_puct, state=sample,
                                  lower_policy=self._lower_policy,
                                  upper_policy=self._upper_policy)
                tree_node.add_subtree(action, sample_str, subtree)

            for action, node in subtree.select():
                previous_key.append(action)
                if action is None:
                    break
                playout_path.append([node, action])
                sample.do_move(action)

            previous_actions = tuple(previous_key)
            tree_node = tree_node.to_next_node(previous_actions,
                                               sample,
                                               self._policy)

        #import pdb; pdb.set_trace()
        if sample.is_end_of_game():
            leaf_value = sample.get_score(self._main_player)
            leaf_value /= abs(leaf_value)
        else:
            _, leaf_value = self._policy_value(sample)
            tree_node.expand()
        double = 1
        for node, action in reversed(playout_path):
            double *= (2 ** action.is_bomb())
            pos = node.get_pos()
            to_update = (leaf_value if is_same_team(self._main_player, pos)
                        else -leaf_value) * double
            node.update(action, to_update)

    def get_move(self, state):
        pos = state.get_current_player()
        root_probas, _ = self._policy_value(state)
        self._root = TreeNode(player_pos=pos, actions_probas=root_probas,
                              c_puct=self._c_puct)

        samples = self._sampler.get_samples(state)
        for _ in range(self._playout_num + 1):
            sample = random.choice(samples)
            self.playout(sample=sample.copy())
        print({k: round(v, 3) for k, v in self._root.actions_values.items()})
        print(self._root.actions_visits)
        return max(self._root.actions_visits, key=self._root.actions_visits.get)


from pyminiddz.miniddz import GameState
from pyminiddz.utils import card_trans_np
cards = [card_trans_np('9JJ'), card_trans_np('TTQQAA'), card_trans_np('TKKA2')]
game = GameState(player_cards=cards)
mcts = ISMCTS(0)
mcts.get_move(game)