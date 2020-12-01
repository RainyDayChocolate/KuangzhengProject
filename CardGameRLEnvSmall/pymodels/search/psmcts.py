"""This script a monte carlo tree search
for three player in perfect miniddz game

This tree search also have Four phases

    * Selection
    * Simulation
    * Expansion
    * Backpropgation

Selection:
    The method in selection could be PUCT or UCT
    For PUCT, there should exist a Policy network.
    For UCT, nothing needs here

Simulation:
    Simulation, rollout play, a fast-play used in
    Similar to that of AlphaGo

Expansion:
    From the player's view to the next player's view.

Backpropagation:
    Trivial, value removed previous double effect.
"""


import random
from collections import defaultdict

import numpy as np

from pyminiddz.miniddz import GameState
from pymodels.history.random_sample import Sampler as Random_Sampler


def dummy_policy(state):

    actions = state.get_legal_moves()
    prob = np.ones(len(actions), float) / len(actions)
    return dict(zip(actions, prob))


def rollout(state):

    state_cp = state.copy()
    double = 2 ** (state_cp.get_bomb_num())
    while not state_cp.is_end_of_game():
        move = random.choice(state_cp.get_legal_moves())
        state_cp.do_move(move)

    winner = state_cp.get_winner()
    score = state_cp.get_score(winner)

    return winner, abs(score) / double

dummy_value = lambda state: 0


def choose_from_random_move(move_probs):
    """Move_probs is a dict = {move: prob}
    """

    moves, probs = zip(*[(move, prob)
                         for move, prob in move_probs.items()])
    # probs need to be normalized
    probs = np.array(probs)
    probs /= probs.sum()

    return np.random.choice(moves, p=probs)


def choose_best_move(move_probs):
    """Move_probs is a dict = {move: prob}
    """

    return max(move_probs, key=move_probs.get)


def is_same_team(player_A, player_B):

    if player_A != 0:
        return player_B != 0
    return player_B == 0

class TreeNode:
    """The current player node in the MCTS tree.
    Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent=None,
                 action=None, c_puct=2.8, prior_prob=1,
                 player=0, policy=None, is_selfplay=False):
        """Params:
        ------
        parent(ChanceNode):
            Where the treenode was selected
        c_uct(float):
            Determine the balance between exploration and exploitation in UCT
        """
        self._P = prior_prob
        if not policy:
            self._policy = (dummy_policy,) * 3
        else:
            self._policy = policy
        self._player = player
        self._parent = parent
        self._parent_action = action
        self._children = {}
        # a map from action to ChanceNode children
        self._visits = 0
        # This value for u will be overwritten in the first call to update(), but is useful for
        self._Q = 0
        # The total visit number from a ChanceNode
        self._c_puct = c_puct
        self._is_selfplay = is_selfplay

    def expand(self, state):
        """Expand tree by creating new children.

        Params:
        ------
        state(GameState):
            The real state in the Search Tree.

        Returns:
        ------
        None
        """

        # to make sure this node got no children
        if state.is_end_of_game():
            self._Q = state.get_score(state.get_pov())
            self._Q /= abs(self._Q)
            return

        my_policy = self._policy[self._player]
        actions_prob = my_policy(state)
        for action, prior_prob in actions_prob.items():
            next_player = (self._player + 1) % 3
            self._children[action] = TreeNode(self, action, self._c_puct, prior_prob,
                                              next_player, self._policy)

    def select(self, is_selfplay=False):
        """UCT selection same to that of David.Silver described
        During self-play, the root node contains Dirichlet Noise
        Reference:
        https://stats.stackexchange.com/questions/322831/purpose-of-dirichlet-noise-in-the-alphazero-paper

        Returns:
        ------
        action(Move):
            Selected action by UCT
        treenode(Treenode):
            The node below the current tree node
        """

        if is_selfplay and self.is_root():
            dir_noise = np.random.dirichlet([0.3] * len(self._children))
        else:
            dir_noise = np.zeros(len(self._children))

        child_value = [(action, child, child.get_value(noise=dir_noise[ind]))
                        for ind, (action, child) in enumerate(self._children.items())]
        action, tree_node, _ = max(child_value, key=lambda helper: helper[2])

        return action, tree_node

    def update(self, leaf_value, winner):
        """Update node values from leaf evaluation.

        Params:
        ------
        leaf_value(Float):
            the value of subtree evaluation from the current player's perspective.

        Returns:
        ------
        None
        """

        # Count visit.
        self._visits += 1
        if not self.is_root():
            if not is_same_team(winner, self._parent.get_player()):
                leaf_value = -leaf_value
        # Update average _Q.
        self._Q += (leaf_value - self._Q) / self._visits

    def update_recursive(self, leaf_value, winner):
        """Like a call to update(), but applied recursively for all ancestors.

        Params:
        ------
        leaf_value(Float):
            the value of subtree evaluation from the current player's perspective.

        Returns:
        ------
        None
        """

        # If it is not root, this node's parent should be updated first.
        beishu = 1
        if not self.is_root():
            if self._parent_action.is_bomb():
                beishu *= 2
        # Final Score contains double effect from current state
        leaf_value *= beishu
        if not self.is_root():
            # there is a chance node inside, do not need to update
            self._parent.update_recursive(leaf_value, winner)
        self.update(leaf_value, winner)

    def is_root(self):
        """Check if root node"""

        return self._parent is None

    def get_visits(self):
        """Get the visit number of this node
        """

        return self._visits

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """

        return self._children == {}

    def get_children(self):
        """Get the children of the node
        """

        return self._children

    def get_player(self):

        return self._player

    def set_parent(self, parent):
        """Set parent
        """

        self._parent = parent

    def get_value(self, noise, noise_weight=0.35):

        if self._is_selfplay and self.is_root():
            _P = noise_weight * noise + (1 - noise_weight) * self._P
        else:
            _P = self._P
        explore = _P * np.sqrt(self._parent.get_visits()) / (self._visits + 1)
        return self._Q + explore


class MCTS:
    """A simple (and slow) single-threaded implementation of the FPMCTS described in the
    DeltaDou paper.

    Search works by exploring moves randomly according to the given policy up to a certain
    depth, which is relatively small given the search space. "Leaves" at this depth are
    assigned a value by the policy-value function. The leaf value would be updated to the
    path from the root to the leaf.
    After each playout, the leaf node would be expanded one-step forward if possible

    The term "playout" refers to a simulation in the search tree.
    """

    def __init__(self, player_pos, c_puct=2.8, playout_num=100,
                 policy=None, is_selfplay=False):
        """
        Params:
        ------
        player_pos(Int):
            The making decision choice
        c_uct(Float):
            A coefficient controls the exploitation
        playout_num(Int):
            Simulation number in MCTS
        """
        if not policy:
            self._policy = [dummy_policy] * 3
        else:
            self._policy = policy
        self._root = TreeNode(None, c_puct=c_puct, player=player_pos,
                              policy=self._policy)
        self._c_puct = c_puct
        self._playout_num = playout_num
        self._main_player = player_pos
        self._is_selfplay = is_selfplay

    def reset_root(self):
        """reset the root of mcts
        """
        self._root = TreeNode(None, c_uct=self._c_puct,
                              player=self._main_player, policy=self._policy,
                              is_selfplay=self._is_selfplay)

    def get_root(self):
        """Return the player of
        """
        return self._root

    def _playout(self, state):
        """Run a single playout from the root to the given depth, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.

        Params:
        ------
        state(GameState or GameStatePov):
            A copy of the state.

        Returns:
        ------
        None
        """
        tree_node = self._root

        double_effect = 2 ** state.get_bomb_num()
        while not tree_node.is_leaf():
            action, tree_node = tree_node.select()
            state.do_move(action)

        #Operations on tree frontiers
        if state.is_end_of_game():
            winner = state.get_winner()
            score = state.get_score(winner) / double_effect
        else:
            winner, score = rollout(state)
            # rollout should be replaced by a perfect-info value function
            # Only expand node if it has not already been done. Existing nodes already know their
            #winner = rollout_state.get_winner()
            tree_node.expand(state)

        tree_node.update_recursive(abs(score), winner)

    def simulation(self, state):
        """Runs all playouts sequentially and returns the most visited action.

        Params:
        ------
        state(GameState or GameStatePov):
            The current state, including both game state and the current player.

        Returns:
        ------
        move(Move):
            the selected action
        """

        #Playout
        for _ in range(self._playout_num + 1):
            if self._root.get_visits() > self._playout_num:
                break
            state_copy = state.copy()
            self._playout(state_copy)

        #Select move
        actions_visit = {action: chance_child.get_visits()
                         for action, chance_child
                         in self._root.get_children().items()}

        return actions_visit


class PIMC:

    def __init__(self, sampler=Random_Sampler(),
                 policy=None, c_puct=2.8, playout_num=100,
                 player_pos=0, is_selfplay=False,
                 temperature=0, samples_num=10):

        self._sampler = sampler
        if not policy:
            self._policy = [dummy_policy] * 3
        else:
            self._policy = policy
        self._root = TreeNode(None, c_puct=c_puct, player=player_pos)
        self._c_puct = c_puct
        self._playout_num = playout_num
        self._is_selfplay = is_selfplay
        self.temperature = temperature
        self._samples_num = samples_num

    def set_temperature(self, temperature):
        """Change PIMC temperature in self-play
        """

        self._temperature = temperature
        if self._temperature:
            self._is_selfplay = True

    def get_pimc_policy(self):

        return self._pimc_policy

    def get_move(self, state, **kwargs):

        legal_moves = state.get_legal_moves()
        if len(legal_moves) == 1:
            return legal_moves[0]

        visit_count = defaultdict(int)
        current_player = state.get_current_player()
        samples = self._sampler.get_samples(state, num=self._samples_num)
        for sample in samples:
            mcts = MCTS(player_pos=current_player,
                        c_puct=self._c_puct,
                        playout_num=self._playout_num,
                        policy=self._policy,
                        is_selfplay=self._is_selfplay,
                        **kwargs)
            action_visit = mcts.simulation(sample)
            for action, visit in action_visit.items():
                visit_count[action] += visit

        if self._is_selfplay:
            max_visit = max(visit_count.values())
            _pimc_policy = {action: (visit / max_visit) ** (1 / self._temperature)
                                 for action, visit in visit_count.items()}
            _helper = sum(_pimc_policy.values())
            self._pimc_policy = {action: round(poli / _helper, 3)
                                 for action, poli in _pimc_policy.items()}

            move = choose_from_random_move(self._pimc_policy)

        else:
            move = max(visit_count, key=visit_count.get)

        return move