
"""Monte Carlo Tree Search, as described in the Deltadou paper.

ddz implementation

The struture is like
                            root                                         TreeNode
                            /   \
                           a_1  a_2                                      ChanceNode
                          /       \
                       (b_1, c_1) (b_2, c_2)  sample generate            TreeNode
"""

import numpy as np

from pyminiddz.miniddz import GameStatePov
from pymodels.history.random_sample import Sampler


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


class TreeNode:
    """The current player node in the MCTS tree.
    Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, c_puct, states_sampled=None, actions=None):
        """Params:
        ------
        parent(ChanceNode):
            Where the treenode was selected
        c_puct(float):
            Determine the balance between exploration and exploitation in PUCT
        state_sampled(List):
            A list contains several states(GameState) sampled for further search
        actions(Tuple):
            the (lower action, upper action) sampled from their parent(Chance Node). If is None
            means the Node is the Root Node
        """

        self._parent = parent
        if not actions:
            actions = (None, None)
        self._parent_actions = actions

        self._children = {}
        # a map from action to ChanceNode children
        self._visits = 0
        # This value for u will be overwritten in the first call to update(), but is useful for
        # choosing the first action from this node.
        self._Q = 0
        # The total visit number from a ChanceNode
        self._chance_count = 0

        self._c_puct = c_puct
        if not states_sampled:
            self._states_sampled = []
        else:
            self._states_sampled = [s.copy() for s in states_sampled]

    def expand(self, action_priors, state, lower_policy, upper_policy):
        """Expand tree by creating new children.

        Params:
        ------
        action_priors(Dict):
            output from policy function, a dict of actions and their prior
            probability according to the policy function.
            The node expands based on this distribution
        state(GameState or GameStatePov):
            The real state in the Search Tree.
        lower_policy(func):
            A function provides the policy of the lower player
        upper_policy(func):
            A function provides the policy of the upper player

        Returns:
        ------
        None
        """

        current_player = state.get_current_player()
        # to make sure this node got no children
        if state.is_end_of_game():
            self._Q = state.get_score(current_player)
            self._Q /= abs(self._Q)
            return

        # same guess states over all actions
        if not self._states_sampled:
            # no state sampled hit this situ
            self._Q = 0
            return

        for action, prior_prob in action_priors.items():
            guess_states = [state.copy() for state in self._states_sampled]
            if action not in self._children:
                for guess_state in guess_states:
                    guess_state.do_move(action)

                self._children[action] = ChanceNode(self, prior_prob, self._c_puct)
                self._children[action].expand(guess_states,
                                              lower_policy,
                                              upper_policy,
                                              action)

    def select(self, is_selfplay=False):
        """PUCT selection same to that of David.Silver described
        During self-play, the root node contains Dirichlet Noise
        Reference:
        https://stats.stackexchange.com/questions/322831/purpose-of-dirichlet-noise-in-the-alphazero-paper

        Params:
        ------
        is_selfplay(Bool):
            Whether the MCTS is used in selfplay

        Returns:
        ------
        action(Move):
            Selected action by PUCT
        chance_node(ChanceNode):
            The node below the current tree node
        """

        if is_selfplay and self.is_root():
            noise = np.random.dirichlet([0.3] * len(self._children))
        else:
            noise = np.zeros(len(self._children))

        children_with_noise_value = [((action, chance_node), chance_node.get_value(noi))
                                     for (action, chance_node), noi
                                     in zip(self._children.items(), noise)]
        action, chance_node = max(children_with_noise_value, key=lambda x: x[1])[0]

        return action, chance_node

    def update(self, leaf_value):
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
        # Update average _Q.
        self._Q += (leaf_value - self._Q)  / self._visits

    def update_recursive(self, leaf_value):
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
        if self._parent_actions[0] is not None and self._parent_actions[0].is_bomb():
            beishu *= 2
        if self._parent_actions[1] is not None and self._parent_actions[1].is_bomb():
            beishu *= 2
        # Final Score contains double effect from current state
        leaf_value *= beishu
        if self._parent:
            # there is a chance node inside, do not need to update
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_chance_count(self):
        """Get the chance(randomly chosen in chance nodes) count
        of Treenodes
        """

        return self._chance_count

    def add_chance_count(self, value=1):
        """Set chance count of Treenodes
        """

        self._chance_count += value

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

    def add_state(self, state):
        """Append new state to states
        """

        self._states_sampled.append(state)

    def set_parent(self, parent):
        """Set parent
        """

        self._parent = parent

    def get_states_sampled(self):
        """Return states sampled
        """

        return self._states_sampled


class ChanceNode:
    """The other players node in MCTS which could be treated as Environment"""

    def __init__(self, parent, prior_prob, c_puct):
        """Params:
        ------
        Parent(TreeNode):
            Where the treenode was selected
        c_puct(Float):
            Determine the balance between exploration and exploitation in PUCT
        prior_prob(Float):
            Prior possibilty given by policy
        """

        self._parent = parent
        self._parent_action = None

        self._children = {}

        self._Q = 0

        self._P = prior_prob
        self._visits = 0
        self._actions_probs = None

        self._samples_num = 0
        self._c_puct = c_puct
        #For lazy expansion
        self._is_expanded = False
        self._expand_params = None

    def expand(self, states, lower_policy, upper_policy, action):
        """Lazy expansion"""

        self._expand_params = (states, lower_policy,
                               upper_policy, action)

    def do_expand(self):
        """real expansion
        """

        self._expand(*self._expand_params)
        self._is_expanded = True

    def _expand(self, guess_states, lower_policy, upper_policy, action):
        """Expand a chance node, only expand some samples
        """

        self._parent_action = action
        to_predict_lower = list(filter(lambda x: not x.is_end_of_game(),
                                guess_states))
        lower_predict_results = lower_policy(to_predict_lower)

        to_predict_upper = []
        lower_moves = []
        for guess_state in guess_states:
            if guess_state.is_end_of_game():
                lower_moves.append(None)
            else:
                move_lower = choose_from_random_move(lower_predict_results.pop(0))
                lower_moves.append(move_lower)
                guess_state.do_move(move_lower)

            if guess_state.is_end_of_game():
                continue
            else:
                to_predict_upper.append(guess_state)

        upper_predict_results = upper_policy(to_predict_upper)
        for guess_state in guess_states:
            if guess_state.is_end_of_game():
                move_upper = None
            else:
                move_upper = choose_from_random_move(upper_predict_results.pop(0))
                guess_state.do_move(move_upper)

            move_key = (lower_moves.pop(0), move_upper)

            if move_key not in self._children:
                if move_key[0] is None:
                    self._children[move_key] = TreeNode(self, self._c_puct)
                else:
                    self._children[move_key] = TreeNode(self, self._c_puct, actions=move_key)

            self._children[move_key].add_chance_count()
            self._children[move_key].add_state(guess_state)

            self._samples_num += 1
        #Count number
        self._actions_probs = np.array([tree_node.get_chance_count() / self._samples_num
                                        for tree_node in list(self._children.values())])
        self._actions_probs /= self._actions_probs.sum()

    def select(self):
        """select a tree node, return a (lower_action, upper_action), upper)

        Returns:
        actions(Tuple):
            is the format of (lower_action, upper_action)
        tree_node(TreeNode):
            selected tree node
        """

        if not self._is_expanded:
            self.do_expand()

        actions_children = list(self._children.items())
        children_num = len(actions_children)

        if children_num == 1:
            return actions_children[0]

        ind = np.random.choice(np.arange(children_num), p=self._actions_probs)
        actions, player_node = actions_children[ind]
        return actions, player_node

    def _compute_u(self, noise, noise_weight=0.25):
        """Compute exploitation part in the PUCT"""

        if not noise:
            noise_weight = 0
        _p = (1 - noise_weight) * self._P + noise * noise_weight
        return self._c_puct * _p * np.sqrt(self._parent.get_visits()) / (1 + self._visits)

    def get_value(self, noise):
        """Get PUCT value with noise
        """

        # this valu10e is the Q value
        return self._Q + self._compute_u(noise)

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.

        Note: it is important that this happens from the root downward so that 'parent' visit
        counts are correct.
        """

        # If it is not root, this node's parent should be updated first.
        beishu = 2 if self._parent_action.is_bomb() else 1
        leaf_value *= beishu
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def update(self, leaf_value):
        """Update chance node
        """

        self._visits += 1
        self._Q += (leaf_value - self._Q)  / self._visits

    def info(self):
        """Several information
        """

        return (self._visits, self._Q, self._compute_u(0), self._P)

    def get_visits(self):
        """Get the visit number of this node
        """

        return self._visits

    def get_children(self):
        """Get the children of the node
        """

        return self._children

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """

        return self._children == {}


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

    def __init__(self, player_pos, policy_value, lower_policy,
                 upper_policy, c_puct=2.8, playout_num=200,
                 temperature=0, samples_num=20, sampler=Sampler()):
        """
        Params:
        ------
        player_pos(Int):
            The making decision choice
        policy_value:
            Representing the policy-value approximate function of current player
        lower_policy:
            Representing the policy approximate function of the lower player
        upper_policy:
            Representing the policy approximate function of the upper player
        c_puct(float):
            A coefficient controls the exploitation
        """

        self._root = TreeNode(None, c_puct)
        self._policy_value = policy_value
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

        if isinstance(state, GameStatePov):
            assert self._main_player == state.get_pov()

        tree_node = self._root

        while not tree_node.is_leaf():
            my_action, chance_node = tree_node.select(self._is_selfplay)
            (lower_action, upper_action), tree_node = chance_node.select()
            actions = [my_action, lower_action, upper_action]
            for action in actions:
                if action is None:
                    break
                state.do_move(action)
        #Operations on tree frontiers
        if state.is_end_of_game():
            leaf_value = (state.get_score(self._main_player))
            leaf_value /= abs(leaf_value)
        else:
            # Only expand node if it has not already been done. Existing nodes already know their
            # prior.
            action_probs, leaf_value = self._policy_value(state)
            #print(state, leaf_value)
            tree_node.expand(action_probs, state, self._lower_policy, self._upper_policy)

        tree_node.update_recursive(leaf_value)

    def get_move(self, state):
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

        legal_moves = state.get_legal_moves()
        if len(legal_moves) == 1:
            self._mcts_policy = {legal_moves[0]: 1.0}
            return legal_moves[0]

        #Sampling for root nodes
        left_states_num = len(self._root.get_states_sampled())
        if  left_states_num < self._samples_num:
            states_sampled = self._sampler.get_samples(state,
                                                       self._samples_num - left_states_num)
            for state_sampled in states_sampled:
                self._root.add_state(state_sampled)

        #Playout
        for _ in range(self._playout_num + 1):
            if self._root.get_visits() > self._playout_num:
                break
            state_copy = state.copy()
            self._playout(state_copy)

        #Select move
        _actions_visit = {action: chance_child.get_visits()
                          for action, chance_child
                          in self._root.get_children().items()}

        if self._is_selfplay:
            max_visit = max(_actions_visit.values())
            _mcts_policy = {action: (visit / max_visit) ** (1 / self._temperature)
                                 for action, visit in _actions_visit.items()}
            _helper = sum(_mcts_policy.values())
            self._mcts_policy = {action: round(poli / _helper, 3)
                                 for action, poli in _mcts_policy.items()}

            move = choose_from_random_move(self._mcts_policy)
        else:
            move = choose_best_move(_actions_visit)

        return move

    def update_with_move(self, moves):
        """The tree might be reused. We update it manually.
        """

        move, move_lower, move_upper = moves
        root_children = self._root.get_children()
        if move in root_children:
            chance_node = root_children[move]
            chance_node_children = chance_node.get_children()
            if (move_lower, move_upper) in chance_node_children:
                tree_node = chance_node_children[(move_lower, move_upper)]
                states_left_num = len(tree_node.get_states_sampled())
                if states_left_num == self._samples_num:
                    self._root = tree_node
                    self._root.set_parent(None)
                    return

        self._root = TreeNode(None, self._c_puct)

    def set_root(self, tree_node=None):
        """Set tree node
        """

        if not tree_node:
            self._root = TreeNode(None, self._c_puct)
        else:
            self._root = tree_node

    def set_temperature(self, temperature):
        """Change FPMCTS temperature in self-play
        """

        self._temperature = temperature
        if self._temperature:
            self._is_selfplay = True
        else:
            self._is_selfplay = False

    def calc_policy(self):
        """calc π(a|s0) for self-play, with self play

        Returns:
        ------
        action_visits(List):
            A list of  number of visit for each move
        """

        if not self._is_selfplay:
            raise ValueError('Why you call me ???????')
        return self._mcts_policy
