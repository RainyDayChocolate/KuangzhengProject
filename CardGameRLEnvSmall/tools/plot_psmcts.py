"""This script gives the method of drawing an mcts tree

Calling Method:
root = mcts_player.mcts._root
plot(root)
"""
import time
from collections import defaultdict

from graphviz import Digraph

from pyminiddz.utils import card_trans_np_reverse
from pymodels.search.psmcts import TreeNode


def plot(node):
    """Plot MCTS tree
    Params:
    ------
    node: an instance of TreeNode which defined in mcts.py
    Return:
    ------
    MCTS_GRAPHA
    """
    MCTS_GRAPHA = Digraph(comment='mcts_graph')
    counter = defaultdict(int)

    root = str(None)
    counter[root] += 1
    MCTS_GRAPHA = Digraph(str(root) + str(counter[root]), 'ROOT')

    def plot_mcts(node, root=None):
        """Plot mcts
        """
        if not root:
            root = str(None)

        if not node.is_leaf():
            children = node.get_children()
            max_child = max(children, key=lambda child: children[child].get_visits())
            for child, child_node in children.items():
                counter[child] += 1
                child_index = str(child) + '_' + str(counter[child])

                if child_node.get_visits() < 10:
                    continue
                label = ('act: {}'.format(str(child)) + '\nvisit: {}'.format(str(child_node.get_visits())) +
                            '\nvalue:{:.2f}'.format(child_node._Q))
                if node._parent:
                    label += '\n {}'.format(node._parent._player)
                color = ''
                if child_node.is_leaf():
                    color = 'red'
                # 修改label，打印需要显示的结果
                if isinstance(child_node, TreeNode):
                    shape='box'
                    if not color:
                        MCTS_GRAPHA.node(child_index, label, shape=shape)
                    else:
                        MCTS_GRAPHA.node(child_index, label, shape=shape, color=color)
                else:
                    MCTS_GRAPHA.node(child_index, label)

                if child == max_child and (len(children) > 1):
                    MCTS_GRAPHA.edge(root, child_index, color='blue')
                else:
                    MCTS_GRAPHA.edge(root, child_index)
                plot_mcts(child_node, child_index)

    plot_mcts(node)
    current_time = time.clock()
    MCTS_GRAPHA.render('./graphs/mcts_graph', view=True)
    # whether pop out a picture
    return MCTS_GRAPHA


if __name__ == '__main__':
    from pyminiddz.miniddz import GameState
    from pymodels.search.psmcts import MCTS
    game_state = GameState()
    mcts = MCTS(player_pos=0, playout_num=1000)
    print(mcts.get_move(game_state))
    plot(mcts._root)

    import pdb; pdb.set_trace()