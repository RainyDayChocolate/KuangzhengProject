from pyminiddz.miniddz import Move, GameState
from pyminiddz.utils import card_trans_np_reverse

class HumanPlayer:

    def __init__(self, pos):
        self._pos = pos

    def get_move(self, state):

        cards = card_trans_np_reverse(state.get_mycards())
        print('My_cards: ', cards)
        legal_moves = state.get_legal_moves()

        while 1:
            try:
                move = input('Your choice: ')
                move = move.upper()
                move = Move(vec=move)
                if move not in legal_moves:
                    continue
                else:
                    break
            except KeyboardInterrupt:
                break
            except Exception as msg:
                print(msg)
                continue
        return move
