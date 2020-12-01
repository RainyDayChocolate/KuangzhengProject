"""This script shows the PIMC selfplay training
Most components are exactly same as the fpmcts selfplay's
"""
import sys
from itertools import cycle

from agents.pimc_agent import PIMCPlayer
from pyminiddz.miniddz import GameState, GameStatePov
from train.fpmcts_selfplay_training import Trainer, get_flags


class PIMCTrainer(Trainer):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def set_dummy_players(self):

        set_player_helper = (lambda pos:
                             PIMCPlayer(
                                 c_puct=self.c_puct,
                                 playout_num=self.playout_num,
                                 samples_num=self.samples_num))

        self.players = [set_player_helper(idx) for idx in range(3)]

    def set_players(self):
        set_player_helper = (lambda pos:
                             PIMCPlayer(
                                 policy = [network.get_policy
                                           for network in self.networks],
                                 c_puct=self.c_puct,
                                 playout_num=self.playout_num,
                                 samples_num=self.samples_num,
                                 sampler=self.sampler))

        self.players = [set_player_helper(idx) for idx in range(3)]

    def run_a_game(self):

        game_data = {"data" : []}
        game = GameState()

        state_povs = [GameStatePov(0), GameStatePov(1), GameStatePov(2)]
        player_seq = cycle(range(3))
        for player_idx in player_seq:
            current_player = self.players[player_idx]
            state_pov = state_povs[player_idx]
            state_pov.from_pure_state(game)
            _pimc = current_player.get_pimc()
            temperature = self.game_state_temperature(game)
            _pimc.set_temperature(temperature)
            move = current_player.get_move(state_pov)
            pimc_policy = _pimc.get_pimc_policy()
            _policy = {move.get_str(): prob for
                       move, prob in pimc_policy.items()}
            scenario = {"state": state_pov.to_dict(),
                        "move": move.to_dict(),
                        "player": player_idx,
                        "policy": _policy}
            game_data['data'].append(scenario)
            game.do_move(move)
            if game.is_end_of_game():
                break

        game_data['winner'] = int(game.get_winner())
        game_data['abs_score'] = float(abs(game.get_score(0)))

        return game_data


def train(argv):
    """Train
    """
    FLAGS = get_flags()
    FLAGS(argv)
    trainer = PIMCTrainer(selfplay_file=FLAGS.selfplay_fold,
                         model_id=FLAGS.model_id,
                         c_puct=FLAGS.c_puct,
                         playout_num=FLAGS.playout_num,
                         samples_num=FLAGS.samples_num,
                         history_temperature=FLAGS.history_temperature,
                         game_num=FLAGS.game_num,
                         iter_num=FLAGS.iter_num,
                         epochs=FLAGS.epochs,
                         learning_rate=FLAGS.learning_rate,
                         batch_size=FLAGS.batch_size)
    trainer.learning(FLAGS.append_game_data)

train(sys.argv)