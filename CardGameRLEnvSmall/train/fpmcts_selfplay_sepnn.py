"""Seperately policy and value nn training
"""
import sys

from pymodels.policy.sep_policy_value import SepPolicyValue
from train.fpmcts_selfplay_training import Trainer as BaseTrainer
from train.fpmcts_selfplay_training import get_flags

class Trainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_networks(self):
        if self.model_id:
            main_model_path = """./{0}/models/{1}/{2}"""
            main_weights_path = """./{0}/models/{1}/{2}"""
            self.networks = [SepPolicyValue(
                             model_path=main_model_path.format(self.selfplay_file,
                                                               self.model_id,
                                                               player))
                             for player in 'CDE']
        else:
            self.networks = [SepPolicyValue() for _ in range(3)]

def train(argv):
    """Train
    """
    FLAGS = get_flags()
    FLAGS(argv)
    trainer = Trainer(selfplay_file=FLAGS.selfplay_fold,
                      model_id=FLAGS.model_id,
                      c_puct=FLAGS.c_puct,
                      playout_num=FLAGS.playout_num,
                      samples_num=FLAGS.samples_num,
                      history_temperature=FLAGS.history_temperature,
                      game_num=FLAGS.game_num,
                      iter_num=FLAGS.iter_num,
                      epochs=FLAGS.epochs,
                      learning_rate=FLAGS.learning_rate,
                      batch_size=FLAGS.batch_size,
                      continue_play=FLAGS.continue_play)
    trainer.learning(FLAGS.append_game_data)

if __name__ == '__main__':
    train(sys.argv)

