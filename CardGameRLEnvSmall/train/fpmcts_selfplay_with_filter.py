"""FPMCTS selfplay with sampler"""
import json
import os
import sys
from random import shuffle

from pyminiddz.miniddz import GameState, GameStatePov
from pymodels.history.filter_sample import Sampler as FilterSampler
from pymodels.history.filters.naive_filter import SampleFilter
from pymodels.nn_utils import NeuralNetBase
from train.fpmcts_selfplay_training import Trainer as BaseTrainer
from train.fpmcts_selfplay_training import get_flags
from train.utils import filesetting, get_current_time

neural_based = NeuralNetBase()

class Trainer(BaseTrainer):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def set_filter(self):
        if self.model_id:
            model_path = """./{0}/models/{1}/{2}_filter.model"""
            weights_path = """./{0}/models/{1}/{2}_filter.weights"""
            self.sample_filters = [self._neural_base.load_model(
                                        json_file=model_path.format(self.selfplay_file,
                                                                     self.model_id,
                                                                     player),
                                        weights_file=weights_path.format(self.selfplay_file,
                                                                         self.model_id,
                                                                         player))
                                   for player in 'CDE']
        else:
            self.sample_filters = [SampleFilter() for _ in range(3)]

    def set_sampler(self):
        #"""set sampler for Player"""
        self.set_filter()
        policies = [network.get_policy for network in self.networks]
        self.sampler = FilterSampler(self.sample_filters,
                                     policies=policies,
                                     history_temperature=self.history_temperature)

    @staticmethod
    def parse_filter_data(data):
        records = data['data']
        state_bomb_score = lambda state: 2 ** state.get_bomb_num()

        #for C, D, E
        empty_recorders = lambda: [[], [], []]
        states, rationalities = [empty_recorders()
                                 for _ in range(2)]

        game_state_helper = GameState()
        for rec in records:
            player_idx = rec['player']
            game_state_helper.from_dict(rec['real_state'])
            states[player_idx].append(game_state_helper.copy())
            rationalities[player_idx].append(1)
            states[player_idx].append(game_state_helper.sample_upper_lower())
            rationalities[player_idx].append(0)

        return states, rationalities

    def making_filter_training_data(self, game_datas):
        empty_recorders = lambda: [[], [], []]
        all_states, all_rationalities = [empty_recorders()
                                         for _ in range(2)]

        def _update(main_records, to_extend):
            for player, infos in enumerate(to_extend):
                main_records[player].extend(infos)

        for data in game_datas:
            states, rations = self.parse_filter_data(data)
            _update(all_states, states)
            _update(all_rationalities, rations)

        return all_states, all_rationalities

    def optimize_filters(self, all_states, all_rationalities):
        for player_idx in range(3):
            self.sample_filters[player_idx].fit(
                all_states[player_idx],
                all_rationalities[player_idx],
                epochs=self.epochs,
                batch_size=int(self.batch_size),
                shuffle=True,
                validation_split=0.1,
                verbose=2)

    def save_filters(self, model_time):
        """Save models in sucn paths
        ./fpmcts_selfplay/models/time/{C, D, E}_filter.{weights, models}'

        Params:
        ------
        model_time(time):
            Model time assigned
        """

        path = './{0}/models/{1}'.format(self.selfplay_file, model_time)
        filesetting(path)
        def save_model_helper(network, path, pos):
            model_path = os.path.join(path, pos)
            network.save_model(model_path + '_filter.model',
                               model_path + '_filter.weights')

        for player_idx, pos in enumerate('CDE'):
            save_model_helper(self.sample_filters[player_idx], path, pos)

    def learning(self, game_data_file=None):
        all_game_datas = []
        if game_data_file:
            game_file = open(game_data_file)
            for i, line in enumerate(game_file.readlines()):
                #if i == 30000:
                #    break
                try:
                    if i % 1000 == 0 and i:
                        print('{} games has processed'.format(i))
                    all_game_datas.append(json.loads(line))
                except:
                    continue
            game_file.close()
        if self.model_id:
            self.set_players()
        experience_replay_length = int(self.game_num * 6)
        for num in range(self.iter_num):
            current_time = get_current_time()
            if (num == 1) and (not self.model_id):
                self.set_players()
            if self.continue_play:
                game_datas = self.run_games(self.game_num, current_time)
                all_game_datas.extend(game_datas)

            all_game_datas = all_game_datas[-experience_replay_length:]
            self.continue_play=True
            shuffle(all_game_datas)
            filter_training_data = self.making_filter_training_data(all_game_datas)
            self.optimize_filters(*filter_training_data)
            self.save_filters(current_time)
            filter_training_data.clear()

            training_datas = self.make_training_data(all_game_datas)
            self.optimize_networks(*training_datas)
            training_datas.clear()
            self.save_models(current_time)
            self.learning_rate_decay(num)

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
