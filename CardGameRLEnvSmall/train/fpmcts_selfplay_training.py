"""This scripts provides the self play process by using
the FPMCTS and ResidualPolicyValue.

Keys Points Refered in the AlphaZero Paper written by David Silver.etc
https://arxiv.org/pdf/1712.01815.pdf

    Exploration:
        I) During the Game, selfplay agent action a ~ pi(MCTS) where
            pi(a|MCTS) proportionally to the visit count at the root
            with the temperature(
            from 1(average, beiginning) -> 0(greedy) as described ).
            The pi(a|MCTS) would also be used in Neural Network training

        II) Dirichlet noise (Dir(alpha)) added to the root highly depends on
            the game size. The alpha is inverse proportion to the action space size
            (In Chess, Shogi, Go, alpha was set as 0.3, 0.2, 0.03)
        III) PUCT setting which is quite trival

    Learning Rate:
        Learning rate seted as 0.2 initially decreased three times. i.e. to
        0.02, 0.002, 0.0002 during the training process.


Self Play Pipeline

    Model Loading(If not loaded) ---> SelfPlay Game ---> Data Saving ---> Model Training ---> Model Saving
                |                  |                                  |                             |
                |                  |                                  |                             |
                |                  |------------Recent X games--------|                             |
                |                                                                                   |
                |                                                                                   |
                |-------------------------------------N iterations----------------------------------|
"""

import json
import sys
from itertools import cycle
from random import shuffle

import gflags

from agents.fpmcts_agent import FPMCTSPlayer
from pyminiddz.miniddz import GameState, GameStatePov
from pyminiddz.utils import check_wins
from pymodels.history.bayes_sample import Sampler as Bayesian_Sampler
from pymodels.nn_utils import NeuralNetBase
from pymodels.policy.residual_policy import ResidualPolicyValue
from train.utils import filesetting, get_current_time


def get_flags():

    FLAGS = gflags.FLAGS

    gflags.DEFINE_string("selfplay_fold", 'fpmcts_selfplay', 'selfplay path')
    gflags.DEFINE_string("model_id", '', 'self play model id')
    gflags.DEFINE_string("append_game_data", '', 'previous data included')

    gflags.DEFINE_float("c_puct", 2.8, "The exploration coefficient of PUCT")
    gflags.DEFINE_integer("playout_num", 200, 'simulation_num in MCTS')
    gflags.DEFINE_integer("samples_num", 20, 'samples number in MCTS root')
    gflags.DEFINE_float("history_temperature", 2.0, 'History temperature in Bayesian Sampler')

    gflags.DEFINE_integer("game_num", 10000, 'Game number in each iteration')
    gflags.DEFINE_integer("iter_num", 100, 'iteration number during selfplay')
    gflags.DEFINE_boolean('continue_play', False, 'Continue Break point training')

    gflags.DEFINE_float("learning_rate", 0.1, 'Learning rate for the DNN')
    gflags.DEFINE_integer("batch_size", 32, 'Batchsize for the DNN')
    gflags.DEFINE_integer("epochs", 2000, 'Training Epochs for the DNN')

    return FLAGS


class Trainer:
    """For FPMCTS pipeline
    """

    def __init__(self, selfplay_file='fpmcts_selfplay', model_id='',
                 c_puct=2.8, playout_num=200, samples_num=10,
                 history_temperature=0.2,
                 game_num=10000, iter_num=100,
                 epochs=2000, learning_rate=0.1, batch_size=32,
                 continue_play=False):
        """Initial Settings
        Params:
        ------
        model_id(str):
            The model id(time) of initial trainer

        """
        self._neural_base = NeuralNetBase()
        self.selfplay_file = selfplay_file
        self.model_id = model_id

        self.c_puct = c_puct
        self.playout_num = playout_num
        self.samples_num = samples_num

        self.history_temperature = history_temperature

        self.game_num = game_num
        self.iter_num = iter_num

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.continue_play = continue_play

        self.set_networks()
        self.set_sampler()
        self.set_dummy_players()

    def set_networks(self):
        if self.model_id:
            main_model_path = """./{0}/models/{1}/{2}.model"""
            main_weights_path = """./{0}/models/{1}/{2}.weights"""
            self.networks = [self._neural_base.load_model(
                             json_file=main_model_path.format(self.selfplay_file,
                                                              self.model_id,
                                                              player),
                             weights_file=main_weights_path.format(self.selfplay_file,
                                                                   self.model_id,
                                                                   player))
                             for player in 'CDE']
        else:
            self.networks = [ResidualPolicyValue() for _ in range(3)]

    def set_sampler(self):
        """set sampler for Player"""
        self.sampler = Bayesian_Sampler([network.get_policy for network in self.networks],
                                        history_temperature=self.history_temperature)

    def set_dummy_players(self):
        set_player_helper = (lambda pos:
                             FPMCTSPlayer(
                                 pos,
                                 c_puct=self.c_puct,
                                 playout_num=self.playout_num,
                                 samples_num=self.samples_num))

        self.players = [set_player_helper(idx) for idx in range(3)]

    def set_players(self):
        """set trainer players
        """

        set_player_helper = (lambda pos:
                             FPMCTSPlayer(
                                 pos,
                                 policy_value=self.networks[pos].get_policy_value,
                                 upper_policy=self.networks[(pos - 1) % 3].get_policy,
                                 lower_policy=self.networks[(pos + 1) % 3].get_policy,
                                 c_puct=self.c_puct,
                                 playout_num=self.playout_num,
                                 samples_num=self.samples_num,
                                 sampler=self.sampler))

        self.players = [set_player_helper(idx) for idx in range(3)]

    @staticmethod
    def game_state_temperature(game):
        """The temeperature varied with the game stage
        The temperature will decrease with game

        Params:
        ------
        game(GameState):
            The current game

        Returns:
        ------
        temperature(float):
            Temperature for the selfplay mcts selection
        """

        landlord_num = game.get_player_cards(0).sum()
        peasant_num = min(game.get_player_cards(1).sum(),
                          game.get_player_cards(2).sum())

        def helper(num):

            if num >= 7:
                return 1
            if num <= 3:
                return 0.01
            return (num - 3) * 0.99 / 4 + 0.01

        temperture = helper(landlord_num) * helper(peasant_num)

        return temperture

    def run_a_game(self):
        """Run a game with self-play-mcts
        """
        game_data = {"data" : []}
        game = GameState()

        state_povs = [GameStatePov(0), GameStatePov(1), GameStatePov(2)]
        player_seq = cycle(range(3))
        for player_idx in player_seq:
            current_player = self.players[player_idx]
            state_pov = state_povs[player_idx]
            state_pov.from_pure_state(game)
            _mcts = current_player.get_mcts()
            temperature = self.game_state_temperature(game)
            _mcts.set_temperature(temperature)
            move = current_player.get_move(state_pov)
            _policy = {move.get_str(): prob for
                       move, prob in _mcts.calc_policy().items()}
            game_copy = game.copy()
            scenario = {"state": state_pov.to_dict(),
                        "move": move.to_dict(),
                        "player": player_idx,
                        "policy": _policy,
                        "real_state": game_copy.to_dict()}
            game_data['data'].append(scenario)
            game.do_move(move)
            if game.is_end_of_game():
                break

        game_data['winner'] = int(game.get_winner())
        game_data['abs_score'] = float(abs(game.get_score(0)))

        return game_data

    def run_games(self, num, model_time):
        """Run several games for selfplay

        Params:
        ------
        num(Int):
            Game number selfplayed
        model_time(str):
            The model training time
        """

        game_datas = []
        filesetting('./{}/game_data'.format(self.selfplay_file))
        for n in range(num):
            if n and (n % 100 == 0):
                print('{} games has finished'.format(n))
            game_data = self.run_a_game()
            game_datas.append(game_data)
            with open('./{0}/game_data/{1}'.
                      format(self.selfplay_file, model_time), 'a') as game_file:
                json.dump(game_data, game_file, ensure_ascii=False)
                game_file.write('\n')

        return game_datas

    @staticmethod
    def parse_data(data):
        """Parse data to current training data

        Params:
        -----
        data(Dict):
            A recorded data, contains information of a GameStatePov

        Returns:
        ------
        states(List):
            A list of state_povs(GameStatePov)
        policies(List):
            A list of policies(dict) consist with states
        scores(List):
            A list of scores(float) that previous double effect was eliminated
        """

        records, abs_score, winner = data['data'], data['abs_score'], data['winner']
        result = [abs_score if check_wins(winner, pos)
                  else -abs_score
                  for pos in range(3)]
        state_bomb_score = lambda state: 2 ** state.get_bomb_num()

        empty_recorders = lambda: [[], [], []]
        states, policies, scores = [empty_recorders() for _ in range(3)]

        for rec in records:
            player_idx = rec['player']
            game_state = GameStatePov(rec['player'])
            game_state.from_dict(rec['state'])
            states[player_idx].append(game_state)

            policy = rec['policy']
            policies[player_idx].append(policy)

            score = result[rec['player']] / state_bomb_score(game_state)
            scores[player_idx].append(score)

        return states, policies, scores

    def make_training_data(self, game_datas):
        """Aggragate all game_datas

        Params:
        ------
        game_datas(List):
            a List of game datas(Dicts) of each is a game

        Returns:
        ------
        all_state(List):
            [List(C's state), List(D's state), List(E's state)]

        all_policies(List):
            [List(C's policy), List(D's policy), List(E's policy)]
            policies consist with above states

        all_scores(List):
            [List(C's scores), List(D's scores), List(E's scores)]
            scores consist with above states, previous double effect removed
        """

        empty_recorders = lambda: [[], [], []]
        all_states, all_policies, all_scores = [empty_recorders() for _ in range(3)]

        def _update(main_records, to_extend):
            for player, infos in enumerate(to_extend):
                main_records[player].extend(infos)

        for data in game_datas:
            states, policies, scores = self.parse_data(data)
            _update(all_states, states)
            _update(all_policies, policies)
            _update(all_scores, scores)

        return all_states, all_policies, all_scores

    def optimize_networks(self, all_states, all_policies, all_scores):
        """Optimize the neural networks contains

        Params:
        ------
        all_state(List):
            [List(C's state), List(D's state), List(E's state)]

        all_policies(List):
            [List(C's policy), List(D's policy), List(E's policy)]
            policies consist with above states

        all_scores(List):
            [List(C's scores), List(D's scores), List(E's scores)]
            scores consist with above states, previous double effect removed
        """

        for player_idx in range(3):
            self.networks[player_idx].fit(
                all_states[player_idx],
                all_policies[player_idx],
                all_scores[player_idx],
                epochs=self.epochs,
                batch_size=int(self.batch_size),
                shuffle=True,
                validation_split=0.1,
                verbose=2)

    def save_models(self, model_time):
        """Save models in sucn paths
        ./fpmcts_selfplay/models/time/{C, D, E}'

        Params:
        ------
        model_time(time):
            Model time assigned
        """

        path = './{0}/models/{1}'.format(self.selfplay_file, model_time)
        filesetting(path)
        def save_model_helper(network, path, pos):

            model_path = path + '/' + pos
            network.save_model(model_path + '.model', model_path + '.weights')

        for player_idx, pos in enumerate('CDE'):
            save_model_helper(self.networks[player_idx], path, pos)

    def learning_rate_decay(self, iter_num):
        """Learning rate decay during training process

        Params:
        ------
        iter_num(Int):
            learning rate determined by the iter_num,
            iter_num decays when the iter_num in the
            50%, 80%, 95%
        """

        if iter_num in [0,
                        int(0.5 * self.iter_num),
                        int(0.8 * self.iter_num),
                        int(0.95 * self.iter_num)]:
            if iter_num != 0:
                self.learning_rate /= 10
                for player_idx in range(3):
                    self.networks[player_idx].set_learning_rate(self.learning_rate)

    def learning(self, game_data_file=None):
        """Learning phase for this trainer

        Params:
        ------
        game_data_file(Str):
            if game_data_file was not none, includes game datas in that file
            Or just selfplay game data
        """
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
        experience_length = 6

        for num in range(self.iter_num):
            current_time = get_current_time()
            if num == 1:
                self.set_players()
            if self.continue_play or (not all_game_datas):
                game_datas = self.run_games(self.game_num, current_time)
                all_game_datas.extend(game_datas)
            all_game_datas = all_game_datas[-self.game_num * experience_length:]
            self.continue_play = True
            shuffle(all_game_datas)
            training_datas = self.make_training_data(all_game_datas)
            self.optimize_networks(*training_datas)
            self.save_models(current_time)
            training_datas = []
            self.learning_rate_decay(num)

    def learning_from_file(self, game_data_path):
        """Training model merely from file

        Params:
        ------
        game_data_path(Str):
            Game data file
        """
        all_game_datas = []
        with open(game_data_path) as game_file:
            for i, line in enumerate(game_file.readlines()):
                try:
                    print(i)
                    all_game_datas.append(json.loads(line))
                except:
                    continue
        shuffle(all_game_datas)
        training_datas = self.make_training_data(all_game_datas)
        self.optimize_networks(*training_datas)

        self.save_models('test_from_zero_mcts')

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
