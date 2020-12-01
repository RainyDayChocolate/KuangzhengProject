"""This script shows the evaluation section for the FPMCTS selfplay

The ranking systems ultilized the True-Skill
Introduction:
https://zh.wikipedia.org/wiki/TrueSkill%E8%AF%84%E5%88%86%E7%B3%BB%E7%BB%9F
Trueskill Package:
https://trueskill.org/

Each model will played with his opponents in Head vs Head Format.
A_C VS B_DE and B_C VS A_DE, and then compute the score

And then, the score of the random-agent would be set to zero.

Excutation:

cd miniddz-env
export PYTHONPATH=.:$PYTHONPATH
python3 tools/selfplay_evaluation --main_file=['fpmcts_selfplay']

"""

import os
import pprint
import random
import sys
import time
from collections import defaultdict
from itertools import cycle

import gflags
import keras.backend as K
import numpy as np
import pandas as pd
from pandas import DataFrame, Series, read_csv
from trueskill import Rating, rate_1vs1

from agents.policy_agent import PolicyPlayer
from agents.random_agent import RandomPlayer
from pyminiddz.miniddz import GameState, GameStatePov
from train.utils import filesetting

gflags.DEFINE_list('main_folds', ['fpmcts_selfplay'], 'Files stored models')
gflags.DEFINE_string('ranking_file','./results/selfplay_policy_compare.csv', 'The result of ranking')

PLAYERS = 'CDE'

random_players = [RandomPlayer('C'), RandomPlayer('D'), RandomPlayer('E')]


def update_dict(dict_1, dict_2):

    for k, v in dict_2.items():
        dict_1[k] += v


def run_a_game(player_c, player_d, player_e, game=None):
    """Running one game with three given players
    """

    if not game:
        game = GameState()
    else:
        game = game.copy()
    players = [player_c, player_d, player_e]
    state_povs = [GameStatePov(0), GameStatePov(1), GameStatePov(2)]
    player_seq = cycle(range(3))
    for player_idx in player_seq:
        current_player = players[player_idx]
        state_pov = state_povs[player_idx]
        state_pov.from_pure_state(game)
        move = current_player.get_move(state_pov)
        game.do_move(move)
        if game.is_end_of_game():
            break

    winner = PLAYERS[game.get_winner()]
    score = {PLAYERS[idx]: game.get_score(idx) for idx in range(3)}
    return winner, score

def head_to_head(players_A, players_B, num=500):
    """
    Params:
    ------
    players_A(List):
        A 3-dimensioned list [C, D, E]
    players_B(List):
        A 3-dimensioned list [C, D, E]

    Returns:
    ------
    games_result(Dict):
        Final game result of the head 2 head method.
    """
    helper = lambda: defaultdict(lambda: Rating())
    #winners_A_vs_B, winners_B_vs_A = helper(), helper()
    landlord_win_num, peasant_win_num = 0, 0
    scores_A_vs_B, scores_B_vs_A = defaultdict(int), defaultdict(int)

    for _ in range(num):
        game = GameState()
        A_vs_B_winner, A_vs_B_score = run_a_game(players_A[0],
                                                 players_B[1], players_B[2], game)
        if A_vs_B_winner == 'C':
            landlord_win_num += 1
        update_dict(scores_A_vs_B, A_vs_B_score)

        B_vs_A_winner, B_vs_A_score = run_a_game(players_B[0],
                                                 players_A[1], players_A[2], game)
        if B_vs_A_winner != 'C':
            peasant_win_num += 1
        #winners_B_vs_A[B_vs_A_winner] += 1
        update_dict(scores_B_vs_A, B_vs_A_score)
    landlord_wp, peasant_wp = landlord_win_num / num, peasant_win_num / num
    return scores_A_vs_B, scores_B_vs_A, landlord_wp, peasant_wp


def choose_opponent_model(model_one, ratings):
    'Random Select one model to compare with model one'
    if model_one not in ratings:
        ratings[model_one] = Rating()
        return 'Random'
    score_diffs = [(model_two, abs(ratings[model_one].mu - ratings[model_two].mu))
                    for model_two in ratings
                    if (model_two != model_one)]

    rollout = np.random.rand()
    if rollout < 0.2:
        return random.choice(score_diffs)[0]

    score_diffs = sorted(score_diffs, key=lambda s: s[1])[:5]
    return random.choice(score_diffs)[0]


def create_agent(path):
    """Create models from based on the path
    """
    if not path:
        return random_players
    if pd.isnull(path):
        return random_players
    get_pos_agent = lambda pos: PolicyPlayer(policy_path='{0}/{1}'.
                                             format(path, pos))

    agents = [get_pos_agent(pos) for pos in 'CDE']

    return agents


def rating_change(p1, p2, score_diff, ratings):

    if p1 not in ratings:
        ratings[p1] = Rating()
    if p2 not in ratings:
        ratings[p2] = Rating()

    if score_diff > 0:
        ratings[p1], ratings[p2] = rate_1vs1(ratings[p1],
                                             ratings[p2])
    elif score_diff < 0:
        ratings[p2], ratings[p1] = rate_1vs1(ratings[p2],
                                             ratings[p1])
    else:
        ratings[p1], ratings[p2] = rate_1vs1(ratings[p1],
                                             ratings[p2],
                                             drawn=True)


def scores_from_files(file_path):
    """From Score to diff
    """

    ratings = {}
    ratings['Random'] = Rating()
    model_paths = {'Random': None}
    score_result = DataFrame()
    if not file_path:
        return score_result, ratings, model_paths

    score_result = read_csv(file_path)
    model_paths.update(dict(zip(score_result['1P'],
                           score_result['1P_path'])))
    model_paths.update(dict(zip(score_result['2P'],
                            score_result['2P_path'])))

    print('Loading file.....')
    for _, line in score_result.iterrows():
        p1, p2, score_diff = line['1P'], line['2P'], line['score_diff']
        rating_change(p1, p2, score_diff, ratings)

    print('Ratings are')
    pprint.pprint(ratings)
    return score_result, ratings, model_paths


def get_unevaluated_models(main_folds, evaluated_models):

    unevaluated_models = dict()
    for fold in main_folds:
        files = os.listdir('{}/models'.format(fold))
        for model_id in files:
            if len(model_id) != len('20190416-055006'):
                continue
            model_name = '{}_{}'.format(fold, model_id)
            if model_name in evaluated_models:
                continue
            model_path = '{}/models/{}'.format(fold, model_id)
            unevaluated_models[model_name] = model_path

    return unevaluated_models

FLAGS = gflags.FLAGS
def main(argv):

    FLAGS(argv)
    score_result, ratings, model_paths = scores_from_files(FLAGS.ranking_file)
    unevaluated_models = get_unevaluated_models(FLAGS.main_folds, model_paths)
    model_paths.update(unevaluated_models)
    print('Un-evaluated models are')
    pprint.pprint(unevaluated_models)
    unevaluated_models_ids = list(unevaluated_models.keys())
    model_ids = list(model_paths.keys())
    iter_num = max(50 * len(unevaluated_models_ids), 100)
    helper = 0

    def not_exist(model_id):
        if not isinstance(model_paths[model_id], str):
            return False
        return not os.path.exists(model_paths[model_id])

    while helper < iter_num:
        if unevaluated_models:
            model_one = random.choice(unevaluated_models_ids)
        else:
            model_one = random.choice(model_ids)
        model_two = choose_opponent_model(model_one, ratings)
        print(model_paths[model_one], model_paths[model_two])
        if not_exist(model_one) or not_exist(model_two):
            continue
        helper += 1
        print('+++++++++++++++++++++++++++++++++++++++')
        print('{0} VS {1}'.format(model_one, model_two))
        agents_A = create_agent(model_paths[model_one])
        agents_B = create_agent(model_paths[model_two])

        scores_A_vs_B, scores_B_vs_A, landlord_wp, peasants_wp = head_to_head(agents_A, agents_B)
        print("Result is {0} vs {1}".format(scores_A_vs_B, scores_B_vs_A))
        print('Total WP is {:.2%}'.format(landlord_wp + peasants_wp))
        print('---------------------------------------')
        score_diff = scores_A_vs_B['C'] - scores_B_vs_A['C']
        rating_change(model_one, model_two, score_diff, ratings)
        res_series = Series({'1P': model_one,
                             '2P': model_two,
                             '1P_path': model_paths[model_one],
                             '2P_path': model_paths[model_two],
                             'landlord_wp': landlord_wp,
                             'peasants_wp': peasants_wp,
                             'score_diff': score_diff})
        score_result = score_result.append(res_series, ignore_index=True)
        score_result = score_result[['1P', '2P', '1P_path', '2P_path',
                                     'landlord_wp', 'peasants_wp', 'score_diff']]
        score_result.to_csv('./results/selfplay_policy_compare.csv')
        K.clear_session()
        pprint.pprint(ratings)

if __name__ == '__main__':
    for _ in range(1000):
        main(sys.argv)
        time.sleep(3600)
