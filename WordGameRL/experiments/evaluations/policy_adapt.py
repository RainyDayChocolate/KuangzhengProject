import json
import os
import random
from operator import itemgetter

import numpy as np
import pandas as pd
from scipy.spatial import distance

from experiments.utils import (deterministic_guessers, load_records,
                               play_a_word, topic_guesseres, giver12)
from experiments.single.pg_clue_topic import clue_topic_guessers
from src.agents.bert_agent import ActorCriticAgent

JSD = distance.jensenshannon
guessers = deterministic_guessers #  topic_guesseres# +
def extract_states(record):
    states = []
    traj = record['traj']
    for trace in traj:
        if trace['is_finished']:
            continue
        if trace['role'] != 'Giver':
            continue
        clues = trace['clues']
        if len(clues) >= 4:
            continue
        clues.append(trace['action'])
        states.append((trace['target'], clues, trace['guesses']))
    return states


def load_states(json_path):
    states = []
    records = load_records(json_path)
    for rec in records:
        rec_states = extract_states(rec)
        states.extend(rec_states)
    return states


def get_state_new_guesses(target, clues, guesses):
    new_guesses = set()
    new_states = []
    for _ in range(100):
        guesser = random.choice(guessers)
        guess = guesser.guess(clues, guesses)
        if guess not in new_guesses:
            gs = guesses.copy()
            gs.append(guess)
            new_guesses.add(guess)
            new_state = (clues, gs)
            new_states.append(new_state)
        if len(new_states) == 2:
            return {'target': target, 'arouse': new_states}

def get_state_sens(agent, new_guesses):
    agent.set_target(new_guesses['target'])
    state_one, state_two = new_guesses['arouse']
    try:
        _, policy_one = agent.give(state_one[0], state_one[1], to_train=False, only_policy=True)
        _, policy_two = agent.give(state_two[0], state_two[1], to_train=False, only_policy=True)
    except ValueError:
        import pdb; pdb.set_trace()
    return JSD(policy_one, policy_two)

def get_two_trace(target, giver):
    traces = []
    guesses_set = set()
    for _ in range(1000):
        guesser = random.choice(guessers)
        _, is_win, clues, guesses = play_a_word(target, giver, 
                                           guesser, to_train=False)
        print(is_win, clues, guesses)
        guesses = tuple(guesses)
        if guesses not in guesses_set:
            guesses_set.add(guesses)
            traces.append(clues)
            if len(traces) == 2:
                break
        
    return traces

def compare_traces(traces):
    if len(traces) != 2:
        return
    
    tas, tbs = traces
    diff = 0
    for ta, tb in zip(tas, tbs):
        if ta != tb:
            diff += 1
    return diff / len(tas)

def test_on_words(words, giver, guessers, to_print=False):
    all_turns, all_winloses = [], []
    random.seed(90)
    for word in words:
        guesser = random.choice(guessers)
        try:
            finished_turn, winlose, clues, guesses = play_a_word(word,
                                                    giver, 
                                                    guesser,
                                                    to_train=False)
        except ValueError:
            print("ERROR ", word)
        
        if winlose:
            all_turns.append(finished_turn)
        all_winloses.append(winlose)
        if to_print:
            print("Word: {}".format(word))
            print("Clues: {}".format(clues))
            print("Guesses: {}".format(guesses))
            print("Win Or Lose {}".format(winlose))
            #print(TO_PRINT_ATTENTION)
    win_rate = np.array(all_winloses).mean()
    if win_rate:
        avg_turns = np.array(all_turns).mean()
    else:
        avg_turns = 0
    to_print = [win_rate, avg_turns]
    print("Play with Topic Guessers, win_rate {:.2f}, avg_turn {:.2f}".format(*to_print))
    return win_rate, avg_turns

def main():
    path = './remote_models/single_pg_clue/giver'
    #path = './giver_60'
    #path = './single_pg_clue_topic_res/model_0803170750/giver_19'
    #path = './remote_models/single_pg/giver'
    is_restrict = 'naive' in path
    giver = ActorCriticAgent(player_tag='giver', 
                             is_giver=True,
                             restrict_value=is_restrict)
    giver.model.load_model(path)
    # states = load_states('./remote_models/testing.json')
    # new_states_pairs = []
    # for state in states:
    #     _new = get_state_new_guesses(*state)
    #     if not _new:
    #         continue
    #     new_states_pairs.append(_new)
    # random.seed(919)
    # random.shuffle(new_states_pairs)
    # new_states_pairs = new_states_pairs[:1000]
    # distances = []
    # for _new in new_states_pairs:
    #     dist = get_state_sens(giver, _new)
    #     if dist is not None:
    #         distances.append(dist)
    #     print(dist)

    index = list(range(500))
    all_words = giver.get_all_targets()
    targets = itemgetter(*index)(all_words)
    targets = random.sample(targets, 200)
    targets = [w[0] for w in targets]
    # diff_ratios = []
    # for target in targets:
    #     traces = get_two_trace(target, giver)
    #     diff_ratio = compare_traces(traces)
    #     if diff_ratio is not None:
    #         diff_ratios.append(diff_ratio)

    # distances = np.array(distances)
    # distances = distances[~np.isnan(distances)]
    # diff_ratios = np.array(diff_ratios)
    # print(distances.mean(), diff_ratios.mean())

    test_on_words(targets, giver, clue_topic_guessers)
    # test_on_words(targets, giver, deterministic_guessers)
    print(path)
    import pdb; pdb.set_trace()
if __name__ == '__main__':
    main()
