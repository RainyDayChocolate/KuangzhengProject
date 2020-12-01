import datetime
import json
import logging
import os
import pdb
import random
import urllib.request
from datetime import datetime
from operator import itemgetter

import numpy as np
import pandas as pd

from src.agents.agentutil import get_giver_score, get_guesser_scores
from src.agents.bert_agent import ActorCriticAgent, PGACAgent
from src.agents.guesser import MTWDetEvoGuesser, MTWDetEvoTopicGuesser
from src.agents.zoo import (giver12, giver21, giver22, giver31, giver_bert,
                            guesser11, guesser12, guesser12_family, guesser21,
                            guesser22, guesser22_family, guesser31, guesser32,
                            guesser32_family, guesser41, guesser42,
                            guesser42_family, guesser51, guesser52,
                            guesser52_family)
from src.games import game_util
from src.nn.bert_config import flags
from src.resource.evocation import free_association

topic_guesseres = []
topic_guesseres.extend(guesser12_family)
topic_guesseres.extend(guesser22_family)
topic_guesseres.extend(guesser32_family)
topic_guesseres.extend(guesser42_family)
topic_guesseres.extend(guesser52_family)


stochastic_guessers = [guesser11, guesser21, guesser31, guesser41, guesser51]
deterministic_guessers = [guesser12, guesser22, guesser32, guesser42, guesser52]
deterministic_givers = [giver12, giver21, giver22, giver31]
for i, guesser in enumerate(deterministic_guessers):
    guesser.set_name(str(i))

mean_guessers = [guesser31, guesser32]
scheduled_guessers = [ guesser11, guesser12, guesser21, guesser22, guesser41, guesser42, guesser51, guesser52]

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

TO_PRINT_ATTENTION = []

CURRENT_STATE = {'epoch': 0}
CURRENT_TIME = datetime.now()
CURRENT_TIME = CURRENT_TIME.strftime("%m%d%H%M%S")

TOPIC_CACHE = {}

def make_new_topic(target, strength_dict, topic_index=1, evo_data=free_association):
    if target in TOPIC_CACHE:
        topic_scores = TOPIC_CACHE[target]
    else:
        fsg = evo_data.normwords_fsg_dict
        bsg_dict = strength_dict[target]
        bsg_dict = {k: v 
                    for k, v in bsg_dict.items() 
                    if v < 1}
        bsg_dict = sorted(bsg_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        bsg_dict = dict(bsg_dict)
        topic_scores = {}
        for topic_cand, bsg_score in bsg_dict.items():
            
            corr = 0
            for corr_topic in bsg_dict:
                corr += (fsg[topic_cand].get(corr_topic, 0) + 
                        fsg[corr_topic].get(topic_cand, 0))
            topic_scores[topic_cand] = bsg_score / (corr + 0.001)
        # import pdb; pdb.set_trace()
        topic_scores = sorted(topic_scores.items(), 
                        key=lambda x: x[1], reverse=True)
        TOPIC_CACHE[target] = topic_scores
    
    return topic_scores[topic_index][0]


def get_normalized_value(strength, key):
    max_s = max(strength.values())
    return strength.get(key, 0) / max_s
    
def get_bsg_score(giver, clue):
    strength_dict = giver.strength_dict
    target = giver.target
    if clue not in strength_dict:
        return 0
    return get_normalized_value(strength_dict[clue], 
                                target)

def get_fsg_score(guesser, clue, guess):
    if clue not in guesser.strength_dict:
        return 0
    return get_normalized_value(guesser.strength_dict[clue], 
                                guess)

def load_records(json_path):
    records = []
    with open(json_path) as f:
        lines = f.readlines()
        records = [json.loads(line) for line in lines]
    return records

to_writes = {'traj': []}
def play_a_word(target, giver, guesser, to_train=True):
    
    clues, guesses = [],[]
    #if isinstance(giver, (ActorCriticAgent, PGACAgent)):
    giver.set_target(target)
    if (isinstance(guesser, MTWDetEvoTopicGuesser) and 
        isinstance(giver, (ActorCriticAgent, PGACAgent))):
        topic = make_new_topic(target, giver.strength_dict)
        guesser.set_topic(topic)
        guesser.set_avoid(target)
    res = None

    for i in range(5):
        if isinstance(giver, (ActorCriticAgent, PGACAgent)):
            if to_train:
                clue = giver.give(clues, guesses, to_train=to_train)
            else:
                clue, attention = giver.give(clues, guesses, to_train=to_train)
            
                score = get_bsg_score(giver, clue)
                to_write = CURRENT_STATE.copy()
                to_write['attention'] = attention
                to_write['role'] = 'Giver'
                to_write['clues'] = clues.copy()
                to_write['guesses'] = guesses.copy()
                to_write['turn'] = i + 1
                to_write['is_finished'] = res is not None
                to_write['target'] = giver.target
                to_write['score'] = score
                to_write['action'] = clue
                if isinstance(guesser, MTWDetEvoGuesser):
                    to_write['oppo'] = guesser.name
                else:
                    to_write['oppo'] = guesser.__class__.__name__
    
                to_writes['traj'].append(to_write)
                
        else:
            clue = giver.give(clues, guesses)
        
        if clue is not None:
            clues.append(clue)
        if isinstance(guesser, (ActorCriticAgent, PGACAgent)):
            if to_train:
                guess = guesser.guess(clues, guesses, to_train=to_train)
            else:
                guess, attention = guesser.guess(clues, guesses, 
                                                 to_train=to_train)
    
                score = get_fsg_score(guesser, clue, guess)
                to_write = CURRENT_STATE.copy()
                to_write['attention'] = attention
                to_write['role'] = 'Guesser'
                to_write['clues'] = clues.copy()
                to_write['guesses'] = guesses.copy()
                to_write['action'] = guess
                to_write['turn'] = i + 1
                to_write['is_finished'] = res is not None
                to_write['target'] = giver.target
                to_write['score'] = score
                to_write['oppo'] = giver.__class__.__name__

                to_writes['traj'].append(to_write)
        else:
            if not isinstance(guesser, MTWDetEvoTopicGuesser):
                guess = guesser.guess(clues, guesses)
            else:
                guess = guesser.guess_with_topic(clues, guesses)
        if guess is not None:
            guesses.append(guess)
            h = game_util.hit(guess, target)
            if h and res is None and not to_train:
                res = (i+1, True, clues, guesses)
            if h and to_train:
                return (i+1, True, clues, guesses)

    if res is None:
        res = (i+1, False, clues, guesses)
    if not to_train:
        to_writes['result'] = res
        to_writes['epoch'] = CURRENT_STATE['epoch']
    return res


def test_on_words(words, giver, guesser, 
                  to_print_traj=True, epoch=0, path=None):
    CURRENT_STATE['epoch'] = epoch
    all_turns, all_winloses = [], []
    #random.shuffle(words)
    words = words[:200]
    for word in words:
        try:
            finished_turn, winlose, clues, guesses = play_a_word(word,
                                                    giver, 
                                                    guesser,
                                                    to_train=False)
            if path:
                with open(path, 'a', encoding='utf-8') as f:
                    json.dump(to_writes, f, ensure_ascii=False)
                    f.write('\n')
                to_writes.clear()
                to_writes['traj'] = []
        except ValueError:
            print("ERROR ", word)
        except KeyError:
            continue
        
        if winlose:
            all_turns.append(finished_turn)
        all_winloses.append(winlose)
        if to_print_traj:#and finished_turn >= 2:
            print("Word: {}".format(word))
            print("Clues: {}".format(clues))
            print("Guesses: {}".format(guesses))
            print("Win Or Lose {}".format(winlose))
            #print(TO_PRINT_ATTENTION)
            
        TO_PRINT_ATTENTION.clear()
    win_rate = np.array(all_winloses).mean()
    if win_rate:
        avg_turns = np.array(all_turns).mean()
    else:
        avg_turns = 0
    to_print = [giver.__class__.__name__,
                guesser.__class__.__name__,
                win_rate,
                avg_turns]
    try:
        print("{} play with {}, win_rate {:.2f}, avg_turn {:.2f}".format(*to_print))
    except:
        import pdb; pdb.set_trace()
    return win_rate, avg_turns


if __name__ == '__main__':
    from src.nn.models.model_utils import BasicNNModel

    bm = BasicNNModel(player_tag='filter')
    vocab = bm.vocab_words
    vocab = set(vocab)
    import pdb; pdb.set_trace()
