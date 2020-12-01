import logging
import os
import random
import time
from datetime import datetime
from operator import itemgetter

import numpy as np
import tensorflow as tf

from pandas import DataFrame
from src.agents.agentutil import get_giver_score, get_guesser_scores
from src.agents.zoo import (giver6, giver12,
                            giver22, giver31, giver32, giver_bert, guesser11,
                            guesser12, guesser21, guesser22, guesser31,
                            guesser32, guesser41, guesser42, guesser51,
                            guesser52)
from src.games import game_util
from src.nn.bert_config import flags

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



stochastic_guessers = [guesser11, guesser21, guesser31, guesser41, guesser51]
deterministic_guessers = [guesser12, guesser22, guesser32, guesser42, guesser52]

mean_guessers = [guesser31, guesser32]
scheduled_guessers = [ guesser11, guesser12, guesser21, guesser22, guesser41, guesser42, guesser51, guesser52]

FLAGS = flags.FLAGS

CURRENT_TIMESTAMP = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def play_a_word(target, giver, guesser, userrole=None, to_train=True):
    giver.set_target(target)
    clues, guesses = [],[]
    # print("Target word is {}".format(target))

    for i in range(5):
        if 'BERT' in giver.__class__.__name__:
            clue = giver.give(clues, guesses, to_train=to_train) if userrole != 'giver' else input('>')
        else:
            clue = giver.give(clues, guesses) if userrole != 'giver' else input('>')
        #clue = giver.give(clues,guesses) if userrole != 'giver' else input('>')
        if clue is not None:
            clues.append(clue)
        # if not to_train:
        #     logging.debug('---turn {}'.format(i))
        #     logging.debug('giver: "{}"'.format(clue))
        #     print("Giver gives the clue {}".format(clue))
        if 'BERT' in guesser.__class__.__name__:
            guess = guesser.guess(clues, guesses, 
                               to_train=to_train) if userrole != 'guesser' else input('>')
        else:
            guess = guesser.guess(clues, guesses) if userrole != 'guesser' else input('>')
        
        if guess is not None:
            guesses.append(guess)
        # if not to_train:
        #     print("Guesser guess {}".format(guess))

        #     logging.debug('guesser: "{}"'.format(guess))
        if guess is not None:
            
            #h, s = game_util.hit(guess, target)
            h = game_util.hit(guess, target)
            if h:
                logging.debug('win!')
                return i+1, True, clues, guesses
    return i+1, False, clues, guesses

def game_analysis(epoch, giver, guesser, target, clues, guesses, winlose):
    analysis = {}
    analysis['epoch'] = epoch
    analysis['target'] = target
    analysis['game_len'] = len(clues)
    analysis['winlose'] = winlose
    
    clues_qualities = get_giver_score(giver, target, clues)
    clues_qualities = np.array([round(v, 2) for v in clues_qualities])
    guesses_qualities = get_guesser_scores(guesser, clues, guesses)
    guesses_qualities = np.array([round(v, 2) for v in guesses_qualities])

    analysis['clue_quality'] = clues_qualities.mean()
    analysis['guess_quality'] =  guesses_qualities.mean()
    analysis['clues'] = '|'.join(clues)
    analysis['guesses'] = '|'.join(guesses)
    analysis['clues_qualities'] = str(list(clues_qualities))
    analysis['guesses_qualities'] = str(list(guesses_qualities))
    return analysis


def run_policy_gradient_guesser_train(giver, bert_guesser, words_indexes, args, to_train=True):
    """Test Guesser 1 vs 1. Giver
    """
    allturns, allwinlose = [], []
    played_targets = []
    
    bert_guesser.set_possile_target(dict((giver.get_all_targets()[:2000])))
    for wi, (target, _) in enumerate(itemgetter(*words_indexes)(giver.get_all_targets())):
        if to_train and (np.random.rand() > 0.1):
            continue
        turns, winlose, clues, guesses = play_a_word(
            target,
            giver,
            bert_guesser,
            userrole=args.userrole,
            to_train=to_train
        )

        if not to_train:
            #print('target is {}'.format(target))
            print("""playing word {} --- {}\nclues are: {}\nguesses are {}""".format(
                wi, 
                '' if args.userrole == 'guesser' else target,
                clues, guesses))
            
        if to_train:
            bert_guesser.update_memory(target, clues, guesses, winlose)
            bert_guesser.train_memory_with_evocation()
            bert_guesser.memory = []
    
        allwinlose.append(int(winlose))
        if winlose:
            allturns.append(turns)

        played_targets.append(target)
        
    if not to_train:
        logging.info('bert guessers id: win rate: {}, avg # win turns : {}'.format(
            sum(allwinlose) / len(allwinlose),
            sum(allturns) / (len(allturns) + 0.001))
        )

    return played_targets

def run_single_DQN_exp(giver, guesser, words_indexes, args, epoch, to_train=True):

    allturns, allwinlose = [], []
    played_targets = []
    guesser.set_possile_target(dict((guesser.get_all_targets()[:2000])))
    if to_train:
        candidates = list(enumerate(itemgetter(*words_indexes)(guesser.get_all_targets())))
        for iter_num in range(1000):
            wi, (target, _) = random.choice(candidates)
            turns, winlose, clues, guesses = play_a_word(
                target,
                giver,
                guesser,
                userrole=args.userrole,
                to_train=to_train
            )       
            if (iter_num % 100 == 99):
                print("{} games have been simulated".format(iter_num))
            guesser.update_memory(target, clues, guesses, winlose)
        print("Start training with memory")
        guesser.train_memory_with_evocation()
    else:
        results = []
        for  wi, (target, _) in enumerate(itemgetter(*words_indexes)(guesser.get_all_targets())):
            guesser.model.to_print_attention = True
            turns, winlose, clues, guesses = play_a_word(
                target,
                giver,
                guesser,
                userrole=args.userrole,
                to_train=to_train)
            guesser.model.to_print_attention = False
            print("""playing word {} --- {}\nclues are: {}\nguesses are {}""".format(
                wi, 
                '' if args.userrole == 'guesser' else target,
                clues, guesses))
        
            allwinlose.append(int(winlose))
            if winlose:
                allturns.append(turns)

            played_targets.append(target)
            res = game_analysis(epoch=epoch, target=target, 
                                giver=giver, guesser=guesser, 
                                clues=clues, guesses=guesses, 
                                winlose=winlose)
            results.append(res)
        
        # Write to csv
        to_path = "./result/DQN_with_fsg_{}.csv".format(CURRENT_TIMESTAMP)
    
        results = DataFrame(results)
        if os.path.exists(to_path):
            results.to_csv(to_path, mode='a')
        else:
            results.to_csv(to_path)

        print('DQN guessers id: win rate: {}, avg # win turns : {}'.format(
            sum(allwinlose) / len(allwinlose),
            sum(allturns) / (len(allturns) if allturns else 100))
        )

    return played_targets

def run_single_DQN_giver_exp(giver, guesser, words_indexes, args, epoch, to_train=True):

    allturns, allwinlose = [], []
    played_targets = []
    giver.set_possile_target(dict((giver.get_all_targets()[:2000])))
    if to_train:
        candidates = list(enumerate(itemgetter(*words_indexes)(giver.get_all_targets())))
        for iter_num in range(1000):
            wi, (target, _) = random.choice(candidates)
            turns, winlose, clues, guesses = play_a_word(
                target,
                giver,
                guesser,
                userrole=args.userrole,
                to_train=to_train
            )       
            if (iter_num % 100 == 99):
                print("{} games have been simulated".format(iter_num))
            giver.update_memory(target, clues, guesses, winlose)
        print("Start training with memory")
        giver.train_memory_with_evocation()
    else:
        results = []
        for  wi, (target, _) in enumerate(itemgetter(*words_indexes)(giver.get_all_targets())):
            giver.model.to_print_attention = True
            turns, winlose, clues, guesses = play_a_word(
                target,
                giver,
                guesser,
                userrole=args.userrole,
                to_train=to_train)
            giver.model.to_print_attention = False
            print("""playing word {} --- {}\nclues are: {}\nguesses are {}""".format(
                wi, 
                '' if args.userrole == 'guesser' else target,
                clues, guesses))
        
            allwinlose.append(int(winlose))
            if winlose:
                allturns.append(turns)

            played_targets.append(target)
        
        # Write to csv
        to_path = "./result/DQN_with_fsg_{}.csv".format(CURRENT_TIMESTAMP)
    
        results = DataFrame(results)
        if os.path.exists(to_path):
            results.to_csv(to_path, mode='a')
        else:
            results.to_csv(to_path)

        print(' win rate: {}, avg # win turns : {}'.format(
            sum(allwinlose) / len(allwinlose),
            sum(allturns) / (len(allturns) if allturns else 100))
        )

    return played_targets

def run_DQN_exp(giver, guesser, words_indexes, args, epoch, to_train=True):

    allturns, allwinlose = [], []
    played_targets = []
    giver.set_possile_target(dict((giver.get_all_targets()[:2000])))
    guesser.set_possile_target(dict((guesser.get_all_targets()[:2000])))
    if to_train:
        candidates = list(enumerate(itemgetter(*words_indexes)(giver.get_all_targets())))

        for iter_num in range(1000):
            wi, (target, _) = random.choice(candidates)
            turns, winlose, clues, guesses = play_a_word(
                target,
                giver,
                guesser,
                userrole=args.userrole,
                to_train=to_train
            )       
            if (iter_num % 100 == 99):
                print("{} games have been simulated".format(iter_num))
            giver.update_memory(target, clues, guesses, winlose)
            guesser.update_memory(target, clues, guesses, winlose)
        print("Start training with memory")
        giver.train_memory_with_evocation()
        guesser.train_memory_with_evocation()
    else:

        results = []
        giver.model.to_print_attention = True
        guesser.model.to_print_attention = True

        for  wi, (target, _) in enumerate(itemgetter(*words_indexes)(giver.get_all_targets())):
            turns, winlose, clues, guesses = play_a_word(
                target,
                giver,
                guesser,
                userrole=args.userrole,
                to_train=to_train)
            print("""playing word {} --- {}\nclues are: {}\nguesses are {}""".format(
                wi, 
                '' if args.userrole == 'guesser' else target,
                clues, guesses))
        
            allwinlose.append(int(winlose))
            if winlose:
                allturns.append(turns)

            played_targets.append(target)
            res = game_analysis(epoch=epoch, target=target, 
                                giver=giver, guesser=guesser, 
                                clues=clues, guesses=guesses, 
                                winlose=winlose)
            results.append(res)
        
        # Write to csv
        to_path = "./result/DQN_with_fsg_{}.csv".format(CURRENT_TIMESTAMP)
    
        results = DataFrame(results)
        if os.path.exists(to_path):
            results.to_csv(to_path, mode='a')
        else:
            results.to_csv(to_path)

        print('DQN guessers id: win rate: {}, avg # win turns : {}'.format(
            sum(allwinlose) / len(allwinlose),
            sum(allturns) / (len(allturns) if allturns else 100))
        )
        giver.model.to_print_attention = False
        guesser.model.to_print_attention = False
    return played_targets

def run_exp(guessers, giver, index, args, step, train_giver):
    ids = ['{} with {}'.format(guesser.__class__ , guesser.strategy.__name__) for guesser in guessers]

    allturns = [[] for i in guessers]
    allwinlose = [[] for i in guessers]
    played_targets = []
    for wi, (target, _) in enumerate(itemgetter(*index)(giver.get_all_targets())):
        logging.info('--- playing word {} --- {}'.format(
            wi, '' if args.userrole == 'guesser' else target))
        played_targets.append(target)
        for id, guesser in enumerate(guessers):
            turns, winlose, clues, guesses = play_a_word(
                target,
                giver,
                guesser,
                userrole=args.userrole,
            )
            allwinlose[id].append(int(winlose))
            if winlose:
                allturns[id].append(turns)
            logging.debug('target is {}'.format(target))
            total_reward = giver._total_reward((target, clues, guesses, winlose))
            logging.debug('total reward is {}'.format(total_reward))
            giver.update_memory((target, clues, guesses, winlose, total_reward))
            if step % FLAGS.learn_every == 0:
                if train_giver:
                    giver.train_memory_with_evocation()
            step += 1

    for winlose, turns, id in zip(allwinlose, allturns, ids):
        logging.info('{} guessers id {}: win rate: {}, avg # win turns : {}'.format(
            'Train ' if train_giver else "Evaluate",
            id,
            sum(winlose) / len(winlose),
            sum(turns) / len(turns))
        )

    return played_targets


class Args():
    def __init__(self):
        self.userrole=None # 'guesser'
        self.epochs=100
        self.verbose=False
        self.train=True

def main(_):
    logging.basicConfig(filename=FLAGS.logger_file,level=logging.DEBUG)

    '''
    parser = ArgumentParser()
    parser.add_argument('--role', type=str, default=None, help='role type, giver/guesser. If not specified, auto-play')
    parser.add_argument('--epochs', type=int, default=500, help='auto-play default to 500. Use 1 if human is involved')
    parser.add_argument('--verbose', action='store_true', help='verbose if play interactively with human.')
    parser.add_argument('--train', action='store_true', help='verbose if play interactively with human.')
    args = parser.parse_args()
    '''
    from src.nn.guesser_model import GuesserDQNModel

    args = Args()
    index = range(500)
#    index = list(range(1000,1100))
#    index = list(range(0,1000))
#    index += [ i for i in range(50,100) if i%5==0]
#    index += [i for i in range(100,1000) if i %10==0]

    #giver = giver_bert # giver31
    giver = giver12
    # train_guessers = deterministic_guessers #stochastic_guessers #deterministic_guessers # [guesser32]
    # test_guessers = deterministic_guessers #stochastic_guessers

    # step = 0
    to_path = "./result/DQN_with_fsg_{}.csv".format(CURRENT_TIMESTAMP)
    if os.path.exists(to_path):
        os.remove(to_path)
    #giver = bert_dqn_giver
    #guesser = guesser12
    #bert_dqn_guesser = bert_dqn_guesser

    for epoch in range(args.epochs):

        logging.info('epoch {}'.format(epoch))
        run_single_DQN_exp(
            guesser=guesser,
            giver=giver,
            words_indexes=range(50),
            args=args,
            to_train=False,
            epoch=epoch)

        run_single_DQN_exp(
            guesser=guesser,
            giver=giver,
            words_indexes=range(200),
            args=args,
            to_train=True,
            epoch=epoch)
        

if __name__=='__main__':

    tf.app.run()
